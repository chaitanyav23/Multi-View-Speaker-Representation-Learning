"""Developer API and Plugin Integration Endpoints."""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException

from backend.database.models import (
    JobRepository,
    PluginAnalysisRecord,
    PluginRepository,
    RecordingRepository,
)
from backend.plugins.registry import PluginRegistry
from backend.services.feedback_service import FeedbackService
from backend.services.job_service import JobService
from backend.services.transcription_service import TranscriptionService
from engine.phase20_engine import sanitize_credential_text
from engine.schemas import (
    JobStatusResponse,
    PluginAnalysisResult,
    PluginContextResponse,
    PluginManifest,
    PluginRunRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def build_recording_context(recording_id: str) -> PluginContextResponse:
    """Consolidate recording metadata, transcript, turns, and vocabulary into a typed context."""
    record = RecordingRepository.get(recording_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Recording '{recording_id}' not found.")

    transcript = TranscriptionService.get_transcript(recording_id)
    job = JobService.get_job_by_recording(recording_id)
    vocab = FeedbackService.get_accumulated_vocabulary()

    if transcript:
        words = transcript.words
        mean_conf = (
            round(sum(w.confidence for w in words) / len(words), 3)
            if words
            else 0.0
        )
        return PluginContextResponse(
            recording_id=record.recording_id,
            filename=record.filename,
            duration_seconds=record.duration,
            status="ready",
            has_transcript=True,
            transcript=transcript,
            speakers=transcript.speakers,
            total_words=len(words),
            mean_confidence=mean_conf,
            turns=transcript.segments,
            vocabulary=vocab,
            metadata={
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
                "created_at": record.created_at,
            },
        )
    else:
        status = "untranscribed"
        if job and job.status in ("queued", "processing"):
            status = job.status
        elif job and job.status == "failed":
            status = "failed"

        return PluginContextResponse(
            recording_id=record.recording_id,
            filename=record.filename,
            duration_seconds=record.duration,
            status=status,
            has_transcript=False,
            transcript=None,
            speakers=[],
            total_words=0,
            mean_confidence=None,
            turns=[],
            vocabulary=vocab,
            metadata={
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
                "created_at": record.created_at,
            },
        )


@router.get(
    "/recordings/{recording_id}/context",
    response_model=PluginContextResponse,
    tags=["Developer API"],
)
def get_recording_context(recording_id: str) -> PluginContextResponse:
    """Retrieve consolidated, typed recording and transcript context for plugins."""
    return build_recording_context(recording_id)


@router.get(
    "/plugins",
    response_model=List[PluginManifest],
    tags=["Developer API", "Plugins"],
)
def list_plugins() -> List[PluginManifest]:
    """List manifests of all registered and enabled plugins."""
    return PluginRegistry.list_manifests()


@router.post(
    "/plugins/{plugin_id}/runs",
    response_model=PluginAnalysisResult,
    tags=["Developer API", "Plugins"],
)
def run_plugin(plugin_id: str, request: PluginRunRequest) -> PluginAnalysisResult:
    """Execute a trusted plugin analysis against a recorded transcript."""
    plugin = PluginRegistry.get(plugin_id)
    if not plugin:
        raise HTTPException(
            status_code=404, detail=f"Plugin '{plugin_id}' not registered."
        )

    record = RecordingRepository.get(request.recording_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Recording '{request.recording_id}' not found.",
        )

    context = build_recording_context(request.recording_id)
    if not context.has_transcript or not context.transcript:
        raise HTTPException(
            status_code=400,
            detail=f"Recording '{request.recording_id}' has not been transcribed yet. Plugins require a completed transcript.",
        )

    start_time = time.perf_counter()
    now_iso = datetime.now(timezone.utc).isoformat()
    result_id = str(uuid.uuid4())

    try:
        raw_result = plugin.run(context, request.parameters)
        execution_duration = round(time.perf_counter() - start_time, 4)

        if not isinstance(raw_result, dict):
            raise TypeError(
                f"Plugin '{plugin_id}' returned invalid type {type(raw_result).__name__}; expected dict."
            )

        # Ensure result is serializable
        json_payload = json.dumps(raw_result)

        db_record = PluginAnalysisRecord(
            result_id=result_id,
            plugin_id=plugin_id,
            recording_id=request.recording_id,
            status="completed",
            result_json=json_payload,
            plugin_version=plugin.manifest.version,
            created_at=now_iso,
            execution_time_seconds=execution_duration,
            error=None,
        )
        PluginRepository.create(db_record)

        return PluginAnalysisResult(
            result_id=result_id,
            plugin_id=plugin_id,
            recording_id=request.recording_id,
            status="completed",
            result_json=raw_result,
            plugin_version=plugin.manifest.version,
            created_at=now_iso,
            execution_time_seconds=execution_duration,
            error=None,
        )

    except HTTPException:
        raise
    except Exception as exc:
        execution_duration = round(time.perf_counter() - start_time, 4)
        sanitized_error = sanitize_credential_text(str(exc))
        logger.error(
            "Plugin execution error in %s for recording %s: %s",
            plugin_id,
            request.recording_id,
            sanitized_error,
            exc_info=True,
        )

        db_record = PluginAnalysisRecord(
            result_id=result_id,
            plugin_id=plugin_id,
            recording_id=request.recording_id,
            status="failed",
            result_json=json.dumps({"error": sanitized_error}),
            plugin_version=plugin.manifest.version,
            created_at=now_iso,
            execution_time_seconds=execution_duration,
            error=sanitized_error,
        )
        PluginRepository.create(db_record)

        raise HTTPException(
            status_code=500,
            detail=f"Plugin '{plugin_id}' execution failed: {sanitized_error}",
        )


@router.get(
    "/plugins/{plugin_id}/results/{recording_id}",
    response_model=PluginAnalysisResult,
    tags=["Developer API", "Plugins"],
)
def get_plugin_result(
    plugin_id: str, recording_id: str
) -> PluginAnalysisResult:
    """Retrieve the latest computed plugin result for a recording."""
    record = PluginRepository.get_latest(plugin_id, recording_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for plugin '{plugin_id}' on recording '{recording_id}'.",
        )

    try:
        parsed_json = json.loads(record.result_json)
    except Exception:
        parsed_json = {"raw": record.result_json}

    return PluginAnalysisResult(
        result_id=record.result_id,
        plugin_id=record.plugin_id,
        recording_id=record.recording_id,
        status=record.status,
        result_json=parsed_json,
        plugin_version=record.plugin_version,
        created_at=record.created_at,
        execution_time_seconds=record.execution_time_seconds,
        error=record.error,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
)
def get_job_by_id(job_id: str) -> JobStatusResponse:
    """Unambiguously retrieve job progress and status strictly by job ID."""
    job = JobService.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' not found."
        )

    return JobStatusResponse(
        job_id=job.job_id,
        recording_id=job.recording_id,
        status=job.status,
        progress=job.progress,
        stage=job.stage,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
