"""
ConvoFuse REST API Endpoints (Part 6.2).

Extends the Phase20 FastAPI backend with dedicated ConvoFuse endpoints for:
- Job submission and 15-stage status polling
- Recording metadata & processing manifest retrieval
- Complete conversational transcript retrieval
- Word-level JSON retrieval (with baseline vs refined speakers, overlap metadata, TPST decisions)
- Chronological turn-level JSON retrieval
- Speaker summary index JSON retrieval
- Post-inference evaluation & baseline-vs-refined comparison retrieval
- Multi-format TXT / JSON export downloads
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from backend.services.convofuse_service import BENCHMARK_RECORDING_MAP, ConvoFuseService
from backend.services.job_service import JobService

router = APIRouter()


class ConvoFuseJobRequest(BaseModel):
    recording_id: str = Field(..., description="Recording ID or benchmark short ID (e.g., 'R01' or 'R01_sara_sharif')")
    execution_mode: str = Field(
        default="FROZEN_INPUT_REPLAY",
        description="Execution mode: 'FROZEN_INPUT_REPLAY' or 'FULL_PROCESSING'",
    )
    condition_preference: str = Field(
        default="auto",
        description="Diarization condition preference ('auto', 'E20', 'Condition_J', 'Condition_K')",
    )
    enable_refinement: bool = Field(
        default=True,
        description="Enable ground-truth-isolated DiarizationLM-Gemma generative refinement",
    )
    use_locality_gating: bool = Field(
        default=True,
        description="Enable Locality-Gated TPST (LG-TPST) span gating",
    )
    sync: bool = Field(
        default=False,
        description="Wait for job completion synchronously before returning response",
    )


@router.get("/benchmarks")
def list_convofuse_benchmarks() -> Dict[str, Any]:
    """Return available benchmark recordings (R01-R08) for frozen-input replay."""
    return {
        "benchmarks": ConvoFuseService.list_benchmark_recordings(),
    }


@router.post("/jobs")
def create_convofuse_job(req: ConvoFuseJobRequest) -> Dict[str, Any]:
    """Start a ConvoFuse processing or replay job."""
    try:
        job = ConvoFuseService.submit_convofuse_job(
            recording_id=req.recording_id,
            execution_mode=req.execution_mode,
            condition_preference=req.condition_preference,
            enable_refinement=req.enable_refinement,
            use_locality_gating=req.use_locality_gating,
            sync=req.sync,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ConvoFuse job failed: {exc}") from exc

    return {
        "job_id": job.job_id,
        "recording_id": job.recording_id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@router.get("/jobs/{job_or_rec_id}/status")
def get_convofuse_job_status(job_or_rec_id: str) -> Dict[str, Any]:
    """Retrieve ConvoFuse job status and 15-stage pipeline history."""
    canonical_id = BENCHMARK_RECORDING_MAP.get(job_or_rec_id, job_or_rec_id)
    job = JobService.get_job(job_or_rec_id) or JobService.get_job_by_recording(canonical_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"No ConvoFuse job found for '{job_or_rec_id}'.")

    manifest = ConvoFuseService.get_artifact_json(job.recording_id, "processing_manifest")
    stage_history = manifest.get("stage_history", []) if isinstance(manifest, dict) else []

    return {
        "job_id": job.job_id,
        "recording_id": job.recording_id,
        "status": job.status,
        "progress": job.progress,
        "stage": job.stage,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "stage_history": stage_history,
    }


@router.get("/{recording_id}/metadata")
def get_convofuse_metadata(recording_id: str) -> Dict[str, Any]:
    """Retrieve ConvoFuse processing manifest and audio/model provenance."""
    manifest = ConvoFuseService.get_artifact_json(recording_id, "processing_manifest")
    if not manifest:
        raise HTTPException(
            status_code=404,
            detail=f"ConvoFuse processing manifest not found for '{recording_id}'.",
        )
    return manifest


@router.get("/{recording_id}/transcript")
def get_convofuse_transcript(recording_id: str) -> Dict[str, Any]:
    """Retrieve the complete ConvoFuse conversational transcript bundle."""
    bundle = ConvoFuseService.get_artifact_json(recording_id, "bundle")
    txt = ConvoFuseService.get_readable_txt(recording_id)
    if not bundle:
        raise HTTPException(
            status_code=404,
            detail=f"ConvoFuse transcript bundle not found for '{recording_id}'.",
        )
    out = dict(bundle)
    if txt is not None:
        out["readable_transcript_txt"] = txt
    return out


@router.get("/{recording_id}/words")
def get_convofuse_words(recording_id: str) -> Dict[str, Any]:
    """Retrieve word-level JSON with baseline vs refined speakers, overlap metadata, and TPST decisions."""
    words_data = ConvoFuseService.get_artifact_json(recording_id, "words")
    if not words_data:
        raise HTTPException(
            status_code=404,
            detail=f"ConvoFuse word-level JSON not found for '{recording_id}'.",
        )
    return words_data


@router.get("/{recording_id}/turns")
def get_convofuse_turns(recording_id: str) -> Dict[str, Any]:
    """Retrieve chronological turn-level JSON."""
    turns_data = ConvoFuseService.get_artifact_json(recording_id, "turns")
    if not turns_data:
        raise HTTPException(
            status_code=404,
            detail=f"ConvoFuse turn-level JSON not found for '{recording_id}'.",
        )
    return turns_data


@router.get("/{recording_id}/speakers")
def get_convofuse_speakers(recording_id: str) -> Dict[str, Any]:
    """Retrieve speaker summary index JSON."""
    speakers_data = ConvoFuseService.get_artifact_json(recording_id, "speakers")
    if not speakers_data:
        raise HTTPException(
            status_code=404,
            detail=f"ConvoFuse speaker summary JSON not found for '{recording_id}'.",
        )
    return speakers_data


@router.get("/{recording_id}/evaluation")
def get_convofuse_evaluation(recording_id: str) -> Dict[str, Any]:
    """Retrieve post-inference evaluation report and baseline-vs-refined comparison."""
    eval_data = ConvoFuseService.get_artifact_json(recording_id, "evaluation_report")
    comp_data = ConvoFuseService.get_artifact_json(recording_id, "comparison")
    if not eval_data and not comp_data:
        raise HTTPException(
            status_code=404,
            detail=f"ConvoFuse evaluation/comparison report not found for '{recording_id}'.",
        )
    return {
        "recording_id": BENCHMARK_RECORDING_MAP.get(recording_id, recording_id),
        "evaluation_report": eval_data,
        "baseline_vs_refined_comparison": comp_data,
    }


@router.get("/{recording_id}/export")
def export_convofuse_artifact(
    recording_id: str,
    format: str = Query(
        default="txt",
        description="Export format: txt, json, words, turns, speakers, comparison, manifest, evaluation",
    ),
) -> Response:
    """Download ConvoFuse transcript or structured artifacts in TXT or JSON format."""
    canonical_id = BENCHMARK_RECORDING_MAP.get(recording_id, recording_id)
    fmt = format.lower().strip()

    if fmt == "txt":
        txt = ConvoFuseService.get_readable_txt(canonical_id)
        if txt is None:
            raise HTTPException(status_code=404, detail=f"Readable TXT not found for '{recording_id}'.")
        return PlainTextResponse(
            content=txt,
            headers={"Content-Disposition": f'attachment; filename="{canonical_id}__transcript.txt"'},
        )

    suffix_map = {
        "json": "bundle",
        "bundle": "bundle",
        "words": "words",
        "turns": "turns",
        "speakers": "speakers",
        "comparison": "comparison",
        "manifest": "processing_manifest",
        "evaluation": "evaluation_report",
    }
    if fmt not in suffix_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Supported: txt, json, words, turns, speakers, comparison, manifest, evaluation.",
        )

    suffix = suffix_map[fmt]
    data = ConvoFuseService.get_artifact_json(canonical_id, suffix)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Artifact '{suffix}' not found for '{recording_id}'.")

    return Response(
        content=json.dumps(data, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{canonical_id}__{suffix}.json"'},
    )
