"""Speech Transcription Dispatch, Job Status, and Retrieval Endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse

from backend.database.models import RecordingRepository
from backend.services.export_service import ExportService
from backend.services.job_service import JobService
from backend.services.transcription_service import TranscriptionService
from engine.schemas import JobStatusResponse, TranscriptionJobRequest, TranscriptResult

router = APIRouter()


@router.post("/transcriptions", response_model=JobStatusResponse, status_code=202, tags=["Transcriptions"])
def trigger_transcription(request: TranscriptionJobRequest) -> JobStatusResponse:
    """Dispatch an asynchronous speech transcription and diarization job."""
    record = RecordingRepository.get(request.recording_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Recording '{request.recording_id}' not found.")

    # Create or retrieve active job
    job = JobService.create_job(request.recording_id)

    # If job is already processing or finished, return its status
    if job.status in ("completed", "processing"):
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

    # Submit task to background worker thread
    def _run_task(job_id: str, progress_cb):
        TranscriptionService.execute_transcription(
            recording_id=request.recording_id,
            progress_cb=progress_cb,
            use_mock=request.use_mock,
            initial_prompt=request.initial_prompt,
        )

    JobService.submit_task(job.job_id, _run_task)

    return JobStatusResponse(
        job_id=job.job_id,
        recording_id=job.recording_id,
        status="queued",
        progress=0.0,
        stage="Queued for processing",
        error=None,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/transcriptions/{job_or_rec_id}/status", response_model=JobStatusResponse, tags=["Transcriptions"])
def get_job_status(job_or_rec_id: str) -> JobStatusResponse:
    """Retrieve progress and execution status of a processing job."""
    job = JobService.get_job(job_or_rec_id)
    if not job:
        job = JobService.get_job_by_recording(job_or_rec_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found for ID '{job_or_rec_id}'.")

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


@router.get("/transcriptions/{recording_id}", response_model=TranscriptResult, tags=["Transcriptions"])
def get_transcript(recording_id: str) -> TranscriptResult:
    """Retrieve the canonical transcript, speaker attribution, and confidence analytics."""
    transcript = TranscriptionService.get_transcript(recording_id)
    if not transcript:
        # Check if job is still in progress
        job = JobService.get_job_by_recording(recording_id)
        if job and job.status in ("queued", "processing"):
            raise HTTPException(status_code=202, detail=f"Processing in progress (stage: {job.stage}, {int(job.progress*100)}%).")
        elif job and job.status == "failed":
            raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")
        raise HTTPException(status_code=404, detail=f"No transcript found for recording '{recording_id}'.")

    return transcript


@router.get("/transcriptions/{recording_id}/export", tags=["Transcriptions"])
def export_transcript(
    recording_id: str,
    format: str = Query(default="txt", pattern="^(txt|json|srt|vtt)$"),
) -> Response:
    """Export formatted transcript in TXT, JSON, SRT, or VTT format."""
    transcript = TranscriptionService.get_transcript(recording_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found.")

    fmt = format.lower()
    if fmt == "json":
        content = ExportService.to_json(transcript)
        return Response(content=content, media_type="application/json")
    elif fmt == "srt":
        content = ExportService.to_srt(transcript)
        return PlainTextResponse(content=content, media_type="text/plain")
    elif fmt == "vtt":
        content = ExportService.to_vtt(transcript)
        return PlainTextResponse(content=content, media_type="text/vtt")
    else:
        content = ExportService.to_txt(transcript)
        return PlainTextResponse(content=content, media_type="text/plain")
