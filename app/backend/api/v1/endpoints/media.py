"""Media Ingestion, Upload, and Audio Streaming Endpoints."""
from __future__ import annotations

import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.database.models import RecordingRepository
from backend.services.audio_service import AudioService
from engine.schemas import MediaUploadResponse

router = APIRouter()


@router.post("/media/upload", response_model=MediaUploadResponse, tags=["Media"])
async def upload_media(file: UploadFile = File(...)) -> MediaUploadResponse:
    """Accept and validate a pre-recorded audio or video recording."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    # Save to temporary file to calculate hash and metadata safely
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp_path.write_bytes(content)

    try:
        record = AudioService.save_and_ingest(tmp_path, file.filename)
        return MediaUploadResponse(
            recording_id=record.recording_id,
            filename=record.filename,
            size_bytes=record.size_bytes,
            duration_seconds=record.duration,
            sha256=record.sha256,
            status="ingested",
            message=f"Media file successfully validated and ingested ({record.duration:.1f}s).",
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {exc}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/media", tags=["Media"])
def list_recordings() -> list[dict[str, str | float | int]]:
    """List all ingested recordings."""
    records = RecordingRepository.list_all()
    return [
        {
            "recording_id": r.recording_id,
            "filename": r.filename,
            "duration": r.duration,
            "size_bytes": r.size_bytes,
            "sha256": r.sha256,
            "created_at": r.created_at,
        }
        for r in records
    ]


@router.get("/media/{recording_id}/stream", tags=["Media"])
def stream_media(recording_id: str) -> FileResponse:
    """Stream media file directly for audio playback in frontend."""
    record = RecordingRepository.get(recording_id)
    if not record or not Path(record.filepath).exists():
        raise HTTPException(status_code=404, detail="Recording media not found.")

    ext = Path(record.filepath).suffix.lower()
    media_type = "audio/wav" if ext == ".wav" else "audio/mpeg" if ext == ".mp3" else "application/octet-stream"
    return FileResponse(record.filepath, media_type=media_type, filename=record.filename)
