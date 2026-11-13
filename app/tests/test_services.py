"""Unit tests for Application Services (Job recovery, Exports, Deduplication)."""
from __future__ import annotations

from datetime import datetime, timezone
from backend.database.models import JobRecord, JobRepository
from backend.services.export_service import ExportService
from backend.services.job_service import JobService


def test_job_recovery_on_startup():
    now_iso = datetime.now(timezone.utc).isoformat()
    # Create an interrupted job in queued and processing states
    stale_job_1 = JobRecord(
        job_id="stale_job_1",
        recording_id="rec_stale_1",
        status="queued",
        progress=0.0,
        stage="Queued",
        error=None,
        created_at=now_iso,
        updated_at=now_iso,
    )
    stale_job_2 = JobRecord(
        job_id="stale_job_2",
        recording_id="rec_stale_2",
        status="processing",
        progress=0.45,
        stage="VAD",
        error=None,
        created_at=now_iso,
        updated_at=now_iso,
    )
    JobRepository.create(stale_job_1)
    JobRepository.create(stale_job_2)

    # Recover stale jobs
    recovered_count = JobService.recover_stale_jobs()
    assert recovered_count >= 2

    j1 = JobRepository.get("stale_job_1")
    j2 = JobRepository.get("stale_job_2")
    assert j1.status == "failed"
    assert "interrupted" in j1.error.lower()
    assert j2.status == "failed"
    assert "interrupted" in j2.error.lower()


def test_export_service_formats():
    from engine.schemas import (
        AlignedWord,
        CalibrationMetadata,
        ConfidenceFeatures,
        ConversationTurn,
        TranscriptResult,
    )

    feat = ConfidenceFeatures(acoustic_score=0.92, duration=0.4)
    words = [
        AlignedWord(word="Hello", start=0.0, end=0.4, speaker="Officer 1", confidence=0.95, raw_confidence=0.95, features=feat),
        AlignedWord(word="driver", start=0.5, end=0.9, speaker="Officer 1", confidence=0.90, raw_confidence=0.90, features=feat),
    ]
    turns = [
        ConversationTurn(turn_id="turn_001", speaker="Officer 1", start=0.0, end=0.9, text="Hello driver", words=words, avg_confidence=0.925)
    ]
    result = TranscriptResult(
        recording_id="rec_export",
        filename="test.wav",
        duration=1.0,
        calibration=CalibrationMetadata(method="isotonic_regression"),
        speakers=["Officer 1"],
        segments=turns,
        words=words,
        summary={"overall_confidence": 0.925},
    )

    # Test JSON
    json_out = ExportService.to_json(result)
    assert '"recording_id": "rec_export"' in json_out

    # Test TXT
    txt_out = ExportService.to_txt(result)
    assert "# Transcript: test.wav" in txt_out
    assert "Officer 1" in txt_out

    # Test SRT
    srt_out = ExportService.to_srt(result)
    assert "-->" in srt_out
    assert "1\n" in srt_out

    # Test VTT
    vtt_out = ExportService.to_vtt(result)
    assert "WEBVTT" in vtt_out
