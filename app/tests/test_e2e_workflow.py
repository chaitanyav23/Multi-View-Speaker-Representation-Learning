"""Full End-to-End Workflow Integration Test."""
from __future__ import annotations

import io
import time
from fastapi.testclient import TestClient

from backend.services.transcription_service import TranscriptionService
from engine.base import BaseSpeechEngine
from engine.schemas import (
    AlignedWord,
    CalibrationMetadata,
    ConfidenceFeatures,
    ConversationTurn,
    TranscriptResult,
)


class TestStubEngine(BaseSpeechEngine):
    """Minimal test stub for ultra-fast API workflow verification without network/GPU."""

    def process(self, audio_path, recording_id, filename=None, duration=None, progress_callback=None, options=None):
        if progress_callback:
            progress_callback(0.5, "Processing")
            progress_callback(1.0, "Complete")
        feat = ConfidenceFeatures(acoustic_score=0.92, duration=0.4)
        words_text = ["good", "morning", "officer", "how", "can", "i", "help", "you", "license", "and", "registration", "please"]
        words = []
        for idx, w in enumerate(words_text):
            spk = "SPEAKER_00" if idx < 8 else "SPEAKER_01"
            words.append(
                AlignedWord(
                    word=w,
                    start=round(idx * 0.4, 2),
                    end=round((idx + 1) * 0.4, 2),
                    speaker=spk,
                    confidence=0.92,
                    raw_confidence=0.92,
                    features=feat,
                )
            )

        turns = [
            ConversationTurn(
                turn_id="turn_001",
                speaker="SPEAKER_00",
                start=0.0,
                end=3.2,
                text="good morning officer how can i help you",
                words=words[:8],
                avg_confidence=0.92,
            ),
            ConversationTurn(
                turn_id="turn_002",
                speaker="SPEAKER_01",
                start=3.2,
                end=4.8,
                text="license and registration please",
                words=words[8:],
                avg_confidence=0.92,
            ),
        ]
        return TranscriptResult(
            recording_id=recording_id,
            filename=filename or "test.wav",
            duration=1.0,
            calibration=CalibrationMetadata(method="isotonic_regression"),
            speakers=["SPEAKER_00", "SPEAKER_01"],
            segments=turns,
            words=words,
            summary={"overall_confidence": 0.90, "mean_confidence": 0.90},
        )


def test_full_user_journey(client: TestClient):
    # Ensure fast, deterministic execution during API contract tests
    TranscriptionService.set_engine(TestStubEngine())

    # 1. User Uploads a Media File
    file_bytes = b"RIFFsimulatedaudiocontent16kmono"
    file_tuple = ("bodycam_encounter.wav", io.BytesIO(file_bytes), "audio/wav")

    upload_resp = client.post("/api/v1/media/upload", files={"file": file_tuple})
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    recording_id = upload_data["recording_id"]
    assert recording_id is not None

    # 2. User Triggers Transcription Job
    job_req = {"recording_id": recording_id, "use_mock": True, "language": "en"}
    job_resp = client.post("/api/v1/transcriptions", json=job_req)
    assert job_resp.status_code == 202
    job_data = job_resp.json()
    job_id = job_data["job_id"]
    assert job_id is not None

    # 3. User / Frontend Polls Job Status until Completion
    max_wait = 15
    start_t = time.time()
    final_status = "queued"

    while time.time() - start_t < max_wait:
        poll_resp = client.get(f"/api/v1/transcriptions/{job_id}/status")
        assert poll_resp.status_code == 200
        status_info = poll_resp.json()
        final_status = status_info["status"]
        if final_status in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert final_status == "completed"

    # 4. User Retrieves the Full Canonical Transcript Result
    tx_resp = client.get(f"/api/v1/transcriptions/{recording_id}")
    assert tx_resp.status_code == 200
    transcript = tx_resp.json()

    assert transcript["recording_id"] == recording_id
    assert len(transcript["speakers"]) >= 2
    assert len(transcript["segments"]) >= 2
    assert len(transcript["words"]) > 10
    assert transcript["summary"]["mean_confidence"] > 0.0

    # 5. User Explores Analytics & Reliability
    analytics_resp = client.get(f"/api/v1/analytics/{recording_id}")
    assert analytics_resp.status_code == 200
    analytics = analytics_resp.json()

    assert "brier_score" in analytics
    assert "ece" in analytics
    assert "reliability_diagram" in analytics
    assert len(analytics["reliability_diagram"]) == 10

    # 6. User Exports Transcripts in All Formats
    for fmt in ["txt", "json", "srt", "vtt"]:
        exp_resp = client.get(f"/api/v1/transcriptions/{recording_id}/export?format={fmt}")
        assert exp_resp.status_code == 200
        assert len(exp_resp.text) > 20

    # 7. User Submits Correction Revision (Feedback Loop)
    feedback_req = {
        "recording_id": recording_id,
        "turn_id": transcript["segments"][0]["turn_id"],
        "original_text": transcript["segments"][0]["text"],
        "corrected_text": "Good morning driver. Officer Davis Highway Patrol.",
        "original_speaker": transcript["segments"][0]["speaker"],
        "corrected_speaker": "Officer Davis",
    }
    fb_resp = client.post("/api/v1/feedback", json=feedback_req)
    assert fb_resp.status_code == 200
    assert fb_resp.json()["status"] == "recorded"
