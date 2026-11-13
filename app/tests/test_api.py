"""Integration tests for FastAPI REST API endpoints."""
from __future__ import annotations

import io
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "CS698B" in data["app"]


def test_upload_and_list_media(client: TestClient):
    file_bytes = b"RIFFmockwavcontent"
    file_tuple = ("test_audio.wav", io.BytesIO(file_bytes), "audio/wav")
    
    resp = client.post("/api/v1/media/upload", files={"file": file_tuple})
    assert resp.status_code == 200
    data = resp.json()
    assert "recording_id" in data
    assert data["filename"] == "test_audio.wav"
    assert data["status"] == "ingested"

    # List recordings
    list_resp = client.get("/api/v1/media")
    assert list_resp.status_code == 200
    records = list_resp.json()
    assert any(r["recording_id"] == data["recording_id"] for r in records)


def test_feedback_endpoints(client: TestClient):
    payload = {
        "recording_id": "rec_test_feedback",
        "turn_id": "turn_001",
        "original_text": "officer Davis",
        "corrected_text": "Officer Davis",
        "original_speaker": "SPEAKER_00",
        "corrected_speaker": "Officer Davis",
        "notes": "Spelling and capitalization",
    }
    resp = client.post("/api/v1/feedback", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "recorded"
    assert "feedback_id" in res_data

    # List feedback
    list_resp = client.get("/api/v1/feedback?recording_id=rec_test_feedback")
    assert list_resp.status_code == 200
    items = list_resp.json()
    assert len(items) >= 1
    assert items[0]["corrected_text"] == "Officer Davis"
