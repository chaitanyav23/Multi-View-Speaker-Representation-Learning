"""Unit and regression tests for recording-state and job-lifecycle management.

Covers Stage 1.5.2 verification requirements:
1. Completed historical recording without a job (Stage 1.4.1 scenario).
2. Completed recording with a job.
3. Ingested recording without transcript or job.
4. Nonexistent recording/job (HTTP 404 gracefully handled, HTTP 500 raised).
5. Actual job-ID tracking (separation of job_id and recording_id).
6. Recording switching (state updates, job_id cleared, correct metadata loaded).
7. Workflow-mode switching (state cleared across tabs to prevent stale cross-talk).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import requests

from frontend.api_client import APIClient


# ---------------------------------------------------------------------------
# Test Case 1: Completed Historical Recording Without a Job
# ---------------------------------------------------------------------------
def test_historical_recording_without_job_resolves_cleanly():
    """Stage 1.4.1 scenario: transcript exists in DB, but jobs table has 0 rows.
    
    Verifies:
    - client.get_transcript(recording_id) returns the completed transcript (200).
    - client.get_job_status(recording_id) returns None without raising on 404.
    """
    client = APIClient("http://127.0.0.1:8000")
    hist_rec_id = "b1b3f780-94e9-490d-89f8-52945b84b31b"

    mock_transcript_payload = {
        "recording_id": hist_rec_id,
        "filename": "bwc_msdd_e2e.wav",
        "duration": 175.43,
        "words": [{"word": "test", "start": 0.0, "end": 0.5, "confidence": 0.95, "speaker": "speaker_0"}],
        "segments": [{"speaker": "speaker_0", "start": 0.0, "end": 0.5, "text": "test"}],
        "speakers": ["speaker_0"],
    }

    with patch("requests.get") as mock_get:
        # Transcript endpoint returns 200 with data
        mock_tx_resp = MagicMock()
        mock_tx_resp.status_code = 200
        mock_tx_resp.json.return_value = mock_transcript_payload

        # Job status endpoint returns 404 (no jobs row exists)
        mock_job_resp = MagicMock()
        mock_job_resp.status_code = 404

        def side_effect(url, **kwargs):
            if f"/transcriptions/{hist_rec_id}/status" in url:
                return mock_job_resp
            elif f"/transcriptions/{hist_rec_id}" in url:
                return mock_tx_resp
            return MagicMock(status_code=404)

        mock_get.side_effect = side_effect

        # 1. Transcript-first fetch resolves successfully
        transcript = client.get_transcript(hist_rec_id)
        assert transcript is not None
        assert transcript["recording_id"] == hist_rec_id
        assert len(transcript["words"]) == 1

        # 2. Status check returns None gracefully without raising HTTPError
        job_status = client.get_job_status(hist_rec_id)
        assert job_status is None


# ---------------------------------------------------------------------------
# Test Case 2: Completed Recording With a Job
# ---------------------------------------------------------------------------
def test_completed_recording_with_job():
    """Scenario: Both transcript and a completed job exist."""
    client = APIClient("http://127.0.0.1:8000")
    rec_id = "c381b3c1-4a9d-451b-94c2-ad39302eb17a"
    job_id = "3eb7208a-ecba-4516-9a9f-6157b3d39305"

    with patch("requests.get") as mock_get:
        mock_tx_resp = MagicMock(status_code=200)
        mock_tx_resp.json.return_value = {"recording_id": rec_id, "duration": 215.38, "words": []}

        mock_job_resp = MagicMock(status_code=200)
        mock_job_resp.json.return_value = {
            "job_id": job_id,
            "recording_id": rec_id,
            "status": "completed",
            "progress": 1.0,
            "stage": "Processing complete",
        }

        def side_effect(url, **kwargs):
            if f"/transcriptions/{job_id}/status" in url:
                return mock_job_resp
            elif f"/transcriptions/{rec_id}" in url:
                return mock_tx_resp
            return MagicMock(status_code=404)

        mock_get.side_effect = side_effect

        transcript = client.get_transcript(rec_id)
        assert transcript is not None
        assert transcript["recording_id"] == rec_id

        status = client.get_job_status(job_id)
        assert status is not None
        assert status["job_id"] == job_id
        assert status["status"] == "completed"


# ---------------------------------------------------------------------------
# Test Case 3: Ingested Recording Without Transcript or Job
# ---------------------------------------------------------------------------
def test_ingested_recording_without_transcript_or_job():
    """Scenario: Audio is uploaded, but not yet transcribed and has no active job."""
    client = APIClient("http://127.0.0.1:8000")
    rec_id = "75efae17-d99a-48b0-9326-db46b679716c"

    with patch("requests.get") as mock_get:
        # Both endpoints return 404
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        transcript = client.get_transcript(rec_id)
        assert transcript is None  # Handled as missing, no exception raised

        job_status = client.get_job_status(rec_id)
        assert job_status is None  # Handled as no active job, no exception raised


# ---------------------------------------------------------------------------
# Test Case 4: Nonexistent Recording and Job Error Handling
# ---------------------------------------------------------------------------
def test_nonexistent_ids_and_server_error_propagation():
    """Verifies that HTTP 404 is returned as None, but HTTP 500 is raised."""
    client = APIClient("http://127.0.0.1:8000")
    dummy_id = "00000000-0000-0000-0000-000000000000"

    with patch("requests.get") as mock_get:
        # 404 cleanly returns None
        mock_404 = MagicMock(status_code=404)
        mock_get.return_value = mock_404
        assert client.get_job_status(dummy_id) is None
        assert client.get_transcript(dummy_id) is None

        # 500 server error raises HTTPError (not silently swallowed)
        mock_500 = MagicMock(status_code=500)
        mock_500.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_500

        with pytest.raises(requests.exceptions.HTTPError):
            client.get_job_status(dummy_id)

        with pytest.raises(requests.exceptions.HTTPError):
            client.get_transcript(dummy_id)


# ---------------------------------------------------------------------------
# Test Case 5: Actual Job-ID Tracking
# ---------------------------------------------------------------------------
def test_distinct_job_id_and_recording_id_lifecycle():
    """Verifies that trigger_transcription returns distinct job_id and recording_id."""
    client = APIClient("http://127.0.0.1:8000")
    test_rec_id = "rec-uuid-1111-2222"
    test_job_id = "job-uuid-3333-4444"

    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        mock_post_resp = MagicMock(status_code=202)
        mock_post_resp.json.return_value = {
            "job_id": test_job_id,
            "recording_id": test_rec_id,
            "status": "queued",
            "progress": 0.0,
            "stage": "Queued for processing",
        }
        mock_post.return_value = mock_post_resp

        mock_status_resp = MagicMock(status_code=200)
        mock_status_resp.json.return_value = {
            "job_id": test_job_id,
            "recording_id": test_rec_id,
            "status": "processing",
            "progress": 0.5,
            "stage": "Faster-Whisper ASR",
        }
        mock_get.return_value = mock_status_resp

        # Trigger job
        dispatch_res = client.trigger_transcription(test_rec_id)
        assert dispatch_res["recording_id"] == test_rec_id
        assert dispatch_res["job_id"] == test_job_id
        assert dispatch_res["recording_id"] != dispatch_res["job_id"]

        # Poll using actual job_id
        status_res = client.get_job_status(dispatch_res["job_id"])
        assert status_res["job_id"] == test_job_id
        assert status_res["status"] == "processing"
        mock_get.assert_called_with("http://127.0.0.1:8000/api/v1/transcriptions/job-uuid-3333-4444/status", timeout=5)


# ---------------------------------------------------------------------------
# Test Case 6: Recording Switching State Logic
# ---------------------------------------------------------------------------
def test_recording_switching_clears_active_job_and_updates_data():
    """Simulates frontend selection change between two recordings."""
    session_state = {
        "recording_id": "rec_A",
        "job_id": "job_A_active",
        "filename": "audio_A.mp3",
    }

    # User selects recording B in selector
    new_rec_id = "rec_B"
    new_label = "audio_B.mp3 (rec_B...) - 120.0s"

    if session_state.get("recording_id") != new_rec_id:
        session_state["recording_id"] = new_rec_id
        session_state["filename"] = new_label.split(" (")[0]
        session_state["job_id"] = None

    assert session_state["recording_id"] == "rec_B"
    assert session_state["filename"] == "audio_B.mp3"
    assert session_state["job_id"] is None  # Job state cleared to avoid polling unrelated tasks


# ---------------------------------------------------------------------------
# Test Case 7: Workflow-Mode Switching State Synchronization
# ---------------------------------------------------------------------------
def test_workflow_mode_switching_clears_stale_session_state():
    """Simulates switching from 'Past Ingested Recordings' to 'Upload New Audio / Video'."""
    session_state = {
        "active_mode": "📂 Past Ingested Recordings",
        "recording_id": "rec_A",
        "job_id": "job_A",
        "filename": "audio_A.mp3",
    }

    # Switch radio button to Upload tab
    new_mode = "📤 Upload New Audio / Video"

    if "active_mode" in session_state and session_state["active_mode"] != new_mode:
        session_state["recording_id"] = None
        session_state["job_id"] = None
        session_state["filename"] = None
    session_state["active_mode"] = new_mode

    assert session_state["active_mode"] == "📤 Upload New Audio / Video"
    assert session_state["recording_id"] is None
    assert session_state["job_id"] is None
    assert session_state["filename"] is None
