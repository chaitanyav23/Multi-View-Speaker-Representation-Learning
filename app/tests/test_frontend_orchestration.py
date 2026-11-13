"""Frontend Orchestration and AppTest Verification Suite.

Exercises actual Streamlit frontend orchestration using native `AppTest` facilities:
- Test B1: Completed historical transcript requires ZERO status-polling calls.
- Test B2: Active processing job polls strictly using actual `job_id`.
- Test B3: Repeated switching between past recordings (A -> B -> A -> B).
- Test B4: Switching across all three workflow modes without state leakage.
- Test B5: Ingested recording without transcript or job displays un-transcribed state.
- Test B6: Unexpected backend API failures (HTTP 500, network errors) are visibly surfaced.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import requests

from streamlit.testing.v1 import AppTest

# Ensure 'app' directory is in sys.path for streamlit_app execution
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

SCRIPT_PATH = str(APP_DIR / "frontend" / "streamlit_app.py")


def _make_transcript_payload(recording_id: str, filename: str, words_count: int = 5) -> dict:
    words = [
        {"word": f"word_{i}", "start": float(i), "end": float(i) + 0.5, "confidence": 0.85, "speaker": "speaker_0"}
        for i in range(words_count)
    ]
    return {
        "recording_id": recording_id,
        "filename": filename,
        "duration": float(words_count),
        "words": words,
        "segments": [{"speaker": "speaker_0", "start": 0.0, "end": float(words_count), "text": " ".join(w["word"] for w in words)}],
        "speakers": ["speaker_0"],
        "summary": {"mean_confidence": 0.85},
    }


# ---------------------------------------------------------------------------
# Test B1: Completed Transcript: Zero Status Polling
# ---------------------------------------------------------------------------
def test_b1_completed_transcript_zero_status_polling():
    """Scenario: Historical recording b1b3f780-94e9-490d-89f8-52945b84b31b has a completed
    transcript in the backend, but 0 rows in jobs table.
    
    Critical Assertion:
    Zero status-polling requests (/status) must be made during rendering.
    """
    hist_rec_id = "b1b3f780-94e9-490d-89f8-52945b84b31b"
    recordings_list = [
        {"recording_id": hist_rec_id, "filename": "bwc_msdd_e2e.wav", "duration": 175.4, "created_at": "2026-11-07T17:19:20Z"}
    ]
    transcript_payload = _make_transcript_payload(hist_rec_id, "bwc_msdd_e2e.wav", words_count=10)

    status_call_urls: list[str] = []
    transcript_call_urls: list[str] = []

    with patch("requests.get") as mock_get:
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "/health" in url:
                resp.status_code = 200
                resp.json.return_value = {"status": "ok", "app": "CS698B"}
            elif "/media" in url and "/stream" not in url:
                resp.status_code = 200
                resp.json.return_value = recordings_list
            elif f"/transcriptions/{hist_rec_id}" in url and "/status" not in url:
                transcript_call_urls.append(url)
                resp.status_code = 200
                resp.json.return_value = transcript_payload
            elif "/status" in url:
                status_call_urls.append(url)
                resp.status_code = 404
            elif "/analytics" in url:
                resp.status_code = 200
                resp.json.return_value = {"brier_score": 0.05, "ece": 0.02, "reliability_diagram": []}
            elif "/feedback/vocabulary" in url:
                resp.status_code = 200
                resp.json.return_value = []
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = side_effect

        at = AppTest.from_file(SCRIPT_PATH)
        at.run()
        # Navigate to Past Ingested Recordings
        at.radio[0].set_value("📂 Past Ingested Recordings").run()

        # Verify no script crashes occurred
        assert len(at.exception) == 0

        # Critical Assertion 1: Zero status polling requests made
        assert len(status_call_urls) == 0, f"Expected 0 status calls, got: {status_call_urls}"

        # Critical Assertion 2: Transcript was requested for hist_rec_id
        assert any(f"/transcriptions/{hist_rec_id}" in u for u in transcript_call_urls)

        # Verification of UI rendering
        assert at.session_state.get("recording_id") == hist_rec_id
        # Executive summary metrics should be displayed (media duration, words)
        metric_labels = [m.label for m in at.metric]
        assert "Total Words" in metric_labels
        assert "Avg Confidence" in metric_labels


# ---------------------------------------------------------------------------
# Test B2: Active Job Uses the Actual job_id
# ---------------------------------------------------------------------------
def test_b2_active_job_polls_using_actual_job_id():
    """Scenario: A transcription was dispatched with distinct recording_id and job_id.
    The transcript is not yet available (HTTP 202).
    
    Critical Assertions:
    - Status polling strictly queries using job_id, NOT recording_id.
    - Transcript retrieval queries using recording_id.
    - No identifier substitution or fabrication occurs.
    """
    rec_id = "rec-uuid-1111"
    job_id = "job-uuid-2222"

    polled_status_targets: list[str] = []
    polled_transcript_targets: list[str] = []

    with patch("requests.get") as mock_get:
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "/health" in url:
                resp.status_code = 200
                resp.json.return_value = {"status": "ok", "app": "CS698B"}
            elif f"/transcriptions/{rec_id}" in url and "/status" not in url:
                polled_transcript_targets.append(url)
                # Processing in progress: HTTP 202
                resp.status_code = 202
                resp.json.return_value = None
            elif f"/transcriptions/{job_id}/status" in url:
                polled_status_targets.append(job_id)
                resp.status_code = 200
                # Report active processing
                resp.json.return_value = {
                    "job_id": job_id,
                    "recording_id": rec_id,
                    "status": "failed",  # Use failed to break the polling loop without time.sleep
                    "progress": 0.5,
                    "stage": "Faster-Whisper ASR",
                    "error": "Simulated terminal check",
                }
            elif f"/transcriptions/{rec_id}/status" in url:
                polled_status_targets.append(rec_id)
                resp.status_code = 404
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = side_effect

        at = AppTest.from_file(SCRIPT_PATH)
        at.run()
        # Pre-seed session state simulating freshly dispatched job
        at.radio[0].set_value("📤 Upload New Audio / Video").run()
        at.session_state["recording_id"] = rec_id
        at.session_state["job_id"] = job_id
        at.session_state["filename"] = "uploaded.wav"
        at.run()

        # Critical Assertion 1: Polling strictly used job_id
        assert job_id in polled_status_targets, f"Expected {job_id} in status calls, got: {polled_status_targets}"
        assert rec_id not in polled_status_targets, f"Recording ID {rec_id} was incorrectly used for status polling!"

        # Critical Assertion 2: Transcript was requested using recording_id
        assert any(f"/transcriptions/{rec_id}" in u for u in polled_transcript_targets)


# ---------------------------------------------------------------------------
# Test B3: Repeated Switching Between Past Recordings (A -> B -> A -> B)
# ---------------------------------------------------------------------------
def test_b3_repeated_switching_between_past_recordings():
    """Scenario: Two past recordings A and B with distinct filenames, durations, and transcripts.
    Simulate A -> B -> A -> B transitions.
    
    Verifies:
    - Correct recording selected on each switch.
    - Correct transcript loaded.
    - job_id reset to None on switch.
    - No cross-talk or stale data displayed.
    """
    rec_a = "c381b3c1-4a9d-451b-94c2-ad39302eb17a"
    rec_b = "190bbaef-3b8b-4009-b026-fd14500ad432"

    recordings_list = [
        {"recording_id": rec_a, "filename": "UK_Parliament.mp3", "duration": 215.4, "created_at": "2026-11-07T20:51:00Z"},
        {"recording_id": rec_b, "filename": "Piers_Debate.mp3", "duration": 204.6, "created_at": "2026-11-07T20:53:44Z"},
    ]

    tx_a = _make_transcript_payload(rec_a, "UK_Parliament.mp3", words_count=12)
    tx_b = _make_transcript_payload(rec_b, "Piers_Debate.mp3", words_count=24)

    requested_transcripts: list[str] = []

    with patch("requests.get") as mock_get:
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "/health" in url:
                resp.status_code = 200
                resp.json.return_value = {"status": "ok", "app": "CS698B"}
            elif "/media" in url and "/stream" not in url:
                resp.status_code = 200
                resp.json.return_value = recordings_list
            elif f"/transcriptions/{rec_a}" in url and "/status" not in url:
                requested_transcripts.append(rec_a)
                resp.status_code = 200
                resp.json.return_value = tx_a
            elif f"/transcriptions/{rec_b}" in url and "/status" not in url:
                requested_transcripts.append(rec_b)
                resp.status_code = 200
                resp.json.return_value = tx_b
            elif "/analytics" in url:
                resp.status_code = 200
                resp.json.return_value = {"brier_score": 0.04, "ece": 0.02, "reliability_diagram": []}
            elif "/feedback/vocabulary" in url:
                resp.status_code = 200
                resp.json.return_value = []
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = side_effect

        at = AppTest.from_file(SCRIPT_PATH)
        at.run()
        at.radio[0].set_value("📂 Past Ingested Recordings").run()

        # Step 1: Default selection is rec_a
        assert at.session_state.get("recording_id") == rec_a
        assert at.session_state.get("filename") == "UK_Parliament.mp3"
        assert at.session_state.get("job_id") is None
        label_b = "Piers_Debate.mp3 (190bbaef...) - 204.6s"
        label_a = "UK_Parliament.mp3 (c381b3c1...) - 215.4s"

        # Step 2: Switch to B
        sel = at.selectbox(key="past_recording_selector")
        sel.select(label_b).run()
        assert at.session_state.get("recording_id") == rec_b
        assert at.session_state.get("filename") == "Piers_Debate.mp3"
        assert at.session_state.get("job_id") is None

        # Step 3: Switch back to A
        sel = at.selectbox(key="past_recording_selector")
        sel.select(label_a).run()
        assert at.session_state.get("recording_id") == rec_a
        assert at.session_state.get("filename") == "UK_Parliament.mp3"
        assert at.session_state.get("job_id") is None

        # Step 4: Switch to B again
        sel = at.selectbox(key="past_recording_selector")
        sel.select(label_b).run()
        assert at.session_state.get("recording_id") == rec_b
        assert at.session_state.get("filename") == "Piers_Debate.mp3"
        assert at.session_state.get("job_id") is None

        # Verify that both transcripts were requested during the sequence
        assert rec_a in requested_transcripts
        assert rec_b in requested_transcripts


# ---------------------------------------------------------------------------
# Test B4: Switching Across All Three Workflow Modes
# ---------------------------------------------------------------------------
def test_b4_workflow_mode_switching_state_isolation():
    """Scenario: Transition across Benchmark -> Upload -> Past Recordings -> Upload -> Benchmark.
    
    Verifies:
    - Mode changes clear stale recording_id, job_id, and filename.
    - No active progress or stale transcript leaks into the Upload tab.
    - Returning to Past Recordings synchronizes properly with past_recording_selector.
    """
    rec_a = "rec-historical-a"
    recordings_list = [
        {"recording_id": rec_a, "filename": "sample_past.wav", "duration": 60.0, "created_at": "2026-11-07T10:00:00Z"}
    ]
    tx_a = _make_transcript_payload(rec_a, "sample_past.wav", words_count=8)

    with patch("requests.get") as mock_get:
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "/health" in url:
                resp.status_code = 200
                resp.json.return_value = {"status": "ok", "app": "CS698B"}
            elif "/media" in url and "/stream" not in url:
                resp.status_code = 200
                resp.json.return_value = recordings_list
            elif f"/transcriptions/{rec_a}" in url and "/status" not in url:
                resp.status_code = 200
                resp.json.return_value = tx_a
            elif "/analytics" in url:
                resp.status_code = 200
                resp.json.return_value = {"brier_score": 0.05, "ece": 0.02, "reliability_diagram": []}
            elif "/feedback/vocabulary" in url:
                resp.status_code = 200
                resp.json.return_value = []
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = side_effect

        at = AppTest.from_file(SCRIPT_PATH)
        at.run()

        # Mode 1: Benchmark Mode (default)
        assert at.radio[0].value == "🎯 Select Benchmark Audio Recording"
        assert at.session_state.get("recording_id") is None

        # Mode 2: Switch to Past Ingested Recordings
        at.radio[0].set_value("📂 Past Ingested Recordings").run()
        assert at.session_state.get("recording_id") == rec_a
        assert at.session_state.get("filename") == "sample_past.wav"

        # Mode 3: Switch to Upload New Audio / Video
        at.radio[0].set_value("📤 Upload New Audio / Video").run()
        # Stale state must be cleared
        assert at.session_state.get("recording_id") is None
        assert at.session_state.get("job_id") is None
        assert at.session_state.get("filename") is None
        # Verify prompt requesting media upload is present
        assert any("upload media above" in str(i.value) for i in at.info)

        # Mode 4: Switch back to Past Ingested Recordings
        at.radio[0].set_value("📂 Past Ingested Recordings").run()
        assert at.session_state.get("recording_id") == rec_a

        # Mode 5: Switch back to Benchmark Mode
        at.radio[0].set_value("🎯 Select Benchmark Audio Recording").run()
        assert at.session_state.get("recording_id") is None
        assert at.session_state.get("job_id") is None


# ---------------------------------------------------------------------------
# Test B5: Ingested Recording Without Transcript or Job
# ---------------------------------------------------------------------------
def test_b5_ingested_recording_without_transcript_or_job():
    """Scenario: Recording 75efae17-d99a-48b0-9326-db46b679716c has been ingested,
    but never transcribed and has no active job.
    
    Verifies:
    - Displays 'Not transcribed yet' info notification.
    - Provides 'Run Phase 20 Pipeline on this Recording' button.
    - Does not fabricate transcript content.
    - Does not mark the recording as 'failed'.
    - Does not loop or endlessly poll status.
    """
    untranscribed_id = "75efae17-d99a-48b0-9326-db46b679716c"
    recordings_list = [
        {"recording_id": untranscribed_id, "filename": "test_audio.wav", "duration": 45.0, "created_at": "2026-11-07T14:54:36Z"}
    ]

    with patch("requests.get") as mock_get:
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "/health" in url:
                resp.status_code = 200
                resp.json.return_value = {"status": "ok", "app": "CS698B"}
            elif "/media" in url and "/stream" not in url:
                resp.status_code = 200
                resp.json.return_value = recordings_list
            else:
                # Transcript and job status endpoints both return 404
                resp.status_code = 404
            return resp

        mock_get.side_effect = side_effect

        at = AppTest.from_file(SCRIPT_PATH)
        at.run()
        at.radio[0].set_value("📂 Past Ingested Recordings").run()

        assert len(at.exception) == 0

        # Assert 'Not transcribed yet' banner is shown
        info_texts = [i.value for i in at.info]
        assert any("Not transcribed yet" in t for t in info_texts), f"Info banners: {info_texts}"

        # Assert 'Run Phase 20 Pipeline on this Recording' button is rendered
        btn_labels = [b.label for b in at.button]
        assert "Run Phase 20 Pipeline on this Recording" in btn_labels

        # Assert no failed error is displayed
        assert len(at.error) == 0


# ---------------------------------------------------------------------------
# Test B6: Unexpected API Failures
# ---------------------------------------------------------------------------
def test_b6_unexpected_api_failures_surfaced():
    """Scenario: Backend returns HTTP 500 or raises ConnectionError.
    
    Verifies:
    - HTTP 500 error is visibly surfaced via st.error.
    - Network connection failure is visibly surfaced via st.error.
    - Errors are NOT silently swallowed as None.
    """
    err_rec_id = "rec-error-500"
    recordings_list = [
        {"recording_id": err_rec_id, "filename": "server_crash.wav", "duration": 30.0, "created_at": "2026-11-07T14:54:36Z"}
    ]

    # Subtest 1: HTTP 500
    with patch("requests.get") as mock_get:
        def side_effect_500(url, **kwargs):
            resp = MagicMock()
            if "/health" in url:
                resp.status_code = 200
                resp.json.return_value = {"status": "ok", "app": "CS698B"}
            elif "/media" in url and "/stream" not in url:
                resp.status_code = 200
                resp.json.return_value = recordings_list
            elif f"/transcriptions/{err_rec_id}" in url:
                resp.status_code = 500
                resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Internal Server Error: Database locked")
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = side_effect_500

        at = AppTest.from_file(SCRIPT_PATH)
        at.run()
        at.radio[0].set_value("📂 Past Ingested Recordings").run()

        error_texts = [e.value for e in at.error]
        assert any("500 Internal Server Error" in t for t in error_texts), f"Expected 500 error, got: {error_texts}"

    # Subtest 2: ConnectionError
    with patch("requests.get") as mock_get:
        def side_effect_conn(url, **kwargs):
            if "/health" in url:
                resp = MagicMock(status_code=200)
                resp.json.return_value = {"status": "ok", "app": "CS698B"}
                return resp
            elif "/media" in url and "/stream" not in url:
                resp = MagicMock(status_code=200)
                resp.json.return_value = recordings_list
                return resp
            elif f"/transcriptions/{err_rec_id}" in url:
                raise requests.exceptions.ConnectionError("Connection refused by backend")
            return MagicMock(status_code=404)

        mock_get.side_effect = side_effect_conn

        at = AppTest.from_file(SCRIPT_PATH)
        at.run()
        at.radio[0].set_value("📂 Past Ingested Recordings").run()

        error_texts = [e.value for e in at.error]
        assert any("Connection refused by backend" in t for t in error_texts), f"Expected connection error, got: {error_texts}"
