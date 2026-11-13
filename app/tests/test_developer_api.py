"""Automated Test Suite for Developer API, Plugin Framework, and Test Isolation.

Covers Stage 1.6.1 verification requirements:
1. Pydantic schema validation (manifest, context, request, result, analytics).
2. Consolidated recording-context endpoint (/recordings/{recording_id}/context).
3. Unambiguous job retrieval endpoint (/jobs/{job_id}).
4. Plugin registry and manifest discovery (/plugins).
5. Plugin execution, error sanitization, and output isolation (/plugins/{id}/runs).
6. Plugin result persistence and retrieval (/plugins/{id}/results/{recording_id}).
7. Transcript immutability guarantees.
8. Database and filesystem isolation regressions.
9. Streamlit plugin tab rendering and empty state handling.
"""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from backend.config import settings
from backend.database.db import (
    DEFAULT_DB_PATH,
    get_db_connection,
    get_db_path,
    init_db,
)
from backend.database.models import (
    JobRecord,
    JobRepository,
    PluginAnalysisRecord,
    PluginRepository,
    RecordingRecord,
    RecordingRepository,
    TranscriptRepository,
)
from backend.plugins.base import BasePlugin
from backend.plugins.registry import PluginRegistry
from engine.schemas import (
    AlignedWord,
    AnalyticsResponse,
    AnalyticsTiers,
    AnalyticsTierStats,
    CalibrationMetadata,
    ConversationTurn,
    JobStatusResponse,
    PluginAnalysisResult,
    PluginContextResponse,
    PluginManifest,
    PluginRunRequest,
    TranscriptResult,
)

SCRIPT_PATH = str(Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py")


# ---------------------------------------------------------------------------
# Deterministic Genuine Test Plugin
# ---------------------------------------------------------------------------
class LexicalDiversityTestPlugin(BasePlugin):
    """Deterministic reference plugin computing real lexical metrics from input text."""

    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="lexical-analytics-reference",
            name="Lexical Diversity & Dialogue Metrics",
            version="1.0.0",
            description="Computes type-token ratio, turn counts, and lexical statistics from genuine transcript tokens.",
            author="CS698B Test Infrastructure Team",
            capabilities=["lexical_diversity", "turn_metrics", "speaker_word_distribution"],
            input_requirements=["transcript"],
            output_types=["json"],
            api_version="1.0.0",
            enabled=True,
        )

    def run(self, context: PluginContextResponse, parameters: dict[str, Any]) -> dict[str, Any]:
        if not context.has_transcript or not context.transcript:
            raise ValueError("Transcript is missing; cannot compute lexical metrics.")

        words = [w.word.lower().strip() for w in context.transcript.words if w.word.strip()]
        total_tokens = len(words)
        unique_tokens = len(set(words))
        ttr = round(unique_tokens / max(1, total_tokens), 4)

        # Speaker distribution
        speaker_tokens: dict[str, int] = {}
        for w in context.transcript.words:
            speaker_tokens[w.speaker] = speaker_tokens.get(w.speaker, 0) + 1

        min_length = int(parameters.get("min_word_length", 1))
        filtered_words = [w for w in words if len(w) >= min_length]

        return {
            "recording_id": context.recording_id,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "type_token_ratio": ttr,
            "turn_count": len(context.turns),
            "speaker_distribution": speaker_tokens,
            "filtered_word_count": len(filtered_words),
            "applied_min_length": min_length,
        }


class CrashingTestPlugin(BasePlugin):
    """Fault-injection plugin that raises an exception containing secret-like strings."""

    @property
    def manifest(self) -> PluginManifest:
        return PluginManifest(
            plugin_id="crashing-test-plugin",
            name="Crashing Plugin for Sanitization Testing",
            version="0.1.0",
            description="Raises a simulated failure to test error boundary sanitization.",
            author="Test Infrastructure",
            capabilities=["fault_injection"],
            input_requirements=["transcript"],
            output_types=["json"],
            api_version="1.0.0",
            enabled=True,
        )

    def run(self, context: PluginContextResponse, parameters: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Worker died: token hf_secretcredential123456789012345 in /root/secret/path.py")


# ---------------------------------------------------------------------------
# Helper Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure plugin registry is clean before and after each test."""
    PluginRegistry.clear()
    yield
    PluginRegistry.clear()


def _seed_recording_and_transcript(rec_id: str, filename: str, has_job: bool = False, job_id: str = "job_seed_001"):
    """Helper to populate the isolated test database with a valid recording and transcript."""
    now_iso = datetime.now(timezone.utc).isoformat()
    rec = RecordingRecord(
        recording_id=rec_id,
        filename=filename,
        filepath=f"/test/storage/{filename}",
        duration=120.0,
        size_bytes=10240,
        sha256="abc123sha",
        created_at=now_iso,
    )
    RecordingRepository.create(rec)

    if has_job:
        job = JobRecord(
            job_id=job_id,
            recording_id=rec_id,
            status="completed",
            progress=1.0,
            stage="Complete",
            error=None,
            created_at=now_iso,
            updated_at=now_iso,
        )
        JobRepository.create(job)

    words = [
        AlignedWord(
            word="good", start=0.0, end=0.4, speaker="SPEAKER_00", confidence=0.92, raw_confidence=0.90
        ),
        AlignedWord(
            word="morning", start=0.4, end=0.9, speaker="SPEAKER_00", confidence=0.95, raw_confidence=0.94
        ),
        AlignedWord(
            word="counsel", start=1.0, end=1.5, speaker="SPEAKER_01", confidence=0.88, raw_confidence=0.85
        ),
    ]
    turns = [
        ConversationTurn(turn_id="turn_01", speaker="SPEAKER_00", start=0.0, end=0.9, text="good morning", words=words[:2], avg_confidence=0.935),
        ConversationTurn(turn_id="turn_02", speaker="SPEAKER_01", start=1.0, end=1.5, text="counsel", words=words[2:], avg_confidence=0.88),
    ]
    tx = TranscriptResult(
        recording_id=rec_id,
        filename=filename,
        duration=120.0,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        segments=turns,
        words=words,
        calibration=CalibrationMetadata(method="isotonic_regression", brier_score=0.04, ece=0.02),
        summary={"total_words": 3},
    )
    TranscriptRepository.save(rec_id, tx.model_dump(), now_iso)
    return rec, tx


# ---------------------------------------------------------------------------
# Section 1: Schema Validation Tests
# ---------------------------------------------------------------------------
def test_plugin_manifest_validation():
    """Verify PluginManifest requires valid identifiers and enforces field constraints."""
    # Valid
    manifest = PluginManifest(
        plugin_id="valid-plugin_123",
        name="Valid Plugin",
        version="1.0.0",
        description="A valid plugin manifest",
    )
    assert manifest.plugin_id == "valid-plugin_123"

    # Invalid plugin_id with spaces or special characters
    with pytest.raises(ValidationError):
        PluginManifest(
            plugin_id="invalid plugin with spaces!",
            name="Invalid",
            description="Bad ID",
        )

    # Empty name
    with pytest.raises(ValidationError):
        PluginManifest(
            plugin_id="valid-id",
            name="",
            description="Bad name",
        )


def test_plugin_context_response_schema():
    """Verify PluginContextResponse validates both transcribed and untranscribed states."""
    # Untranscribed state
    ctx_untranscribed = PluginContextResponse(
        recording_id="rec_untranscribed",
        filename="audio.wav",
        duration_seconds=45.0,
        status="untranscribed",
        has_transcript=False,
    )
    assert ctx_untranscribed.total_words == 0
    assert ctx_untranscribed.mean_confidence is None
    assert ctx_untranscribed.transcript is None

    # Transcribed state
    tx = TranscriptResult(
        recording_id="rec_ready",
        filename="audio.wav",
        duration=45.0,
        words=[],
    )
    ctx_ready = PluginContextResponse(
        recording_id="rec_ready",
        filename="audio.wav",
        duration_seconds=45.0,
        status="ready",
        has_transcript=True,
        transcript=tx,
        total_words=10,
        mean_confidence=0.89,
    )
    assert ctx_ready.has_transcript is True
    assert ctx_ready.mean_confidence == 0.89


def test_analytics_response_schema():
    """Verify AnalyticsResponse validates structured confidence tiers and calibration metrics."""
    resp = AnalyticsResponse(
        recording_id="rec_001",
        filename="test.wav",
        total_words=100,
        mean_confidence=0.85,
        mean_raw_confidence=0.82,
        calibration=CalibrationMetadata(),
        brier_score=0.035,
        ece=0.015,
        tiers=AnalyticsTiers(
            low=AnalyticsTierStats(count=5, ratio=0.05),
            mid=AnalyticsTierStats(count=20, ratio=0.20),
            high=AnalyticsTierStats(count=75, ratio=0.75),
        ),
        reliability_diagram=[{"bin": 1, "conf": 0.9, "acc": 0.88}],
        speakers_count=2,
        duration=60.0,
    )
    assert resp.tiers.high.count == 75
    assert resp.brier_score == 0.035


# ---------------------------------------------------------------------------
# Section 2: Recording Context Endpoint Tests
# ---------------------------------------------------------------------------
def test_recording_context_existing_with_transcript(client: TestClient):
    """GET /api/v1/recordings/{recording_id}/context returns consolidated payload for completed recordings."""
    rec_id = "rec_ctx_test_001"
    _seed_recording_and_transcript(rec_id, "hearing.mp3", has_job=True)

    resp = client.get(f"/api/v1/recordings/{rec_id}/context")
    assert resp.status_code == 200
    data = resp.json()
    assert data["recording_id"] == rec_id
    assert data["has_transcript"] is True
    assert data["status"] == "ready"
    assert data["total_words"] == 3
    assert len(data["turns"]) == 2
    assert "SPEAKER_00" in data["speakers"]


def test_recording_context_historical_without_job(client: TestClient):
    """GET /api/v1/recordings/{recording_id}/context resolves historical recordings with zero job rows."""
    rec_id = "hist_rec_no_job_001"
    _seed_recording_and_transcript(rec_id, "historical_benchmark.wav", has_job=False)

    # Ensure 0 job rows exist for this recording
    assert JobRepository.get_by_recording(rec_id) is None

    resp = client.get(f"/api/v1/recordings/{rec_id}/context")
    assert resp.status_code == 200
    data = resp.json()
    assert data["recording_id"] == rec_id
    assert data["has_transcript"] is True
    assert data["status"] == "ready"


def test_recording_context_untranscribed(client: TestClient):
    """GET /api/v1/recordings/{recording_id}/context handles recordings with no transcript."""
    rec_id = "rec_raw_no_tx_001"
    now_iso = datetime.now(timezone.utc).isoformat()
    rec = RecordingRecord(
        recording_id=rec_id,
        filename="raw.wav",
        filepath="/test/raw.wav",
        duration=30.0,
        size_bytes=2048,
        sha256="raw_sha",
        created_at=now_iso,
    )
    RecordingRepository.create(rec)

    resp = client.get(f"/api/v1/recordings/{rec_id}/context")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_transcript"] is False
    assert data["status"] == "untranscribed"
    assert data["total_words"] == 0
    assert data["transcript"] is None


def test_recording_context_unknown_recording_returns_404(client: TestClient):
    """GET /api/v1/recordings/{recording_id}/context returns 404 for unknown IDs."""
    resp = client.get("/api/v1/recordings/nonexistent-uuid-999/context")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Section 3: Job Status Retrieval Endpoint Tests
# ---------------------------------------------------------------------------
def test_job_retrieval_by_actual_job_id(client: TestClient):
    """GET /api/v1/jobs/{job_id} retrieves status using explicit job identifier."""
    job_id = "job_test_explicit_001"
    now_iso = datetime.now(timezone.utc).isoformat()
    job = JobRecord(
        job_id=job_id,
        recording_id="rec_for_job_001",
        status="processing",
        progress=0.45,
        stage="Aligning phonemes",
        error=None,
        created_at=now_iso,
        updated_at=now_iso,
    )
    JobRepository.create(job)

    resp = client.get(f"/api/v1/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["status"] == "processing"
    assert data["progress"] == 0.45
    assert data["stage"] == "Aligning phonemes"


def test_job_retrieval_rejects_recording_id(client: TestClient):
    """GET /api/v1/jobs/{job_id} returns 404 when querying with a recording_id."""
    rec_id = "rec_not_a_job_id"
    _seed_recording_and_transcript(rec_id, "audio.wav", has_job=True, job_id="real_job_id_123")

    # Querying /jobs/ with recording_id must return 404
    resp = client.get(f"/api/v1/jobs/{rec_id}")
    assert resp.status_code == 404


def test_job_retrieval_unknown_job_returns_404(client: TestClient):
    """GET /api/v1/jobs/{job_id} returns 404 for nonexistent job."""
    resp = client.get("/api/v1/jobs/nonexistent-job-uuid")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Section 4: Plugin Registry and Discovery Tests
# ---------------------------------------------------------------------------
def test_plugin_registry_empty_by_default(client: TestClient):
    """GET /api/v1/plugins returns empty list when no plugins are registered."""
    resp = client.get("/api/v1/plugins")
    assert resp.status_code == 200
    assert resp.json() == []


def test_plugin_registration_and_manifest_discovery(client: TestClient):
    """Plugins register explicitly and appear in the discovery endpoint."""
    plugin = LexicalDiversityTestPlugin()
    PluginRegistry.register(plugin)

    assert PluginRegistry.get("lexical-analytics-reference") is plugin

    resp = client.get("/api/v1/plugins")
    assert resp.status_code == 200
    manifests = resp.json()
    assert len(manifests) == 1
    assert manifests[0]["plugin_id"] == "lexical-analytics-reference"
    assert "lexical_diversity" in manifests[0]["capabilities"]


def test_duplicate_plugin_registration_rejected():
    """Registering duplicate plugin IDs raises ValueError."""
    plugin = LexicalDiversityTestPlugin()
    PluginRegistry.register(plugin)

    with pytest.raises(ValueError, match="already registered"):
        PluginRegistry.register(plugin)


# ---------------------------------------------------------------------------
# Section 5: Plugin Execution, Error Sanitization & Immutability Tests
# ---------------------------------------------------------------------------
def test_plugin_execution_success(client: TestClient):
    """POST /api/v1/plugins/{plugin_id}/runs executes deterministic computation and persists output."""
    plugin = LexicalDiversityTestPlugin()
    PluginRegistry.register(plugin)

    rec_id = "rec_plugin_exec_001"
    _seed_recording_and_transcript(rec_id, "deposition.mp3")

    payload = {
        "recording_id": rec_id,
        "parameters": {"min_word_length": 4},
    }
    resp = client.post(f"/api/v1/plugins/{plugin.manifest.plugin_id}/runs", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["plugin_id"] == plugin.manifest.plugin_id
    assert data["recording_id"] == rec_id

    # Verify genuine computation
    insights = data["result_json"]
    assert insights["total_tokens"] == 3
    assert insights["unique_tokens"] == 3
    assert insights["type_token_ratio"] == 1.0
    assert insights["applied_min_length"] == 4
    # 'good', 'morning', 'counsel' -> all >= 4 chars
    assert insights["filtered_word_count"] == 3


def test_plugin_result_retrieval(client: TestClient):
    """GET /api/v1/plugins/{plugin_id}/results/{recording_id} retrieves stored result."""
    plugin = LexicalDiversityTestPlugin()
    PluginRegistry.register(plugin)

    rec_id = "rec_result_retrieval_001"
    _seed_recording_and_transcript(rec_id, "test.wav")

    # Run analysis
    client.post(f"/api/v1/plugins/{plugin.manifest.plugin_id}/runs", json={"recording_id": rec_id})

    # Retrieve analysis
    resp = client.get(f"/api/v1/plugins/{plugin.manifest.plugin_id}/results/{rec_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["recording_id"] == rec_id
    assert data["plugin_id"] == plugin.manifest.plugin_id
    assert data["result_json"]["total_tokens"] == 3


def test_plugin_execution_missing_transcript_returns_400(client: TestClient):
    """Executing plugin against untranscribed recording returns 400 error."""
    plugin = LexicalDiversityTestPlugin()
    PluginRegistry.register(plugin)

    rec_id = "rec_no_transcript_plugin"
    now_iso = datetime.now(timezone.utc).isoformat()
    RecordingRepository.create(
        RecordingRecord(rec_id, "no_tx.wav", "/p/no_tx.wav", 10.0, 500, "sha", now_iso)
    )

    resp = client.post(f"/api/v1/plugins/{plugin.manifest.plugin_id}/runs", json={"recording_id": rec_id})
    assert resp.status_code == 400
    assert "has not been transcribed yet" in resp.json()["detail"]


def test_plugin_execution_error_sanitization(client: TestClient):
    """Plugin exceptions must be sanitized so credentials/paths are not leaked in responses."""
    plugin = CrashingTestPlugin()
    PluginRegistry.register(plugin)

    rec_id = "rec_crash_sanitization"
    _seed_recording_and_transcript(rec_id, "crash_test.wav")

    resp = client.post(f"/api/v1/plugins/{plugin.manifest.plugin_id}/runs", json={"recording_id": rec_id})
    assert resp.status_code == 500
    detail = resp.json()["detail"]

    # Verify secret token pattern is redacted
    assert "hf_secretcredential123456789012345" not in detail
    assert "[REDACTED_HF_TOKEN]" in detail

    # Check persistence records error state separately
    latest = PluginRepository.get_latest(plugin.manifest.plugin_id, rec_id)
    assert latest is not None
    assert latest.status == "failed"
    assert "hf_secretcredential" not in latest.error


def test_canonical_transcript_immutability(client: TestClient):
    """Verify executing a plugin never mutates canonical transcript in transcripts table."""
    plugin = LexicalDiversityTestPlugin()
    PluginRegistry.register(plugin)

    rec_id = "rec_immutability_check"
    _, original_tx = _seed_recording_and_transcript(rec_id, "immutable.wav")

    original_json = json.dumps(TranscriptRepository.get(rec_id), sort_keys=True)

    # Execute plugin
    client.post(f"/api/v1/plugins/{plugin.manifest.plugin_id}/runs", json={"recording_id": rec_id})

    # Post-execution canonical transcript check
    post_json = json.dumps(TranscriptRepository.get(rec_id), sort_keys=True)
    assert original_json == post_json, "Canonical transcript was mutated during plugin execution!"


# ---------------------------------------------------------------------------
# Section 6: Typed Analytics Endpoint Verification
# ---------------------------------------------------------------------------
def test_typed_analytics_endpoint(client: TestClient):
    """GET /api/v1/analytics/{recording_id} conforms to typed AnalyticsResponse."""
    rec_id = "rec_analytics_verify"
    _seed_recording_and_transcript(rec_id, "analytics.wav")

    resp = client.get(f"/api/v1/analytics/{rec_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "tiers" in data
    assert "low" in data["tiers"]
    assert "high" in data["tiers"]
    assert "brier_score" in data
    assert "ece" in data
    assert data["total_words"] == 3


# ---------------------------------------------------------------------------
# Section 7: Database & Filesystem Isolation Regression Tests
# ---------------------------------------------------------------------------
def test_test_environment_is_isolated():
    """Verify test database is separate from production database path."""
    current_db = get_db_path().resolve()
    assert current_db != DEFAULT_DB_PATH.resolve()
    assert "cs698b_isolated_test" in str(current_db)


def test_test_storage_is_isolated():
    """Verify test storage directory is separate from production storage path."""
    current_storage = settings.storage_dir.resolve()
    prod_storage = (Path(__file__).resolve().parents[1] / "storage").resolve()
    assert current_storage != prod_storage
    assert "cs698b_isolated_test" in str(current_storage)


def test_upload_isolation_does_not_affect_production(client: TestClient):
    """Uploading a media file via test client writes strictly into temporary test storage."""
    file_bytes = b"RIFFtestaudiobytes16k"
    resp = client.post("/api/v1/media/upload", files={"file": ("isolation_check.wav", io.BytesIO(file_bytes), "audio/wav")})
    assert resp.status_code == 200
    data = resp.json()
    rec_id = data["recording_id"]

    # Verify file exists in isolated storage, NOT production storage
    prod_uploads = Path(__file__).resolve().parents[1] / "storage" / "uploads"
    prod_files = [f.name for f in prod_uploads.glob("*")]
    assert not any(rec_id in name for name in prod_files)


# ---------------------------------------------------------------------------
# Section 8: Streamlit Plugin Extension Tab AppTest Tests
# ---------------------------------------------------------------------------
def test_streamlit_plugin_tab_empty_notice():
    """Streamlit plugin tab renders clear static notice when no plugins are registered."""
    from unittest.mock import MagicMock

    at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
    rec_id = "rec_st_test_001"
    at.session_state["recording_id"] = rec_id
    at.session_state["filename"] = "meeting.wav"

    mock_tx = {
        "recording_id": rec_id,
        "filename": "meeting.wav",
        "duration": 10.0,
        "words": [{"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9, "speaker": "spk0"}],
        "segments": [{"turn_id": "t1", "speaker": "spk0", "start": 0.0, "end": 0.5, "text": "hello"}],
        "speakers": ["spk0"],
    }

    with patch("requests.get") as mock_get:
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "/health" in url:
                resp.status_code = 200
                resp.json.return_value = {"status": "ok"}
            elif "/media" in url and "/stream" not in url:
                resp.status_code = 200
                resp.json.return_value = [{"recording_id": rec_id, "filename": "meeting.wav"}]
            elif f"/transcriptions/{rec_id}" in url and "/status" not in url:
                resp.status_code = 200
                resp.json.return_value = mock_tx
            elif "/plugins" in url and "/results" not in url:
                resp.status_code = 200
                resp.json.return_value = []
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
        at.run()
        assert len(at.exception) == 0
        info_texts = [i.value for i in at.info]
        assert any("no plugin installed" in t.lower() for t in info_texts)
