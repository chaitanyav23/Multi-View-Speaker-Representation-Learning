"""
Integration Tests for ConvoFuse FastAPI Endpoints (`/api/v1/convofuse/*`) (Part 6.2 & Part 7.1).
"""

from __future__ import annotations

from pathlib import Path
import sys
from fastapi.testclient import TestClient

from backend.config import APP_DIR
from backend.main import app
from backend.services.convofuse_service import RESEARCH_REPO_ROOT

client = TestClient(app)


def test_convofuse_repository_b_self_contained_independence():
    """Verify the application and ConvoFuse run strictly inside Repository B with zero imports from Repository A."""
    repo_b_root = Path("/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning").resolve()
    repo_a_root = Path("/home/priyanka/chaitanya/Video-Transcription-with-Confidence-Analytics").resolve()

    assert RESEARCH_REPO_ROOT.resolve() == repo_b_root
    assert APP_DIR.resolve() == (repo_b_root / "app").resolve()

    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name.split(".")[0] in ("backend", "frontend", "engine", "adaptive", "convofuse"):
            mod_file = getattr(mod, "__file__", None)
            if mod_file:
                mod_path = Path(mod_file).resolve()
                assert not str(mod_path).startswith(str(repo_a_root)), (
                    f"Module {mod_name} imported from Repository A: {mod_path}"
                )
                assert str(mod_path).startswith(str(repo_b_root)), (
                    f"Module {mod_name} not inside Repository B: {mod_path}"
                )


def test_convofuse_r02_to_r08_benchmark_artifact_loading():
    """Verify R01-R08 benchmark artifacts are accessible through Repository B's ConvoFuse API."""
    for rec_id in (
        "R01_sara_sharif",
        "R02_dennis_skinner",
        "R03_disabled_parking",
        "R04_piers_robinson",
        "R05_southwest_pilot",
        "R06_rich_karen",
        "R07_bwc_cheap_shot",
        "R08_uk_parliament",
    ):
        meta_resp = client.get(f"/api/v1/convofuse/{rec_id}/metadata")
        assert meta_resp.status_code == 200, f"Metadata failed for {rec_id}: {meta_resp.text}"
        meta = meta_resp.json()
        assert meta["recording_id"] == rec_id
        assert meta["ground_truth_isolation_enforced"] is True

        tx_resp = client.get(f"/api/v1/convofuse/{rec_id}/transcript")
        assert tx_resp.status_code == 200, f"Transcript failed for {rec_id}: {tx_resp.text}"
        tx = tx_resp.json()
        assert tx["recording_id"] == rec_id
        assert tx["word_count"] > 0
        assert tx["turn_count"] > 0
        assert len(tx["readable_transcript_txt"]) > 0


def test_convofuse_list_benchmarks():
    resp = client.get("/api/v1/convofuse/benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    assert "benchmarks" in data
    assert len(data["benchmarks"]) == 8
    rec_ids = [b["recording_id"] for b in data["benchmarks"]]
    assert "R01_sara_sharif" in rec_ids
    assert "R08_uk_parliament" in rec_ids


def test_convofuse_r01_end_to_end_job_and_artifact_endpoints():
    # 1. Trigger R01 frozen-input replay job synchronously (with live DiarizationLM-Gemma via Ollama)
    job_resp = client.post(
        "/api/v1/convofuse/jobs",
        json={
            "recording_id": "R01",
            "execution_mode": "FROZEN_INPUT_REPLAY",
            "condition_preference": "E20",
            "enable_refinement": True,
            "use_locality_gating": True,
            "sync": True,
        },
    )
    assert job_resp.status_code == 200, job_resp.text
    job_data = job_resp.json()
    assert job_data["recording_id"] == "R01_sara_sharif"
    assert job_data["status"] == "completed"
    assert job_data["progress"] == 1.0
    job_id = job_data["job_id"]

    # 2. Check job status endpoint (includes 15-stage history)
    status_resp = client.get(f"/api/v1/convofuse/jobs/{job_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "completed"
    assert len(status_data["stage_history"]) >= 14
    stage_names = [s["stage"] for s in status_data["stage_history"]]
    for expected_stage in (
        "UPLOADED",
        "QUEUED",
        "AUDIO_EXTRACTION",
        "VAD",
        "ASR",
        "WORD_ALIGNMENT",
        "SPEAKER_REPRESENTATION",
        "AFFINITY_FUSION",
        "CLUSTERING",
        "WORD_SPEAKER_ASSIGNMENT",
        "GENERATIVE_REFINEMENT",
        "TRANSCRIPT_RECONSTRUCTION",
        "EVALUATION",
        "COMPLETED",
    ):
        assert expected_stage in stage_names

    # 3. Check metadata endpoint
    meta_resp = client.get("/api/v1/convofuse/R01_sara_sharif/metadata")
    assert meta_resp.status_code == 200
    meta = meta_resp.json()
    assert meta["recording_id"] == "R01_sara_sharif"
    assert meta["ground_truth_isolation_enforced"] is True
    assert meta["frozen_inputs_sha256_verified"] is True

    # 4. Check complete transcript bundle endpoint
    tx_resp = client.get("/api/v1/convofuse/R01_sara_sharif/transcript")
    assert tx_resp.status_code == 200
    tx = tx_resp.json()
    assert tx["word_count"] == 266
    assert tx["turn_count"] in (11, 13)
    assert "SPEAKER_00" in tx["speaker_inventory"]
    assert "[00:00:02.052 – 00:00:02.377] UNK: Yes sir." in tx["readable_transcript_txt"]

    # 5. Check word-level JSON endpoint
    words_resp = client.get("/api/v1/convofuse/R01_sara_sharif/words")
    assert words_resp.status_code == 200
    words_data = words_resp.json()
    assert len(words_data["words"]) == 266
    # Word 0 ("Yes") was UNK and must remain UNK
    assert words_data["words"][0]["baseline_speaker_id"] == "UNK"
    assert words_data["words"][0]["refined_speaker_id"] == "UNK"

    # 6. Check turn-level JSON endpoint
    turns_resp = client.get("/api/v1/convofuse/R01_sara_sharif/turns")
    assert turns_resp.status_code == 200
    turns_data = turns_resp.json()
    assert len(turns_data["turns"]) in (11, 13)
    assert len(turns_data["baseline_turns"]) == 11

    # 7. Check speaker summary index endpoint
    spks_resp = client.get("/api/v1/convofuse/R01_sara_sharif/speakers")
    assert spks_resp.status_code == 200
    spks_data = spks_resp.json()
    assert len(spks_data["speaker_summaries"]) == 4  # SPEAKER_00, SPEAKER_01, SPEAKER_02, UNK

    # 8. Check evaluation & comparison endpoint
    eval_resp = client.get("/api/v1/convofuse/R01_sara_sharif/evaluation")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert eval_data["evaluation_report"]["has_reference_annotations"] is True
    assert eval_data["evaluation_report"]["integrity_verification"]["word_sequence_intact"] is True
    assert eval_data["evaluation_report"]["integrity_verification"]["unk_invariance_verified"] is True

    # 9. Check export endpoints (txt & json)
    exp_txt = client.get("/api/v1/convofuse/R01_sara_sharif/export?format=txt")
    assert exp_txt.status_code == 200
    assert "ConvoFuse Conversational Transcript" in exp_txt.text

    for fmt in ("json", "words", "turns", "speakers", "comparison", "manifest", "evaluation"):
        exp_j = client.get(f"/api/v1/convofuse/R01_sara_sharif/export?format={fmt}")
        assert exp_j.status_code == 200

    # 10. Check backward compatibility with Phase20 `/api/v1/transcriptions/R01_sara_sharif`
    p20_resp = client.get("/api/v1/transcriptions/R01_sara_sharif")
    assert p20_resp.status_code == 200
    p20_data = p20_resp.json()
    assert p20_data["recording_id"] == "R01_sara_sharif"
    assert len(p20_data["words"]) == 266
    assert "convofuse" in p20_data["summary"]
