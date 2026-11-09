#!/usr/bin/env python3
"""
Integration Scenarios Test Suite (Section 5):
Synthetic non-inference cross-component scenarios A through G.
Verifies full end-to-end control flow across preflight, model factory, cache,
overlap provenance, RTTM output guard, and orchestrator.

Scenarios:
- Scenario A: Valid synthetic production-style M0 path.
- Scenario B: Valid synthetic M4 path.
- Scenario C: Invalid source provenance fails closed.
- Scenario D: Invalid model identity fails closed.
- Scenario E: Checkpoint mutation fails closed (TOCTOU).
- Scenario F: RTTM symlink race defeated.
- Scenario G: Historical mono provenance fails closed for production M4.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_preflight import (
    ExecutionMode,
    VerifiedCheckpoint,
    verify_encoder_checkpoint_strict,
    create_synthetic_checkpoint_fixture,
    compute_file_sha256,
)
from model_factory import (
    ControlledModelFactory,
    VerifiedModelHandle,
    ModelFactoryError,
)
from cache_manager_remediated import IsolatedEmbeddingCacheManagerRemediated
from clustering_backends import SyntheticSpectralClusteringBackend
from evaluation_adapters_remediated import SyntheticEvaluationAdapter
from vad_adapter_remediated import VADAdapterRemediated
from output_path_guard import (
    validate_final_rttm_path,
    revalidate_and_write_rttm,
    OutputSecurityError,
)
from overlap_provenance import (
    OverlapProvenanceVerifier,
    OverlapProvenanceError,
    TRUSTED_PYANNET_POSTERIOR_SPEC,
    PYANNET_FIRST_FRAME_CENTER,
    PYANNET_FRAME_STEP_SEC,
    PYANNET_SAMPLE_RATE,
)
from overlap_posterior_loader import OverlapPosteriorLoaderStrict
from orchestrator_remediated import FreshDiarizationOrchestratorRemediated


class MockTitaNet(torch.nn.Module):
    def forward(self, input_signal=None, input_signal_length=None, **kwargs):
        return None, torch.randn(1, 192)


class MockWeSpeaker(torch.nn.Module):
    def forward(self, feats=None, **kwargs):
        return None, torch.randn(1, 256)


@pytest.fixture
def base_env(tmp_path):
    tita_file = tmp_path / "tita.pt"
    tita_token = create_synthetic_checkpoint_fixture("titanet_large", tita_file)
    tita_handle = ControlledModelFactory.build_synthetic_model(tita_token, MockTitaNet())

    wes_file = tmp_path / "wes.pt"
    wes_token = create_synthetic_checkpoint_fixture("wespeaker_resnet293", wes_file)
    wes_handle = ControlledModelFactory.build_synthetic_model(wes_token, MockWeSpeaker())

    cache_dir = tmp_path / "cache"
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(cache_dir, allow_custom_root=True)
    vad_adapter = VADAdapterRemediated()
    eval_adapter = SyntheticEvaluationAdapter()
    cluster_backend = SyntheticSpectralClusteringBackend()
    out_dir = tmp_path / "outputs"

    orch = FreshDiarizationOrchestratorRemediated(
        titanet_handle=tita_handle,
        wespeaker_handle=wes_handle,
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=cluster_backend,
        evaluation_adapter=eval_adapter,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=out_dir,
    )

    return {
        "tmp_path": tmp_path,
        "orch": orch,
        "out_dir": out_dir,
        "tita_file": tita_file,
        "wes_file": wes_file,
        "tita_token": tita_token,
        "wes_token": wes_token,
    }


def test_scenario_a_valid_synthetic_m0_path(base_env):
    """
    Scenario A: Valid synthetic production-style M0 path:
    M0 condition executes, RTTM is written, m0_execution_eligible=True,
    overlap_provenance_verified=False, overlap_production_eligible=False.
    """
    orch = base_env["orch"]
    rec_id = "syn_scenario_a"
    wav = np.zeros(16000 * 2, dtype=np.float32)

    f_times = PYANNET_FIRST_FRAME_CENTER + np.arange(100) * PYANNET_FRAME_STEP_SEC
    p_speech = np.ones(100, dtype=np.float32)

    stats = orch.process_recording(
        condition_id="E00",
        recording_id=rec_id,
        dfn_waveform=wav,
        mono_waveform=wav,
        posteriors_override=(f_times, p_speech, None),
        sample_rate=16000,
    )

    assert stats.m0_execution_eligible is True
    assert stats.overlap_provenance_verified is False
    assert stats.overlap_production_eligible is False
    assert stats.m4_execution_eligible is False
    assert stats.structurally_valid is True

    # Confirm RTTM written to canonical layout
    expected_rttm = base_env["out_dir"] / "E00" / f"{rec_id}.rttm"
    assert expected_rttm.exists()
    assert expected_rttm.stat().st_size > 0


def test_scenario_b_valid_synthetic_m4_path(base_env):
    """
    Scenario B: Valid synthetic M4 path:
    M4 condition executes with synthetic arrays, structural validation passes,
    but production_eligible is False because synthetic fixture can never be production eligible.
    """
    orch = base_env["orch"]
    rec_id = "syn_scenario_b"
    wav = np.zeros(16000 * 3, dtype=np.float32)

    f_times = PYANNET_FIRST_FRAME_CENTER + np.arange(120) * PYANNET_FRAME_STEP_SEC
    p_speech = np.ones(120, dtype=np.float32)
    p_overlap = np.zeros(120, dtype=np.float32)

    stats = orch.process_recording(
        condition_id="E01",
        recording_id=rec_id,
        dfn_waveform=wav,
        mono_waveform=wav,
        posteriors_override=(f_times, p_speech, p_overlap),
        sample_rate=16000,
    )

    assert stats.m0_execution_eligible is False
    assert stats.overlap_provenance_verified is False
    assert stats.overlap_production_eligible is False
    assert stats.m4_execution_eligible is False
    assert stats.structurally_valid is True
    assert stats.overlap_method == "M4_conservative_abstention"

    expected_rttm = base_env["out_dir"] / "E01" / f"{rec_id}.rttm"
    assert expected_rttm.exists()


def test_scenario_c_invalid_source_provenance_fails_closed(tmp_path, monkeypatch):
    """
    Scenario C: Invalid source audio provenance:
    Physical source audio bytes mismatch the manifest SHA-256.
    Verification fails closed, preventing production eligibility.
    """
    post_file = tmp_path / "rec_01_stream_headset_overlap.npz"
    f_times = PYANNET_FIRST_FRAME_CENTER + np.arange(100) * PYANNET_FRAME_STEP_SEC
    p_ovl = np.zeros(100, dtype=np.float32)
    np.savez(post_file, frame_times=f_times, p_overlap=p_ovl)
    post_sha = compute_file_sha256(post_file)

    # Real source audio on disk
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"CORRUPTED_MODIFIED_AUDIO_CONTENT")

    ckpt_file = tmp_path / "pytorch_model.bin"
    ckpt_file.write_bytes(b"PYANNET_CKPT_BYTES" * 10)
    ckpt_sha = compute_file_sha256(ckpt_file)
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", ckpt_sha)

    manifest_data = {
        "schema_version": "1.0",
        "artifacts": {
            "rec_01__stream_headset": {
                "posterior_file_rel": post_file.name,
                "posterior_file_sha256": post_sha,
                "posterior_file_size_bytes": post_file.stat().st_size,
                "model_architecture": "pyannote/segmentation",
                "model_checkpoint_sha256": ckpt_sha,
                "generation_config_id": "pyannote_3_1_continuous_top2",
                "posterior_semantics": "continuous_top2_second_highest",
                "timestamp_convention": "sliding_window_middle_centering",
                "frame_step_sec": PYANNET_FRAME_STEP_SEC,
                "first_frame_center_sec": PYANNET_FIRST_FRAME_CENTER,
                "source_sample_rate": PYANNET_SAMPLE_RATE,
                "generation_code_version": "stage2_baseline_v1",
                "source_audio_rel": "audio.wav",
                "source_audio_sha256": hashlib.sha256(b"DIFFERENT_ORIGINAL_AUDIO").hexdigest(),
                "source_duration_sec": 5.0,
            }
        }
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    verifier = OverlapProvenanceVerifier(
        manifest_path=manifest_file,
        source_audio_root=tmp_path,
        pyannet_checkpoint_path=ckpt_file,
    )
    state = verifier.verify_provenance("rec_01", "stream_headset", post_file, ExecutionMode.PRODUCTION)
    assert state.provenance_verified is False
    assert any("Source audio file hash mismatch" in err for err in state.provenance_errors)


def test_scenario_d_invalid_model_identity_fails_closed(tmp_path):
    """
    Scenario D: Invalid model identity:
    Attempting to forge a VerifiedModelHandle or inject unauthorized model fails closed.
    """
    ckpt_file = tmp_path / "mock.pt"
    token = create_synthetic_checkpoint_fixture("titanet_large", ckpt_file)

    with pytest.raises(ModelFactoryError, match="FORGERY DETECTED"):
        VerifiedModelHandle(
            encoder_name="titanet_large",
            verified_checkpoint=token,
            model_instance=MockTitaNet(),
            creation_timestamp="2026-11-09T12:00:00Z",
            is_synthetic=False,
            _handle_signature="FAKE_ATTACKER_SIGNATURE",
        )


def test_scenario_e_checkpoint_mutation_detected(tmp_path, monkeypatch):
    """
    Scenario E: Checkpoint mutation:
    File is mutated on disk after preflight token is generated.
    ControlledModelFactory detects mutation during in-memory buffer read and aborts.
    """
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [tmp_path.resolve()],
    )
    ckpt_file = tmp_path / "model.pt"
    orig_bytes = b"ORIGINAL_BYTES_1234567890" * 8
    ckpt_file.write_bytes(orig_bytes)
    sha = hashlib.sha256(orig_bytes).hexdigest()

    token = verify_encoder_checkpoint_strict(
        encoder_name="wespeaker_resnet293",
        checkpoint_path=ckpt_file,
        expected_sha256=sha,
        execution_mode=ExecutionMode.PRODUCTION,
    )

    # Mutate checkpoint file on disk with same size
    ckpt_file.write_bytes(b"X" * len(orig_bytes))

    with pytest.raises(ModelFactoryError, match="TOCTOU validation failure"):
        ControlledModelFactory.read_and_verify_checkpoint_bytes(token)


def test_scenario_f_rttm_symlink_attack_defeated(tmp_path):
    """
    Scenario F: RTTM symlink race:
    Target RTTM path replaced with symlink to sentinel file.
    revalidate_and_write_rttm rejects write with O_NOFOLLOW and sentinel file is untouched.
    """
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("SENSITIVE_DATA", encoding="utf-8")

    path_obj = validate_final_rttm_path(
        recording_id="rec_attack",
        condition_id="E21",
        custom_output_root=out_root,
        execution_mode=ExecutionMode.SYNTHETIC,
    )

    # Symlink race injection
    path_obj.final_path.symlink_to(sentinel)

    with pytest.raises(OutputSecurityError):
        revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "ATTACKER"}])

    assert sentinel.read_text(encoding="utf-8") == "SENSITIVE_DATA"


def test_scenario_g_historical_mono_provenance_fails_closed(tmp_path, monkeypatch):
    """
    Scenario G: Historical mono provenance limitation:
    Historical mono audio has UNRECORDED_IN_HISTORICAL_MANIFEST in manifest.
    Provenance verification fails closed, and production M4 execution is rejected.
    """
    post_file = tmp_path / "rec_mono_stream_mono_overlap.npz"
    f_times = PYANNET_FIRST_FRAME_CENTER + np.arange(100) * PYANNET_FRAME_STEP_SEC
    p_ovl = np.zeros(100, dtype=np.float32)
    np.savez(post_file, frame_times=f_times, p_overlap=p_ovl)
    post_sha = compute_file_sha256(post_file)

    ckpt_file = tmp_path / "pytorch_model.bin"
    ckpt_file.write_bytes(b"PYANNET_CKPT_BYTES" * 10)
    ckpt_sha = compute_file_sha256(ckpt_file)
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", ckpt_sha)

    manifest_data = {
        "schema_version": "1.0",
        "artifacts": {
            "rec_mono__stream_mono": {
                "posterior_file_rel": post_file.name,
                "posterior_file_sha256": post_sha,
                "posterior_file_size_bytes": post_file.stat().st_size,
                "model_architecture": "pyannote/segmentation",
                "model_checkpoint_sha256": ckpt_sha,
                "generation_config_id": "pyannote_3_1_continuous_top2",
                "posterior_semantics": "continuous_top2_second_highest",
                "timestamp_convention": "sliding_window_middle_centering",
                "frame_step_sec": PYANNET_FRAME_STEP_SEC,
                "first_frame_center_sec": PYANNET_FIRST_FRAME_CENTER,
                "source_sample_rate": PYANNET_SAMPLE_RATE,
                "generation_code_version": "stage2_baseline_v1",
                "source_audio_rel": "mono.wav",
                "source_audio_sha256": "UNRECORDED_IN_HISTORICAL_MANIFEST",
                "source_duration_sec": 5.0,
            }
        }
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=tmp_path,
        provenance_manifest_path=manifest_file,
        source_audio_root=tmp_path,
        pyannet_checkpoint_path=ckpt_file,
        execution_mode=ExecutionMode.PRODUCTION,
    )

    container = loader.load_and_validate("rec_mono", stream="stream_mono")
    assert container.provenance_state.provenance_verified is False
    assert "source_audio_sha256" in container.provenance_state.unverified_fields
    assert container.overall_state.production_eligible is False
