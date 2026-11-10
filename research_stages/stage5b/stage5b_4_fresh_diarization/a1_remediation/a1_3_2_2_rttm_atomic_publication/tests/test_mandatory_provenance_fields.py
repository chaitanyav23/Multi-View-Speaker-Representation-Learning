#!/usr/bin/env python3
"""
Unit tests for Stage 5B.4-A.1.3.2 Finding 3:
Mandatory Generation-Code Identity & Complete Provenance Field Validation.

Verifies:
1. Complete valid manifest passes verification.
2. Missing generation_code_version fails closed (no default substitution).
3. Null generation_code_version fails closed.
4. Empty generation_code_version fails closed.
5. Incorrect generation_code_version fails closed.
6. Missing each other mandatory field fails closed:
   - schema_version
   - model_architecture
   - model_checkpoint_sha256
   - generation_config_id
   - posterior_semantics
   - timestamp_convention
   - frame_step_sec
   - first_frame_center_sec
   - source_sample_rate
7. M4 production eligibility remains false when provenance is incomplete.
8. M0 execution remains independent of overlap posterior provenance.
"""

import copy
import hashlib
import json
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
    compute_file_sha256,
    create_synthetic_checkpoint_fixture,
)
from model_factory import ControlledModelFactory
from cache_manager_remediated import IsolatedEmbeddingCacheManagerRemediated
from clustering_backends import SyntheticSpectralClusteringBackend
from evaluation_adapters_remediated import SyntheticEvaluationAdapter
from vad_adapter_remediated import VADAdapterRemediated
from overlap_provenance import (
    OverlapProvenanceVerifier,
    OverlapProvenanceError,
    TRUSTED_PYANNET_POSTERIOR_SPEC,
    MANDATORY_PROVENANCE_FIELDS,
    PYANNET_FIRST_FRAME_CENTER,
    PYANNET_FRAME_STEP_SEC,
    PYANNET_SAMPLE_RATE,
)
from overlap_posterior_loader import OverlapPosteriorLoaderStrict
from orchestrator_remediated import FreshDiarizationOrchestratorRemediated


@pytest.fixture
def valid_manifest_fixture(tmp_path, monkeypatch):
    """Sets up a complete, valid manifest matching all 10 mandatory fields."""
    post_file = tmp_path / "rec_test_stream_headset_overlap.npz"
    f_times = PYANNET_FIRST_FRAME_CENTER + np.arange(100) * PYANNET_FRAME_STEP_SEC
    p_ovl = np.zeros(100, dtype=np.float32)
    np.savez(post_file, frame_times=f_times, p_overlap=p_ovl)
    post_sha = compute_file_sha256(post_file)

    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF____WAVEfmt " + b"\x00" * 200)
    audio_sha = compute_file_sha256(audio_file)

    ckpt_file = tmp_path / "pytorch_model.bin"
    ckpt_file.write_bytes(b"PYANNET_CKPT_BYTES" + b"\x44" * 128)
    ckpt_sha = compute_file_sha256(ckpt_file)
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", ckpt_sha)

    manifest_data = {
        "schema_version": "1.0",
        "artifacts": {
            "rec_test__stream_headset": {
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
                "source_audio_sha256": audio_sha,
                "source_duration_sec": 10.0,
            }
        }
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    return {
        "tmp_path": tmp_path,
        "post_file": post_file,
        "audio_file": audio_file,
        "ckpt_file": ckpt_file,
        "ckpt_sha": ckpt_sha,
        "manifest_file": manifest_file,
        "manifest_data": manifest_data,
    }


def test_1_complete_valid_manifest_passes(valid_manifest_fixture):
    """Verifies that a complete valid manifest passes verification with 0 errors."""
    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_manifest_fixture["manifest_file"],
        source_audio_root=valid_manifest_fixture["tmp_path"],
        pyannet_checkpoint_path=valid_manifest_fixture["ckpt_file"],
    )
    state = verifier.verify_provenance("rec_test", "stream_headset", valid_manifest_fixture["post_file"])
    assert state.provenance_verified is True
    assert len(state.provenance_errors) == 0
    assert len(state.unverified_fields) == 0


def test_2_missing_generation_code_version_fails(valid_manifest_fixture):
    """
    Finding 3 Critical: Missing generation_code_version MUST fail closed.
    Proves that silent default substitution is eliminated!
    """
    data = copy.deepcopy(valid_manifest_fixture["manifest_data"])
    del data["artifacts"]["rec_test__stream_headset"]["generation_code_version"]
    valid_manifest_fixture["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_manifest_fixture["manifest_file"],
        source_audio_root=valid_manifest_fixture["tmp_path"],
        pyannet_checkpoint_path=valid_manifest_fixture["ckpt_file"],
    )
    state = verifier.verify_provenance("rec_test", "stream_headset", valid_manifest_fixture["post_file"])
    assert state.provenance_verified is False
    assert any("generation_code_version' is missing" in err for err in state.provenance_errors)
    assert "generation_code_version_missing" in state.unverified_fields


def test_3_null_generation_code_version_fails(valid_manifest_fixture):
    """Null generation_code_version must fail closed."""
    data = copy.deepcopy(valid_manifest_fixture["manifest_data"])
    data["artifacts"]["rec_test__stream_headset"]["generation_code_version"] = None
    valid_manifest_fixture["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_manifest_fixture["manifest_file"],
        source_audio_root=valid_manifest_fixture["tmp_path"],
        pyannet_checkpoint_path=valid_manifest_fixture["ckpt_file"],
    )
    state = verifier.verify_provenance("rec_test", "stream_headset", valid_manifest_fixture["post_file"])
    assert state.provenance_verified is False
    assert any("generation_code_version' is null" in err for err in state.provenance_errors)


def test_4_empty_generation_code_version_fails(valid_manifest_fixture):
    """Empty string generation_code_version must fail closed."""
    data = copy.deepcopy(valid_manifest_fixture["manifest_data"])
    data["artifacts"]["rec_test__stream_headset"]["generation_code_version"] = "   "
    valid_manifest_fixture["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_manifest_fixture["manifest_file"],
        source_audio_root=valid_manifest_fixture["tmp_path"],
        pyannet_checkpoint_path=valid_manifest_fixture["ckpt_file"],
    )
    state = verifier.verify_provenance("rec_test", "stream_headset", valid_manifest_fixture["post_file"])
    assert state.provenance_verified is False
    assert any("generation_code_version' is empty" in err for err in state.provenance_errors)


def test_5_incorrect_generation_code_version_fails(valid_manifest_fixture):
    """Mismatched generation_code_version must fail closed."""
    data = copy.deepcopy(valid_manifest_fixture["manifest_data"])
    data["artifacts"]["rec_test__stream_headset"]["generation_code_version"] = "unauthorized_custom_script_v2"
    valid_manifest_fixture["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_manifest_fixture["manifest_file"],
        source_audio_root=valid_manifest_fixture["tmp_path"],
        pyannet_checkpoint_path=valid_manifest_fixture["ckpt_file"],
    )
    state = verifier.verify_provenance("rec_test", "stream_headset", valid_manifest_fixture["post_file"])
    assert state.provenance_verified is False
    assert any("Generation code version mismatch" in err for err in state.provenance_errors)


@pytest.mark.parametrize("missing_field", [
    "model_architecture",
    "model_checkpoint_sha256",
    "generation_config_id",
    "posterior_semantics",
    "timestamp_convention",
    "frame_step_sec",
    "first_frame_center_sec",
    "source_sample_rate",
])
def test_6_missing_other_mandatory_fields_fail(valid_manifest_fixture, missing_field):
    """Verifies that missing each other mandatory field in entry fails closed."""
    data = copy.deepcopy(valid_manifest_fixture["manifest_data"])
    del data["artifacts"]["rec_test__stream_headset"][missing_field]
    valid_manifest_fixture["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_manifest_fixture["manifest_file"],
        source_audio_root=valid_manifest_fixture["tmp_path"],
        pyannet_checkpoint_path=valid_manifest_fixture["ckpt_file"],
    )
    state = verifier.verify_provenance("rec_test", "stream_headset", valid_manifest_fixture["post_file"])
    assert state.provenance_verified is False
    assert any(f"'{missing_field}' is missing" in err for err in state.provenance_errors)


def test_6b_missing_schema_version_fails(valid_manifest_fixture):
    """Verifies that missing top-level schema_version fails closed."""
    data = copy.deepcopy(valid_manifest_fixture["manifest_data"])
    del data["schema_version"]
    valid_manifest_fixture["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_manifest_fixture["manifest_file"],
        source_audio_root=valid_manifest_fixture["tmp_path"],
        pyannet_checkpoint_path=valid_manifest_fixture["ckpt_file"],
    )
    state = verifier.verify_provenance("rec_test", "stream_headset", valid_manifest_fixture["post_file"])
    assert state.provenance_verified is False
    assert any("schema_version" in err for err in state.provenance_errors)


def test_7_m4_production_eligibility_false_on_incomplete_provenance(valid_manifest_fixture):
    """Incomplete provenance causes OverlapPosteriorLoaderStrict to set production_eligible=False."""
    data = copy.deepcopy(valid_manifest_fixture["manifest_data"])
    del data["artifacts"]["rec_test__stream_headset"]["generation_code_version"]
    valid_manifest_fixture["manifest_file"].write_text(json.dumps(data))

    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=valid_manifest_fixture["tmp_path"],
        provenance_manifest_path=valid_manifest_fixture["manifest_file"],
        source_audio_root=valid_manifest_fixture["tmp_path"],
        pyannet_checkpoint_path=valid_manifest_fixture["ckpt_file"],
        execution_mode=ExecutionMode.PRODUCTION,
    )

    container = loader.load_and_validate("rec_test", stream="stream_headset")
    assert container.provenance_state.provenance_verified is False
    assert container.overall_state.production_eligible is False


class MockTitaNet(torch.nn.Module):
    def forward(self, input_signal=None, input_signal_length=None, **kwargs):
        return None, torch.randn(1, 192)


class MockWeSpeaker(torch.nn.Module):
    def forward(self, feats=None, **kwargs):
        return None, torch.randn(1, 256)


def test_8_m0_execution_independent_of_overlap_provenance(tmp_path):
    """Verifies that Condition M0 execution is completely unaffected by missing overlap provenance."""
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

    orch = FreshDiarizationOrchestratorRemediated(
        titanet_handle=tita_handle,
        wespeaker_handle=wes_handle,
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=cluster_backend,
        evaluation_adapter=eval_adapter,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=tmp_path / "outputs",
    )

    wav = np.zeros(16000 * 2, dtype=np.float32)
    f_times = PYANNET_FIRST_FRAME_CENTER + np.arange(100) * PYANNET_FRAME_STEP_SEC
    p_speech = np.ones(100, dtype=np.float32)

    # Condition E00 = V0 (Oracle) + M0 (No overlap)
    stats = orch.process_recording(
        condition_id="E00",
        recording_id="syn_m0_test",
        dfn_waveform=wav,
        mono_waveform=wav,
        posteriors_override=(f_times, p_speech, None),
        sample_rate=16000,
    )

    assert stats.m0_execution_eligible is True
    assert stats.overlap_provenance_verified is False
    assert stats.overlap_production_eligible is False
    assert stats.m4_execution_eligible is False
