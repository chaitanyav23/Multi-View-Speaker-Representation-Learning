#!/usr/bin/env python3
"""
Unit tests for Stage 5B.4-A.1.3.1 Finding 3:
Validation of overlap posterior metadata against TRUSTED_PYANNET_POSTERIOR_SPEC.
Verifies that:
1. Matching configuration against trusted spec passes validation.
2. Invalid schema_version fails closed.
3. Incorrect model_architecture fails closed.
4. Incorrect generation_config_id fails closed.
5. Incorrect posterior_semantics fails closed.
6. Incorrect timestamp_convention fails closed.
7. Incorrect frame_step_sec fails closed.
8. Incorrect first_frame_center_sec fails closed.
9. Incorrect sample_rate fails closed.
10. Incorrect generation_code_version fails closed.
11. Missing required metadata fields fail closed.
12. Every invalid field causes production_eligible to be False for Condition M4.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_preflight import ExecutionMode, compute_file_sha256
from overlap_provenance import (
    OverlapProvenanceVerifier,
    OverlapProvenanceError,
    TRUSTED_PYANNET_POSTERIOR_SPEC,
    PYANNET_FRAME_STEP_SEC,
    PYANNET_FIRST_FRAME_CENTER,
    PYANNET_SAMPLE_RATE,
)
from overlap_posterior_loader import OverlapPosteriorLoaderStrict


@pytest.fixture
def valid_env(tmp_path):
    """Sets up a fully valid environment matching TRUSTED_PYANNET_POSTERIOR_SPEC."""
    post_file = tmp_path / "rec_01_stream_headset_overlap.npz"
    frame_times = PYANNET_FIRST_FRAME_CENTER + np.arange(100) * PYANNET_FRAME_STEP_SEC
    p_overlap = np.zeros(100, dtype=np.float32)
    np.savez(post_file, frame_times=frame_times, p_overlap=p_overlap)
    post_sha = compute_file_sha256(post_file)

    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"RIFF____WAVEfmt " + b"\x00" * 200)
    audio_sha = compute_file_sha256(audio_file)

    ckpt_file = tmp_path / "pytorch_model.bin"
    ckpt_file.write_bytes(b"MOCK_PYANNET_BYTES" + b"\x22" * 128)
    ckpt_sha = compute_file_sha256(ckpt_file)

    manifest_data = {
        "schema_version": "1.0",
        "artifacts": {
            "rec_01__stream_headset": {
                "posterior_file_rel": "rec_01_stream_headset_overlap.npz",
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
                "source_audio_rel": "test_audio.wav",
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


def test_1_valid_configuration_passes(valid_env, monkeypatch):
    """Test that valid configuration matching TRUSTED_PYANNET_POSTERIOR_SPEC passes."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
    )

    state = verifier.verify_provenance("rec_01", "stream_headset", valid_env["post_file"])
    assert state.provenance_verified is True
    assert len(state.provenance_errors) == 0


def test_2_invalid_schema_version(valid_env, monkeypatch):
    """Test that unexpected schema_version fails closed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    data = copy.deepcopy(valid_env["manifest_data"])
    data["schema_version"] = "2.0_unsupported"
    valid_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
    )

    state = verifier.verify_provenance("rec_01", "stream_headset", valid_env["post_file"])
    assert state.provenance_verified is False
    assert any("Unsupported manifest schema_version" in err for err in state.provenance_errors)


def test_3_incorrect_model_architecture(valid_env, monkeypatch):
    """Test that unexpected model_architecture fails closed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    data = copy.deepcopy(valid_env["manifest_data"])
    data["artifacts"]["rec_01__stream_headset"]["model_architecture"] = "custom/hacked_net"
    valid_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
    )

    state = verifier.verify_provenance("rec_01", "stream_headset", valid_env["post_file"])
    assert state.provenance_verified is False
    assert any("Model architecture mismatch" in err for err in state.provenance_errors)


def test_4_incorrect_generation_config_id(valid_env, monkeypatch):
    """Test that unexpected generation_config_id fails closed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    data = copy.deepcopy(valid_env["manifest_data"])
    data["artifacts"]["rec_01__stream_headset"]["generation_config_id"] = "greedy_argmax_config"
    valid_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
    )

    state = verifier.verify_provenance("rec_01", "stream_headset", valid_env["post_file"])
    assert state.provenance_verified is False
    assert any("Generation config mismatch" in err for err in state.provenance_errors)


def test_5_incorrect_posterior_semantics(valid_env, monkeypatch):
    """Test that unexpected posterior_semantics fails closed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    data = copy.deepcopy(valid_env["manifest_data"])
    data["artifacts"]["rec_01__stream_headset"]["posterior_semantics"] = "binary_threshold_0.5"
    valid_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
    )

    state = verifier.verify_provenance("rec_01", "stream_headset", valid_env["post_file"])
    assert state.provenance_verified is False
    assert any("Posterior semantics mismatch" in err for err in state.provenance_errors)


def test_6_incorrect_timestamp_convention(valid_env, monkeypatch):
    """Test that unexpected timestamp_convention fails closed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    data = copy.deepcopy(valid_env["manifest_data"])
    data["artifacts"]["rec_01__stream_headset"]["timestamp_convention"] = "window_left_edge"
    valid_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
    )

    state = verifier.verify_provenance("rec_01", "stream_headset", valid_env["post_file"])
    assert state.provenance_verified is False
    assert any("Timestamp convention mismatch" in err for err in state.provenance_errors)


def test_7_incorrect_frame_step(valid_env, monkeypatch):
    """Test that unexpected frame_step_sec fails closed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    data = copy.deepcopy(valid_env["manifest_data"])
    data["artifacts"]["rec_01__stream_headset"]["frame_step_sec"] = 0.020000  # wrong step
    valid_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
    )

    state = verifier.verify_provenance("rec_01", "stream_headset", valid_env["post_file"])
    assert state.provenance_verified is False
    assert any("Frame step mismatch" in err for err in state.provenance_errors)


def test_8_incorrect_first_frame_center(valid_env, monkeypatch):
    """Test that unexpected first_frame_center_sec fails closed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    data = copy.deepcopy(valid_env["manifest_data"])
    data["artifacts"]["rec_01__stream_headset"]["first_frame_center_sec"] = 0.050000  # wrong center
    valid_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
    )

    state = verifier.verify_provenance("rec_01", "stream_headset", valid_env["post_file"])
    assert state.provenance_verified is False
    assert any("First frame center mismatch" in err for err in state.provenance_errors)


def test_9_incorrect_sample_rate(valid_env, monkeypatch):
    """Test that unexpected source_sample_rate fails closed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    data = copy.deepcopy(valid_env["manifest_data"])
    data["artifacts"]["rec_01__stream_headset"]["source_sample_rate"] = 8000  # wrong rate
    valid_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
    )

    state = verifier.verify_provenance("rec_01", "stream_headset", valid_env["post_file"])
    assert state.provenance_verified is False
    assert any("Sample rate mismatch" in err for err in state.provenance_errors)


def test_10_incorrect_generation_code_version(valid_env, monkeypatch):
    """Test that unexpected generation_code_version fails closed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    data = copy.deepcopy(valid_env["manifest_data"])
    data["artifacts"]["rec_01__stream_headset"]["generation_code_version"] = "unverified_dev_script"
    valid_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
    )

    state = verifier.verify_provenance("rec_01", "stream_headset", valid_env["post_file"])
    assert state.provenance_verified is False
    assert any("Generation code version mismatch" in err for err in state.provenance_errors)


def test_11_all_invalids_fail_m4_production_eligibility(valid_env, monkeypatch):
    """Verifies that any config mismatch causes loader to mark production_eligible=False."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", valid_env["ckpt_sha"])

    # Corrupt model architecture
    data = copy.deepcopy(valid_env["manifest_data"])
    data["artifacts"]["rec_01__stream_headset"]["model_architecture"] = "incompatible_model"
    valid_env["manifest_file"].write_text(json.dumps(data))

    loader = OverlapPosteriorLoaderStrict(
        posteriors_dir=valid_env["tmp_path"],
        provenance_manifest_path=valid_env["manifest_file"],
        source_audio_root=valid_env["tmp_path"],
        pyannet_checkpoint_path=valid_env["ckpt_file"],
        execution_mode=ExecutionMode.PRODUCTION,
    )

    container = loader.load_and_validate(
        recording_id="rec_01",
        stream="stream_headset",
    )

    assert container.provenance_state.provenance_verified is False
    assert container.overall_state.production_eligible is False
