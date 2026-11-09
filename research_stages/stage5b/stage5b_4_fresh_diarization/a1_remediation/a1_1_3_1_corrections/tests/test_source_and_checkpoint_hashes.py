#!/usr/bin/env python3
"""
Unit tests for Stage 5B.4-A.1.3.1 Finding 2:
Physical byte-level hashing of source audio and model checkpoints.
Verifies:
1. Matching source audio bytes and digest.
2. Modified source audio bytes detected as mismatch.
3. Incorrect source audio digest in manifest.
4. Missing source audio file.
5. Path traversal ('..') in source audio path.
6. Source audio symlink escape.
7. Matching PyanNet checkpoint bytes.
8. Modified PyanNet checkpoint bytes.
9. Missing PyanNet checkpoint file.
10. Posterior file hash mismatch and placeholder hashes.
11. Historically unrecorded mono source hash ('UNRECORDED_IN_HISTORICAL_MANIFEST').
12. Manifest string alone without physical byte hashing fails verification.
"""

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
    ProvenanceValidationState,
    TRUSTED_PYANNET_POSTERIOR_SPEC,
    PYANNET_FRAME_STEP_SEC,
    PYANNET_FIRST_FRAME_CENTER,
    PYANNET_SAMPLE_RATE,
)


@pytest.fixture
def test_env(tmp_path):
    """Creates a temporary testing environment with synthetic posterior, audio, and checkpoint."""
    # 1. Posterior file
    post_file = tmp_path / "test_posterior.npy"
    data = np.zeros((100, 2), dtype=np.float32)
    np.save(post_file, data)
    post_sha = compute_file_sha256(post_file)
    post_size = post_file.stat().st_size

    # 2. Source audio file
    audio_file = tmp_path / "test_audio.wav"
    audio_bytes = b"RIFF____WAVEfmt " + b"\x00" * 200
    audio_file.write_bytes(audio_bytes)
    audio_sha = compute_file_sha256(audio_file)

    # 3. PyanNet model checkpoint
    ckpt_file = tmp_path / "pytorch_model.bin"
    ckpt_bytes = b"MOCK_PYANNET_CHECKPOINT_BYTES_" + b"\x11" * 128
    ckpt_file.write_bytes(ckpt_bytes)
    ckpt_sha = compute_file_sha256(ckpt_file)

    # 4. Manifest
    manifest_data = {
        "schema_version": "1.0",
        "description": "Test overlap provenance manifest",
        "artifacts": {
            "rec_test__stream_headset": {
                "posterior_file_rel": "test_posterior.npy",
                "posterior_file_sha256": post_sha,
                "posterior_file_size_bytes": post_size,
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
            },
            "rec_mono__stream_mono": {
                "posterior_file_rel": "test_posterior.npy",
                "posterior_file_sha256": post_sha,
                "posterior_file_size_bytes": post_size,
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
                "source_audio_sha256": "UNRECORDED_IN_HISTORICAL_MANIFEST",
                "source_duration_sec": 10.0,
            },
        },
    }

    manifest_file = tmp_path / "test_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data, indent=2))

    return {
        "tmp_path": tmp_path,
        "post_file": post_file,
        "audio_file": audio_file,
        "ckpt_file": ckpt_file,
        "manifest_file": manifest_file,
        "ckpt_sha": ckpt_sha,
        "audio_sha": audio_sha,
        "post_sha": post_sha,
    }


def test_1_matching_source_audio_bytes_and_digest(test_env, monkeypatch):
    """Verifies that matching physical source audio bytes and model checkpoint succeed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])
    
    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )
    
    state = verifier.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    assert state.provenance_verified is True
    assert len(state.provenance_errors) == 0
    assert len(state.unverified_fields) == 0
    assert state.source_audio_sha256 == test_env["audio_sha"]
    assert state.model_checkpoint_sha256 == test_env["ckpt_sha"]


def test_2_modified_source_audio_bytes_detected(test_env, monkeypatch):
    """Verifies that mutating physical source audio bytes on disk is detected."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])
    
    # Mutate audio bytes
    test_env["audio_file"].write_bytes(b"CORRUPTED_BYTES" + b"\x99" * 100)

    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state = verifier.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    assert state.provenance_verified is False
    assert any("Source audio file hash mismatch" in err for err in state.provenance_errors)


def test_3_incorrect_source_audio_digest_in_manifest(test_env, monkeypatch):
    """Verifies that a manifest containing a forged/wrong source audio digest fails."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])
    
    # Alter manifest entry with incorrect audio digest
    data = json.loads(test_env["manifest_file"].read_text())
    data["artifacts"]["rec_test__stream_headset"]["source_audio_sha256"] = hashlib.sha256(b"non_matching_audio").hexdigest()
    test_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state = verifier.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    assert state.provenance_verified is False
    assert any("Source audio file hash mismatch" in err for err in state.provenance_errors)


def test_4_missing_source_audio_file(test_env, monkeypatch):
    """Verifies that missing source audio file causes immediate failure."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])
    
    test_env["audio_file"].unlink()

    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state = verifier.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    assert state.provenance_verified is False
    assert any("Source audio file does not exist" in err for err in state.provenance_errors)


def test_5_path_traversal_in_source_audio_rejected(test_env, monkeypatch):
    """Verifies that path traversal ('..') in source audio relative path is rejected."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])

    data = json.loads(test_env["manifest_file"].read_text())
    data["artifacts"]["rec_test__stream_headset"]["source_audio_rel"] = "../outside.wav"
    test_env["manifest_file"].write_text(json.dumps(data))

    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state = verifier.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    assert state.provenance_verified is False
    assert any("Path traversal detected" in err for err in state.provenance_errors)


def test_6_source_audio_symlink_escape_rejected(test_env, monkeypatch):
    """Verifies that a source audio file that is a symlink is rejected."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])

    real_audio = test_env["tmp_path"] / "real_audio.wav"
    real_audio.write_bytes(test_env["audio_file"].read_bytes())
    test_env["audio_file"].unlink()
    test_env["audio_file"].symlink_to(real_audio)

    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state = verifier.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    assert state.provenance_verified is False
    assert any("Symlink escape: source audio file is a symlink" in err for err in state.provenance_errors)


def test_7_pyannet_checkpoint_byte_hash_verification(test_env, monkeypatch):
    """Verifies that PyanNet checkpoint byte hash must match trusted spec."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])

    # Mutate checkpoint bytes
    test_env["ckpt_file"].write_bytes(b"MUTATED_CKPT_BYTES")

    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state = verifier.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    assert state.provenance_verified is False
    assert any("PyanNet physical checkpoint bytes hash mismatch" in err for err in state.provenance_errors)


def test_8_pyannet_checkpoint_missing(test_env, monkeypatch):
    """Verifies that missing PyanNet checkpoint file fails closed."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])
    
    test_env["ckpt_file"].unlink()

    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state = verifier.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    assert state.provenance_verified is False
    assert any("PyanNet checkpoint file does not exist" in err for err in state.provenance_errors)


def test_9_posterior_hash_mismatch_and_placeholders(test_env, monkeypatch):
    """Verifies that posterior file hash mismatch and placeholder hashes fail."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])

    # Corrupt posterior bytes
    test_env["post_file"].write_bytes(b"CORRUPTED_POSTERIOR_DATA")

    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state = verifier.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    assert state.provenance_verified is False
    assert any("Posterior file hash mismatch" in err for err in state.provenance_errors)

    # Test placeholder hash
    data = json.loads(test_env["manifest_file"].read_text())
    data["artifacts"]["rec_test__stream_headset"]["posterior_file_sha256"] = "PLACEHOLDER_HASH"
    test_env["manifest_file"].write_text(json.dumps(data))

    verifier2 = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state2 = verifier2.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )
    assert state2.provenance_verified is False
    assert any("Placeholder SHA-256 in manifest" in err for err in state2.provenance_errors)


def test_10_unrecorded_historical_mono_fails_verification(test_env, monkeypatch):
    """Verifies that historical mono audio marked UNRECORDED_IN_HISTORICAL_MANIFEST fails verification."""
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])

    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state = verifier.verify_provenance(
        recording_id="rec_mono",
        stream_name="stream_mono",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    assert state.provenance_verified is False
    assert "source_audio_sha256" in state.unverified_fields
    assert any("Historical source audio SHA-256 is unrecorded" in err for err in state.provenance_errors)


def test_11_manifest_string_alone_insufficient(test_env, monkeypatch):
    """
    CRITICAL PROOF TEST:
    Proves that manifest string alone without physical byte hashing fails verification.
    If the manifest claims a valid digest, but the physical file on disk has different bytes,
    the verifier MUST reject it by directly computing SHA-256 from disk bytes.
    """
    monkeypatch.setitem(TRUSTED_PYANNET_POSTERIOR_SPEC, "model_checkpoint_sha256", test_env["ckpt_sha"])

    # Overwrite the physical source audio on disk with different content
    test_env["audio_file"].write_bytes(b"ATTACKER_CONTROLLED_AUDIO_BYTES_SUBSTITUTED_ON_DISK")

    verifier = OverlapProvenanceVerifier(
        manifest_path=test_env["manifest_file"],
        source_audio_root=test_env["tmp_path"],
        pyannet_checkpoint_path=test_env["ckpt_file"],
    )

    state = verifier.verify_provenance(
        recording_id="rec_test",
        stream_name="stream_headset",
        posterior_path=test_env["post_file"],
        execution_mode=ExecutionMode.PRODUCTION,
        is_synthetic=False,
    )

    # Even though manifest records a SHA-256, verification MUST fail because physical disk bytes were checked!
    assert state.provenance_verified is False
    assert any("Source audio file hash mismatch" in err for err in state.provenance_errors)
