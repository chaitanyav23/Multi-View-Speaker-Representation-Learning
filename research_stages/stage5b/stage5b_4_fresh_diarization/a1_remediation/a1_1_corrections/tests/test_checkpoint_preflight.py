#!/usr/bin/env python3
"""Tests for Task 2: Checkpoint Preflight Verification."""

import hashlib
from pathlib import Path
import pytest
import sys

CORRECTIONS_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(CORRECTIONS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTIONS_SCRIPTS))

from checkpoint_preflight import (
    verify_encoder_checkpoint_preflight,
    run_preflight_verification,
    CheckpointPreflightError,
)


def create_test_checkpoint(tmp_path: Path, filename: str, content: bytes) -> tuple[Path, str]:
    ckpt = tmp_path / filename
    ckpt.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()
    return ckpt, expected_hash


def test_checkpoint_preflight_success(tmp_path):
    ckpt, h = create_test_checkpoint(tmp_path, "titanet.nemo", b"mock titanet model weights bytes 12345")
    res = verify_encoder_checkpoint_preflight("titanet_large", ckpt, h)
    assert res.is_verified
    assert res.computed_sha256 == h


def test_checkpoint_preflight_rejects_truncated_placeholder(tmp_path):
    ckpt, _ = create_test_checkpoint(tmp_path, "titanet.nemo", b"mock")
    # Old 32-character MD5 hash should be rejected
    truncated_hash = "450f6bd279a3f783687dbd15814fd0f1"
    with pytest.raises(CheckpointPreflightError, match="64-character hexadecimal string"):
        verify_encoder_checkpoint_preflight("titanet_large", ckpt, truncated_hash)


def test_checkpoint_preflight_missing_file(tmp_path):
    missing_ckpt = tmp_path / "nonexistent.onnx"
    dummy_hash = "a" * 64
    with pytest.raises(CheckpointPreflightError, match="does not exist"):
        verify_encoder_checkpoint_preflight("wespeaker_resnet293", missing_ckpt, dummy_hash)


def test_checkpoint_preflight_hash_mismatch(tmp_path):
    ckpt, _ = create_test_checkpoint(tmp_path, "wespeaker.onnx", b"real bytes")
    corrupt_hash = "0" * 64
    with pytest.raises(CheckpointPreflightError, match="Hash mismatch"):
        verify_encoder_checkpoint_preflight("wespeaker_resnet293", ckpt, corrupt_hash)


def test_dual_checkpoint_preflight_selective_failure(tmp_path):
    tita_ckpt, tita_h = create_test_checkpoint(tmp_path, "titanet.nemo", b"titanet")
    res_ckpt, res_h = create_test_checkpoint(tmp_path, "wespeaker.onnx", b"wespeaker")

    # Both succeed
    summary = run_preflight_verification(tita_ckpt, tita_h, res_ckpt, res_h)
    assert summary.is_all_verified

    # Corrupt only WeSpeaker
    with pytest.raises(CheckpointPreflightError, match="wespeaker_resnet293"):
        run_preflight_verification(tita_ckpt, tita_h, res_ckpt, "f" * 64)
