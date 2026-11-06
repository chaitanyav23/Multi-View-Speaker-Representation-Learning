#!/usr/bin/env python3
"""Tests for Task 6: Checkpoint Verification Utility."""

import pytest
import hashlib
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from checkpoint_verifier import (
    verify_checkpoint,
    compute_file_sha256,
    CheckpointStatus,
)


def test_missing_checkpoint():
    p = Path("/nonexistent/checkpoint.pt")
    dummy_hash = "0" * 64
    res = verify_checkpoint(p, dummy_hash)
    assert res.status == CheckpointStatus.MISSING_FILE
    assert res.is_verified is False


def test_invalid_expected_hash_format(tmp_path):
    f = tmp_path / "model.pt"
    f.write_bytes(b"dummy")
    res = verify_checkpoint(f, "short_hash")
    assert res.status == CheckpointStatus.INVALID_EXPECTED_HASH
    assert res.is_verified is False


def test_hash_mismatch(tmp_path):
    f = tmp_path / "model.pt"
    f.write_bytes(b"actual model bytes")
    wrong_hash = "f" * 64
    res = verify_checkpoint(f, wrong_hash)
    assert res.status == CheckpointStatus.HASH_MISMATCH
    assert res.is_verified is False
    assert res.computed_sha256 == hashlib.sha256(b"actual model bytes").hexdigest()


def test_successful_verification(tmp_path):
    f = tmp_path / "model.pt"
    data = b"synthetic checkpoint weight payload"
    f.write_bytes(data)
    expected_hash = hashlib.sha256(data).hexdigest()

    res = verify_checkpoint(f, expected_hash)
    assert res.status == CheckpointStatus.VERIFIED
    assert res.is_verified is True
    assert res.file_size_bytes == len(data)
    assert res.computed_sha256 == expected_hash


def test_real_file_hashing_without_loading():
    """Verify that compute_file_sha256 hashes a file on disk without importing/loading it."""
    # WhisperX asset file on disk
    target_file = Path("research_stages/.venv_phase20/lib/python3.10/site-packages/whisperx/assets/pytorch_model.bin")
    if target_file.exists():
        h = compute_file_sha256(target_file)
        assert len(h) == 64
        # Matches verified hash from audit
        assert h == "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea"
