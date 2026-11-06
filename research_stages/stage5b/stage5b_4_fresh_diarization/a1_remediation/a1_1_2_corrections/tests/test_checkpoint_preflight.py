#!/usr/bin/env python3
"""
Tests for Correction 1: Mandatory Checkpoint Verification for Production Execution.
Validates strict preflight verification, rejection of placeholders, symlink escapes,
and isolation between production and synthetic execution modes.
"""

import hashlib
import os
from pathlib import Path
import pytest
import sys

CORRECTIONS_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(CORRECTIONS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTIONS_SCRIPTS))

from checkpoint_preflight import (
    ExecutionMode,
    CheckpointPreflightError,
    VerifiedCheckpoint,
    verify_encoder_checkpoint_strict,
    run_production_checkpoint_preflight,
    is_placeholder_hash,
)


def create_mock_checkpoint_file(path: Path, content: bytes) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()
    return path, expected_hash


def test_missing_titanet_path_in_production():
    with pytest.raises(CheckpointPreflightError, match="Checkpoint path must be explicitly provided"):
        run_production_checkpoint_preflight(
            titanet_path=None,
            titanet_expected_sha256="450f6bd279a3f78348f6e883359f9b3b34687dbd15814fd0f10e446ca0f708f6",
            wespeaker_path=Path("/tmp/some_path"),
            wespeaker_expected_sha256="dbb1ccc7754caff552ebc46347a51aaee2669bb24efc740e665d1a1133d20e98",
        )


def test_missing_wespeaker_path_in_production(tmp_path):
    t_file, t_hash = create_mock_checkpoint_file(tmp_path / "titanet.nemo", b"tita_content")
    with pytest.raises(CheckpointPreflightError, match="Checkpoint path must be explicitly provided"):
        run_production_checkpoint_preflight(
            titanet_path=t_file,
            titanet_expected_sha256=t_hash,
            wespeaker_path=None,
            wespeaker_expected_sha256="dbb1ccc7754caff552ebc46347a51aaee2669bb24efc740e665d1a1133d20e98",
            custom_permitted_roots=[tmp_path],
        )


def test_missing_expected_hash_in_production(tmp_path):
    t_file, _ = create_mock_checkpoint_file(tmp_path / "titanet.nemo", b"tita_content")
    w_file, _ = create_mock_checkpoint_file(tmp_path / "wespeaker.onnx", b"wes_content")
    with pytest.raises(CheckpointPreflightError, match="Expected SHA-256 must be explicitly provided"):
        run_production_checkpoint_preflight(
            titanet_path=t_file,
            titanet_expected_sha256=None,
            wespeaker_path=w_file,
            wespeaker_expected_sha256="dbb1ccc7754caff552ebc46347a51aaee2669bb24efc740e665d1a1133d20e98",
            custom_permitted_roots=[tmp_path],
        )


def test_malformed_expected_hash(tmp_path):
    t_file, _ = create_mock_checkpoint_file(tmp_path / "titanet.nemo", b"tita_content")
    # Truncated 32-character hash
    with pytest.raises(CheckpointPreflightError, match="64-character hexadecimal string"):
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=t_file,
            expected_sha256="450f6bd279a3f783687dbd15814fd0f1",
            execution_mode=ExecutionMode.PRODUCTION,
            custom_permitted_roots=[tmp_path],
        )


def test_placeholder_hash_rejection(tmp_path):
    t_file, _ = create_mock_checkpoint_file(tmp_path / "titanet.nemo", b"tita_content")

    # All identical characters
    with pytest.raises(CheckpointPreflightError, match="unauthorized placeholder hash"):
        verify_encoder_checkpoint_strict(
            "titanet_large", t_file, "0" * 64, ExecutionMode.PRODUCTION, [tmp_path]
        )

    # 32 zeros padding
    with pytest.raises(CheckpointPreflightError, match="unauthorized placeholder hash"):
        verify_encoder_checkpoint_strict(
            "titanet_large", t_file, "450f6bd279a3f783687dbd15814fd0f1" + "0" * 32, ExecutionMode.PRODUCTION, [tmp_path]
        )

    # Repeated pattern
    with pytest.raises(CheckpointPreflightError, match="unauthorized placeholder hash"):
        verify_encoder_checkpoint_strict(
            "titanet_large", t_file, "12345678" * 8, ExecutionMode.PRODUCTION, [tmp_path]
        )


def test_hash_mismatch(tmp_path):
    t_file, _ = create_mock_checkpoint_file(tmp_path / "titanet.nemo", b"real_file_bytes")
    # Real SHA-256 of completely different content (valid 64-char hex, non-placeholder, but wrong)
    wrong_hash = hashlib.sha256(b"completely_different_content_bytes").hexdigest()
    with pytest.raises(CheckpointPreflightError, match="Hash mismatch"):
        verify_encoder_checkpoint_strict(
            "titanet_large", t_file, wrong_hash, ExecutionMode.PRODUCTION, [tmp_path]
        )


def test_correct_hash_verification(tmp_path):
    t_file, t_hash = create_mock_checkpoint_file(tmp_path / "titanet.nemo", b"verified_model_weights")
    verified = verify_encoder_checkpoint_strict(
        "titanet_large", t_file, t_hash, ExecutionMode.PRODUCTION, [tmp_path]
    )
    assert verified.is_verified
    assert verified.computed_sha256 == t_hash
    assert verified.file_size_bytes == len(b"verified_model_weights")
    assert not verified.is_synthetic


def test_missing_file_fails(tmp_path):
    missing_file = tmp_path / "missing_model.nemo"
    valid_hash = hashlib.sha256(b"dummy").hexdigest()
    with pytest.raises(CheckpointPreflightError, match="checkpoint file not found"):
        verify_encoder_checkpoint_strict(
            "titanet_large", missing_file, valid_hash, ExecutionMode.PRODUCTION, [tmp_path]
        )


def test_directory_supplied_fails(tmp_path):
    dir_target = tmp_path / "fake_dir_model.nemo"
    dir_target.mkdir()
    valid_hash = hashlib.sha256(b"dummy").hexdigest()
    with pytest.raises(CheckpointPreflightError, match="target .* is a directory"):
        verify_encoder_checkpoint_strict(
            "titanet_large", dir_target, valid_hash, ExecutionMode.PRODUCTION, [tmp_path]
        )


def test_symlink_escape_rejection(tmp_path):
    permitted_root = tmp_path / "permitted_checkpoints"
    permitted_root.mkdir()
    unauthorized_outside = tmp_path / "unauthorized_external"
    unauthorized_outside.mkdir()

    outside_file, outside_hash = create_mock_checkpoint_file(
        unauthorized_outside / "secret_weights.nemo", b"unauthorized_secret_weights"
    )

    symlink_in_root = permitted_root / "link_to_external.nemo"
    os.symlink(str(outside_file), str(symlink_in_root))

    with pytest.raises(CheckpointPreflightError, match="outside permitted roots|Symlink escape"):
        verify_encoder_checkpoint_strict(
            "titanet_large", symlink_in_root, outside_hash, ExecutionMode.PRODUCTION, [permitted_root]
        )


def test_production_mode_rejects_synthetic_bypass(tmp_path):
    # In production mode, both files must be provided and verified; no bypass
    with pytest.raises(CheckpointPreflightError):
        run_production_checkpoint_preflight(
            titanet_path=None,
            titanet_expected_sha256=None,
            wespeaker_path=None,
            wespeaker_expected_sha256=None,
        )


def test_verified_checkpoint_identity_passed_unchanged(tmp_path):
    t_file, t_hash = create_mock_checkpoint_file(tmp_path / "titanet.nemo", b"tita_binary")
    w_file, w_hash = create_mock_checkpoint_file(tmp_path / "wespeaker.onnx", b"wes_binary")

    v_tita, v_wes = run_production_checkpoint_preflight(
        titanet_path=t_file,
        titanet_expected_sha256=t_hash,
        wespeaker_path=w_file,
        wespeaker_expected_sha256=w_hash,
        custom_permitted_roots=[tmp_path],
    )

    assert v_tita.verified_path == t_file.resolve()
    assert v_tita.computed_sha256 == t_hash
    assert v_wes.verified_path == w_file.resolve()
    assert v_wes.computed_sha256 == w_hash
