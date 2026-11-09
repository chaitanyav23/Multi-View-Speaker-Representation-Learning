#!/usr/bin/env python3
"""
Unit tests for Stage 5B.4-A.1.3.2 Finding 1:
TitaNet / NeMo Checkpoint Identity & Secure Staging Loader.

Verifies that:
1. Correct synthetic checkpoint loads through mocked NeMo loader function.
2. Checkpoint digest mismatch is rejected.
3. Checkpoint mutation before loading is rejected.
4. Pathname replacement does not result in unverified bytes being loaded.
5. Loader receives the exact bytes that passed verification.
6. Arbitrary checkpoint paths are rejected.
7. Symlink substitution is rejected.
8. Loader exceptions are propagated safely.
9. Temporary staging files and directories are guaranteed to be cleaned up.
10. No production-mode model injection bypass exists.
"""

import hashlib
import os
from pathlib import Path
import stat
import sys
import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_preflight import (
    ExecutionMode,
    VerifiedCheckpoint,
    verify_encoder_checkpoint_strict,
    get_immutable_standard_checkpoint_roots,
    CheckpointPreflightError,
    create_synthetic_checkpoint_fixture,
    compute_file_sha256,
)
from model_factory import (
    ControlledModelFactory,
    VerifiedModelHandle,
    ModelFactoryError,
    load_titanet_model_from_verified_bytes,
)


@pytest.fixture
def titanet_env(tmp_path, monkeypatch):
    """Sets up an authorized checkpoint environment for TitaNet testing."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [tmp_path.resolve()],
    )

    ckpt_file = tmp_path / "titanet_model.nemo"
    content = b"SYNTHETIC_TITANET_CHECKPOINT_BYTES_AUTHENTIC_" + b"\x77" * 256
    ckpt_file.write_bytes(content)
    ckpt_sha = hashlib.sha256(content).hexdigest()

    token = verify_encoder_checkpoint_strict(
        encoder_name="titanet_large",
        checkpoint_path=ckpt_file,
        expected_sha256=ckpt_sha,
        execution_mode=ExecutionMode.PRODUCTION,
    )

    return {
        "tmp_path": tmp_path,
        "ckpt_file": ckpt_file,
        "ckpt_sha": ckpt_sha,
        "content": content,
        "token": token,
    }


def test_1_correct_synthetic_checkpoint_loads(titanet_env):
    """Verifies that a valid checkpoint loads through the mocked NeMo adapter."""
    captured_paths = []

    def mock_nemo_loader(staging_path: str, map_location: str):
        captured_paths.append(staging_path)
        # Verify staging file exists and contains the exact bytes
        p = Path(staging_path)
        assert p.exists()
        assert p.is_file()
        assert p.read_bytes() == titanet_env["content"]
        # Return mock model
        return torch.nn.Linear(10, 192)

    ControlledModelFactory.nemo_loader_override = mock_nemo_loader
    try:
        handle = ControlledModelFactory.build_production_model(titanet_env["token"])
        assert isinstance(handle, VerifiedModelHandle)
        assert handle.encoder_name == "titanet_large"
        assert len(captured_paths) == 1
    finally:
        ControlledModelFactory.nemo_loader_override = None


def test_2_checkpoint_digest_mismatch_rejected(titanet_env):
    """Verifies that if memory buffer digest does not match token, loading fails."""
    tampered_bytes = b"TAMPERED_BYTES" + b"\x00" * 200

    with pytest.raises(ModelFactoryError, match="digest mismatch"):
        load_titanet_model_from_verified_bytes(
            verified_checkpoint=titanet_env["token"],
            ckpt_bytes=tampered_bytes,
        )


def test_3_checkpoint_mutation_before_loading_rejected(titanet_env):
    """Verifies that mutating the physical file on disk before loading fails TOCTOU check."""
    # Mutate file on disk after token generation
    titanet_env["ckpt_file"].write_bytes(b"MUTATED_DISK_BYTES" + b"\x99" * (len(titanet_env["content"]) - 18))

    with pytest.raises(ModelFactoryError, match="TOCTOU validation failure"):
        ControlledModelFactory.build_production_model(titanet_env["token"])


def test_4_pathname_replacement_cannot_load_unverified_bytes(titanet_env):
    """
    CRITICAL PROOF TEST:
    Verifies that replacing the original file path on disk with an attacker-controlled file
    DOES NOT cause unverified bytes to be loaded.
    The loader consumes the exact verified bytes via secure staging!
    """
    received_bytes = []

    def mock_nemo_loader(staging_path: str, map_location: str):
        # Record what was actually delivered to the loader
        received_bytes.append(Path(staging_path).read_bytes())
        return torch.nn.Linear(10, 192)

    # Attacker tries to swap disk file after verification
    raw_bytes = ControlledModelFactory.read_and_verify_checkpoint_bytes(titanet_env["token"])

    # Overwrite original path on disk with malicious content
    titanet_env["ckpt_file"].write_bytes(b"ATTACKER_MALICIOUS_PAYLOAD_ON_DISK")

    # Load from verified bytes
    load_titanet_model_from_verified_bytes(
        verified_checkpoint=titanet_env["token"],
        ckpt_bytes=raw_bytes,
        loader_fn=mock_nemo_loader,
    )

    assert len(received_bytes) == 1
    # Loader received AUTHENTIC bytes, NOT the attacker payload on disk!
    assert received_bytes[0] == titanet_env["content"]
    assert received_bytes[0] != b"ATTACKER_MALICIOUS_PAYLOAD_ON_DISK"


def test_5_loader_receives_exact_verified_bytes(titanet_env):
    """Verifies that loader receives exact byte-for-byte copy with exact SHA-256."""
    captured_hashes = []

    def mock_nemo_loader(staging_path: str, map_location: str):
        captured_hashes.append(hashlib.sha256(Path(staging_path).read_bytes()).hexdigest())
        return torch.nn.Linear(10, 192)

    ControlledModelFactory.nemo_loader_override = mock_nemo_loader
    try:
        ControlledModelFactory.build_production_model(titanet_env["token"])
        assert len(captured_hashes) == 1
        assert captured_hashes[0] == titanet_env["ckpt_sha"]
    finally:
        ControlledModelFactory.nemo_loader_override = None


def test_6_arbitrary_checkpoint_paths_rejected(titanet_env):
    """Passing arbitrary unverified paths or invalid tokens is strictly rejected."""
    with pytest.raises(ModelFactoryError, match="Expected VerifiedCheckpoint instance"):
        ControlledModelFactory.read_and_verify_checkpoint_bytes("not_a_token")

    with pytest.raises(ModelFactoryError, match="Expected VerifiedCheckpoint instance"):
        load_titanet_model_from_verified_bytes("not_a_token", b"bytes")


def test_7_symlink_substitution_rejected(titanet_env):
    """Replacing the checkpoint with a symlink before loading fails closed."""
    target_outside = titanet_env["tmp_path"] / "outside.nemo"
    target_outside.write_bytes(titanet_env["content"])
    titanet_env["ckpt_file"].unlink()
    titanet_env["ckpt_file"].symlink_to(target_outside)

    with pytest.raises(ModelFactoryError, match="TOCTOU validation failure"):
        ControlledModelFactory.read_and_verify_checkpoint_bytes(titanet_env["token"])


def test_8_loader_exceptions_propagated_safely(titanet_env):
    """If the loader raises an exception, it propagates out cleanly without being masked."""
    def failing_loader(staging_path: str, map_location: str):
        raise ValueError("NeMo internal deserialization failure")

    with pytest.raises(ValueError, match="NeMo internal deserialization failure"):
        load_titanet_model_from_verified_bytes(
            verified_checkpoint=titanet_env["token"],
            ckpt_bytes=titanet_env["content"],
            loader_fn=failing_loader,
        )


def test_9_temporary_staging_files_cleaned_up(titanet_env):
    """
    Verifies that the staging file and staging directory are guaranteed to be cleaned up,
    both on success and on failure!
    """
    captured_staging_path = []

    def mock_loader(staging_path: str, map_location: str):
        captured_staging_path.append(staging_path)
        assert Path(staging_path).exists()
        return torch.nn.Linear(10, 192)

    load_titanet_model_from_verified_bytes(
        verified_checkpoint=titanet_env["token"],
        ckpt_bytes=titanet_env["content"],
        loader_fn=mock_loader,
    )

    staged = Path(captured_staging_path[0])
    # Must NOT exist after function returns!
    assert not staged.exists()
    assert not staged.parent.exists()

    # Also test cleanup on failure
    captured_fail_path = []

    def fail_loader(staging_path: str, map_location: str):
        captured_fail_path.append(staging_path)
        raise RuntimeError("Crash during load")

    with pytest.raises(RuntimeError):
        load_titanet_model_from_verified_bytes(
            verified_checkpoint=titanet_env["token"],
            ckpt_bytes=titanet_env["content"],
            loader_fn=fail_loader,
        )

    staged_fail = Path(captured_fail_path[0])
    assert not staged_fail.exists()
    assert not staged_fail.parent.exists()


def test_10_no_production_mode_model_injection_bypass(tmp_path):
    """Synthetic tokens passed to build_production_model are strictly rejected."""
    synth_file = tmp_path / "synth.pt"
    synth_token = create_synthetic_checkpoint_fixture("titanet_large", synth_file)

    with pytest.raises(ModelFactoryError, match="Cannot build production model from synthetic checkpoint"):
        ControlledModelFactory.build_production_model(synth_token)
