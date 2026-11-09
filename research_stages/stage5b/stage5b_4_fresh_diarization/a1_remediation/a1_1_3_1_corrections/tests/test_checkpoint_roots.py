#!/usr/bin/env python3
"""
Test Suite: Finding 1 — Checkpoint Root Allowlist Containment & Preflight Security
Validates:
1. Checkpoint inside the standard allowlist.
2. Checkpoint outside the allowlist.
3. Caller-supplied custom root outside the allowlist in production mode (rejected).
4. Custom root that is a parent of an authorized root in production mode (rejected).
5. Symlink pointing outside the allowlist (rejected).
6. Relative-path traversal (rejected).
7. Non-existent checkpoint (rejected).
8. Directory passed instead of a file (rejected).
9. Attempt to change allowlist through orchestrator arguments in production mode.
10. Production mode rejecting any unsupported custom-root override.
11. Proves a caller cannot expand production authorization by supplying a custom directory.
"""

import hashlib
import os
from pathlib import Path
import sys
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from checkpoint_preflight import (
    ExecutionMode,
    VerifiedCheckpoint,
    verify_encoder_checkpoint_strict,
    get_immutable_standard_checkpoint_roots,
    CheckpointPreflightError,
    create_synthetic_checkpoint_fixture,
)
from model_factory import ControlledModelFactory, ModelFactoryError
from orchestrator_remediated import FreshDiarizationOrchestratorRemediated
from cache_manager_remediated import IsolatedEmbeddingCacheManagerRemediated
from clustering_backends import SyntheticSpectralClusteringBackend
from evaluation_adapters_remediated import SyntheticEvaluationAdapter
from vad_adapter_remediated import VADAdapterRemediated


@pytest.fixture
def test_roots(tmp_path):
    # Standard authorized root
    std_root = tmp_path / "authorized_models"
    std_root.mkdir()
    
    # Unauthorized external directory
    unauth_root = tmp_path / "unauthorized_evil"
    unauth_root.mkdir()
    
    # Authorized checkpoint file
    valid_ckpt = std_root / "model_valid.pt"
    valid_content = b"VALID_ENCODER_BYTES_PAYLOAD_1234567890" * 4
    valid_ckpt.write_bytes(valid_content)
    valid_sha = hashlib.sha256(valid_content).hexdigest()

    # Unauthorized checkpoint file
    evil_ckpt = unauth_root / "evil_model.pt"
    evil_content = b"UNAUTHORIZED_MALICIOUS_BYTES_PAYLOAD" * 4
    evil_ckpt.write_bytes(evil_content)
    evil_sha = hashlib.sha256(evil_content).hexdigest()

    return {
        "std_root": std_root,
        "unauth_root": unauth_root,
        "valid_ckpt": valid_ckpt,
        "valid_sha": valid_sha,
        "evil_ckpt": evil_ckpt,
        "evil_sha": evil_sha,
        "tmp_path": tmp_path,
    }


def test_1_checkpoint_inside_allowlist(monkeypatch, test_roots):
    """A checkpoint inside the immutable allowlist passes verification."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [test_roots["std_root"].resolve()],
    )
    v = verify_encoder_checkpoint_strict(
        encoder_name="titanet_large",
        checkpoint_path=test_roots["valid_ckpt"],
        expected_sha256=test_roots["valid_sha"],
        execution_mode=ExecutionMode.PRODUCTION,
    )
    assert v.verification_status == "VERIFIED"
    assert v.canonical_path == test_roots["valid_ckpt"].resolve()


def test_2_checkpoint_outside_allowlist(monkeypatch, test_roots):
    """A checkpoint outside the immutable allowlist is strictly rejected."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [test_roots["std_root"].resolve()],
    )
    with pytest.raises(CheckpointPreflightError) as exc_info:
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=test_roots["evil_ckpt"],
            expected_sha256=test_roots["evil_sha"],
            execution_mode=ExecutionMode.PRODUCTION,
        )
    assert "escapes authorized checkpoint roots" in str(exc_info.value)


def test_3_custom_root_outside_allowlist_rejected(monkeypatch, test_roots):
    """In production mode, supplying a custom root outside the allowlist is rejected."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [test_roots["std_root"].resolve()],
    )
    with pytest.raises(CheckpointPreflightError) as exc_info:
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=test_roots["evil_ckpt"],
            expected_sha256=test_roots["evil_sha"],
            execution_mode=ExecutionMode.PRODUCTION,
            custom_permitted_roots=[test_roots["unauth_root"]],
        )
    assert "is outside the immutable standard allowlist" in str(exc_info.value)


def test_4_custom_root_parent_of_authorized_root_rejected(monkeypatch, test_roots):
    """Supplying a parent of an authorized root (which would expand the boundary) is rejected."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [test_roots["std_root"].resolve()],
    )
    # tmp_path is the parent of std_root
    with pytest.raises(CheckpointPreflightError) as exc_info:
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=test_roots["evil_ckpt"],
            expected_sha256=test_roots["evil_sha"],
            execution_mode=ExecutionMode.PRODUCTION,
            custom_permitted_roots=[test_roots["tmp_path"]],
        )
    assert "is a parent of standard root" in str(exc_info.value)
    assert "illegally expands the permitted boundary" in str(exc_info.value)


def test_5_symlink_pointing_outside_allowlist_rejected(monkeypatch, test_roots):
    """Symlinks on the checkpoint path or parents are strictly rejected."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [test_roots["std_root"].resolve()],
    )
    # Create symlink inside std_root pointing to evil_ckpt outside
    sym_path = test_roots["std_root"] / "sym_evil.pt"
    sym_path.symlink_to(test_roots["evil_ckpt"])

    with pytest.raises(CheckpointPreflightError) as exc_info:
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=sym_path,
            expected_sha256=test_roots["evil_sha"],
            execution_mode=ExecutionMode.PRODUCTION,
        )
    assert "Checkpoint path is a symlink" in str(exc_info.value)


def test_6_relative_path_traversal_rejected(monkeypatch, test_roots):
    """Paths containing relative traversal '..' are strictly rejected."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [test_roots["std_root"].resolve()],
    )
    traversal_path = test_roots["std_root"] / ".." / "unauthorized_evil" / "evil_model.pt"

    with pytest.raises(CheckpointPreflightError) as exc_info:
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=traversal_path,
            expected_sha256=test_roots["evil_sha"],
            execution_mode=ExecutionMode.PRODUCTION,
        )
    assert "Relative path traversal ('..') is prohibited" in str(exc_info.value)


def test_7_nonexistent_checkpoint_rejected(monkeypatch, test_roots):
    """Nonexistent checkpoint files fail preflight."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [test_roots["std_root"].resolve()],
    )
    missing = test_roots["std_root"] / "does_not_exist.pt"
    with pytest.raises(CheckpointPreflightError) as exc_info:
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=missing,
            expected_sha256=test_roots["valid_sha"],
            execution_mode=ExecutionMode.PRODUCTION,
        )
    assert "File does not exist" in str(exc_info.value)


def test_8_directory_passed_as_checkpoint_rejected(monkeypatch, test_roots):
    """Passing a directory instead of a file fails preflight."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [test_roots["std_root"].resolve()],
    )
    with pytest.raises(CheckpointPreflightError) as exc_info:
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=test_roots["std_root"],
            expected_sha256=test_roots["valid_sha"],
            execution_mode=ExecutionMode.PRODUCTION,
        )
    assert "Target path is a directory, not a file" in str(exc_info.value)


def test_9_orchestrator_rejects_unauthorized_custom_roots(monkeypatch, test_roots):
    """FreshDiarizationOrchestrator rejects caller attempt to expand allowlist via custom_permitted_checkpoint_roots."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [test_roots["std_root"].resolve()],
    )
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(test_roots["tmp_path"] / "cache", allow_custom_root=True)
    vad_adapter = VADAdapterRemediated()
    clustering = SyntheticSpectralClusteringBackend()
    eval_adapter = SyntheticEvaluationAdapter()

    with pytest.raises(CheckpointPreflightError) as exc_info:
        FreshDiarizationOrchestratorRemediated(
            cache_manager=cache_mgr,
            vad_adapter=vad_adapter,
            clustering_backend=clustering,
            evaluation_adapter=eval_adapter,
            titanet_checkpoint_path=test_roots["evil_ckpt"],
            titanet_checkpoint_sha256=test_roots["evil_sha"],
            wespeaker_checkpoint_path=test_roots["valid_ckpt"],
            wespeaker_checkpoint_sha256=test_roots["valid_sha"],
            execution_mode=ExecutionMode.PRODUCTION,
            custom_permitted_checkpoint_roots=[test_roots["unauth_root"]],
        )
    assert "is outside the immutable standard allowlist" in str(exc_info.value)


def test_10_malicious_caller_cannot_expand_allowlist(monkeypatch, test_roots):
    """Demonstrates that a caller passing arbitrary directories cannot authorize foreign models."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [test_roots["std_root"].resolve()],
    )
    # Attempt with root directory /
    with pytest.raises(CheckpointPreflightError):
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=test_roots["evil_ckpt"],
            expected_sha256=test_roots["evil_sha"],
            execution_mode=ExecutionMode.PRODUCTION,
            custom_permitted_roots=[Path("/")],
        )
    # Attempt with user home
    with pytest.raises(CheckpointPreflightError):
        verify_encoder_checkpoint_strict(
            encoder_name="titanet_large",
            checkpoint_path=test_roots["evil_ckpt"],
            expected_sha256=test_roots["evil_sha"],
            execution_mode=ExecutionMode.PRODUCTION,
            custom_permitted_roots=[Path("/home")],
        )
