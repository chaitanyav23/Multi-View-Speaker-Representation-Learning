#!/usr/bin/env python3
"""
Unit tests for Stage 5B.4-A.1.3.1 Finding 4:
WeSpeaker model construction and TOCTOU elimination.
Verifies that:
1. WeSpeaker ResNet293 model architecture instantiates and state_dict loads with projection key filtering.
2. In-memory byte buffer verification eliminates TOCTOU races.
3. Mutation of checkpoint bytes between preflight and loading raises ModelFactoryError.
4. Symlink replacement of checkpoint between preflight and loading raises ModelFactoryError.
5. Deletion of checkpoint file between preflight and loading raises ModelFactoryError.
6. Checkpoint file size change between preflight and loading raises ModelFactoryError.
7. Synthetic checkpoint passed to build_production_model fails closed.
8. VerifiedCheckpoint signature forgery is prevented.
9. VerifiedModelHandle signature forgery is prevented.
"""

import hashlib
import io
import os
from pathlib import Path
import sys
import pytest
import torch
import torch.nn as nn

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
    Bottleneck,
    TSTP,
    WeSpeakerResNet293,
    ControlledModelFactory,
    VerifiedModelHandle,
    ModelFactoryError,
)


@pytest.fixture
def checkpoint_fixture(tmp_path, monkeypatch):
    """Creates a temporary checkpoint and preflight token."""
    monkeypatch.setattr(
        "checkpoint_preflight.get_immutable_standard_checkpoint_roots",
        lambda: [tmp_path.resolve()],
    )
    ckpt_file = tmp_path / "mock_model.pt"
    content = b"AUTHENTIC_CHECKPOINT_CONTENT_" + b"\x55" * 256
    ckpt_file.write_bytes(content)
    ckpt_sha = hashlib.sha256(content).hexdigest()
    ckpt_size = len(content)

    token = verify_encoder_checkpoint_strict(
        encoder_name="wespeaker_resnet293",
        checkpoint_path=ckpt_file,
        expected_sha256=ckpt_sha,
        execution_mode=ExecutionMode.PRODUCTION,
    )

    return {
        "tmp_path": tmp_path,
        "ckpt_file": ckpt_file,
        "ckpt_sha": ckpt_sha,
        "ckpt_size": ckpt_size,
        "token": token,
    }


def test_1_wespeaker_architecture_and_projection_filtering(tmp_path):
    """Verifies that WeSpeaker ResNet293 instantiates and loads state_dict with projection keys filtered."""
    model = WeSpeakerResNet293(
        num_blocks=[1, 1, 1, 1],  # Small config for quick testing
        m_channels=8,
        feat_dim=40,
        embed_dim=64,
        two_emb_layer=False,
    )
    assert isinstance(model, nn.Module)

    # Generate state dict including projection head keys
    sd = model.state_dict()
    sd["projection.weight"] = torch.randn(10, 64)
    sd["projection.bias"] = torch.randn(10)

    # Save to buffer
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    saved_bytes = buf.read()

    # Emulate factory loading: filter projection keys
    loaded_sd = torch.load(io.BytesIO(saved_bytes), map_location="cpu")
    model_keys = set(model.state_dict().keys())
    filtered_sd = {k: v for k, v in loaded_sd.items() if k in model_keys}
    assert "projection.weight" not in filtered_sd
    assert "projection.bias" not in filtered_sd
    assert set(filtered_sd.keys()) == model_keys

    model.load_state_dict(filtered_sd, strict=True)


def test_2_toctou_read_and_verify_checkpoint_bytes(checkpoint_fixture):
    """Verifies that read_and_verify_checkpoint_bytes reads exact bytes and verifies digest."""
    token = checkpoint_fixture["token"]
    raw_bytes = ControlledModelFactory.read_and_verify_checkpoint_bytes(token)
    assert len(raw_bytes) == checkpoint_fixture["ckpt_size"]
    assert hashlib.sha256(raw_bytes).hexdigest() == checkpoint_fixture["ckpt_sha"]


def test_3_checkpoint_mutation_between_preflight_and_load(checkpoint_fixture):
    """Verifies that byte modification on disk between preflight and loading raises ModelFactoryError."""
    token = checkpoint_fixture["token"]
    ckpt_file = checkpoint_fixture["ckpt_file"]

    # Mutate file on disk
    corrupted_content = b"TAMPERED_CHECKPOINT_CONTENT_" + b"\x99" * (checkpoint_fixture["ckpt_size"] - 28)
    ckpt_file.write_bytes(corrupted_content)

    with pytest.raises(ModelFactoryError, match="TOCTOU validation failure: Checkpoint physical bytes modified"):
        ControlledModelFactory.read_and_verify_checkpoint_bytes(token)


def test_4_checkpoint_symlink_replacement_fails(checkpoint_fixture):
    """Verifies that replacing the checkpoint with a symlink before loading fails immediately."""
    token = checkpoint_fixture["token"]
    ckpt_file = checkpoint_fixture["ckpt_file"]

    outside_file = checkpoint_fixture["tmp_path"] / "outside.pt"
    outside_file.write_bytes(ckpt_file.read_bytes())
    ckpt_file.unlink()
    ckpt_file.symlink_to(outside_file)

    with pytest.raises(ModelFactoryError, match="TOCTOU validation failure"):
        ControlledModelFactory.read_and_verify_checkpoint_bytes(token)


def test_5_checkpoint_file_deleted_fails(checkpoint_fixture):
    """Verifies that deleting the checkpoint file before loading fails."""
    token = checkpoint_fixture["token"]
    checkpoint_fixture["ckpt_file"].unlink()

    with pytest.raises(ModelFactoryError, match="TOCTOU validation failure: Checkpoint file disappeared"):
        ControlledModelFactory.read_and_verify_checkpoint_bytes(token)


def test_6_checkpoint_size_change_fails(checkpoint_fixture):
    """Verifies that altering checkpoint file size before loading fails."""
    token = checkpoint_fixture["token"]
    # Truncate file
    checkpoint_fixture["ckpt_file"].write_bytes(b"TRUNCATED")

    with pytest.raises(ModelFactoryError, match="TOCTOU validation failure: Checkpoint file size changed"):
        ControlledModelFactory.read_and_verify_checkpoint_bytes(token)


def test_7_synthetic_checkpoint_in_production_fails(tmp_path):
    """Verifies that a synthetic checkpoint cannot be used to build a production model."""
    synth_file = tmp_path / "synth_model.pt"
    synth_token = create_synthetic_checkpoint_fixture(
        encoder_name="wespeaker_resnet293",
        target_file=synth_file,
    )

    with pytest.raises(ModelFactoryError, match="Cannot build production model from synthetic checkpoint"):
        ControlledModelFactory.build_production_model(synth_token)


def test_8_verified_checkpoint_signature_forgery_prevented(checkpoint_fixture):
    """Verifies that an attacker cannot forge a VerifiedCheckpoint instance."""
    token = checkpoint_fixture["token"]

    # Attempt to forge token with manipulated fields or fake signature
    with pytest.raises(CheckpointPreflightError, match="FORGERY DETECTED"):
        VerifiedCheckpoint(
            encoder_name=token.encoder_name,
            canonical_path=token.canonical_path,
            expected_sha256=token.expected_sha256,
            computed_sha256=token.computed_sha256,
            file_size_bytes=token.file_size_bytes,
            verification_timestamp=token.verification_timestamp,
            verification_status=token.verification_status,
            authorized_root=token.authorized_root,
            execution_mode=ExecutionMode.PRODUCTION,
            _security_signature="FORGED_FAKE_SIGNATURE_HEX",
        )


def test_9_verified_model_handle_signature_forgery_prevented(checkpoint_fixture):
    """Verifies that an attacker cannot forge a VerifiedModelHandle instance."""
    token = checkpoint_fixture["token"]

    with pytest.raises(ModelFactoryError, match="FORGERY DETECTED"):
        VerifiedModelHandle(
            encoder_name=token.encoder_name,
            verified_checkpoint=token,
            model_instance=torch.nn.Identity(),
            creation_timestamp="2026-11-09T00:00:00Z",
            is_synthetic=False,
            _handle_signature="FORGED_HANDLE_SIGNATURE",
        )
