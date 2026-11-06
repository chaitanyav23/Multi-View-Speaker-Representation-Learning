#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3 Remediation: Controlled Model Factory & Verified Handle Binding
Enforces that model construction occurs exclusively from cryptographic VerifiedCheckpoint tokens.
Performs immediate pre-load TOCTOU revalidation of physical checkpoint bytes.
Binds the resulting model instance into an immutable VerifiedModelHandle.
Strictly prohibits arbitrary preloaded model injection in production.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import secrets
from typing import Any, Callable, Dict, List, Optional

from checkpoint_preflight import (
    ExecutionMode,
    VerifiedCheckpoint,
    CheckpointPreflightError,
    compute_file_sha256,
)


class ModelFactoryError(Exception):
    """Raised when model construction or identity binding fails."""
    pass


_HANDLE_PEPPER = secrets.token_hex(32)


def _generate_handle_signature(
    encoder_name: str,
    checkpoint_sha256: str,
    creation_timestamp: str,
    is_synthetic: bool,
) -> str:
    h = hashlib.sha256()
    h.update(_HANDLE_PEPPER.encode("utf-8"))
    h.update(encoder_name.encode("utf-8"))
    h.update(checkpoint_sha256.encode("utf-8"))
    h.update(creation_timestamp.encode("utf-8"))
    h.update(str(is_synthetic).encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class VerifiedModelHandle:
    """
    Immutable binding between a verified model instance and its verified checkpoint.
    Production pipelines strictly require this handle, prohibiting arbitrary preloaded models.
    """
    encoder_name: str
    verified_checkpoint: VerifiedCheckpoint
    model_instance: Any
    creation_timestamp: str
    is_synthetic: bool
    _handle_signature: str

    def __post_init__(self):
        # Validate that the model handle signature is valid
        expected_sig = _generate_handle_signature(
            encoder_name=self.encoder_name,
            checkpoint_sha256=self.verified_checkpoint.computed_sha256,
            creation_timestamp=self.creation_timestamp,
            is_synthetic=self.is_synthetic,
        )
        if not secrets.compare_digest(self._handle_signature, expected_sig):
            raise ModelFactoryError("FORGERY DETECTED: VerifiedModelHandle was not created by the authorized ModelFactory.")

        # Validate encoder consistency
        if self.encoder_name != self.verified_checkpoint.encoder_name:
            raise ModelFactoryError(
                f"Model handle encoder '{self.encoder_name}' does not match "
                f"checkpoint encoder '{self.verified_checkpoint.encoder_name}'."
            )

        # In production mode, synthetic handles are prohibited
        if not self.is_synthetic and self.verified_checkpoint.execution_mode != ExecutionMode.PRODUCTION:
            raise ModelFactoryError(
                "Inconsistent state: Non-synthetic VerifiedModelHandle cannot wrap a synthetic VerifiedCheckpoint."
            )

    @property
    def model(self) -> Any:
        """Returns the underlying model instance."""
        return self.model_instance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder_name": self.encoder_name,
            "checkpoint": self.verified_checkpoint.to_dict(),
            "creation_timestamp": self.creation_timestamp,
            "is_synthetic": self.is_synthetic,
            "model_type": type(self.model_instance).__name__,
        }


class ControlledModelFactory:
    """
    Controlled Factory for constructing encoder model handles.
    Guarantees:
    1. Pre-load TOCTOU revalidation of checkpoint file on disk.
    2. Binds exact checkpoint identity to model handle.
    3. Prevents arbitrary preloaded model injection in production.
    """

    @classmethod
    def revalidate_checkpoint_physical_bytes(cls, verified_checkpoint: VerifiedCheckpoint) -> None:
        """
        Immediately revalidates physical checkpoint file before loading.
        Guards against file substitution, symlink swaps, or disk mutations between preflight and load.
        """
        if not isinstance(verified_checkpoint, VerifiedCheckpoint):
            raise ModelFactoryError(
                f"Expected VerifiedCheckpoint instance, got {type(verified_checkpoint)}."
            )

        raw_path = verified_checkpoint.canonical_path
        if not raw_path.exists():
            raise ModelFactoryError(
                f"TOCTOU validation failure: Checkpoint file disappeared before loading: {raw_path}"
            )
        if raw_path.is_dir():
            raise ModelFactoryError(
                f"TOCTOU validation failure: Target path is a directory: {raw_path}"
            )
        if not raw_path.is_file():
            raise ModelFactoryError(
                f"TOCTOU validation failure: Target is not a regular file: {raw_path}"
            )

        # Verify symlink containment
        canonical_now = raw_path.resolve()
        auth_root = verified_checkpoint.authorized_root
        if not (canonical_now == auth_root or canonical_now.is_relative_to(auth_root)):
            raise ModelFactoryError(
                f"TOCTOU validation failure: Checkpoint {canonical_now} escapes authorized root {auth_root}"
            )

        # Re-verify size
        curr_size = canonical_now.stat().st_size
        if curr_size != verified_checkpoint.file_size_bytes:
            raise ModelFactoryError(
                f"TOCTOU validation failure: Checkpoint file size changed between preflight and load: "
                f"expected {verified_checkpoint.file_size_bytes} bytes, found {curr_size} bytes."
            )

        # Re-compute SHA-256 from physical bytes
        curr_sha256 = compute_file_sha256(canonical_now)
        if curr_sha256 != verified_checkpoint.computed_sha256:
            raise ModelFactoryError(
                f"TOCTOU validation failure: Checkpoint physical bytes modified between preflight and load: "
                f"computed {curr_sha256} != expected {verified_checkpoint.computed_sha256}"
            )

    @classmethod
    def build_production_model(
        cls,
        verified_checkpoint: VerifiedCheckpoint,
        device: str = "cpu",
    ) -> VerifiedModelHandle:
        """
        Constructs a verified production model handle from a VerifiedCheckpoint.
        Performs immediate pre-load revalidation.
        """
        if verified_checkpoint.execution_mode != ExecutionMode.PRODUCTION:
            raise ModelFactoryError(
                f"Cannot build production model from synthetic checkpoint ({verified_checkpoint.encoder_name})."
            )

        # 1. Immediate physical file revalidation
        cls.revalidate_checkpoint_physical_bytes(verified_checkpoint)

        # 2. Production dispatch to framework loaders (TitaNet / WeSpeaker)
        encoder_name = verified_checkpoint.encoder_name
        ckpt_path = str(verified_checkpoint.canonical_path)

        if encoder_name == "titanet_large":
            # NeMo EncDecSpeakerLabelModel
            from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
            model_instance = EncDecSpeakerLabelModel.restore_from(restore_path=ckpt_path, map_location=device)
            model_instance.eval()
            for p in model_instance.parameters():
                p.requires_grad_(False)

        elif encoder_name in ("wespeaker_resnet293", "wespeaker_resnet293_lm"):
            if ckpt_path.endswith(".onnx"):
                import onnxruntime as ort
                providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
                model_instance = ort.InferenceSession(ckpt_path, providers=providers)
            else:
                import torch
                # PyTorch checkpoint loading
                state_dict = torch.load(ckpt_path, map_location=device)
                model_instance = state_dict  # Or instantiated WeSpeakerResNet293
        else:
            raise ModelFactoryError(f"Unsupported production encoder: '{encoder_name}'")

        timestamp = datetime.now(timezone.utc).isoformat()
        signature = _generate_handle_signature(
            encoder_name=encoder_name,
            checkpoint_sha256=verified_checkpoint.computed_sha256,
            creation_timestamp=timestamp,
            is_synthetic=False,
        )

        return VerifiedModelHandle(
            encoder_name=encoder_name,
            verified_checkpoint=verified_checkpoint,
            model_instance=model_instance,
            creation_timestamp=timestamp,
            is_synthetic=False,
            _handle_signature=signature,
        )

    @classmethod
    def build_synthetic_model(
        cls,
        verified_checkpoint: VerifiedCheckpoint,
        mock_model_instance: Any,
    ) -> VerifiedModelHandle:
        """
        Constructs a verified synthetic model handle for non-inference testing.
        Requires a synthetic VerifiedCheckpoint.
        """
        if verified_checkpoint.execution_mode != ExecutionMode.SYNTHETIC:
            raise ModelFactoryError(
                "Cannot build synthetic model from production checkpoint."
            )

        # Pre-load revalidation on synthetic fixture
        cls.revalidate_checkpoint_physical_bytes(verified_checkpoint)

        encoder_name = verified_checkpoint.encoder_name
        timestamp = datetime.now(timezone.utc).isoformat()
        signature = _generate_handle_signature(
            encoder_name=encoder_name,
            checkpoint_sha256=verified_checkpoint.computed_sha256,
            creation_timestamp=timestamp,
            is_synthetic=True,
        )

        return VerifiedModelHandle(
            encoder_name=encoder_name,
            verified_checkpoint=verified_checkpoint,
            model_instance=mock_model_instance,
            creation_timestamp=timestamp,
            is_synthetic=True,
            _handle_signature=signature,
        )


def validate_model_handle_strict(
    handle: Any,
    expected_encoder: str,
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
) -> VerifiedModelHandle:
    """
    Validates that a candidate model handle satisfies all security and identity requirements.
    Fails closed on any arbitrary, preloaded, unverified, or mismatched instance.
    """
    if not isinstance(handle, VerifiedModelHandle):
        raise ModelFactoryError(
            f"Arbitrary preloaded model rejected: Expected VerifiedModelHandle, got {type(handle)}."
        )

    if handle.encoder_name != expected_encoder:
        raise ModelFactoryError(
            f"Encoder identity mismatch: handle is '{handle.encoder_name}', expected '{expected_encoder}'."
        )

    if execution_mode == ExecutionMode.PRODUCTION:
        if handle.is_synthetic:
            raise ModelFactoryError(
                f"Security violation: Synthetic model handle for '{expected_encoder}' passed into PRODUCTION execution."
            )
        if handle.verified_checkpoint.execution_mode != ExecutionMode.PRODUCTION:
            raise ModelFactoryError(
                f"Security violation: Model handle wraps non-production checkpoint for '{expected_encoder}'."
            )
    elif execution_mode == ExecutionMode.SYNTHETIC:
        if not handle.is_synthetic:
            # Synthetic mode can test production-style handles if desired, but synthetic handles are normal
            pass
    else:
        raise ModelFactoryError(f"Unsupported execution mode: {execution_mode}")

    return handle
