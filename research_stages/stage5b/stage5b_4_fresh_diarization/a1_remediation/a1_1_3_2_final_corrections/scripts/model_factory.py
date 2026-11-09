#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3.2 Remediation: Controlled Model Factory & Verified Handle Binding
Enforces that model construction occurs exclusively from cryptographic VerifiedCheckpoint tokens.

Remediates Finding 1 (TitaNet Checkpoint Identity & Deserialization):
1. Reads checkpoint file into immutable Python bytes buffer and cryptographically revalidates SHA-256.
2. For WeSpeaker: loads state_dict directly from io.BytesIO(ckpt_bytes) in-memory buffer.
3. For TitaNet / NeMo:
   - Uses an isolated, securely created staging directory (mode 0700).
   - Writes verified bytes into a new staging file using exclusive creation (O_EXCL | O_NOFOLLOW, mode 0600).
   - Re-verifies file descriptor, link count, and SHA-256 immediately before loader invocation.
   - Consumes the staging path through the NeMo loader.
   - Cleans up (wipes and unlinks) staging file and directory in a guaranteed finally block.
   - Prevents pathname substitution attacks from loading unverified bytes.
4. Binds model instance into an immutable VerifiedModelHandle.
5. Strictly prohibits arbitrary preloaded model injection in production.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import secrets
import stat
import tempfile
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ============================================================================
# Official WeSpeaker ResNet293 Architecture (Stage 3 Parity)
# ============================================================================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TSTP(nn.Module):
    """Temporal Statistics Pooling: concatenates temporal mean and standard deviation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        mean = x.mean(dim=-1)
        std = torch.sqrt(torch.clamp(x.var(dim=-1, unbiased=True), min=1e-10))
        return torch.cat([mean, std], dim=-1)


class WeSpeakerResNet293(nn.Module):
    def __init__(
        self,
        num_blocks: List[int] = [10, 20, 64, 3],
        m_channels: int = 32,
        feat_dim: int = 80,
        embed_dim: int = 256,
        two_emb_layer: bool = False,
    ) -> None:
        super().__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.two_emb_layer = two_emb_layer

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(Bottleneck, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, m_channels * 8, num_blocks[3], stride=2)

        out_channels = m_channels * 8 * Bottleneck.expansion
        stats_dim = out_channels * (feat_dim // 8) * 2

        self.pool = TSTP()
        self.seg_1 = nn.Linear(stats_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block: Any, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        stats = self.pool(out)
        embed = self.seg_1(stats)
        if self.two_emb_layer:
            embed = self.seg_bn_1(embed)
            embed = self.seg_2(embed)
        return embed


@dataclass(frozen=True)
class VerifiedModelHandle:
    """
    Cryptographically bound handle coupling an instantiated model to its verified checkpoint.
    """
    encoder_name: str
    verified_checkpoint: VerifiedCheckpoint
    model_instance: Any
    creation_timestamp: str
    is_synthetic: bool
    _handle_signature: str

    def __post_init__(self):
        expected_sig = _generate_handle_signature(
            encoder_name=self.encoder_name,
            checkpoint_sha256=self.verified_checkpoint.computed_sha256,
            creation_timestamp=self.creation_timestamp,
            is_synthetic=self.is_synthetic,
        )
        if not secrets.compare_digest(self._handle_signature, expected_sig):
            raise ModelFactoryError(
                "FORGERY DETECTED: VerifiedModelHandle signature mismatch."
            )

    @property
    def model(self) -> Any:
        return self.model_instance

    def to_dict(self) -> Dict:
        return {
            "encoder_name": self.encoder_name,
            "checkpoint": self.verified_checkpoint.to_dict(),
            "creation_timestamp": self.creation_timestamp,
            "is_synthetic": self.is_synthetic,
            "model_type": type(self.model_instance).__name__,
        }


# ============================================================================
# Finding 1: Secure TitaNet Staging Loader
# ============================================================================

def load_titanet_model_from_verified_bytes(
    verified_checkpoint: VerifiedCheckpoint,
    ckpt_bytes: bytes,
    device: str = "cpu",
    loader_fn: Optional[Callable[[str, str], Any]] = None,
) -> Any:
    """
    Securely stages verified in-memory checkpoint bytes and loads TitaNet/NeMo model.
    Remediates Finding 1:
    - Never passes original caller pathname to NeMo.
    - Writes exact verified bytes into an isolated, exclusive staging file (mode 0600)
      inside an isolated directory (mode 0700).
    - Verifies file descriptor link count and byte hash immediately before loading.
    - Consumes the staging path.
    - In a guaranteed finally block, wipes and deletes the staging file and directory.
    """
    if not isinstance(verified_checkpoint, VerifiedCheckpoint):
        raise ModelFactoryError(
            f"Expected VerifiedCheckpoint instance, got {type(verified_checkpoint)}."
        )

    # Validate memory buffer hash
    buf_hash = hashlib.sha256(ckpt_bytes).hexdigest().lower()
    if buf_hash != verified_checkpoint.computed_sha256.strip().lower():
        raise ModelFactoryError(
            f"TitaNet memory buffer digest mismatch: {buf_hash} != {verified_checkpoint.computed_sha256}"
        )

    stage_dir = tempfile.mkdtemp(prefix="nemo_titanet_stage_")
    staging_file = Path(stage_dir) / "model.nemo"

    try:
        # Secure directory permissions
        os.chmod(stage_dir, 0o700)

        # Exclusive file creation with O_NOFOLLOW
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        fd = os.open(str(staging_file), flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(ckpt_bytes)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            raise ModelFactoryError(f"Failed writing TitaNet verified bytes to staging: {e}") from e

        # Re-verify staging file descriptor and link count immediately before loader invocation
        open_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW

        fd_check = os.open(str(staging_file), open_flags)
        try:
            st = os.fstat(fd_check)
            if not stat.S_ISREG(st.st_mode):
                raise ModelFactoryError("TitaNet staging target is not a regular file.")
            if st.st_nlink != 1:
                raise ModelFactoryError(
                    f"TitaNet staging security violation: multiple hard links detected ({st.st_nlink})."
                )
            if st.st_size != len(ckpt_bytes):
                raise ModelFactoryError(
                    f"TitaNet staging size mismatch: expected {len(ckpt_bytes)} bytes, found {st.st_size}."
                )
            with os.fdopen(fd_check, "rb") as f_in:
                read_back = f_in.read()
                read_hash = hashlib.sha256(read_back).hexdigest().lower()
                if read_hash != verified_checkpoint.computed_sha256.strip().lower():
                    raise ModelFactoryError(
                        f"TitaNet staging file corrupted between write and load: {read_hash} != {verified_checkpoint.computed_sha256}"
                    )
        except Exception:
            raise

        # Invoke model loader
        if loader_fn is not None:
            model_instance = loader_fn(str(staging_file), device)
        else:
            from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
            model_instance = EncDecSpeakerLabelModel.restore_from(
                restore_path=str(staging_file),
                map_location=device,
            )

        if hasattr(model_instance, "eval"):
            model_instance.eval()
        if hasattr(model_instance, "parameters"):
            for p in model_instance.parameters():
                p.requires_grad_(False)

        return model_instance

    finally:
        # Guaranteed cleanup: overwrite staging bytes, unlink file, rmdir
        try:
            if staging_file.exists():
                try:
                    staging_file.write_bytes(b"\x00" * min(4096, len(ckpt_bytes)))
                except Exception:
                    pass
                staging_file.unlink(missing_ok=True)
        except Exception:
            pass

        try:
            if Path(stage_dir).exists():
                os.rmdir(stage_dir)
        except Exception:
            pass


class ControlledModelFactory:
    """
    Controlled Factory for constructing encoder model handles.
    Guarantees:
    1. Pre-load TOCTOU revalidation of checkpoint file on disk.
    2. Exact-byte loading: Reads into an immutable memory buffer, verifies SHA-256 on that buffer.
    3. Proper WeSpeaker ResNet293 architecture instantiation and state_dict loading directly from memory.
    4. Secure staging loader for TitaNet/NeMo consuming verified bytes with exclusive staging and cleanup.
    5. Binds exact checkpoint identity to model handle.
    6. Prevents arbitrary preloaded model injection in production.
    """

    # Optional test hook for NeMo loader without importing heavy weights
    nemo_loader_override: Optional[Callable[[str, str], Any]] = None

    @classmethod
    def read_and_verify_checkpoint_bytes(cls, verified_checkpoint: VerifiedCheckpoint) -> bytes:
        """
        Reads checkpoint file into an immutable bytes buffer and re-verifies SHA-256.
        Guarantees that the exact bytes passed to the model loader match the verified digest,
        defeating Time-of-Check / Time-of-Use (TOCTOU) races.
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
        if raw_path.is_symlink():
            raise ModelFactoryError(
                f"TOCTOU validation failure: Checkpoint path was replaced with a symlink: {raw_path}"
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

        # Open securely with O_NOFOLLOW
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        try:
            fd = os.open(str(raw_path), flags)
        except OSError as e:
            raise ModelFactoryError(
                f"TOCTOU validation failure: Could not open checkpoint securely without symlinks: {e}"
            ) from e

        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ModelFactoryError(
                    f"TOCTOU validation failure: Checkpoint descriptor is not a regular file."
                )
            if st.st_nlink != 1:
                raise ModelFactoryError(
                    f"TOCTOU validation failure: Checkpoint file has multiple hard links ({st.st_nlink})."
                )
            with os.fdopen(fd, "rb") as f:
                content = f.read()
        except Exception as e:
            if isinstance(e, ModelFactoryError):
                raise
            raise ModelFactoryError(
                f"TOCTOU validation failure: Failed reading checkpoint bytes: {e}"
            ) from e

        # Re-verify size
        if len(content) != verified_checkpoint.file_size_bytes:
            raise ModelFactoryError(
                f"TOCTOU validation failure: Checkpoint file size changed between preflight and load: "
                f"expected {verified_checkpoint.file_size_bytes} bytes, found {len(content)} bytes."
            )

        # Re-verify SHA-256 on the exact bytes read into memory
        computed_sha256 = hashlib.sha256(content).hexdigest().lower()
        if computed_sha256 != verified_checkpoint.computed_sha256:
            raise ModelFactoryError(
                f"TOCTOU validation failure: Checkpoint physical bytes modified between preflight and load: "
                f"computed {computed_sha256} != expected {verified_checkpoint.computed_sha256}"
            )

        return content

    @classmethod
    def revalidate_checkpoint_physical_bytes(cls, verified_checkpoint: VerifiedCheckpoint) -> None:
        """Compatibility wrapper that verifies bytes on disk."""
        cls.read_and_verify_checkpoint_bytes(verified_checkpoint)

    @classmethod
    def build_production_model(
        cls,
        verified_checkpoint: VerifiedCheckpoint,
        device: str = "cpu",
    ) -> VerifiedModelHandle:
        """
        Constructs a verified production model handle from a VerifiedCheckpoint.
        Loads strictly from verified in-memory byte buffer (and secure staging for TitaNet).
        """
        if verified_checkpoint.execution_mode != ExecutionMode.PRODUCTION:
            raise ModelFactoryError(
                f"Cannot build production model from synthetic checkpoint ({verified_checkpoint.encoder_name})."
            )

        # 1. Read and revalidate exact checkpoint bytes into immutable buffer
        ckpt_bytes = cls.read_and_verify_checkpoint_bytes(verified_checkpoint)

        # 2. Production dispatch to framework loaders (TitaNet / WeSpeaker)
        encoder_name = verified_checkpoint.encoder_name
        ckpt_path = str(verified_checkpoint.canonical_path)

        if encoder_name == "titanet_large":
            # NeMo EncDecSpeakerLabelModel loaded through secure staging
            model_instance = load_titanet_model_from_verified_bytes(
                verified_checkpoint=verified_checkpoint,
                ckpt_bytes=ckpt_bytes,
                device=device,
                loader_fn=cls.nemo_loader_override,
            )

        elif encoder_name in ("wespeaker_resnet293", "wespeaker_resnet293_lm"):
            if ckpt_path.endswith(".onnx"):
                import onnxruntime as ort
                providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
                model_instance = ort.InferenceSession(ckpt_bytes, providers=providers)
            else:
                resnet_model = WeSpeakerResNet293(
                    num_blocks=[10, 20, 64, 3],
                    m_channels=32,
                    feat_dim=80,
                    embed_dim=256,
                    two_emb_layer=False,
                )
                state_dict = torch.load(io.BytesIO(ckpt_bytes), map_location=device)
                
                model_keys = set(resnet_model.state_dict().keys())
                filtered_sd = {k: v for k, v in state_dict.items() if k in model_keys}
                missing_keys = model_keys - set(filtered_sd.keys())
                if missing_keys:
                    raise ModelFactoryError(
                        f"STRICT FAILURE: Missing keys in WeSpeaker ResNet293 state_dict: {missing_keys}"
                    )
                resnet_model.load_state_dict(filtered_sd, strict=True)
                resnet_model.to(device)
                resnet_model.eval()
                for p in resnet_model.parameters():
                    p.requires_grad_(False)
                model_instance = resnet_model
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
        Constructs a verified synthetic model handle for unit tests.
        Pre-load revalidation is still strictly performed on the synthetic checkpoint.
        """
        if not verified_checkpoint.is_synthetic:
            raise ModelFactoryError(
                "Cannot build synthetic model from production checkpoint. Use build_production_model."
            )

        # Pre-load byte verification on synthetic fixture
        cls.read_and_verify_checkpoint_bytes(verified_checkpoint)

        timestamp = datetime.now(timezone.utc).isoformat()
        signature = _generate_handle_signature(
            encoder_name=verified_checkpoint.encoder_name,
            checkpoint_sha256=verified_checkpoint.computed_sha256,
            creation_timestamp=timestamp,
            is_synthetic=True,
        )

        return VerifiedModelHandle(
            encoder_name=verified_checkpoint.encoder_name,
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
        pass
    else:
        raise ModelFactoryError(f"Unsupported execution mode: {execution_mode}")

    return handle
