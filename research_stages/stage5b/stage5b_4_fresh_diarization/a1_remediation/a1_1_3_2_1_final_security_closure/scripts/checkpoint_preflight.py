#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3.1 Remediation: Hardened Checkpoint Preflight Verifier & Identity Token
Enforces mandatory immutable allowlist containment for production checkpoint authorization.
Eliminates caller-expanded checkpoint boundaries in production mode.
Rejects placeholder hashes, missing files, directory targets, symlink escapes,
relative-path traversal, and unauthorized synthetic bypasses in production mode.
Issues cryptographically signed, immutable VerifiedCheckpoint tokens.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import os
from pathlib import Path
import secrets
from typing import Dict, List, Optional, Tuple

from path_utils import get_repo_root, get_research_stages_dir


class ExecutionMode(str, Enum):
    PRODUCTION = "production"
    SYNTHETIC = "synthetic"


class CheckpointStatus(str, Enum):
    VERIFIED = "VERIFIED"
    MISSING_PATH = "MISSING_PATH"
    MISSING_FILE = "MISSING_FILE"
    DIRECTORY_SUPPLIED = "DIRECTORY_SUPPLIED"
    SYMLINK_ESCAPE = "SYMLINK_ESCAPE"
    UNREADABLE_FILE = "UNREADABLE_FILE"
    INVALID_EXPECTED_HASH = "INVALID_EXPECTED_HASH"
    PLACEHOLDER_HASH = "PLACEHOLDER_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"


class CheckpointPreflightError(Exception):
    """Raised when checkpoint preflight verification fails."""
    pass


# Private internal module secret token ensuring VerifiedCheckpoint
# cannot be forged by external callers passing arbitrary fields.
_INTERNAL_SECURITY_PEPPER = secrets.token_hex(32)


def _generate_checkpoint_signature(
    encoder_name: str,
    canonical_path: str,
    computed_sha256: str,
    file_size_bytes: int,
    execution_mode: str,
) -> str:
    h = hashlib.sha256()
    h.update(_INTERNAL_SECURITY_PEPPER.encode("utf-8"))
    h.update(encoder_name.encode("utf-8"))
    h.update(canonical_path.encode("utf-8"))
    h.update(computed_sha256.encode("utf-8"))
    h.update(str(file_size_bytes).encode("utf-8"))
    h.update(execution_mode.encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True)
class VerifiedCheckpoint:
    """
    Immutable representation of an encoder checkpoint that has undergone
    rigorous cryptographic and path verification.
    Cannot be forged with arbitrary caller-controlled booleans.
    """
    encoder_name: str
    canonical_path: Path
    expected_sha256: str
    computed_sha256: str
    file_size_bytes: int
    verification_timestamp: str
    verification_status: str
    authorized_root: Path
    execution_mode: ExecutionMode
    _security_signature: str

    def __post_init__(self):
        # Validate that signature matches internal pepper
        expected_sig = _generate_checkpoint_signature(
            encoder_name=self.encoder_name,
            canonical_path=str(self.canonical_path),
            computed_sha256=self.computed_sha256,
            file_size_bytes=self.file_size_bytes,
            execution_mode=self.execution_mode.value,
        )
        if not secrets.compare_digest(self._security_signature, expected_sig):
            raise CheckpointPreflightError(
                "FORGERY DETECTED: VerifiedCheckpoint instance was constructed outside the authorized preflight verifier."
            )

    @property
    def is_synthetic(self) -> bool:
        return self.execution_mode == ExecutionMode.SYNTHETIC

    @property
    def is_production(self) -> bool:
        return self.execution_mode == ExecutionMode.PRODUCTION

    @property
    def verified_path(self) -> Path:
        """Alias for canonical_path for backward compatibility."""
        return self.canonical_path

    def to_dict(self) -> Dict:
        return {
            "encoder_name": self.encoder_name,
            "canonical_path": str(self.canonical_path),
            "expected_sha256": self.expected_sha256,
            "computed_sha256": self.computed_sha256,
            "file_size_bytes": self.file_size_bytes,
            "verification_timestamp": self.verification_timestamp,
            "verification_status": self.verification_status,
            "authorized_root": str(self.authorized_root),
            "execution_mode": self.execution_mode.value,
            "is_synthetic": self.is_synthetic,
        }


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash by reading file in binary chunks without model deserialization."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest().lower()


def get_immutable_standard_checkpoint_roots() -> List[Path]:
    """
    Returns the fixed, canonical immutable allowlist of standard checkpoint root directories.
    Production checkpoints must reside strictly within these roots.
    """
    roots = [
        get_research_stages_dir() / "stage3_dual_encoder" / "models",
        Path.home() / ".cache" / "torch",
        Path.home() / ".cache" / "huggingface",
        get_repo_root() / "whisperx" / "assets",
    ]
    return [r.resolve() for r in roots]


def get_permitted_checkpoint_roots() -> List[Path]:
    """Alias for get_immutable_standard_checkpoint_roots for backward compatibility."""
    return get_immutable_standard_checkpoint_roots()


def is_placeholder_hash(hash_str: str) -> bool:
    """
    Detects obvious placeholder hashes:
    - All identical characters (e.g., '0' * 64, 'a' * 64, 'f' * 64)
    - Padded 32-char hashes (e.g. 32-char prefix followed by 32 zeros)
    - Sequential or trivial patterns
    """
    clean = hash_str.strip().lower()
    if len(clean) != 64:
        return True
    if len(set(clean)) <= 1:
        return True
    if clean.endswith("0" * 32) or clean.startswith("0" * 32):
        return True
    for pattern_len in [4, 8, 16]:
        if clean == clean[:pattern_len] * (64 // pattern_len):
            return True
    return False


def verify_encoder_checkpoint_strict(
    encoder_name: str,
    checkpoint_path: Optional[Path],
    expected_sha256: Optional[str],
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
    custom_permitted_roots: Optional[List[Path]] = None,
) -> VerifiedCheckpoint:
    """
    Strictly verifies an encoder checkpoint file on disk.

    Finding 1 Hardening:
    In PRODUCTION mode:
    - Checkpoint path and expected_sha256 are mandatory.
    - Path must not be a symlink and cannot traverse parent directories.
    - Checkpoint must exist as a regular, non-empty file.
    - Checkpoint must reside strictly inside the immutable standard allowlist.
    - Caller-supplied custom_permitted_roots cannot expand the allowlist:
      if provided, each custom root must resolve strictly within an immutable standard root
      and cannot be a parent of any standard root. Any custom root outside the allowlist
      is strictly rejected.
    - Expected hash must be a valid, non-placeholder 64-char hexadecimal string.
    - Physical bytes SHA-256 must match expected hash.

    In SYNTHETIC mode:
    - Dedicated test fixtures permitted in custom roots (e.g. tmp_path).
    - Returns a synthetic VerifiedCheckpoint explicitly marked is_synthetic=True.
    """
    if execution_mode == ExecutionMode.PRODUCTION:
        if checkpoint_path is None:
            raise CheckpointPreflightError(
                f"Production preflight failure for '{encoder_name}': Checkpoint path must be explicitly provided."
            )
        if expected_sha256 is None:
            raise CheckpointPreflightError(
                f"Production preflight failure for '{encoder_name}': Expected SHA-256 must be explicitly provided."
            )

    if checkpoint_path is None:
        raise CheckpointPreflightError(f"Checkpoint path missing for '{encoder_name}'.")
    if expected_sha256 is None:
        raise CheckpointPreflightError(f"Expected SHA-256 missing for '{encoder_name}'.")

    # 1. Validate expected hash format
    expected_clean = expected_sha256.strip().lower()
    if len(expected_clean) != 64 or not all(c in "0123456789abcdef" for c in expected_clean):
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': "
            f"Expected SHA-256 must be a 64-character hexadecimal string, got '{expected_sha256}'."
        )

    # 2. Reject placeholder hashes
    if is_placeholder_hash(expected_clean):
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': "
            f"Expected SHA-256 '{expected_sha256}' is an unauthorized placeholder hash."
        )

    # 3. Path existence, traversal, and symlink checks
    raw_path = Path(checkpoint_path)
    
    # Reject relative traversal
    if any(part == ".." for part in raw_path.parts):
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': Relative path traversal ('..') is prohibited: {raw_path}."
        )

    # Reject null bytes
    if "\x00" in str(raw_path):
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': Null byte in path is prohibited."
        )

    # Reject symlinks on the path itself or any of its parents
    if raw_path.is_symlink():
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': Checkpoint path is a symlink: {raw_path}."
        )
    for parent in raw_path.parents:
        if parent.is_symlink():
            raise CheckpointPreflightError(
                f"Preflight validation failed for '{encoder_name}': Checkpoint ancestor is a symlink: {parent}."
            )

    if not raw_path.exists():
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': File does not exist at {raw_path}."
        )
    if raw_path.is_dir():
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': Target path is a directory, not a file: {raw_path}."
        )

    canonical_path = raw_path.resolve()
    if not canonical_path.is_file():
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': Path does not resolve to a regular file: {canonical_path}."
        )

    # 4. Checkpoint root containment & allowlist enforcement (Finding 1)
    standard_roots = get_immutable_standard_checkpoint_roots()

    if execution_mode == ExecutionMode.PRODUCTION:
        if custom_permitted_roots is not None:
            # Validate every custom root against the immutable standard allowlist
            validated_roots: List[Path] = []
            for r in custom_permitted_roots:
                r_raw = Path(r)
                if r_raw.is_symlink():
                    raise CheckpointPreflightError(
                        f"Production preflight failure for '{encoder_name}': Custom root cannot be a symlink: {r_raw}"
                    )
                r_canon = r_raw.resolve()

                # Reject any custom root that expands the allowlist by being a parent of a standard root
                for std in standard_roots:
                    if std.is_relative_to(r_canon) and std != r_canon:
                        raise CheckpointPreflightError(
                            f"Production preflight failure for '{encoder_name}': Custom root {r_canon} is a parent of "
                            f"standard root {std} and illegally expands the permitted boundary."
                        )

                # Custom root must be strictly inside or equal to a standard root
                is_contained = any(
                    r_canon == std or r_canon.is_relative_to(std)
                    for std in standard_roots
                )
                if not is_contained:
                    raise CheckpointPreflightError(
                        f"Production preflight failure for '{encoder_name}': Custom root {r_canon} is outside "
                        f"the immutable standard allowlist: {[str(s) for s in standard_roots]}."
                    )
                validated_roots.append(r_canon)
            permitted_roots = validated_roots
        else:
            permitted_roots = standard_roots
    elif execution_mode == ExecutionMode.SYNTHETIC:
        if custom_permitted_roots is not None:
            permitted_roots = [Path(r).resolve() for r in custom_permitted_roots]
        else:
            permitted_roots = standard_roots
    else:
        raise CheckpointPreflightError(f"Unsupported execution mode: {execution_mode}")

    matched_root: Optional[Path] = None
    for root in permitted_roots:
        if canonical_path == root or canonical_path.is_relative_to(root):
            matched_root = root
            break

    if matched_root is None:
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': Checkpoint {canonical_path} "
            f"escapes authorized checkpoint roots."
        )

    # 5. Compute SHA-256 hash from physical bytes
    try:
        file_size = canonical_path.stat().st_size
        if file_size == 0 and execution_mode == ExecutionMode.PRODUCTION:
            raise CheckpointPreflightError(
                f"Preflight validation failed for '{encoder_name}': Checkpoint file is empty (0 bytes)."
            )
        computed_sha256 = compute_file_sha256(canonical_path)
    except Exception as e:
        if isinstance(e, CheckpointPreflightError):
            raise
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': Could not read file {canonical_path}: {e}"
        ) from e

    # 6. Digest equality check
    if computed_sha256 != expected_clean:
        raise CheckpointPreflightError(
            f"Hash mismatch for '{encoder_name}': computed {computed_sha256} != expected {expected_clean}"
        )

    timestamp = datetime.now(timezone.utc).isoformat()
    signature = _generate_checkpoint_signature(
        encoder_name=encoder_name,
        canonical_path=str(canonical_path),
        computed_sha256=computed_sha256,
        file_size_bytes=file_size,
        execution_mode=execution_mode.value,
    )

    return VerifiedCheckpoint(
        encoder_name=encoder_name,
        canonical_path=canonical_path,
        expected_sha256=expected_clean,
        computed_sha256=computed_sha256,
        file_size_bytes=file_size,
        verification_timestamp=timestamp,
        verification_status=CheckpointStatus.VERIFIED.value,
        authorized_root=matched_root,
        execution_mode=execution_mode,
        _security_signature=signature,
    )


def create_synthetic_checkpoint_fixture(
    encoder_name: str,
    target_file: Path,
    file_bytes: bytes = b"SYNTHETIC_MODEL_CHECKPOINT_FIXTURE_PAYLOAD",
) -> VerifiedCheckpoint:
    """Helper for synthetic test fixtures creating a valid, non-placeholder synthetic checkpoint."""
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_bytes(file_bytes)
    sha = hashlib.sha256(file_bytes).hexdigest()
    return verify_encoder_checkpoint_strict(
        encoder_name=encoder_name,
        checkpoint_path=target_file,
        expected_sha256=sha,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_permitted_roots=[target_file.parent],
    )


def run_production_checkpoint_preflight(
    titanet_path: Optional[Path],
    titanet_expected_sha256: Optional[str],
    wespeaker_path: Optional[Path],
    wespeaker_expected_sha256: Optional[str],
    custom_permitted_roots: Optional[List[Path]] = None,
) -> Tuple[VerifiedCheckpoint, VerifiedCheckpoint]:
    """
    Mandatory preflight entry point for production diarization execution.
    Fails closed on any error or missing checkpoint.
    """
    v_tita = verify_encoder_checkpoint_strict(
        encoder_name="titanet_large",
        checkpoint_path=titanet_path,
        expected_sha256=titanet_expected_sha256,
        execution_mode=ExecutionMode.PRODUCTION,
        custom_permitted_roots=custom_permitted_roots,
    )
    v_wes = verify_encoder_checkpoint_strict(
        encoder_name="wespeaker_resnet293",
        checkpoint_path=wespeaker_path,
        expected_sha256=wespeaker_expected_sha256,
        execution_mode=ExecutionMode.PRODUCTION,
        custom_permitted_roots=custom_permitted_roots,
    )
    return v_tita, v_wes
