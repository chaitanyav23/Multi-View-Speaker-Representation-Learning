#!/usr/bin/env python3
"""
Stage 5B.4-A.1.2 Remediation: Checkpoint Preflight Verifier (Hardened)
Enforces mandatory on-disk byte-level SHA-256 cryptographic verification for production execution.
Rejects placeholder hashes, missing files, directory targets, symlink escapes,
and unauthorized synthetic bypasses in production mode.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from path_utils import get_repo_root


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


@dataclass(frozen=True)
class VerifiedCheckpoint:
    encoder_name: str
    verified_path: Path
    computed_sha256: str
    file_size_bytes: int
    is_synthetic: bool = False

    @property
    def is_verified(self) -> bool:
        return True

    def to_dict(self) -> Dict:
        return {
            "encoder_name": self.encoder_name,
            "verified_path": str(self.verified_path),
            "computed_sha256": self.computed_sha256,
            "file_size_bytes": self.file_size_bytes,
            "is_synthetic": self.is_synthetic,
        }


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash by reading file in binary chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest().lower()


def get_permitted_checkpoint_roots() -> List[Path]:
    """Returns canonical list of permitted root directories for checkpoint storage."""
    roots = [
        Path.home() / ".cache" / "torch",
        Path.home() / ".cache" / "huggingface",
        get_repo_root(),
    ]
    return [r.resolve() for r in roots]


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
    # All identical characters
    if len(set(clean)) <= 1:
        return True
    # Padded with 32 zeros
    if clean.endswith("0" * 32) or clean.startswith("0" * 32):
        return True
    # Repeated 8-char or 16-char sub-patterns (e.g. '1234' * 16)
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

    In PRODUCTION mode:
    - Both checkpoint_path and expected_sha256 are mandatory.
    - Checkpoint must exist as a regular file.
    - Path and any symlink target must reside inside permitted checkpoint roots.
    - Expected hash must be a valid, non-placeholder 64-char hexadecimal string.
    - File bytes SHA-256 must match expected hash.

    In SYNTHETIC mode:
    - Dedicated test fixtures permitted in custom roots (e.g. tmp_path).
    - Returns VerifiedCheckpoint with is_synthetic=True.
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

    # 3. Path existence and file type checks
    p = Path(checkpoint_path)
    if not p.exists():
        raise CheckpointPreflightError(
            f"Preflight checkpoint file not found for '{encoder_name}': {p}"
        )
    if p.is_dir():
        raise CheckpointPreflightError(
            f"Preflight target for '{encoder_name}' is a directory, not a regular file: {p}"
        )
    if not p.is_file():
        raise CheckpointPreflightError(
            f"Preflight target for '{encoder_name}' is not a regular file: {p}"
        )

    # 4. Checkpoint root containment and symlink escape checks
    permitted_roots = custom_permitted_roots or get_permitted_checkpoint_roots()
    real_path = p.resolve()

    if execution_mode == ExecutionMode.PRODUCTION:
        is_contained = any(real_path == r or real_path.is_relative_to(r) for r in permitted_roots)
        if not is_contained:
            raise CheckpointPreflightError(
                f"Checkpoint path containment violation for '{encoder_name}': "
                f"'{real_path}' resolves outside permitted roots: {[str(r) for r in permitted_roots]}"
            )

        # Check symlink escapes
        if p.is_symlink():
            link_dest = os.readlink(p)
            resolved_dest = (p.parent / link_dest).resolve()
            dest_contained = any(resolved_dest == r or resolved_dest.is_relative_to(r) for r in permitted_roots)
            if not dest_contained:
                raise CheckpointPreflightError(
                    f"Symlink escape detected for '{encoder_name}': {p} points outside permitted roots to {resolved_dest}"
                )

    # 5. Byte-level SHA-256 calculation
    try:
        size = p.stat().st_size
        computed = compute_file_sha256(real_path)
    except Exception as e:
        raise CheckpointPreflightError(
            f"Failed to read checkpoint file for '{encoder_name}' at {real_path}: {e}"
        )

    # 6. Verify hash match
    if computed != expected_clean:
        raise CheckpointPreflightError(
            f"Hash mismatch for '{encoder_name}': computed {computed} != expected {expected_clean}"
        )

    return VerifiedCheckpoint(
        encoder_name=encoder_name,
        verified_path=real_path,
        computed_sha256=computed,
        file_size_bytes=size,
        is_synthetic=(execution_mode == ExecutionMode.SYNTHETIC),
    )


def run_production_checkpoint_preflight(
    titanet_path: Optional[Path],
    titanet_expected_sha256: Optional[str],
    wespeaker_path: Optional[Path],
    wespeaker_expected_sha256: Optional[str],
    custom_permitted_roots: Optional[List[Path]] = None,
) -> Tuple[VerifiedCheckpoint, VerifiedCheckpoint]:
    """
    Mandatory production preflight entry point.
    Verifies both checkpoints on disk before any model loading or execution.
    Fails closed if either checkpoint fails.
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
