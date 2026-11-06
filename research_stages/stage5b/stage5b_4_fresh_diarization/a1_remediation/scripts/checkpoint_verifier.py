#!/usr/bin/env python3
"""
Stage 5B.4-A.1 Remediation: Checkpoint Verifier
Cryptographically verifies speaker encoder checkpoints on disk against expected SHA-256 hashes.
Strictly file-hash verification only; zero model loading or weight deserialization.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


class CheckpointStatus(str, Enum):
    VERIFIED = "VERIFIED"
    MISSING_FILE = "MISSING_FILE"
    UNREADABLE_FILE = "UNREADABLE_FILE"
    INVALID_EXPECTED_HASH = "INVALID_EXPECTED_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"


@dataclass(frozen=True)
class CheckpointVerificationResult:
    checkpoint_path: str
    status: CheckpointStatus
    computed_sha256: Optional[str]
    expected_sha256: Optional[str]
    file_size_bytes: Optional[int]
    error_message: Optional[str] = None

    @property
    def is_verified(self) -> bool:
        return self.status == CheckpointStatus.VERIFIED

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash by reading file in binary chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_checkpoint(
    checkpoint_path: Path,
    expected_sha256: str,
) -> CheckpointVerificationResult:
    """
    Verifies that a checkpoint file exists and its SHA-256 digest matches expected_sha256.
    Zero model loading or weight inspection.
    """
    p = Path(checkpoint_path).resolve()

    # Validate expected hash string
    if not expected_sha256 or len(expected_sha256) != 64 or not all(c in "0123456789abcdefABCDEF" for c in expected_sha256):
        return CheckpointVerificationResult(
            checkpoint_path=str(p),
            status=CheckpointStatus.INVALID_EXPECTED_HASH,
            computed_sha256=None,
            expected_sha256=expected_sha256,
            file_size_bytes=None,
            error_message=f"Expected hash '{expected_sha256}' is not a valid 64-char hexadecimal string.",
        )

    expected_clean = expected_sha256.strip().lower()

    if not p.exists() or not p.is_file():
        return CheckpointVerificationResult(
            checkpoint_path=str(p),
            status=CheckpointStatus.MISSING_FILE,
            computed_sha256=None,
            expected_sha256=expected_clean,
            file_size_bytes=None,
            error_message=f"Checkpoint file does not exist or is not a regular file: {p}",
        )

    try:
        size = p.stat().st_size
        computed = compute_file_sha256(p)
    except Exception as e:
        return CheckpointVerificationResult(
            checkpoint_path=str(p),
            status=CheckpointStatus.UNREADABLE_FILE,
            computed_sha256=None,
            expected_sha256=expected_clean,
            file_size_bytes=None,
            error_message=f"Failed to read checkpoint file: {e}",
        )

    if computed != expected_clean:
        return CheckpointVerificationResult(
            checkpoint_path=str(p),
            status=CheckpointStatus.HASH_MISMATCH,
            computed_sha256=computed,
            expected_sha256=expected_clean,
            file_size_bytes=size,
            error_message=f"Hash mismatch: computed {computed} != expected {expected_clean}",
        )

    return CheckpointVerificationResult(
        checkpoint_path=str(p),
        status=CheckpointStatus.VERIFIED,
        computed_sha256=computed,
        expected_sha256=expected_clean,
        file_size_bytes=size,
        error_message=None,
    )
