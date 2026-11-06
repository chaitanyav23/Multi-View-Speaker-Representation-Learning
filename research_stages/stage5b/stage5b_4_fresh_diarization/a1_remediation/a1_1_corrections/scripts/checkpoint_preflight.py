#!/usr/bin/env python3
"""
Stage 5B.4-A.1.1 Remediation: Checkpoint Preflight Verifier
Wired directly into orchestrator preflight before any cache identities are constructed.
Enforces on-disk byte-level SHA-256 verification via checkpoint_verifier.py.
Rejects truncated placeholders and guarantees granular invalidation per encoder.
"""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

from path_utils import get_stage5b_4_dir

A1_SCRIPTS = get_stage5b_4_dir() / "a1_remediation" / "scripts"
if str(A1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A1_SCRIPTS))

from checkpoint_verifier import (
    CheckpointStatus,
    CheckpointVerificationResult,
    verify_checkpoint,
)


class CheckpointPreflightError(Exception):
    """Raised when checkpoint preflight verification fails."""
    pass


@dataclass(frozen=True)
class PreflightEncoderConfig:
    encoder_name: str
    checkpoint_path: Path
    expected_sha256: str


@dataclass(frozen=True)
class PreflightVerificationSummary:
    titanet_result: CheckpointVerificationResult
    wespeaker_result: CheckpointVerificationResult

    @property
    def is_all_verified(self) -> bool:
        return self.titanet_result.is_verified and self.wespeaker_result.is_verified


def verify_encoder_checkpoint_preflight(
    encoder_name: str,
    checkpoint_path: Path,
    expected_sha256: str,
) -> CheckpointVerificationResult:
    """
    Verifies a single encoder checkpoint file on disk.
    Requires a valid 64-character hex string for expected_sha256.
    Fails immediately on missing file, unreadable file, or hash mismatch.
    """
    p = Path(checkpoint_path)

    # 1. Reject invalid or truncated hash strings immediately
    if not expected_sha256 or len(expected_sha256) != 64 or not all(c in "0123456789abcdefABCDEF" for c in expected_sha256):
        raise CheckpointPreflightError(
            f"Preflight validation failed for '{encoder_name}': "
            f"Expected SHA-256 must be a 64-character hexadecimal string, got '{expected_sha256}'."
        )

    # 2. Verify on disk
    result = verify_checkpoint(p, expected_sha256)
    if not result.is_verified:
        raise CheckpointPreflightError(
            f"Preflight checkpoint verification failed for '{encoder_name}' at {p}: "
            f"status={result.status.value}, error={result.error_message}"
        )

    return result


def run_preflight_verification(
    titanet_path: Path,
    titanet_expected_sha256: str,
    wespeaker_path: Path,
    wespeaker_expected_sha256: str,
) -> PreflightVerificationSummary:
    """
    Runs preflight verification for both TitaNet and WeSpeaker checkpoints.
    Fails if either checkpoint is missing or corrupt.
    """
    res_tita = verify_encoder_checkpoint_preflight(
        encoder_name="titanet_large",
        checkpoint_path=titanet_path,
        expected_sha256=titanet_expected_sha256,
    )

    res_res = verify_encoder_checkpoint_preflight(
        encoder_name="wespeaker_resnet293",
        checkpoint_path=wespeaker_path,
        expected_sha256=wespeaker_expected_sha256,
    )

    return PreflightVerificationSummary(
        titanet_result=res_tita,
        wespeaker_result=res_res,
    )
