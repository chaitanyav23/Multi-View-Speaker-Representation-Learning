#!/usr/bin/env python3
"""
Stage 5B.4-A.1.2 Remediation: Final RTTM Path and Identifier Guard
Enforces strict identifier validation and comprehensive path containment
for output RTTMs, manifests, and reports.
Defends against traversal in recording_id and condition_id, symlink escapes,
and redirection into prohibited historical stages.
"""

from enum import Enum
import os
from pathlib import Path
import re
from typing import List, Optional

from path_utils import get_stage5b_4_dir, get_research_stages_dir
from checkpoint_preflight import ExecutionMode

# Authorized condition IDs
AUTHORIZED_CONDITIONS = {"E00", "E01", "E10", "E11", "E20", "E21"}

# Benchmark recording ID regex (canonical: R01_sara_sharif .. R08_uk_parliament)
BENCHMARK_RECORDING_REGEX = re.compile(r"^R0[1-8]_[a-z0-9_]+$")
# Synthetic recording ID regex permitted strictly in synthetic mode
SYNTHETIC_RECORDING_REGEX = re.compile(r"^(syn|test|rec)_[a-zA-Z0-9_]+$")

PROHIBITED_STAGE_NAMES = [
    "stage1_data_prep",
    "stage2_baseline",
    "stage3_dual_encoder",
    "stage5a",
    "stage5b_1_vad_optimization",
    "stage5b_2_overlap_optimization",
    "stage5b_3_speaker_count",
]


class OutputSecurityError(PermissionError):
    """Raised when an output path or identifier violates security boundaries."""
    pass


def validate_condition_id(condition_id: str) -> str:
    """
    Validates that condition_id matches an exact authorized experimental condition.
    Rejects traversal, extensions, or unrecognized identifiers.
    """
    if not isinstance(condition_id, str):
        raise OutputSecurityError(f"condition_id must be a string, got {type(condition_id)}")

    cid = condition_id.strip()
    if "/" in cid or "\\" in cid or ".." in cid or "\x00" in cid:
        raise OutputSecurityError(f"Path traversal characters detected in condition_id: '{condition_id}'")

    if cid not in AUTHORIZED_CONDITIONS:
        raise OutputSecurityError(
            f"Unauthorized condition_id '{condition_id}'. Must be one of: {sorted(AUTHORIZED_CONDITIONS)}"
        )
    return cid


def validate_recording_id(
    recording_id: str,
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
) -> str:
    """
    Validates that recording_id matches documented project naming conventions.
    Never sanitizes invalid strings; strictly rejects them.
    """
    if not isinstance(recording_id, str):
        raise OutputSecurityError(f"recording_id must be a string, got {type(recording_id)}")

    rid = recording_id.strip()

    # Reject path separators, traversal, null bytes, control characters
    if any(c in rid for c in ["/", "\\", "..", "\x00", "\n", "\r", "\t", " "]):
        raise OutputSecurityError(f"Illegal characters or path traversal in recording_id: '{recording_id}'")

    # Reject unexpected suffixes / extensions
    if "." in rid:
        raise OutputSecurityError(f"recording_id cannot contain file extensions or dots: '{recording_id}'")

    if execution_mode == ExecutionMode.PRODUCTION:
        if not BENCHMARK_RECORDING_REGEX.match(rid):
            raise OutputSecurityError(
                f"Production recording_id '{recording_id}' does not match documented benchmark pattern '^R0[1-8]_[a-z0-9_]+$'."
            )
    elif execution_mode == ExecutionMode.SYNTHETIC:
        if not (BENCHMARK_RECORDING_REGEX.match(rid) or SYNTHETIC_RECORDING_REGEX.match(rid)):
            raise OutputSecurityError(
                f"Synthetic recording_id '{recording_id}' does not match benchmark or synthetic pattern."
            )
    else:
        raise OutputSecurityError(f"Unknown execution mode: {execution_mode}")

    return rid


def validate_final_rttm_path(
    output_dir: Path,
    recording_id: str,
    condition_id: str,
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
    custom_allowed_roots: Optional[List[Path]] = None,
) -> Path:
    """
    Validates recording_id and condition_id, constructs the complete final RTTM path,
    and enforces strict containment within designated output roots.

    Returns:
        Validated absolute Path ready for file writing.
    """
    # 1. Validate identifiers
    rec_id = validate_recording_id(recording_id, execution_mode=execution_mode)
    cond_id = validate_condition_id(condition_id)

    # 2. Establish allowed roots
    if execution_mode == ExecutionMode.PRODUCTION:
        if custom_allowed_roots is not None:
            raise OutputSecurityError("Custom allowed_output_roots prohibited in PRODUCTION mode.")
        allowed_roots = [get_stage5b_4_dir().resolve()]
    else:
        # Synthetic mode
        allowed_roots = [Path(r).resolve() for r in (custom_allowed_roots or [get_stage5b_4_dir()])]

    # 3. Construct candidate final path
    filename = f"{rec_id}_{cond_id}.rttm"
    target_dir = Path(output_dir).resolve()
    final_rttm_path = (target_dir / filename).resolve()

    # 4. Prohibit targeting any frozen or historical stages
    for prohibited in PROHIBITED_STAGE_NAMES:
        if prohibited in final_rttm_path.parts:
            raise OutputSecurityError(
                f"Output path escapes into prohibited stage '{prohibited}': {final_rttm_path}"
            )

    # 5. Strict containment check within allowed roots
    is_contained = any(
        final_rttm_path == root or final_rttm_path.is_relative_to(root)
        for root in allowed_roots
    )
    if not is_contained:
        allowed_str = ", ".join(str(r) for r in allowed_roots)
        raise OutputSecurityError(
            f"Output path containment violation: '{final_rttm_path}' is outside allowed roots: [{allowed_str}]"
        )

    # 6. Check for directory collisions (if path already exists as a directory)
    if final_rttm_path.exists() and final_rttm_path.is_dir():
        raise OutputSecurityError(
            f"Cannot write RTTM: destination path already exists as a directory: {final_rttm_path}"
        )

    # 7. Symlink inspection on existing path components
    check_curr = Path(output_dir)
    while check_curr != check_curr.parent:
        if check_curr.is_symlink():
            link_target = check_curr.resolve()
            if not any(link_target == r or link_target.is_relative_to(r) for r in allowed_roots):
                raise OutputSecurityError(
                    f"Symlink escape detected: {check_curr} points outside allowed workspace to {link_target}"
                )
        check_curr = check_curr.parent

    # 8. Create parent directory safely
    final_rttm_path.parent.mkdir(parents=True, exist_ok=True)

    return final_rttm_path
