#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3.1 Remediation: Canonical Output Path & RTTM Containment Guard
Enforces strict containment of production RTTM writes exclusively to the canonical
Stage 5B.4 outputs directory: research_stages/stage5b/stage5b_4_fresh_diarization/outputs/
Layout: outputs/{condition_id}/{recording_id}.rttm

Remediates Finding 5:
- Prohibits custom production output roots.
- Defeats RTTM destination symlink replacement races by opening destination files with os.O_NOFOLLOW.
- Validates descriptors via fstat to ensure target is a regular file.
- Rejects parent and child symlinks, directory traversal, null bytes, and collisions.
"""

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import stat
from typing import Any, Dict, List, Optional

from path_utils import get_stage5b_4_outputs_dir, get_stage5b_4_dir, get_repo_root
from checkpoint_preflight import ExecutionMode

AUTHORIZED_CONDITIONS = {"E00", "E01", "E10", "E11", "E20", "E21"}
BENCHMARK_RECORDING_REGEX = re.compile(r"^R0[1-8]_[a-z0-9_]+$")
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


@dataclass(frozen=True)
class FinalRTTMPath:
    """
    Immutable representation of a verified final RTTM path.
    Prevents path reconstruction or modification between validation and writing.
    """
    recording_id: str
    condition_id: str
    canonical_output_root: Path
    final_path: Path
    execution_mode: ExecutionMode

    def __post_init__(self):
        # Validate that final_path matches canonical layout: {root}/{condition_id}/{recording_id}.rttm
        expected_path = (self.canonical_output_root / self.condition_id / f"{self.recording_id}.rttm").resolve()
        if self.final_path.resolve() != expected_path:
            raise OutputSecurityError(
                f"Path mismatch: final_path '{self.final_path}' does not match expected '{expected_path}'."
            )

    @property
    def path(self) -> Path:
        return self.final_path

    def __str__(self) -> str:
        return str(self.final_path)


def validate_condition_id(condition_id: str) -> str:
    """
    Validates that condition_id matches an exact authorized experimental condition.
    Rejects traversal, extensions, or unrecognized identifiers.
    """
    if not isinstance(condition_id, str):
        raise OutputSecurityError(f"condition_id must be a string, got {type(condition_id)}")

    cid = condition_id.strip()
    if "/" in cid or "\\" in cid or ".." in cid or "\x00" in cid or "\n" in cid:
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
    recording_id: str,
    condition_id: str,
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
    custom_output_root: Optional[Path] = None,
) -> FinalRTTMPath:
    """
    Validates recording_id and condition_id, computes the canonical final RTTM path:
        outputs/{condition_id}/{recording_id}.rttm
    and enforces strict containment within designated output roots.

    In PRODUCTION mode:
    - custom_output_root is strictly PROHIBITED.
    - Root is derived deterministically from repo: research_stages/stage5b/stage5b_4_fresh_diarization/outputs/
    - Layout is strictly: outputs/{condition_id}/{recording_id}.rttm

    In SYNTHETIC mode:
    - custom_output_root permitted (e.g. tmp_path) for isolated non-inference tests.
    """
    # 1. Validate identifiers
    rec_id = validate_recording_id(recording_id, execution_mode=execution_mode)
    cond_id = validate_condition_id(condition_id)

    # 2. Establish authorized output root
    if execution_mode == ExecutionMode.PRODUCTION:
        if custom_output_root is not None:
            raise OutputSecurityError(
                "Security violation: Custom output root is strictly prohibited in PRODUCTION mode. "
                "Production RTTM output must use canonical outputs/ root."
            )
        raw_output_root = get_stage5b_4_outputs_dir()
    else:
        raw_output_root = Path(custom_output_root) if custom_output_root is not None else get_stage5b_4_outputs_dir()

    # 3. Check for symlink on output_root itself
    if raw_output_root.is_symlink() or os.path.islink(str(raw_output_root)):
        target = raw_output_root.resolve()
        if execution_mode == ExecutionMode.PRODUCTION:
            if not target.is_relative_to(get_stage5b_4_dir().resolve()):
                raise OutputSecurityError(
                    f"Symlink escape: output root {raw_output_root} points outside Stage 5B.4 to {target}."
                )
        else:
            if not target.is_relative_to(raw_output_root.parent.resolve()):
                raise OutputSecurityError(
                    f"Symlink escape: synthetic output root {raw_output_root} points to {target}."
                )

    output_root = raw_output_root.resolve()

    # 4. Canonical layout: {output_root}/{condition_id}/{recording_id}.rttm
    condition_dir = output_root / cond_id
    if condition_dir.is_symlink() or os.path.islink(str(condition_dir)):
        raise OutputSecurityError(
            f"Symlink escape: condition directory {condition_dir} is an unauthorized symlink."
        )

    filename = f"{rec_id}.rttm"
    candidate_path = condition_dir / filename

    # Check symlinked final RTTM target before resolve
    if candidate_path.is_symlink() or os.path.islink(str(candidate_path)):
        raise OutputSecurityError(
            f"Symlink escape: destination RTTM {candidate_path} is an unauthorized symlink."
        )

    canonical_final = candidate_path.resolve()

    # 5. Prohibit targeting any frozen or historical stages
    for prohibited in PROHIBITED_STAGE_NAMES:
        if prohibited in canonical_final.parts:
            raise OutputSecurityError(
                f"Output path escapes into prohibited stage '{prohibited}': {canonical_final}"
            )

    # 6. Verify containment within output_root
    if not (canonical_final == output_root or canonical_final.is_relative_to(output_root)):
        raise OutputSecurityError(
            f"Output path containment violation: '{canonical_final}' is outside authorized root '{output_root}'."
        )

    # 7. Check if destination already exists as a directory
    if canonical_final.exists() and canonical_final.is_dir():
        raise OutputSecurityError(
            f"Cannot write RTTM: destination path already exists as a directory: {canonical_final}"
        )

    # 8. Safely ensure parent condition directory exists
    condition_dir.mkdir(parents=True, exist_ok=True)

    return FinalRTTMPath(
        recording_id=rec_id,
        condition_id=cond_id,
        canonical_output_root=output_root,
        final_path=candidate_path,
        execution_mode=execution_mode,
    )


def revalidate_and_write_rttm(
    rttm_path_obj: FinalRTTMPath,
    segments: List[Dict[str, Any]],
) -> Path:
    """
    Revalidates FinalRTTMPath immediately before writing, ensures writer receives
    the exact validated path, and safely writes single/multi-speaker segments.

    Remediates Finding 5:
    Uses os.O_NOFOLLOW and descriptor verification to guarantee destination
    symlink replacement attacks fail safely without following symlinks.
    """
    if not isinstance(rttm_path_obj, FinalRTTMPath):
        raise OutputSecurityError(
            f"Writer expected FinalRTTMPath object, got {type(rttm_path_obj)}."
        )

    final_path = rttm_path_obj.final_path
    output_root = rttm_path_obj.canonical_output_root

    # Ensure parent still exists, is not a symlink, and resolves within output root
    parent_dir = final_path.parent
    if parent_dir.is_symlink() or os.path.islink(str(parent_dir)):
        raise OutputSecurityError(
            f"TOCTOU violation: Parent directory was replaced with a symlink: {parent_dir}"
        )
    if not parent_dir.resolve().is_relative_to(output_root):
        raise OutputSecurityError(
            f"TOCTOU violation: Parent directory escapes authorized output root {output_root}."
        )

    # Pre-write symlink check on destination
    if final_path.is_symlink() or os.path.islink(str(final_path)):
        raise OutputSecurityError(
            f"TOCTOU violation: Destination path is an unauthorized symlink: {final_path}"
        )

    if final_path.exists() and final_path.is_dir():
        raise OutputSecurityError(
            f"TOCTOU violation: Destination was replaced with directory: {final_path}"
        )

    # Format RTTM lines
    lines = []
    sorted_segs = sorted(segments, key=lambda s: (s.get("start", 0.0), s.get("end", 0.0)))
    for s in sorted_segs:
        st = float(s["start"])
        dur = float(s["end"]) - st
        spk = s.get("speaker", "SPEAKER_00")
        line = f"SPEAKER {rttm_path_obj.recording_id} 1 {st:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
        lines.append(line)

    content = "".join(lines)

    # Secure open using O_NOFOLLOW
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        fd = os.open(str(final_path), flags, 0o644)
    except OSError as e:
        raise OutputSecurityError(
            f"Security violation: Failed to open destination path securely (symlink or permission): {e}"
        ) from e

    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(fd)
            raise OutputSecurityError(
                f"Security violation: Destination descriptor is not a regular file: mode {file_stat.st_mode}"
            )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        if isinstance(e, OutputSecurityError):
            raise
        raise OutputSecurityError(f"Failed to write RTTM: {e}") from e

    # Post-write revalidation
    if final_path.is_symlink() or os.path.islink(str(final_path)):
        raise OutputSecurityError(
            f"TOCTOU violation: Destination path became a symlink after write: {final_path}"
        )
    if not final_path.resolve().is_relative_to(output_root):
        raise OutputSecurityError(
            f"TOCTOU violation: Written file {final_path.resolve()} escapes authorized root {output_root}"
        )

    return final_path
