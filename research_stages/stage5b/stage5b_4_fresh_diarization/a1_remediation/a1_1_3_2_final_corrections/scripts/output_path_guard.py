#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3.2 Remediation: Canonical Output Path & Descriptor-Based RTTM Containment Guard
Enforces strict containment of production RTTM writes exclusively to the canonical
Stage 5B.4 outputs directory: research_stages/stage5b/stage5b_4_fresh_diarization/outputs/
Layout: outputs/{condition_id}/{recording_id}.rttm

Remediates Finding 2:
1. Descriptor-based directory traversal:
   - Opens output root securely as a directory file descriptor with O_DIRECTORY | O_NOFOLLOW.
   - Opens condition subdirectory relative to root descriptor using dir_fd with O_DIRECTORY | O_NOFOLLOW.
   - Completely avoids parent-directory substitution races.
2. Hard-link truncation attack elimination:
   - Never opens existing destination with O_TRUNC before descriptor inspection.
   - Attempts exclusive creation (O_CREAT | O_EXCL | O_NOFOLLOW) first.
   - If destination exists, opens without O_TRUNC and checks fstat().
   - Rejects non-regular files and files with st_nlink > 1 immediately, preserving external files untouched.
   - Only calls ftruncate() after verifying st_nlink == 1 and S_ISREG.
3. Rejects parent and child symlinks, directory traversal, null bytes, and collisions.
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
    if "/" in rid or "\\" in rid or ".." in rid or "\x00" in rid or "\n" in rid:
        raise OutputSecurityError(f"Path traversal characters detected in recording_id: '{recording_id}'")

    if execution_mode == ExecutionMode.PRODUCTION:
        if not BENCHMARK_RECORDING_REGEX.match(rid):
            raise OutputSecurityError(
                f"Production recording_id '{recording_id}' does not match canonical benchmark pattern 'R0[1-8]_[a-z0-9_]+'."
            )
    else:
        if not (BENCHMARK_RECORDING_REGEX.match(rid) or SYNTHETIC_RECORDING_REGEX.match(rid)):
            raise OutputSecurityError(
                f"Synthetic recording_id '{recording_id}' does not match benchmark or synthetic naming convention."
            )
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
    """
    rec_id = validate_recording_id(recording_id, execution_mode=execution_mode)
    cond_id = validate_condition_id(condition_id)

    if execution_mode == ExecutionMode.PRODUCTION:
        if custom_output_root is not None:
            raise OutputSecurityError(
                "Security violation: Custom output root is strictly prohibited in PRODUCTION mode. "
                "Production RTTM output must use canonical outputs/ root."
            )
        raw_output_root = get_stage5b_4_outputs_dir()
    else:
        raw_output_root = Path(custom_output_root) if custom_output_root is not None else get_stage5b_4_outputs_dir()

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
    condition_dir = output_root / cond_id
    if condition_dir.is_symlink() or os.path.islink(str(condition_dir)):
        raise OutputSecurityError(
            f"Symlink escape: condition directory {condition_dir} is an unauthorized symlink."
        )

    filename = f"{rec_id}.rttm"
    candidate_path = condition_dir / filename

    if candidate_path.is_symlink() or os.path.islink(str(candidate_path)):
        raise OutputSecurityError(
            f"Symlink escape: destination RTTM {candidate_path} is an unauthorized symlink."
        )

    canonical_final = candidate_path.resolve()

    for prohibited in PROHIBITED_STAGE_NAMES:
        if prohibited in canonical_final.parts:
            raise OutputSecurityError(
                f"Output path escapes into prohibited stage '{prohibited}': {canonical_final}"
            )

    if not (canonical_final == output_root or canonical_final.is_relative_to(output_root)):
        raise OutputSecurityError(
            f"Output path containment violation: '{canonical_final}' is outside authorized root '{output_root}'."
        )

    if canonical_final.exists() and canonical_final.is_dir():
        raise OutputSecurityError(
            f"Cannot write RTTM: destination path already exists as a directory: {canonical_final}"
        )

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
    Descriptor-based secure RTTM writer.
    Remediates Finding 2:
    1. Uses root and condition directory descriptors (dir_fd) with O_DIRECTORY | O_NOFOLLOW
       to completely prevent parent-directory replacement races.
    2. Protects against hard-link truncation attacks:
       - Attempts exclusive creation (O_CREAT | O_EXCL) first.
       - If file exists, opens WITHOUT O_TRUNC to check fstat().
       - If st_nlink > 1, rejects immediately without modifying the file.
       - Truncates only after confirming regular file and st_nlink == 1.
    """
    if not isinstance(rttm_path_obj, FinalRTTMPath):
        raise OutputSecurityError(
            f"Writer expected FinalRTTMPath object, got {type(rttm_path_obj)}."
        )

    final_path = rttm_path_obj.final_path
    output_root = rttm_path_obj.canonical_output_root
    cond_id = rttm_path_obj.condition_id
    rec_id = rttm_path_obj.recording_id
    filename = f"{rec_id}.rttm"

    # Format RTTM lines
    lines = []
    sorted_segs = sorted(segments, key=lambda s: (s.get("start", 0.0), s.get("end", 0.0)))
    for s in sorted_segs:
        st_sec = float(s["start"])
        dur = float(s["end"]) - st_sec
        spk = s.get("speaker", "SPEAKER_00")
        line = f"SPEAKER {rec_id} 1 {st_sec:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
        lines.append(line)
    content = "".join(lines)

    # 1. Open canonical output root as directory descriptor
    root_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        root_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        root_flags |= os.O_NOFOLLOW

    try:
        root_fd = os.open(str(output_root), root_flags)
    except OSError as e:
        raise OutputSecurityError(f"Security violation: Failed to open output root securely: {e}") from e

    try:
        st_root = os.fstat(root_fd)
        if not stat.S_ISDIR(st_root.st_mode):
            raise OutputSecurityError(f"Output root is not a directory: {output_root}")

        # 2. Traverse to condition subdirectory via root_fd
        try:
            os.mkdir(cond_id, mode=0o755, dir_fd=root_fd)
        except FileExistsError:
            pass
        except OSError as e:
            raise OutputSecurityError(f"Failed to create condition directory in output root: {e}") from e

        cond_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            cond_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            cond_flags |= os.O_NOFOLLOW

        try:
            cond_fd = os.open(cond_id, cond_flags, dir_fd=root_fd)
        except OSError as e:
            raise OutputSecurityError(
                f"Security violation: Failed to open condition directory securely (symlink or permission): {e}"
            ) from e

        try:
            st_cond = os.fstat(cond_fd)
            if not stat.S_ISDIR(st_cond.st_mode):
                raise OutputSecurityError(f"Condition path is not a directory: {cond_id}")

            # 3. Open destination file securely via cond_fd
            # Attempt exclusive creation first
            create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                create_flags |= os.O_NOFOLLOW

            try:
                dest_fd = os.open(filename, create_flags, mode=0o644, dir_fd=cond_fd)
            except FileExistsError:
                # File already exists: Open WITHOUT O_TRUNC to inspect descriptor before modifying
                inspect_flags = os.O_WRONLY
                if hasattr(os, "O_NOFOLLOW"):
                    inspect_flags |= os.O_NOFOLLOW

                try:
                    dest_fd = os.open(filename, inspect_flags, dir_fd=cond_fd)
                except OSError as e:
                    raise OutputSecurityError(
                        f"Security violation: Failed to open existing destination securely: {e}"
                    ) from e

                # Verify descriptor attributes BEFORE modifying
                st_dest = os.fstat(dest_fd)
                if not stat.S_ISREG(st_dest.st_mode):
                    os.close(dest_fd)
                    raise OutputSecurityError(
                        f"Security violation: Destination is not a regular file (mode {st_dest.st_mode})."
                    )

                # CRITICAL HARD-LINK CHECK:
                if st_dest.st_nlink > 1:
                    os.close(dest_fd)
                    raise OutputSecurityError(
                        f"Security violation: Destination '{final_path}' has multiple hard links "
                        f"({st_dest.st_nlink}); refusing write to prevent hard-link truncation attack."
                    )

                # Only after proving st_nlink == 1 and regular file: safely truncate
                os.ftruncate(dest_fd, 0)

            # 4. Write verified content through descriptor
            try:
                st_dest_verify = os.fstat(dest_fd)
                if not stat.S_ISREG(st_dest_verify.st_mode):
                    os.close(dest_fd)
                    raise OutputSecurityError("Destination descriptor is not a regular file.")
                if st_dest_verify.st_nlink > 1:
                    os.close(dest_fd)
                    raise OutputSecurityError("Destination descriptor hard-link count increased during write.")

                with os.fdopen(dest_fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                if isinstance(e, OutputSecurityError):
                    raise
                raise OutputSecurityError(f"Failed to write RTTM: {e}") from e

        finally:
            os.close(cond_fd)

    finally:
        os.close(root_fd)

    # 5. Post-write containment check
    if final_path.is_symlink() or os.path.islink(str(final_path)):
        raise OutputSecurityError(
            f"TOCTOU violation: Destination path became a symlink after write: {final_path}"
        )
    if not final_path.resolve().is_relative_to(output_root):
        raise OutputSecurityError(
            f"TOCTOU violation: Written file {final_path.resolve()} escapes authorized root {output_root}"
        )

    return final_path
