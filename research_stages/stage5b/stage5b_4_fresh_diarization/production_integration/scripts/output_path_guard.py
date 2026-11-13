#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3.2.2 Micro-Correction: Canonical Output Path & Descriptor-Bound Atomic RTTM Publication Guard
Enforces strict containment of production RTTM writes exclusively to the canonical
Stage 5B.4 outputs directory: research_stages/stage5b/stage5b_4_fresh_diarization/outputs/
Layout: outputs/{condition_id}/{recording_id}.rttm

Remediates Finding (Stage 5B.4-A.1.3.2.2):
1. Complete elimination of temporary-file pathname substitution races:
   - Retains open condition directory descriptor (cond_fd) from canonical output-root traversal.
   - Creates a private staging subdirectory inside cond_fd with restrictive mode 0700.
   - Opens staging directory with O_DIRECTORY | O_NOFOLLOW to obtain stage_fd.
   - Creates temporary RTTM file relative to stage_fd with O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY and mode 0600.
   - Retains tmp_fd throughout writing, verification, and publication.
   - Verifies the actual inode directly via os.fstat(tmp_fd): regular file, st_nlink == 1, exact byte size, mode 0600.
   - Never reopens or verifies the temporary file through a pathname.
2. Descriptor-bound atomic publication:
   - Uses platform-supported descriptor-bound publication via /proc/self/fd/{tmp_fd} hard-linking into cond_fd.
   - Atomically publishes to destination recording filename via descriptor-relative renameat.
   - Verifies the published file via descriptor-relative no-follow open and asserts (st_dev, st_ino) match the verified tmp_fd.
   - Sets final destination mode to 0644.
   - Removes the staging directory entry and directory only after verified publication.
   - Fails closed if descriptor-bound publication is unsupported.
3. Complete hard-link safety and truncation elimination:
   - Avoids opening or truncating an existing destination inode.
   - Any external file hard-linked to an existing destination is never modified or truncated;
     its directory entry is atomically replaced and its content remains 100% byte-identical.
   - Temporary files, hard links, and staging directories are guaranteed to be cleaned up on every failure path.
4. Strict symlink and directory traversal prevention:
   - Existing destination symlinks are rejected and never followed.
   - Final post-publication containment verifies regular file status and absence of symlinks.
"""

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Callable, Dict, List, Optional

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
    *,
    _test_hook_after_create: Optional[Callable[[int, int], None]] = None,
    _test_hook_after_verify: Optional[Callable[[int, int], None]] = None,
    _test_hook_after_link: Optional[Callable[[int, int], None]] = None,
    _test_hook_after_rename: Optional[Callable[[int, int], None]] = None,
    _test_publication_mechanism_override: Optional[str] = None,
) -> Path:
    """
    Descriptor-bound secure RTTM writer and atomic publisher.
    Stage 5B.4-A.1.3.2.2 Micro-Correction:
    1. Uses root and condition directory descriptors (dir_fd) with O_DIRECTORY | O_NOFOLLOW
       to completely prevent parent-directory replacement races.
    2. Completely eliminates temporary-file pathname substitution races:
       - Creates a private staging subdirectory inside cond_fd with mode 0700.
       - Opens and retains stage_fd with O_DIRECTORY | O_NOFOLLOW.
       - Creates temporary RTTM file relative to stage_fd with O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY
         and mode 0600.
       - Retains tmp_fd throughout writing, verification, and publication.
       - Never reopens or verifies the temporary file through a pathname.
    3. Direct inode verification via os.fstat(tmp_fd):
       - Verifies regular file, st_nlink == 1, exact byte size, and mode 0600.
    4. Descriptor-bound atomic publication:
       - Links open descriptor into cond_fd via /proc/self/fd/{tmp_fd}.
       - Fails closed if descriptor-bound publication is unsupported.
       - Atomically renames temporary link to destination filename in cond_fd.
       - Verifies published destination via descriptor-relative open (O_RDONLY | O_NOFOLLOW)
         and asserts (st_dev, st_ino) match the verified descriptor.
       - Sets final file permissions to 0644.
       - Removes the staging directory entry and staging directory only after verified publication.
    5. Complete hard-link safety and truncation elimination:
       - Destination file is never opened with O_TRUNC or modified in place.
       - Any external hard link to an existing destination retains 100% byte-identical content.
    6. Guaranteed cleanup on every success and failure path:
       - All descriptors (root_fd, cond_fd, stage_fd, tmp_fd, pub_fd) are closed.
       - Staging directory, temporary files, and ephemeral hard links are cleaned up.
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
    content_bytes = content.encode("utf-8")
    expected_bytes = len(content_bytes)

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

            # 3. Check existing destination status via cond_fd without following symlinks
            try:
                st_existing = os.stat(filename, dir_fd=cond_fd, follow_symlinks=False)
                if stat.S_ISLNK(st_existing.st_mode):
                    raise OutputSecurityError(
                        f"Security violation: Destination '{filename}' is a symbolic link; symlinks are strictly prohibited."
                    )
                if stat.S_ISDIR(st_existing.st_mode):
                    raise OutputSecurityError(
                        f"Security violation: Destination '{filename}' is a directory."
                    )
            except FileNotFoundError:
                pass

            # 4. Create private temporary staging directory inside cond_fd (mode 0700)
            stage_name = f".stage_{rec_id}_{os.getpid()}_{secrets.token_hex(8)}"
            try:
                os.mkdir(stage_name, mode=0o700, dir_fd=cond_fd)
            except OSError as e:
                raise OutputSecurityError(f"Security violation: Failed to create private staging directory: {e}") from e

            stage_flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                stage_flags |= os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                stage_flags |= os.O_NOFOLLOW

            try:
                stage_fd = os.open(stage_name, stage_flags, dir_fd=cond_fd)
            except OSError as e:
                try:
                    os.rmdir(stage_name, dir_fd=cond_fd)
                except OSError:
                    pass
                raise OutputSecurityError(f"Security violation: Failed to open staging directory securely: {e}") from e

            tmp_fd = -1
            tmp_filename = f"tmp_{rec_id}_{secrets.token_hex(8)}.rttm"
            pub_tmp_name: Optional[str] = None
            published = False

            try:
                # 5. Create temporary RTTM file relative to stage_fd (mode 0600)
                create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    create_flags |= os.O_NOFOLLOW

                try:
                    tmp_fd = os.open(tmp_filename, create_flags, mode=0o600, dir_fd=stage_fd)
                except OSError as e:
                    raise OutputSecurityError(
                        f"Security violation: Failed to create temporary RTTM file securely: {e}"
                    ) from e

                # 6. Write payload directly to retained tmp_fd
                try:
                    bytes_written = 0
                    while bytes_written < expected_bytes:
                        n = os.write(tmp_fd, content_bytes[bytes_written:])
                        if n <= 0:
                            raise OSError("Zero bytes written during RTTM write.")
                        bytes_written += n
                    os.fsync(tmp_fd)
                except Exception as e:
                    raise OutputSecurityError(f"Failed writing content to temporary RTTM descriptor: {e}") from e

                if _test_hook_after_create is not None:
                    _test_hook_after_create(tmp_fd, stage_fd)

                # 7. Direct inode verification via os.fstat(tmp_fd)
                try:
                    st_tmp = os.fstat(tmp_fd)
                except OSError as e:
                    raise OutputSecurityError(f"Failed to stat temporary RTTM descriptor: {e}") from e

                if not stat.S_ISREG(st_tmp.st_mode):
                    raise OutputSecurityError("Temporary RTTM target is not a regular file.")
                if st_tmp.st_nlink != 1:
                    raise OutputSecurityError(
                        f"Temporary RTTM file has unexpected hard links: {st_tmp.st_nlink} (expected 1)."
                    )
                if st_tmp.st_size != expected_bytes:
                    raise OutputSecurityError(
                        f"Temporary RTTM size mismatch: expected {expected_bytes} bytes, found {st_tmp.st_size}."
                    )
                if stat.S_IMODE(st_tmp.st_mode) != 0o600:
                    raise OutputSecurityError(
                        f"Temporary RTTM permission mismatch: expected 0600, found {oct(stat.S_IMODE(st_tmp.st_mode))}."
                    )

                if _test_hook_after_verify is not None:
                    _test_hook_after_verify(tmp_fd, stage_fd)

                # 8. Check platform descriptor-bound publication support
                if _test_publication_mechanism_override == "unsupported":
                    raise OutputSecurityError(
                        "Platform does not support descriptor-bound publication mechanism (simulated unsupported mechanism)."
                    )

                proc_fd_path = f"/proc/self/fd/{tmp_fd}"
                if not os.path.exists(proc_fd_path):
                    raise OutputSecurityError(
                        "Platform does not support descriptor-bound publication mechanism (missing /proc/self/fd)."
                    )

                if os.rename not in os.supports_dir_fd:
                    import posix
                    if not hasattr(posix, "rename") or posix.rename not in os.supports_dir_fd:
                        raise OutputSecurityError(
                            "Platform does not support atomic descriptor-relative rename (renameat)."
                        )

                # 9. Descriptor-bound publication link inside private staging directory (mode 0700)
                # Creating pub_tmp_name inside stage_fd guarantees that no other process with write
                # access to cond_dir can enumerate, modify, unlink, or swap the publication link.
                pub_tmp_name = f".pub_tmp_{rec_id}_{os.getpid()}_{secrets.token_hex(8)}.rttm"
                try:
                    os.link(proc_fd_path, pub_tmp_name, dst_dir_fd=stage_fd)
                except OSError as e:
                    raise OutputSecurityError(
                        f"Failed creating descriptor-bound publication link: {e}"
                    ) from e

                if _test_hook_after_link is not None:
                    _test_hook_after_link(tmp_fd, stage_fd)

                # 10. Atomic publication from private stage_fd directly into canonical cond_fd
                # Single atomic syscall renameat moves the verified inode from private staging
                # into destination filename. Zero ephemeral entries are ever created in cond_fd.
                try:
                    os.rename(pub_tmp_name, filename, src_dir_fd=stage_fd, dst_dir_fd=cond_fd)
                except OSError as e:
                    raise OutputSecurityError(f"Failed atomic rename to destination: {e}") from e

                if _test_hook_after_rename is not None:
                    _test_hook_after_rename(tmp_fd, cond_fd)

                # 11. Post-publication descriptor verification
                pub_flags = os.O_RDONLY
                if hasattr(os, "O_NOFOLLOW"):
                    pub_flags |= os.O_NOFOLLOW

                try:
                    pub_fd = os.open(filename, pub_flags, dir_fd=cond_fd)
                except OSError as e:
                    raise OutputSecurityError(
                        f"Security violation: Failed to reopen published file securely: {e}"
                    ) from e

                try:
                    st_pub = os.fstat(pub_fd)
                    if st_pub.st_dev != st_tmp.st_dev or st_pub.st_ino != st_tmp.st_ino:
                        # Fail closed without performing an unsafe pathname unlink on destination,
                        # which could race with third-party or concurrent entries.
                        raise OutputSecurityError(
                            f"Integrity violation: Published file inode/device (dev={st_pub.st_dev}, ino={st_pub.st_ino}) "
                            f"does not match verified temporary inode (dev={st_tmp.st_dev}, ino={st_tmp.st_ino})."
                        )
                    if not stat.S_ISREG(st_pub.st_mode):
                        raise OutputSecurityError("Published destination is not a regular file.")
                    if st_pub.st_size != expected_bytes:
                        raise OutputSecurityError(
                            f"Published file size mismatch: expected {expected_bytes}, found {st_pub.st_size}."
                        )
                    # Set final output permissions explicitly (0644)
                    try:
                        os.fchmod(pub_fd, 0o644)
                    except OSError as e:
                        raise OutputSecurityError(f"Failed setting final permissions on published file: {e}") from e
                finally:
                    os.close(pub_fd)

                # 12. Remove staging directory entry only after successful publication and verification
                try:
                    os.unlink(tmp_filename, dir_fd=stage_fd)
                except OSError:
                    pass

                published = True

            finally:
                # Cleanup temporary file descriptor
                if tmp_fd != -1:
                    try:
                        os.close(tmp_fd)
                    except OSError:
                        pass

                # If publication failed, clean up ephemeral publication link in stage_fd if it exists
                if not published and pub_tmp_name is not None:
                    try:
                        os.unlink(pub_tmp_name, dir_fd=stage_fd)
                    except OSError:
                        pass

                # Clean up staging directory entry if still present
                try:
                    os.unlink(tmp_filename, dir_fd=stage_fd)
                except OSError:
                    pass

                # Close staging directory descriptor
                try:
                    os.close(stage_fd)
                except OSError:
                    pass

                # Remove staging directory
                try:
                    os.rmdir(stage_name, dir_fd=cond_fd)
                except OSError:
                    pass

        finally:
            os.close(cond_fd)

    finally:
        os.close(root_fd)

    # 13. Post-write containment check
    if final_path.is_symlink() or os.path.islink(str(final_path)):
        raise OutputSecurityError(
            f"TOCTOU violation: Destination path became a symlink after write: {final_path}"
        )
    if not final_path.resolve().is_relative_to(output_root):
        raise OutputSecurityError(
            f"TOCTOU violation: Written file {final_path.resolve()} escapes authorized root {output_root}"
        )

    return final_path
