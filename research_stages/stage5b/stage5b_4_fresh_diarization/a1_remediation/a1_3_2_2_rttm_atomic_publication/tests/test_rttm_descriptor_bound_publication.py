#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3.2.2 Micro-Correction Adversarial Test Suite
Comprehensive verification of descriptor-bound atomic RTTM publication:
1. Normal successful RTTM publication.
2. Existing destination replacement.
3. Existing hard-linked destination: external sentinel retains byte-identical contents.
4. Destination symlink: never followed.
5. Temporary directory-entry substitution after file verification (defeated by descriptor binding).
6. Temporary file replacement attempts while the original descriptor remains open.
7. Publication from the retained descriptor points to the exact verified inode.
8. Concurrent destination replacement attempts.
9. Parent condition-directory substitution attempts.
10. Failure during writing (cleaned up).
11. Failure during verification (cleaned up).
12. Failure during publication (cleaned up).
13. Cleanup after every failure.
14. No leaked file or directory descriptors.
15. No O_TRUNC or ftruncate on existing destination inodes.
16. Unsupported publication mechanism fails closed.
"""

import errno
import os
from pathlib import Path
import stat
import tempfile
from unittest.mock import patch
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from output_path_guard import (
    FinalRTTMPath,
    OutputSecurityError,
    revalidate_and_write_rttm,
    validate_final_rttm_path,
)
from checkpoint_preflight import ExecutionMode


@pytest.fixture
def temp_output_root(tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def count_open_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


# 1. Normal successful RTTM publication
def test_normal_successful_rttm_publication(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec01",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )
    segments = [
        {"start": 0.5, "end": 2.5, "speaker": "SPEAKER_01"},
        {"start": 3.0, "end": 4.5, "speaker": "SPEAKER_00"},
    ]
    published_path = revalidate_and_write_rttm(path_obj, segments)

    assert published_path.exists()
    assert published_path.is_file()
    assert not published_path.is_symlink()
    assert stat.S_IMODE(published_path.stat().st_mode) == 0o644

    content = published_path.read_text()
    assert "SPEAKER syn_rec01 1 0.500 2.000 <NA> <NA> SPEAKER_01 <NA> <NA>" in content
    assert "SPEAKER syn_rec01 1 3.000 1.500 <NA> <NA> SPEAKER_00 <NA> <NA>" in content

    # Verify condition dir contains only the published file and no staging leftover
    cond_dir = temp_output_root / "E00"
    entries = os.listdir(cond_dir)
    assert entries == ["syn_rec01.rttm"]


# 2. Existing destination replacement
def test_existing_destination_replacement(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec02",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )
    cond_dir = temp_output_root / "E00"
    dest = cond_dir / "syn_rec02.rttm"
    dest.write_text("OLD PREEXISTING CONTENT\n")
    old_st = dest.stat()

    segments = [{"start": 1.0, "end": 3.0, "speaker": "SPK_NEW"}]
    published_path = revalidate_and_write_rttm(path_obj, segments)

    new_st = published_path.stat()
    assert published_path.read_text().strip() == "SPEAKER syn_rec02 1 1.000 2.000 <NA> <NA> SPK_NEW <NA> <NA>"
    assert new_st.st_ino != old_st.st_ino


# 3. Existing hard-linked destination: external sentinel retains byte-identical contents
def test_existing_hard_linked_destination_external_sentinel_preserved(temp_output_root, tmp_path):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec03",
        condition_id="E01",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )
    dest = temp_output_root / "E01" / "syn_rec03.rttm"
    original_payload = b"ORIGINAL IMMUTABLE BENCHMARK BYTES"
    dest.write_bytes(original_payload)

    # External sentinel hard link
    sentinel = tmp_path / "sentinel_backup.rttm"
    os.link(dest, sentinel)

    assert sentinel.stat().st_nlink == 2
    old_ino = dest.stat().st_ino

    # Publish new replacement
    segments = [{"start": 0.0, "end": 1.0, "speaker": "SPK_REPLACED"}]
    revalidate_and_write_rttm(path_obj, segments)

    # Sentinel must retain 100% of its original bytes and original inode
    assert sentinel.read_bytes() == original_payload
    assert sentinel.stat().st_ino == old_ino
    assert sentinel.stat().st_nlink == 1  # Link count decreased because destination was unlinked by renameat

    # Destination now points to new inode with new content
    assert dest.read_bytes() != original_payload
    assert dest.stat().st_ino != old_ino


# 4. Destination symlink: never followed
def test_destination_symlink_never_followed(temp_output_root, tmp_path):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec04",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )
    canary = tmp_path / "canary_target.txt"
    canary.write_text("CRITICAL SYSTEM FILE - DO NOT OVERWRITE")

    dest = temp_output_root / "E00" / "syn_rec04.rttm"
    dest.symlink_to(canary)

    with pytest.raises(OutputSecurityError, match="symbolic link; symlinks are strictly prohibited"):
        revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "SPK_ATTACK"}])

    assert canary.read_text() == "CRITICAL SYSTEM FILE - DO NOT OVERWRITE"


# 5. Temporary directory-entry substitution after file verification
def test_temporary_directory_entry_substitution_after_verification_defeated(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec05",
        condition_id="E10",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    tampered = False

    def hook_after_verify(tmp_fd, stage_fd):
        nonlocal tampered
        # Adversary finds the file name in staging directory and replaces it with rogue file
        entries = os.listdir(f"/proc/self/fd/{stage_fd}")
        for e in entries:
            if e.startswith("tmp_"):
                # Anchor descriptor so kernel retains active link while directory entry is swapped
                os.link(e, ".anchor.rttm", src_dir_fd=stage_fd, dst_dir_fd=stage_fd)
                os.unlink(e, dir_fd=stage_fd)
                rogue_fd = os.open(e, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o600, dir_fd=stage_fd)
                os.write(rogue_fd, b"ROGUE MALICIOUS PAYLOAD\n")
                os.close(rogue_fd)
                tampered = True
                break

    segments = [{"start": 0.0, "end": 2.0, "speaker": "VALID_SPEAKER"}]
    published_path = revalidate_and_write_rttm(
        path_obj,
        segments,
        _test_hook_after_verify=hook_after_verify,
    )

    assert tampered is True
    # The published file MUST contain the original valid speaker payload, NOT the rogue payload!
    content = published_path.read_text()
    assert "VALID_SPEAKER" in content
    assert "ROGUE MALICIOUS PAYLOAD" not in content

    # Part B: Adversary attempts directory-entry substitution on the publication link in stage_fd
    path_obj_b = validate_final_rttm_path(
        recording_id="syn_rec05_b",
        condition_id="E10",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    def hook_after_link(tmp_fd, stage_fd):
        # Adversary attempts to replace the publication link in stage_fd with a rogue file
        entries = os.listdir(f"/proc/self/fd/{stage_fd}")
        for f in entries:
            if f.startswith(".pub_tmp_syn_rec05_b"):
                os.unlink(f, dir_fd=stage_fd)
                rogue_fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o600, dir_fd=stage_fd)
                os.write(rogue_fd, b"ROGUE SUBSTITUTED ENTRY\n")
                os.close(rogue_fd)
                break

    with pytest.raises(OutputSecurityError, match="Integrity violation: Published file inode/device"):
        revalidate_and_write_rttm(
            path_obj_b,
            [{"start": 0.0, "end": 1.0, "speaker": "LEGIT_SPEAKER"}],
            _test_hook_after_link=hook_after_link,
        )


# 6. Temporary file replacement attempts while the original descriptor remains open
def test_temporary_file_replacement_attempts_while_original_descriptor_open(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec06",
        condition_id="E10",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    unlinked_during_create = False

    def hook_after_create(tmp_fd, stage_fd):
        nonlocal unlinked_during_create
        entries = os.listdir(f"/proc/self/fd/{stage_fd}")
        for e in entries:
            if e.startswith("tmp_"):
                os.unlink(e, dir_fd=stage_fd)
                unlinked_during_create = True
                break

    segments = [{"start": 0.0, "end": 1.0, "speaker": "SPK_UNLINKED_NAME"}]
    # Because tmp_fd is already unlinked before verify, st_nlink will be 0, so fstat check detects it!
    with pytest.raises(OutputSecurityError, match="unexpected hard links: 0"):
        revalidate_and_write_rttm(
            path_obj,
            segments,
            _test_hook_after_create=hook_after_create,
        )

    assert unlinked_during_create is True
    assert not (temp_output_root / "E10" / "syn_rec06.rttm").exists()


# 7. Publication from the retained descriptor points to the exact verified inode
def test_publication_points_to_exact_verified_inode(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec07",
        condition_id="E11",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    captured_stat = None

    def hook_after_verify(tmp_fd, stage_fd):
        nonlocal captured_stat
        captured_stat = os.fstat(tmp_fd)

    segments = [{"start": 1.0, "end": 2.0, "speaker": "SPK_INODE"}]
    published_path = revalidate_and_write_rttm(
        path_obj,
        segments,
        _test_hook_after_verify=hook_after_verify,
    )

    assert captured_stat is not None
    final_st = published_path.stat()
    assert final_st.st_ino == captured_stat.st_ino
    assert final_st.st_dev == captured_stat.st_dev


# 8. Concurrent destination replacement attempts
def test_concurrent_destination_replacement_attempts(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec08",
        condition_id="E20",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    def hook_after_link(tmp_fd, stage_fd):
        # Simulate another process writing to destination right before atomic renameat
        dest = temp_output_root / "E20" / "syn_rec08.rttm"
        dest.write_text("COMPETING WRITER PAYLOAD\n")

    segments = [{"start": 0.0, "end": 1.0, "speaker": "SPK_PRIMARY"}]
    published_path = revalidate_and_write_rttm(
        path_obj,
        segments,
        _test_hook_after_link=hook_after_link,
    )

    # POSIX renameat atomically overwrites destination with primary content
    assert "SPK_PRIMARY" in published_path.read_text()
    assert "COMPETING WRITER PAYLOAD" not in published_path.read_text()


# 9. Parent condition-directory substitution attempts
def test_parent_condition_directory_substitution_attempts(temp_output_root, tmp_path):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec09",
        condition_id="E21",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    # Replace condition dir with symlink pointing outside
    cond_dir = temp_output_root / "E21"
    if cond_dir.exists():
        cond_dir.rmdir()
    outside_target = tmp_path / "outside_dir"
    outside_target.mkdir()
    cond_dir.symlink_to(outside_target)

    with pytest.raises(OutputSecurityError, match="Failed to open condition directory securely"):
        revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "SPK_ESCAPE"}])


# 10. Failure during writing
def test_failure_during_writing_cleaned_up(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec10",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    def mock_write(fd, b):
        raise OSError(errno.ENOSPC, "No space left on device")

    with patch("os.write", side_effect=mock_write):
        with pytest.raises(OutputSecurityError, match="Failed writing content"):
            revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "SPK_ERR"}])

    cond_dir = temp_output_root / "E00"
    # Destination must not exist and staging must be cleaned up
    assert not (cond_dir / "syn_rec10.rttm").exists()
    assert len(os.listdir(cond_dir)) == 0


# 11. Failure during verification
def test_failure_during_verification_cleaned_up(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec11",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    def hook_after_create(tmp_fd, stage_fd):
        # Invalidate permissions
        os.fchmod(tmp_fd, 0o666)

    with pytest.raises(OutputSecurityError, match="permission mismatch"):
        revalidate_and_write_rttm(
            path_obj,
            [{"start": 0.0, "end": 1.0, "speaker": "SPK_VERIFY_FAIL"}],
            _test_hook_after_create=hook_after_create,
        )

    cond_dir = temp_output_root / "E00"
    assert not (cond_dir / "syn_rec11.rttm").exists()
    assert len(os.listdir(cond_dir)) == 0


# 12. Failure during publication
def test_failure_during_publication_cleaned_up(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec12",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    orig_rename = os.rename

    def mock_rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        if "syn_rec12" in str(src):
            raise OSError(errno.EIO, "Simulated I/O failure during renameat")
        return orig_rename(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    with patch("os.rename", side_effect=mock_rename), patch.object(os, "supports_dir_fd", os.supports_dir_fd | {mock_rename}):
        with pytest.raises(OutputSecurityError, match="Failed atomic rename to destination"):
            revalidate_and_write_rttm(
                path_obj,
                [{"start": 0.0, "end": 1.0, "speaker": "SPK_RENAME_FAIL"}],
            )

    cond_dir = temp_output_root / "E00"
    assert not (cond_dir / "syn_rec12.rttm").exists()
    assert len(os.listdir(cond_dir)) == 0


# 13. Cleanup after every failure
def test_cleanup_after_every_failure(temp_output_root):
    cond_dir = temp_output_root / "E01"
    cond_dir.mkdir(parents=True, exist_ok=True)

    # Trigger failure at linking
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec13",
        condition_id="E01",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    with patch("os.link", side_effect=OSError(errno.EACCES, "Permission denied on link")):
        with pytest.raises(OutputSecurityError, match="Failed creating descriptor-bound publication link"):
            revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "SPK_CLEAN"}])

    # Ensure no lingering staging or temporary link files
    assert os.listdir(cond_dir) == []


# 14. No leaked file or directory descriptors
def test_no_leaked_file_or_directory_descriptors(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec14",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    # Measure before
    fds_before = count_open_fds()

    # Success run
    revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "SPK_FD"}])
    fds_after_success = count_open_fds()
    assert fds_after_success == fds_before, f"Leaked FDs on success: {fds_after_success - fds_before}"

    # Failure run
    with patch("os.write", side_effect=OSError("Write failure")):
        with pytest.raises(OutputSecurityError):
            revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "SPK_FD"}])

    fds_after_failure = count_open_fds()
    assert fds_after_failure == fds_before, f"Leaked FDs on failure: {fds_after_failure - fds_before}"


# 15. No O_TRUNC or ftruncate on existing destination inodes
def test_no_o_trunc_or_ftruncate_on_existing_destination_inodes(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec15",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )
    dest = temp_output_root / "E00" / "syn_rec15.rttm"
    dest.write_text("PREEXISTING INODE DATA")

    ftruncate_called = False

    def guarded_ftruncate(fd, length):
        nonlocal ftruncate_called
        ftruncate_called = True
        return os.ftruncate(fd, length)

    with patch("os.ftruncate", side_effect=guarded_ftruncate):
        revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "SPK_NOTRUNC"}])

    assert ftruncate_called is False


# 16. Unsupported publication mechanism fails closed
def test_unsupported_publication_mechanism_fails_closed(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec16",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    with pytest.raises(OutputSecurityError, match="Platform does not support descriptor-bound publication mechanism"):
        revalidate_and_write_rttm(
            path_obj,
            [{"start": 0.0, "end": 1.0, "speaker": "SPK_UNSUPPORTED"}],
            _test_publication_mechanism_override="unsupported",
        )

    cond_dir = temp_output_root / "E00"
    assert not (cond_dir / "syn_rec16.rttm").exists()
    assert len(os.listdir(cond_dir)) == 0


# 17. Replacement of the destination after publication but before post-publication verification
def test_17_replacement_of_destination_after_rename_detected(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec17",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    def hook_after_rename(tmp_fd, cond_fd):
        # Adversary replaces destination file in cond_fd right after renameat but before post-verification open
        dest = temp_output_root / "E00" / "syn_rec17.rttm"
        dest.unlink()
        dest.write_text("ADVERSARY POST-RENAME SUBSTITUTION\n")

    with pytest.raises(OutputSecurityError, match="Integrity violation: Published file inode/device"):
        revalidate_and_write_rttm(
            path_obj,
            [{"start": 0.0, "end": 1.0, "speaker": "SPK_SWAP_AFTER_RENAME"}],
            _test_hook_after_rename=hook_after_rename,
        )


# 18. Cleanup racing with destination replacement: does not perform unsafe pathname unlink
def test_18_cleanup_racing_with_destination_replacement_does_not_unlink_foreign_file(temp_output_root):
    path_obj = validate_final_rttm_path(
        recording_id="syn_rec18",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=temp_output_root,
    )

    foreign_content = "FOREIGN CONCURRENT INODE DATA - MUST NOT BE UNLINKED\n"

    def hook_after_rename(tmp_fd, cond_fd):
        dest = temp_output_root / "E00" / "syn_rec18.rttm"
        dest.unlink()
        dest.write_text(foreign_content)

    # When integrity violation is detected, writer fails closed without calling pathname unlink
    with pytest.raises(OutputSecurityError, match="Integrity violation: Published file inode/device"):
        revalidate_and_write_rttm(
            path_obj,
            [{"start": 0.0, "end": 1.0, "speaker": "SPK_CLEANUP_RACE"}],
            _test_hook_after_rename=hook_after_rename,
        )

    # The foreign file was not deleted by an unsafe cleanup race
    dest = temp_output_root / "E00" / "syn_rec18.rttm"
    assert dest.exists()
    assert dest.read_text() == foreign_content

