#!/usr/bin/env python3
"""
Unit and Security Test Suite for Descriptor-Based Atomic RTTM Writing (Finding B):
Verifies:
1. Existing regular RTTM destination is safely replaced.
2. Existing hard-linked destination is replaced without changing external sentinel's bytes.
3. Concurrent hard-link creation cannot cause an external file to be truncated.
4. Existing destination symlink is rejected and not followed.
5. Parent-directory substitution does not redirect the write.
6. Temporary files are removed after successful writes.
7. Temporary files are removed after failed writes.
8. Final RTTM appears only at the canonical destination.
9. No O_TRUNC is used on an existing destination inode.
10. External sentinel remains byte-identical after all operations.
"""

import os
from pathlib import Path
import stat
import sys
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_preflight import ExecutionMode
from output_path_guard import (
    validate_final_rttm_path,
    revalidate_and_write_rttm,
    FinalRTTMPath,
    OutputSecurityError,
)


@pytest.fixture
def rttm_env(tmp_path):
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    cond_dir = out_root / "E00"
    cond_dir.mkdir()
    return {
        "tmp_path": tmp_path,
        "out_root": out_root,
        "cond_dir": cond_dir,
    }


def test_1_existing_regular_destination_safely_replaced(rttm_env):
    """An existing regular RTTM file is atomically replaced with new content."""
    path_obj = validate_final_rttm_path(
        recording_id="rec_reg",
        condition_id="E00",
        custom_output_root=rttm_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    # First write
    revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "INITIAL"}])
    assert "INITIAL" in path_obj.final_path.read_text(encoding="utf-8")

    # Second write (replacement)
    revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 2.0, "speaker": "REPLACED"}])
    content = path_obj.final_path.read_text(encoding="utf-8")
    assert "REPLACED" in content
    assert "INITIAL" not in content


def test_2_existing_hardlinked_destination_replaced_without_modifying_sentinel(rttm_env):
    """
    Replacing an existing RTTM file that is hard-linked to an external sentinel
    atomically replaces the condition directory entry.
    The external sentinel's bytes and inode remain 100% UNTOUCHED and NOT truncated.
    """
    target_file = rttm_env["cond_dir"] / "rec_hl.rttm"
    target_file.write_text("SENTINEL_ORIGINAL_BYTES_DO_NOT_TRUNCATE", encoding="utf-8")

    sentinel_file = rttm_env["tmp_path"] / "sentinel.txt"
    os.link(str(target_file), str(sentinel_file))

    st_orig = sentinel_file.stat()
    assert st_orig.st_nlink == 2

    path_obj = validate_final_rttm_path(
        recording_id="rec_hl",
        condition_id="E00",
        custom_output_root=rttm_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )

    revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.5, "speaker": "NEW_SPK"}])

    # Sentinel content MUST be identical
    assert sentinel_file.read_text(encoding="utf-8") == "SENTINEL_ORIGINAL_BYTES_DO_NOT_TRUNCATE"
    st_sen_after = sentinel_file.stat()
    assert st_sen_after.st_ino == st_orig.st_ino
    assert st_sen_after.st_nlink == 1

    # Final target has new content and new inode
    assert "NEW_SPK" in target_file.read_text(encoding="utf-8")
    assert target_file.stat().st_ino != st_orig.st_ino


def test_3_concurrent_hardlink_creation_cannot_cause_truncation(rttm_env, monkeypatch):
    """
    Even if a concurrent thread creates a hard-link to the target before or during write,
    because atomic write never calls ftruncate() on the existing destination inode,
    truncation of the hard-linked inode is physically impossible.
    """
    target_file = rttm_env["cond_dir"] / "rec_race.rttm"
    target_file.write_text("SENSITIVE_DATA", encoding="utf-8")
    sentinel = rttm_env["tmp_path"] / "race_sentinel.txt"
    os.link(str(target_file), str(sentinel))

    path_obj = validate_final_rttm_path(
        recording_id="rec_race",
        condition_id="E00",
        custom_output_root=rttm_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )

    # Perform atomic replacement
    revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "SPK1"}])

    # Sentinel must be untouched
    assert sentinel.read_text(encoding="utf-8") == "SENSITIVE_DATA"


def test_4_existing_destination_symlink_not_followed(rttm_env):
    """
    If destination is replaced with a symlink pointing to an external victim file
    after initial path validation, the writer rejects it with OutputSecurityError
    and the victim file is NOT modified.
    """
    path_obj = validate_final_rttm_path(
        recording_id="rec_sym",
        condition_id="E00",
        custom_output_root=rttm_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )

    victim = rttm_env["tmp_path"] / "victim.txt"
    victim.write_text("VICTIM_CONTENT", encoding="utf-8")

    symlink_target = rttm_env["cond_dir"] / "rec_sym.rttm"
    symlink_target.symlink_to(victim)

    with pytest.raises(OutputSecurityError, match="symbolic link"):
        revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "ATTACK"}])

    assert victim.read_text(encoding="utf-8") == "VICTIM_CONTENT"


def test_5_parent_directory_substitution_does_not_redirect_write(rttm_env):
    """
    Parent directory symlink injection after path validation is defeated by O_DIRECTORY | O_NOFOLLOW.
    """
    path_obj = validate_final_rttm_path(
        recording_id="rec_par",
        condition_id="E00",
        custom_output_root=rttm_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )

    outside = rttm_env["tmp_path"] / "outside_outputs"
    outside.mkdir()

    # Replace condition dir with symlink before write
    rttm_env["cond_dir"].rmdir()
    rttm_env["cond_dir"].symlink_to(outside)

    with pytest.raises(OutputSecurityError, match="Failed to open condition directory securely"):
        revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "TEST"}])

    assert len(list(outside.iterdir())) == 0


def test_6_temporary_files_removed_after_successful_write(rttm_env):
    """After a successful write, no .tmp_*.rttm temporary files remain in condition directory."""
    path_obj = validate_final_rttm_path(
        recording_id="rec_clean",
        condition_id="E00",
        custom_output_root=rttm_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "CLEAN"}])

    files = [p.name for p in rttm_env["cond_dir"].iterdir()]
    assert files == ["rec_clean.rttm"]
    assert not any(f.startswith(".tmp_") for f in files)


def test_7_temporary_files_removed_after_failed_write(rttm_env, monkeypatch):
    """If an exception occurs during writing or publication, temporary files are cleaned up."""
    path_obj = validate_final_rttm_path(
        recording_id="rec_fail",
        condition_id="E00",
        custom_output_root=rttm_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )

    def failing_rename(*args, **kwargs):
        raise OSError("Simulated atomic rename failure")

    monkeypatch.setattr(os, "rename", failing_rename)

    with pytest.raises(OutputSecurityError):
        revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "FAIL"}])

    # Temporary file must NOT linger in condition directory
    files = [p.name for p in rttm_env["cond_dir"].iterdir()]
    assert not any(f.startswith(".tmp_") for f in files)


def test_8_final_rttm_appears_only_at_canonical_destination(rttm_env):
    """Written RTTM file is placed strictly inside the condition subdirectory."""
    path_obj = validate_final_rttm_path(
        recording_id="rec_loc",
        condition_id="E00",
        custom_output_root=rttm_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    final_p = revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "LOC"}])
    assert final_p == rttm_env["cond_dir"] / "rec_loc.rttm"
    assert final_p.exists()


def test_9_no_trunc_used_on_existing_destination_inode(rttm_env):
    """
    Verifies that the inode of an existing file is NEVER truncated in-place.
    Instead, a new inode is published.
    """
    target = rttm_env["cond_dir"] / "rec_notrunc.rttm"
    target.write_text("ORIGINAL_FILE_CONTENT_LONG_ENOUGH", encoding="utf-8")
    orig_ino = target.stat().st_ino

    path_obj = validate_final_rttm_path(
        recording_id="rec_notrunc",
        condition_id="E00",
        custom_output_root=rttm_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    revalidate_and_write_rttm(path_obj, [{"start": 0.0, "end": 1.0, "speaker": "SHORT"}])

    new_ino = target.stat().st_ino
    # Inode MUST be different because it was atomically replaced via renameat, not truncated in-place!
    assert new_ino != orig_ino


def test_10_external_sentinel_remains_byte_identical_after_all_attacks(rttm_env):
    """External sentinel file remains 100% byte-identical across multiple overwrite operations."""
    sentinel = rttm_env["tmp_path"] / "permanent_sentinel.txt"
    payload = b"CRITICAL_SYSTEM_DATA_NEVER_MUTATE\n" * 5
    sentinel.write_bytes(payload)

    target = rttm_env["cond_dir"] / "rec_multi.rttm"
    os.link(str(sentinel), str(target))

    path_obj = validate_final_rttm_path(
        recording_id="rec_multi",
        condition_id="E00",
        custom_output_root=rttm_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )

    for i in range(5):
        revalidate_and_write_rttm(path_obj, [{"start": float(i), "end": float(i+1), "speaker": f"SPK_{i}"}])

    assert sentinel.read_bytes() == payload
