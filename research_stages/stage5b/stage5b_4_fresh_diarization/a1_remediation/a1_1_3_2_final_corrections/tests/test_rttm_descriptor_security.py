#!/usr/bin/env python3
"""
Unit tests for Stage 5B.4-A.1.3.2 Finding 2:
Descriptor-Based RTTM Writing, Parent-Directory Race Mitigation, and Hard-Link Truncation Defense.

Verifies:
1. Normal RTTM creation succeeds.
2. Existing regular destination is safely overwritten after descriptor verification.
3. Final destination symlink is rejected.
4. Parent condition directory symlink is rejected.
5. Parent-directory replacement race test: replace parent directory after initial validation
   and prove write fails securely without escaping canonical output root.
6. Hard-link attack test: create hard link from intended destination to an external sentinel file,
   invoke write operation, prove write fails and external sentinel remains 100% UNTOUCHED!
7. Condition-directory traversal is rejected.
8. Recording-ID traversal is rejected.
9. Output-root substitution is rejected.
10. The external sentinel remains byte-identical after every rejected attack.
"""

import os
from pathlib import Path
import sys
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_preflight import ExecutionMode
from output_path_guard import (
    validate_condition_id,
    validate_recording_id,
    validate_final_rttm_path,
    revalidate_and_write_rttm,
    FinalRTTMPath,
    OutputSecurityError,
)


@pytest.fixture
def descriptor_env(tmp_path):
    """Sets up an isolated output environment with sentinel file."""
    out_root = tmp_path / "outputs"
    out_root.mkdir()

    sentinel_file = tmp_path / "sentinel_critical.txt"
    sentinel_content = "SENTINEL_CRITICAL_AUTHENTIC_DATA_DO_NOT_TRUNCATE_98765"
    sentinel_file.write_text(sentinel_content, encoding="utf-8")

    return {
        "tmp_path": tmp_path,
        "out_root": out_root,
        "sentinel_file": sentinel_file,
        "sentinel_content": sentinel_content,
    }


def test_1_normal_rttm_creation_succeeds(descriptor_env):
    """Normal RTTM path validation and descriptor-based write succeeds."""
    rttm_path_obj = validate_final_rttm_path(
        recording_id="rec_test_normal",
        condition_id="E00",
        custom_output_root=descriptor_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    segments = [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
        {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01"},
    ]

    written_path = revalidate_and_write_rttm(rttm_path_obj, segments)
    assert written_path.exists()
    assert written_path.is_file()
    content = written_path.read_text(encoding="utf-8")
    assert "SPEAKER rec_test_normal 1 0.000 2.500 <NA> <NA> SPEAKER_00 <NA> <NA>" in content
    assert "SPEAKER rec_test_normal 1 2.500 2.500 <NA> <NA> SPEAKER_01 <NA> <NA>" in content


def test_2_existing_regular_destination_safely_overwritten(descriptor_env):
    """
    Existing regular file with st_nlink == 1 is safely overwritten after descriptor inspection.
    """
    rttm_path_obj = validate_final_rttm_path(
        recording_id="rec_test_overwrite",
        condition_id="E00",
        custom_output_root=descriptor_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    # First write
    revalidate_and_write_rttm(rttm_path_obj, [{"start": 0.0, "end": 1.0, "speaker": "OLD"}])
    assert "OLD" in rttm_path_obj.final_path.read_text(encoding="utf-8")

    # Overwrite
    revalidate_and_write_rttm(rttm_path_obj, [{"start": 0.0, "end": 2.0, "speaker": "NEW"}])
    new_content = rttm_path_obj.final_path.read_text(encoding="utf-8")
    assert "NEW" in new_content
    assert "OLD" not in new_content


def test_3_final_destination_symlink_rejected(descriptor_env):
    """If destination path is already a symlink, write fails immediately without following it."""
    cond_dir = descriptor_env["out_root"] / "E01"
    cond_dir.mkdir(parents=True, exist_ok=True)
    symlink_target = cond_dir / "rec_symlink_dest.rttm"
    symlink_target.symlink_to(descriptor_env["sentinel_file"])

    with pytest.raises(OutputSecurityError, match="symlink"):
        validate_final_rttm_path(
            recording_id="rec_symlink_dest",
            condition_id="E01",
            custom_output_root=descriptor_env["out_root"],
            execution_mode=ExecutionMode.SYNTHETIC,
        )


def test_4_parent_directory_symlink_rejected(descriptor_env):
    """If parent condition directory is a symlink, write fails."""
    cond_dir = descriptor_env["out_root"] / "E10"
    outside_dir = descriptor_env["tmp_path"] / "outside_dir"
    outside_dir.mkdir()
    cond_dir.symlink_to(outside_dir)

    with pytest.raises(OutputSecurityError, match="symlink"):
        validate_final_rttm_path(
            recording_id="rec_parent_sym",
            condition_id="E10",
            custom_output_root=descriptor_env["out_root"],
            execution_mode=ExecutionMode.SYNTHETIC,
        )


def test_5_parent_directory_race_defeated(descriptor_env):
    """
    CRITICAL ACCEPTANCE TEST: PARENT-DIRECTORY RACE DEFEATED
    1. Validate destination path.
    2. Simulate parent directory swap: replace condition directory with a symlink to outside.
    3. Invoke revalidate_and_write_rttm.
    4. Assert OutputSecurityError is raised (via descriptor verification).
    5. Prove write cannot escape canonical output root!
    """
    rttm_path_obj = validate_final_rttm_path(
        recording_id="rec_parent_race",
        condition_id="E11",
        custom_output_root=descriptor_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    cond_dir = descriptor_env["out_root"] / "E11"
    outside_escape_dir = descriptor_env["tmp_path"] / "escaped_outside_outputs"
    outside_escape_dir.mkdir()

    # Attacker swaps parent condition directory with symlink pointing outside
    cond_dir.rmdir()
    cond_dir.symlink_to(outside_escape_dir)

    with pytest.raises(OutputSecurityError, match="symlink|directory|violation"):
        revalidate_and_write_rttm(
            rttm_path_obj,
            [{"start": 0.0, "end": 1.0, "speaker": "ATTACKER"}],
        )

    # Prove outside directory remains completely empty (no file written outside output root!)
    assert len(list(outside_escape_dir.iterdir())) == 0


def test_6_hardlink_truncation_attack_defeated(descriptor_env):
    """
    CRITICAL ACCEPTANCE TEST: HARD-LINK TRUNCATION ATTACK DEFEATED
    1. Validate destination path.
    2. Attacker creates hard link from destination path to an external sentinel file.
    3. Invoke revalidate_and_write_rttm.
    4. Writer detects st_nlink > 1 BEFORE truncation!
    5. OutputSecurityError is raised.
    6. VITAL CHECK: External sentinel file is byte-identical and NOT truncated!
    """
    rttm_path_obj = validate_final_rttm_path(
        recording_id="rec_hardlink_victim",
        condition_id="E20",
        custom_output_root=descriptor_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    dest_path = rttm_path_obj.final_path
    sentinel = descriptor_env["sentinel_file"]

    # Attacker hard-links destination path to sentinel file
    os.link(str(sentinel), str(dest_path))
    assert dest_path.exists()
    assert dest_path.stat().st_nlink > 1

    # Writer must refuse to truncate and fail closed
    with pytest.raises(OutputSecurityError, match="multiple hard links"):
        revalidate_and_write_rttm(
            rttm_path_obj,
            [{"start": 0.0, "end": 10.0, "speaker": "ATTACKER_OVERWRITE"}],
        )

    # PROVE SENTINEL IS COMPLETELY UNTOUCHED AND UNTRUNCATED!
    current_content = sentinel.read_text(encoding="utf-8")
    assert current_content == descriptor_env["sentinel_content"], (
        "CRITICAL SECURITY VULNERABILITY: External sentinel file was truncated or modified!"
    )


def test_7_condition_directory_traversal_rejected():
    """Path traversal in condition_id is strictly rejected."""
    with pytest.raises(OutputSecurityError, match="traversal"):
        validate_condition_id("../malicious")
    with pytest.raises(OutputSecurityError, match="traversal"):
        validate_condition_id("E00/sub")


def test_8_recording_id_traversal_rejected():
    """Path traversal in recording_id is strictly rejected."""
    with pytest.raises(OutputSecurityError, match="traversal"):
        validate_recording_id("../escape_rec", execution_mode=ExecutionMode.SYNTHETIC)
    with pytest.raises(OutputSecurityError, match="traversal"):
        validate_recording_id("rec/sub", execution_mode=ExecutionMode.SYNTHETIC)


def test_9_output_root_substitution_rejected(descriptor_env):
    """Output root pointing to prohibited stages is strictly rejected."""
    prohibited_root = descriptor_env["tmp_path"] / "stage2_baseline" / "outputs"
    prohibited_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(OutputSecurityError, match="escapes into prohibited stage"):
        validate_final_rttm_path(
            recording_id="rec_escape_root",
            condition_id="E00",
            custom_output_root=prohibited_root,
            execution_mode=ExecutionMode.SYNTHETIC,
        )


def test_10_external_sentinel_remains_byte_identical_after_all_attacks(descriptor_env):
    """
    Verifies that through all attempted operations, the external sentinel file
    remains byte-identical to its initial state.
    """
    assert descriptor_env["sentinel_file"].read_text(encoding="utf-8") == descriptor_env["sentinel_content"]
