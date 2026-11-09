#!/usr/bin/env python3
"""
Unit tests for Stage 5B.4-A.1.3.1 Finding 5:
RTTM destination symlink replacement race prevention & output path containment.
Verifies that:
1. Normal RTTM write succeeds and writes valid formatted lines.
2. Destination symlink rejected by pre-write checks.
3. CRITICAL ACCEPTANCE TEST:
   - Validate destination path.
   - Simulate TOCTOU race: replace destination with symlink pointing to external sentinel file.
   - Invoke revalidate_and_write_rttm.
   - Assert OutputSecurityError is raised.
   - Assert external sentinel file content is 100% UNTOUCHED!
4. Parent directory symlink replacement is detected and rejected.
5. Path traversal in condition_id rejected.
6. Path traversal in recording_id rejected.
7. Destination escaping into prohibited stages rejected.
8. Directory supplied as destination target rejected.
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
    validate_condition_id,
    validate_recording_id,
    validate_final_rttm_path,
    revalidate_and_write_rttm,
    FinalRTTMPath,
    OutputSecurityError,
)


@pytest.fixture
def output_env(tmp_path):
    """Sets up an isolated output environment."""
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    sentinel_file = tmp_path / "sentinel_important_data.txt"
    sentinel_original_content = "CRITICAL_SYSTEM_DATA_DO_NOT_OVERWRITE_12345"
    sentinel_file.write_text(sentinel_original_content, encoding="utf-8")

    return {
        "tmp_path": tmp_path,
        "out_root": out_root,
        "sentinel_file": sentinel_file,
        "sentinel_original_content": sentinel_original_content,
    }


def test_1_normal_rttm_write_succeeds(output_env):
    """Normal RTTM path validation and write succeeds."""
    rttm_path_obj = validate_final_rttm_path(
        recording_id="rec_test_01",
        condition_id="E00",
        custom_output_root=output_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    assert isinstance(rttm_path_obj, FinalRTTMPath)

    segments = [
        {"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
        {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
    ]

    written_path = revalidate_and_write_rttm(rttm_path_obj, segments)
    assert written_path.exists()
    assert written_path.is_file()

    content = written_path.read_text(encoding="utf-8")
    assert "SPEAKER rec_test_01 1 0.000 1.500 <NA> <NA> SPEAKER_00 <NA> <NA>" in content
    assert "SPEAKER rec_test_01 1 1.500 1.500 <NA> <NA> SPEAKER_01 <NA> <NA>" in content


def test_2_pre_existing_destination_symlink_rejected(output_env):
    """If destination is already a symlink, validation fails immediately."""
    cond_dir = output_env["out_root"] / "E00"
    cond_dir.mkdir(parents=True, exist_ok=True)
    symlink_dest = cond_dir / "rec_test_symlink.rttm"
    symlink_dest.symlink_to(output_env["sentinel_file"])

    with pytest.raises(OutputSecurityError, match="symlink"):
        validate_final_rttm_path(
            recording_id="rec_test_symlink",
            condition_id="E00",
            custom_output_root=output_env["out_root"],
            execution_mode=ExecutionMode.SYNTHETIC,
        )


def test_3_critical_acceptance_symlink_race_defeated(output_env):
    """
    CRITICAL ACCEPTANCE TEST FOR FINDING 5:
    1. Validate destination path.
    2. Simulate TOCTOU race: replace target path with symlink to external sentinel file.
    3. Invoke revalidate_and_write_rttm.
    4. Assert OutputSecurityError is raised.
    5. Assert external sentinel file content is 100% UNTOUCHED!
    """
    rttm_path_obj = validate_final_rttm_path(
        recording_id="rec_race_target",
        condition_id="E01",
        custom_output_root=output_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )
    dest_path = rttm_path_obj.final_path
    assert not dest_path.exists()

    # Attacker races to replace destination with symlink to sentinel file
    dest_path.symlink_to(output_env["sentinel_file"])
    assert dest_path.is_symlink()

    malicious_segments = [
        {"start": 0.0, "end": 10.0, "speaker": "ATTACKER_OVERWRITE"},
    ]

    # Attempt to write must fail with OutputSecurityError
    with pytest.raises(OutputSecurityError) as exc_info:
        revalidate_and_write_rttm(rttm_path_obj, malicious_segments)

    assert "symlink" in str(exc_info.value).lower() or "violation" in str(exc_info.value).lower()

    # VITAL CHECK: External sentinel file MUST NOT be overwritten or touched!
    current_sentinel_content = output_env["sentinel_file"].read_text(encoding="utf-8")
    assert current_sentinel_content == output_env["sentinel_original_content"], (
        "CRITICAL SECURITY FAILURE: External sentinel file was modified by symlink attack!"
    )


def test_4_parent_directory_symlink_detected(output_env):
    """If parent condition directory is replaced with a symlink, write fails."""
    cond_dir = output_env["out_root"] / "E10"
    cond_dir.mkdir(parents=True, exist_ok=True)

    rttm_path_obj = validate_final_rttm_path(
        recording_id="rec_parent_race",
        condition_id="E10",
        custom_output_root=output_env["out_root"],
        execution_mode=ExecutionMode.SYNTHETIC,
    )

    # Replace parent condition directory with symlink
    outside_dir = output_env["tmp_path"] / "outside_dir"
    outside_dir.mkdir()
    cond_dir.rmdir()
    cond_dir.symlink_to(outside_dir)

    segments = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    with pytest.raises(OutputSecurityError, match="Parent directory was replaced with a symlink"):
        revalidate_and_write_rttm(rttm_path_obj, segments)


def test_5_path_traversal_in_condition_id_rejected(output_env):
    """Path traversal characters in condition_id must be strictly rejected."""
    with pytest.raises(OutputSecurityError, match="traversal"):
        validate_condition_id("../malicious")

    with pytest.raises(OutputSecurityError, match="traversal"):
        validate_condition_id("E00/sub")


def test_6_path_traversal_in_recording_id_rejected(output_env):
    """Path traversal characters in recording_id must be strictly rejected."""
    with pytest.raises(OutputSecurityError, match="traversal"):
        validate_recording_id("../escape_rec", execution_mode=ExecutionMode.SYNTHETIC)

    with pytest.raises(OutputSecurityError, match="traversal"):
        validate_recording_id("rec/sub_part", execution_mode=ExecutionMode.SYNTHETIC)


def test_7_prohibited_stage_escape_rejected(output_env):
    """Path attempting to write into prohibited historical stages is rejected."""
    prohibited_root = output_env["tmp_path"] / "stage1_data_prep" / "outputs"
    prohibited_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(OutputSecurityError, match="escapes into prohibited stage"):
        validate_final_rttm_path(
            recording_id="rec_test_escape",
            condition_id="E00",
            custom_output_root=prohibited_root,
            execution_mode=ExecutionMode.SYNTHETIC,
        )


def test_8_destination_is_directory_rejected(output_env):
    """If destination path exists as a directory, write fails."""
    cond_dir = output_env["out_root"] / "E21"
    cond_dir.mkdir(parents=True, exist_ok=True)
    dir_target = cond_dir / "rec_is_dir.rttm"
    dir_target.mkdir()

    with pytest.raises(OutputSecurityError, match="already exists as a directory"):
        validate_final_rttm_path(
            recording_id="rec_is_dir",
            condition_id="E21",
            custom_output_root=output_env["out_root"],
            execution_mode=ExecutionMode.SYNTHETIC,
        )
