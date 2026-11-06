#!/usr/bin/env python3
"""
Test Suite: Canonical RTTM Output Path and Containment Guard
Tests all requirements from Section 5.3:
- Canonical production root accepted (research_stages/stage5b/stage5b_4_fresh_diarization/outputs/)
- Custom production root rejected
- Temporary synthetic root accepted only in synthetic mode
- Recording ID traversal rejected
- Condition ID traversal rejected
- Unknown condition rejected
- Symlinked output root rejected
- Symlinked condition directory rejected
- Symlinked final RTTM target rejected
- Directory collision rejected
- Final path remains under canonical outputs/
- Writer receives exactly the validated path
- Path substitution between validation and writing detected
"""

import os
from pathlib import Path
import sys
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from output_path_guard import (
    FinalRTTMPath,
    validate_final_rttm_path,
    validate_recording_id,
    validate_condition_id,
    revalidate_and_write_rttm,
    OutputSecurityError,
)
from path_utils import get_stage5b_4_outputs_dir, get_stage5b_4_dir
from checkpoint_preflight import ExecutionMode


# 1. Canonical production root accepted
def test_1_canonical_production_root_accepted():
    rttm_obj = validate_final_rttm_path(
        recording_id="R01_sara_sharif",
        condition_id="E00",
        execution_mode=ExecutionMode.PRODUCTION,
    )
    assert isinstance(rttm_obj, FinalRTTMPath)
    expected_path = get_stage5b_4_outputs_dir() / "E00" / "R01_sara_sharif.rttm"
    assert rttm_obj.final_path == expected_path.resolve()
    assert rttm_obj.execution_mode == ExecutionMode.PRODUCTION


# 2. Custom production root rejected
def test_2_custom_production_root_rejected(tmp_path):
    with pytest.raises(OutputSecurityError, match="Custom output root is strictly prohibited in PRODUCTION mode"):
        validate_final_rttm_path(
            recording_id="R01_sara_sharif",
            condition_id="E00",
            execution_mode=ExecutionMode.PRODUCTION,
            custom_output_root=tmp_path,
        )


# 3. Temporary synthetic root accepted only in synthetic mode
def test_3_temporary_synthetic_root_accepted_in_synthetic_mode(tmp_path):
    rttm_obj = validate_final_rttm_path(
        recording_id="rec_test01",
        condition_id="E01",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=tmp_path,
    )
    expected_path = tmp_path / "E01" / "rec_test01.rttm"
    assert rttm_obj.final_path == expected_path.resolve()
    assert rttm_obj.execution_mode == ExecutionMode.SYNTHETIC


# 4. Recording ID traversal rejected
def test_4_recording_id_traversal_rejected():
    for bad_id in ["../../etc/passwd", "../R01_test", "R01/hack", "R01\x00null"]:
        with pytest.raises(OutputSecurityError, match="Illegal characters or path traversal in recording_id"):
            validate_recording_id(bad_id, execution_mode=ExecutionMode.PRODUCTION)


# 5. Condition ID traversal rejected
def test_5_condition_id_traversal_rejected():
    for bad_cid in ["E00/../escape", "../E01", "E00\\test", "E00\x00"]:
        with pytest.raises(OutputSecurityError, match="Path traversal characters detected in condition_id"):
            validate_condition_id(bad_cid)


# 6. Unknown condition rejected
def test_6_unknown_condition_rejected():
    for unknown in ["E99", "C00", "CONDITION_X", "E00_extra"]:
        with pytest.raises(OutputSecurityError, match="Unauthorized condition_id"):
            validate_condition_id(unknown)


# 7. Symlinked output root escaping workspace rejected
def test_7_symlinked_output_root_escape_rejected(tmp_path):
    safe_dir = tmp_path / "safe_out"
    safe_dir.mkdir()
    danger_dir = tmp_path / "danger_escape"
    danger_dir.mkdir()

    sym_root = safe_dir / "sym_outputs"
    os.symlink(danger_dir, sym_root)

    with pytest.raises(OutputSecurityError, match="Symlink escape"):
        validate_final_rttm_path(
            recording_id="rec_synth01",
            condition_id="E00",
            execution_mode=ExecutionMode.SYNTHETIC,
            custom_output_root=sym_root,
        )


# 8. Symlinked condition directory escaping root rejected
def test_8_symlinked_condition_directory_rejected(tmp_path):
    out_root = tmp_path / "outputs"
    out_root.mkdir()
    danger_escape = tmp_path / "escaped_dir"
    danger_escape.mkdir()

    sym_cond = out_root / "E00"
    os.symlink(danger_escape, sym_cond)

    with pytest.raises(OutputSecurityError, match="Symlink escape: condition directory"):
        validate_final_rttm_path(
            recording_id="rec_synth01",
            condition_id="E00",
            execution_mode=ExecutionMode.SYNTHETIC,
            custom_output_root=out_root,
        )


# 9. Symlinked final RTTM target escaping root rejected
def test_9_symlinked_final_rttm_target_rejected(tmp_path):
    out_root = tmp_path / "outputs"
    cond_dir = out_root / "E00"
    cond_dir.mkdir(parents=True)

    secret_target = tmp_path / "secret.txt"
    secret_target.write_text("protected")

    sym_target = cond_dir / "rec_synth01.rttm"
    os.symlink(secret_target, sym_target)

    with pytest.raises(OutputSecurityError, match="Symlink escape: destination RTTM"):
        validate_final_rttm_path(
            recording_id="rec_synth01",
            condition_id="E00",
            execution_mode=ExecutionMode.SYNTHETIC,
            custom_output_root=out_root,
        )


# 10. Directory collision rejected
def test_10_directory_collision_rejected(tmp_path):
    out_root = tmp_path / "outputs"
    collision_path = out_root / "E00" / "rec_synth01.rttm"
    collision_path.mkdir(parents=True)  # Created as a directory instead of a file

    with pytest.raises(OutputSecurityError, match="destination path already exists as a directory"):
        validate_final_rttm_path(
            recording_id="rec_synth01",
            condition_id="E00",
            execution_mode=ExecutionMode.SYNTHETIC,
            custom_output_root=out_root,
        )


# 11. Final path remains strictly under canonical outputs/
def test_11_final_path_remains_under_canonical_outputs():
    rttm_obj = validate_final_rttm_path(
        recording_id="R02_dennis_skinner",
        condition_id="E21",
        execution_mode=ExecutionMode.PRODUCTION,
    )
    canonical_root = get_stage5b_4_outputs_dir().resolve()
    assert rttm_obj.final_path.is_relative_to(canonical_root)
    assert rttm_obj.final_path.parent == canonical_root / "E21"
    assert rttm_obj.final_path.name == "R02_dennis_skinner.rttm"


# 12. Writer receives exactly the validated path
def test_12_writer_receives_exact_validated_path(tmp_path):
    rttm_obj = validate_final_rttm_path(
        recording_id="rec_01",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=tmp_path,
    )
    segments = [
        {"speaker": "SPK1", "start": 0.0, "end": 1.5},
        {"speaker": "SPK2", "start": 1.5, "end": 3.0},
    ]

    written_path = revalidate_and_write_rttm(rttm_obj, segments)
    assert written_path == rttm_obj.final_path
    assert written_path.exists()

    content = written_path.read_text().strip().splitlines()
    assert len(content) == 2
    assert content[0] == "SPEAKER rec_01 1 0.000 1.500 <NA> <NA> SPK1 <NA> <NA>"
    assert content[1] == "SPEAKER rec_01 1 1.500 1.500 <NA> <NA> SPK2 <NA> <NA>"


# 13. Path substitution between validation and writing detected
def test_13_path_substitution_detected_by_writer(tmp_path):
    rttm_obj = validate_final_rttm_path(
        recording_id="rec_02",
        condition_id="E00",
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=tmp_path,
    )

    # Simulate filesystem substitution: destination replaced by a directory before write
    rttm_obj.final_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(OutputSecurityError, match="TOCTOU violation: Destination was replaced with directory"):
        revalidate_and_write_rttm(rttm_obj, [{"speaker": "SPK1", "start": 0.0, "end": 1.0}])
