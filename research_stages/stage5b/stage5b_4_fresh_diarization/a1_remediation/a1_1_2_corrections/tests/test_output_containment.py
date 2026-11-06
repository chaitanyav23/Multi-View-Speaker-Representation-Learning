#!/usr/bin/env python3
"""
Tests for Correction 2: Final RTTM Filename and Identifier Containment.
Verifies strict validation of recording_id and condition_id,
prevention of directory traversal through filenames, rejection of symlink escapes,
and containment of the full final RTTM path.
"""

import os
from pathlib import Path
import pytest
import sys

CORRECTIONS_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(CORRECTIONS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTIONS_SCRIPTS))

from output_path_guard import (
    validate_recording_id,
    validate_condition_id,
    validate_final_rttm_path,
    OutputSecurityError,
)
from checkpoint_preflight import ExecutionMode
from path_utils import get_stage5b_4_dir, get_research_stages_dir


def test_valid_recording_ids():
    valid_ids = [
        "R01_sara_sharif",
        "R02_dennis_skinner",
        "R03_disabled_parking",
        "R04_piers_robinson",
        "R05_southwest_pilot",
        "R06_rich_karen",
        "R07_bwc_cheap_shot",
        "R08_uk_parliament",
    ]
    for rid in valid_ids:
        assert validate_recording_id(rid, execution_mode=ExecutionMode.PRODUCTION) == rid


def test_valid_registered_condition_ids():
    valid_cids = ["E00", "E01", "E10", "E11", "E20", "E21"]
    for cid in valid_cids:
        assert validate_condition_id(cid) == cid


def test_unknown_condition_id_rejection():
    for invalid_cid in ["E99", "C00", "M0_no_overlap", "UNKNOWN", ""]:
        with pytest.raises(OutputSecurityError, match="Unauthorized condition_id"):
            validate_condition_id(invalid_cid)


def test_traversal_in_recording_id():
    traversal_ids = [
        "../escape",
        "../../etc/passwd",
        "R01/../../escape",
        "subdir/R01_sara_sharif",
        "R01\\windows\\style",
    ]
    for tid in traversal_ids:
        with pytest.raises(OutputSecurityError, match="Illegal characters or path traversal"):
            validate_recording_id(tid, execution_mode=ExecutionMode.PRODUCTION)


def test_absolute_recording_id():
    with pytest.raises(OutputSecurityError, match="Illegal characters or path traversal"):
        validate_recording_id("/tmp/R01_sara_sharif", execution_mode=ExecutionMode.PRODUCTION)


def test_path_separators_in_condition_id():
    with pytest.raises(OutputSecurityError, match="Path traversal characters detected"):
        validate_condition_id("../E00")
    with pytest.raises(OutputSecurityError, match="Path traversal characters detected"):
        validate_condition_id("E00/E01")


def test_null_bytes_and_control_characters():
    with pytest.raises(OutputSecurityError):
        validate_recording_id("R01_sara\x00_sharif", execution_mode=ExecutionMode.PRODUCTION)
    with pytest.raises(OutputSecurityError):
        validate_recording_id("R01_sara\n_sharif", execution_mode=ExecutionMode.PRODUCTION)
    with pytest.raises(OutputSecurityError):
        validate_condition_id("E00\x00")


def test_unexpected_suffixes_in_recording_id():
    # File extensions like .rttm or .txt must be rejected in recording_id
    with pytest.raises(OutputSecurityError, match="cannot contain file extensions or dots"):
        validate_recording_id("R01_sara_sharif.rttm", execution_mode=ExecutionMode.PRODUCTION)
    with pytest.raises(OutputSecurityError, match="cannot contain file extensions or dots"):
        validate_recording_id("R01_sara_sharif.wav", execution_mode=ExecutionMode.PRODUCTION)


def test_symlinked_output_directory_escape(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside_forbidden"
    outside.mkdir()

    symlink_dir = sandbox / "link_to_outside"
    os.symlink(str(outside), str(symlink_dir))

    with pytest.raises(OutputSecurityError, match="Symlink escape detected|outside allowed roots"):
        validate_final_rttm_path(
            output_dir=symlink_dir,
            recording_id="rec_synthetic01",
            condition_id="E00",
            execution_mode=ExecutionMode.SYNTHETIC,
            custom_allowed_roots=[sandbox],
        )


def test_symlinked_final_rttm_path(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside_file = tmp_path / "outside_rttm.rttm"
    outside_file.write_text("dummy")

    # Create symlink named rec_synthetic01_E00.rttm inside sandbox pointing to outside_file
    target_link = sandbox / "rec_synthetic01_E00.rttm"
    os.symlink(str(outside_file), str(target_link))

    with pytest.raises(OutputSecurityError, match="containment violation"):
        validate_final_rttm_path(
            output_dir=sandbox,
            recording_id="rec_synthetic01",
            condition_id="E00",
            execution_mode=ExecutionMode.SYNTHETIC,
            custom_allowed_roots=[sandbox],
        )


def test_existing_directory_collision(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    # Create directory with the EXACT name that the RTTM file would have
    target_dir_name = sandbox / "rec_synthetic01_E00.rttm"
    target_dir_name.mkdir()

    with pytest.raises(OutputSecurityError, match="already exists as a directory"):
        validate_final_rttm_path(
            output_dir=sandbox,
            recording_id="rec_synthetic01",
            condition_id="E00",
            execution_mode=ExecutionMode.SYNTHETIC,
            custom_allowed_roots=[sandbox],
        )


def test_custom_output_root_prohibited_in_production(tmp_path):
    with pytest.raises(OutputSecurityError, match="Custom allowed_output_roots prohibited in PRODUCTION mode"):
        validate_final_rttm_path(
            output_dir=tmp_path / "custom_out",
            recording_id="R01_sara_sharif",
            condition_id="E00",
            execution_mode=ExecutionMode.PRODUCTION,
            custom_allowed_roots=[tmp_path],
        )


def test_valid_production_output_path():
    stage5b4_dir = get_stage5b_4_dir()
    out_dir = stage5b4_dir / "outputs" / "fresh_run"
    final_path = validate_final_rttm_path(
        output_dir=out_dir,
        recording_id="R01_sara_sharif",
        condition_id="E00",
        execution_mode=ExecutionMode.PRODUCTION,
    )
    assert final_path == (out_dir / "R01_sara_sharif_E00.rttm").resolve()
    assert final_path.is_relative_to(stage5b4_dir.resolve())
