#!/usr/bin/env python3
"""Tests for Task 6: Output RTTM and Artifact Path Containment."""

import os
from pathlib import Path
import pytest
import sys

CORRECTIONS_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(CORRECTIONS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTIONS_SCRIPTS))

from output_path_guard import (
    validate_output_path,
    OutputSecurityError,
)
from path_utils import get_stage5b_4_dir, get_research_stages_dir


def test_valid_output_path_inside_stage5b4():
    s5b4_dir = get_stage5b_4_dir()
    test_out = s5b4_dir / "rttm_outputs" / "test_run"
    validated = validate_output_path(test_out, allowed_roots=[s5b4_dir], allow_creation=False)
    assert validated == test_out.resolve()


def test_valid_custom_test_root(tmp_path):
    custom_root = tmp_path / "custom_test_outputs"
    custom_root.mkdir()
    target_sub = custom_root / "run_01"
    validated = validate_output_path(target_sub, allowed_roots=[custom_root], allow_creation=True)
    assert validated == target_sub.resolve()
    assert target_sub.exists()


def test_rejection_of_prohibited_stages():
    rs_dir = get_research_stages_dir()
    s5b4_dir = get_stage5b_4_dir()

    # Stage 3
    stage3_path = rs_dir / "stage3_dual_encoder" / "rttm"
    with pytest.raises(OutputSecurityError, match="escapes into prohibited stage"):
        validate_output_path(stage3_path, allowed_roots=[s5b4_dir, rs_dir], allow_creation=False)

    # Stage 5B.1
    stage5b1_path = rs_dir / "stage5b" / "stage5b_1_vad_optimization" / "output"
    with pytest.raises(OutputSecurityError, match="escapes into prohibited stage"):
        validate_output_path(stage5b1_path, allowed_roots=[s5b4_dir, rs_dir], allow_creation=False)

    # Stage 5B.2
    stage5b2_path = rs_dir / "stage5b" / "stage5b_2_overlap_optimization" / "output"
    with pytest.raises(OutputSecurityError, match="escapes into prohibited stage"):
        validate_output_path(stage5b2_path, allowed_roots=[s5b4_dir, rs_dir], allow_creation=False)


def test_rejection_of_directory_traversal(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    traversal_path = root / "subdir" / ".." / ".." / "escaped"
    with pytest.raises(OutputSecurityError, match="containment violation"):
        validate_output_path(traversal_path, allowed_roots=[root], allow_creation=False)


def test_rejection_of_symlink_escape(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside_secret"
    outside.mkdir()

    symlink_dir = sandbox / "link_to_outside"
    os.symlink(str(outside), str(symlink_dir))

    target = symlink_dir / "output_file"
    with pytest.raises(OutputSecurityError):
        validate_output_path(target, allowed_roots=[sandbox], allow_creation=False)
