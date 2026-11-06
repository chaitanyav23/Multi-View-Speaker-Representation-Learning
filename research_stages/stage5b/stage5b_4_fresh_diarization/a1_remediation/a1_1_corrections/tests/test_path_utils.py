#!/usr/bin/env python3
"""Tests for Task 3: Centralized Repository Root and Path Utilities."""

import os
from pathlib import Path
import pytest
import sys

CORRECTIONS_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(CORRECTIONS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTIONS_SCRIPTS))

from path_utils import (
    get_repo_root,
    get_research_stages_dir,
    get_stage5b_dir,
    get_stage5b_4_dir,
    get_stage5b_4_cache_dir,
    get_stage2_posteriors_dir,
    get_phase20_scripts_dir,
)


def test_repo_root_resolution():
    root = get_repo_root()
    assert (root / ".git").exists()
    assert root.name == "Multi-View-Speaker-Representation-Learning"


def test_path_resolution_no_duplication():
    root = get_repo_root()
    rs_dir = get_research_stages_dir()
    assert rs_dir == root / "research_stages"
    assert "research_stages/research_stages" not in str(rs_dir)

    stage5b_dir = get_stage5b_dir()
    assert stage5b_dir == rs_dir / "stage5b"

    stage5b4_dir = get_stage5b_4_dir()
    assert stage5b4_dir == stage5b_dir / "stage5b_4_fresh_diarization"

    posteriors_dir = get_stage2_posteriors_dir()
    assert posteriors_dir == rs_dir / "stage2_baseline" / "posteriors"
    assert posteriors_dir.exists()

    p20_scripts = get_phase20_scripts_dir()
    assert p20_scripts == root / "phase20_standalone" / "scripts"
    assert p20_scripts.exists()


def test_resolution_from_arbitrary_working_directory(monkeypatch, tmp_path):
    # Change current working directory to an arbitrary temp directory
    monkeypatch.chdir(tmp_path)
    # Without start_path, resolves from __file__ location
    root = get_repo_root()
    assert (root / ".git").exists()
    assert root.name == "Multi-View-Speaker-Representation-Learning"


def test_resolution_with_mock_git_tree(tmp_path):
    mock_repo = tmp_path / "mock_project"
    mock_git = mock_repo / ".git"
    mock_git.mkdir(parents=True)
    sub_deep = mock_repo / "a" / "b" / "c" / "d"
    sub_deep.mkdir(parents=True)

    resolved = get_repo_root(sub_deep)
    assert resolved == mock_repo.resolve()
