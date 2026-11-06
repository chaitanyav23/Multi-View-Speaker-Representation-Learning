#!/usr/bin/env python3
"""Tests for Stage 5B.4 CLI Safety Mechanism."""

import pytest
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
MAIN_SCRIPT = SCRIPTS_DIR / "run_stage5b_4_fresh_diarization.py"
PYTHON_BIN = sys.executable


def test_validate_config_mode():
    res = subprocess.run(
        [PYTHON_BIN, str(MAIN_SCRIPT), "validate-config"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "[OK] Registry valid: 6 conditions registered" in res.stdout


def test_validate_cache_mode():
    res = subprocess.run(
        [PYTHON_BIN, str(MAIN_SCRIPT), "validate-cache"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "[OK] Stage 3 prohibited path check passed" in res.stdout


def test_test_synthetic_mode():
    res = subprocess.run(
        [PYTHON_BIN, str(MAIN_SCRIPT), "test-synthetic"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "[SUCCESS] All synthetic pipeline steps executed cleanly" in res.stdout


def test_run_mode_without_authorization_flag_fails():
    res = subprocess.run(
        [PYTHON_BIN, str(MAIN_SCRIPT), "run", "--recordings", "R01_sara_sharif", "--conditions", "E00"],
        capture_output=True,
        text=True,
    )
    # Must fail with non-zero exit code
    assert res.returncode != 0
    assert "Execution of 'run' mode is BLOCKED!" in res.stderr
    assert "ACCESS DENIED" in res.stderr
