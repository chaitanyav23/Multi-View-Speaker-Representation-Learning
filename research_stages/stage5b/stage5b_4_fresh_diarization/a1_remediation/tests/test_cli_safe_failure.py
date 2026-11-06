#!/usr/bin/env python3
"""Tests for Task 3: CLI Safe Failure and Backend Availability Enforcement."""

import pytest
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
MAIN_SCRIPT = SCRIPTS_DIR / "orchestrator.py"
PYTHON_BIN = sys.executable


def test_validate_config_success():
    res = subprocess.run(
        [PYTHON_BIN, str(MAIN_SCRIPT), "validate-config"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "[OK] Registry valid: 6 conditions registered." in res.stdout


def test_validate_cache_success():
    res = subprocess.run(
        [PYTHON_BIN, str(MAIN_SCRIPT), "validate-cache"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "[OK] Cache root validated" in res.stdout


def test_run_mode_without_authorization_fails():
    res = subprocess.run(
        [PYTHON_BIN, str(MAIN_SCRIPT), "run", "--recordings", "R01", "--conditions", "E00"],
        capture_output=True,
        text=True,
    )
    # Must exit with non-zero code
    assert res.returncode != 0
    assert "Execution of 'run' mode is BLOCKED!" in res.stderr


def test_run_mode_with_authorization_fails_when_backend_not_ready():
    """
    CRITICAL AUDIT CASE:
    Running with --authorize-inference MUST NOT print false success
    when production inference models are not available.
    Must exit with non-zero code (2) and clear NotImplementedError!
    """
    res = subprocess.run(
        [PYTHON_BIN, str(MAIN_SCRIPT), "run", "--authorize-inference", "--recordings", "R01", "--conditions", "E00"],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "NotImplementedError: Production fresh-diarization execution backend is not available" in res.stderr
