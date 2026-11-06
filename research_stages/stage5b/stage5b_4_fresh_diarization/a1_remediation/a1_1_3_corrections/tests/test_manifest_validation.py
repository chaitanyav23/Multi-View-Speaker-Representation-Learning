#!/usr/bin/env python3
"""
Test Suite: Manifest and Protected Artifact Validator Unit Tests
Validates that verify_manifest.py correctly detects:
- Missing files
- File size mismatches
- Cryptographic SHA-256 hash mismatches
- Verifies that all 9 protected baseline manifests are currently bit-exact.
"""

import hashlib
import json
from pathlib import Path
import sys
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_manifest import verify_manifest, verify_protected_baselines, sha256_file


def test_1_protected_baselines_integrity():
    """All 9 protected baseline manifests must pass bit-exact validation."""
    ok, errors = verify_protected_baselines()
    assert ok is True, f"Protected baselines modified: {errors}"
    assert len(errors) == 0


def test_2_validator_detects_missing_file(tmp_path):
    """Manifest verifier must report MISSING when a registered file does not exist."""
    fake_manifest = tmp_path / "manifest.json"
    data = {
        "schema_version": "1.0",
        "files": {
            "non_existent_file.txt": {
                "size_bytes": 123,
                "sha256": "0" * 64,
            }
        }
    }
    fake_manifest.write_text(json.dumps(data))
    ok, errors = verify_manifest(fake_manifest)
    assert ok is False
    assert any("MISSING" in e for e in errors)


def test_3_validator_detects_size_mismatch(tmp_path):
    """Manifest verifier must report SIZE MISMATCH when actual byte size differs."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello world")
    actual_size = test_file.stat().st_size
    actual_sha = sha256_file(test_file)

    fake_manifest = tmp_path / "manifest.json"
    data = {
        "schema_version": "1.0",
        "files": {
            str(test_file): {
                "size_bytes": actual_size + 10,
                "sha256": actual_sha,
            }
        }
    }
    fake_manifest.write_text(json.dumps(data))
    ok, errors = verify_manifest(fake_manifest)
    assert ok is False
    assert any("SIZE MISMATCH" in e for e in errors)


def test_4_validator_detects_hash_mismatch(tmp_path):
    """Manifest verifier must report HASH MISMATCH when content does not match expected sha256."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("hello world")
    actual_size = test_file.stat().st_size

    fake_manifest = tmp_path / "manifest.json"
    data = {
        "schema_version": "1.0",
        "files": {
            str(test_file): {
                "size_bytes": actual_size,
                "sha256": hashlib.sha256(b"different_content").hexdigest(),
            }
        }
    }
    fake_manifest.write_text(json.dumps(data))
    ok, errors = verify_manifest(fake_manifest)
    assert ok is False
    assert any("HASH MISMATCH" in e for e in errors)


def test_5_validator_passes_on_valid_manifest(tmp_path):
    """Manifest verifier must return True and zero errors on a valid manifest."""
    test_file = tmp_path / "file.txt"
    test_file.write_text("antigravity deterministic verification test")
    actual_size = test_file.stat().st_size
    actual_sha = sha256_file(test_file)

    fake_manifest = tmp_path / "manifest.json"
    data = {
        "schema_version": "1.0",
        "files": {
            str(test_file): {
                "size_bytes": actual_size,
                "sha256": actual_sha,
            }
        }
    }
    fake_manifest.write_text(json.dumps(data))
    ok, errors = verify_manifest(fake_manifest)
    assert ok is True
    assert len(errors) == 0
