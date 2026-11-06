#!/usr/bin/env python3
"""
Test Suite: Manifest Validation and Protected Baseline Integrity
Tests the manifest verification engine against valid manifests, synthetic corruptions,
and verifies that all protected historical manifests remain cryptographically intact.
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


def test_protected_baselines_intact():
    """Verify that all protected baseline manifests have unchanged SHA-256 digests."""
    ok, errors = verify_protected_baselines()
    assert ok is True, f"Protected baselines were compromised: {errors}"
    assert len(errors) == 0


def test_manifest_verification_on_valid_data(tmp_path):
    """Verify that verify_manifest succeeds on valid files."""
    f1 = tmp_path / "sample1.txt"
    f1.write_text("sample content one")
    f2 = tmp_path / "sample2.txt"
    f2.write_text("sample content two")

    manifest_data = {
        "manifest_version": "1.0",
        "files": {
            str(f1): {
                "sha256": hashlib.sha256(b"sample content one").hexdigest(),
                "size_bytes": f1.stat().st_size,
            },
            str(f2): {
                "sha256": hashlib.sha256(b"sample content two").hexdigest(),
                "size_bytes": f2.stat().st_size,
            },
        },
    }
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    ok, errors = verify_manifest(manifest_path)
    assert ok is True
    assert len(errors) == 0


def test_manifest_verification_detects_hash_mismatch(tmp_path):
    """Verify that verify_manifest detects hash corruption."""
    f1 = tmp_path / "sample.txt"
    f1.write_text("valid content")

    manifest_data = {
        "manifest_version": "1.0",
        "files": {
            str(f1): {
                "sha256": "0" * 64,  # wrong hash
                "size_bytes": f1.stat().st_size,
            },
        },
    }
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    ok, errors = verify_manifest(manifest_path)
    assert ok is False
    assert any("HASH MISMATCH" in e for e in errors)


def test_manifest_verification_detects_size_mismatch(tmp_path):
    """Verify that verify_manifest detects size discrepancy."""
    f1 = tmp_path / "sample.txt"
    f1.write_text("valid content")

    manifest_data = {
        "manifest_version": "1.0",
        "files": {
            str(f1): {
                "sha256": hashlib.sha256(b"valid content").hexdigest(),
                "size_bytes": 99999,  # wrong size
            },
        },
    }
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    ok, errors = verify_manifest(manifest_path)
    assert ok is False
    assert any("SIZE MISMATCH" in e for e in errors)


def test_manifest_verification_detects_missing_file(tmp_path):
    """Verify that verify_manifest detects missing file entries."""
    missing_file = tmp_path / "ghost.txt"

    manifest_data = {
        "manifest_version": "1.0",
        "files": {
            str(missing_file): {
                "sha256": "abcdef" * 10 + "1234",
                "size_bytes": 100,
            },
        },
    }
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    ok, errors = verify_manifest(manifest_path)
    assert ok is False
    assert any("MISSING" in e for e in errors)
