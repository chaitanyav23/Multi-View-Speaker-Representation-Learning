#!/usr/bin/env python3
"""
Unit tests for Stage 5B.4-A.1.3.1 Manifest & Baseline Verification.
Verifies:
1. All 10 protected baseline manifests pass integrity verification.
2. Compromised baseline manifest is immediately detected.
3. Deliverables manifest verifier detects valid manifests.
4. File hash mismatch is detected and reported.
5. Missing file in manifest is detected and reported.
"""

import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_manifest import (
    verify_protected_baselines,
    verify_manifest,
    PROTECTED_MANIFEST_DIGESTS,
)


def test_1_protected_baselines_integrity():
    """All 10 baseline manifests are verified 100% untouched."""
    ok, errors = verify_protected_baselines()
    assert ok is True
    assert len(errors) == 0


def test_2_baseline_mutation_detected(monkeypatch):
    """If a baseline manifest is tampered with, verifier catches it."""
    first_key = next(iter(PROTECTED_MANIFEST_DIGESTS.keys()))
    monkeypatch.setitem(PROTECTED_MANIFEST_DIGESTS, first_key, "0" * 64)

    ok, errors = verify_protected_baselines()
    assert ok is False
    assert any("PROTECTED FILE COMPROMISED" in e for e in errors)


def test_3_manifest_verifier_success(tmp_path):
    """A valid manifest with correct hashes passes."""
    f1 = tmp_path / "file1.txt"
    f1.write_text("Hello World", encoding="utf-8")
    f1_sha = hashlib.sha256(b"Hello World").hexdigest()

    manifest_data = {
        "files": {
            str(f1.name): {
                "sha256": f1_sha,
                "size_bytes": f1.stat().st_size,
            }
        }
    }
    manifest_file = tmp_path / "test_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    # verify with monkeypatched A1_3_1_DIR
    import verify_manifest as vm
    old_dir = vm.A1_3_1_DIR
    try:
        vm.A1_3_1_DIR = tmp_path
        ok, errors = verify_manifest(manifest_file)
        assert ok is True
        assert len(errors) == 0
    finally:
        vm.A1_3_1_DIR = old_dir


def test_4_manifest_hash_mismatch_detected(tmp_path):
    """A manifest entry with mismatched hash is detected."""
    f1 = tmp_path / "file1.txt"
    f1.write_text("Hello World", encoding="utf-8")

    manifest_data = {
        "files": {
            str(f1.name): {
                "sha256": "0" * 64,
                "size_bytes": f1.stat().st_size,
            }
        }
    }
    manifest_file = tmp_path / "test_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    import verify_manifest as vm
    old_dir = vm.A1_3_1_DIR
    try:
        vm.A1_3_1_DIR = tmp_path
        ok, errors = verify_manifest(manifest_file)
        assert ok is False
        assert any("HASH MISMATCH" in e for e in errors)
    finally:
        vm.A1_3_1_DIR = old_dir


def test_5_manifest_missing_file_detected(tmp_path):
    """A manifest entry pointing to a missing file is detected."""
    manifest_data = {
        "files": {
            "nonexistent_file.txt": {
                "sha256": "0" * 64,
                "size_bytes": 100,
            }
        }
    }
    manifest_file = tmp_path / "test_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    import verify_manifest as vm
    old_dir = vm.A1_3_1_DIR
    try:
        vm.A1_3_1_DIR = tmp_path
        ok, errors = verify_manifest(manifest_file)
        assert ok is False
        assert any("MISSING" in e for e in errors)
    finally:
        vm.A1_3_1_DIR = old_dir
