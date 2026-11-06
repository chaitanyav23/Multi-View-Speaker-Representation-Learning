#!/usr/bin/env python3
"""
Manifest Verification Script for Stage 5B.4-A.1.2.
Verifies cryptographic SHA-256 checksums and file sizes for all files
registered in manifests/A1_2_MANIFEST.json, and checks protected baseline files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

from path_utils import get_repo_root

SCRIPT_DIR = Path(__file__).resolve().parent
A1_2_DIR = SCRIPT_DIR.parent
REPO_ROOT = get_repo_root().resolve()

# Protected baseline manifest digests recorded at start of A.1.2
PROTECTED_MANIFEST_DIGESTS = {
    "research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_corrections/manifests/A1_1_MANIFEST.json": "2b6aeef078cee49ee426afd7d8ce0ba08fd08852597601979f963abacec99241",
    "research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/manifests/A1_MANIFEST.json": "4a80f20d9a598c88a675bace4fe07909f23e1451390f1033c37331b4d68c17f5",
    "research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4A_MANIFEST.json": "5216ff7066d84d440eb3911543f42580e4cae5ba0c3708d5ab128a1f7cf56b43",
    "research_stages/stage5b/STAGE5B_BASELINE_MANIFEST.json": "4ad738d7d67f9af9be2d4d9589fb293931847b05fce8c428f13b929791123a6c",
    "research_stages/stage5b/stage5b_1_vad_optimization/STAGE5B_1_MANIFEST.json": "3fe9becb9a72a24b99548bdfc7629d36309625abb0f164c047f3d564f214ffe9",
    "research_stages/stage5b/stage5b_2_overlap_optimization/STAGE5B_2_MANIFEST.json": "4ad3e03a6ac161836c0fe596e40977173f2f5f005683114ff10df7faccce7abf",
    "research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_2/V4_2_MANIFEST.json": "56ba622c83f81fb99b9359af80cdedfd042ced8b905bba09fa728e476d831919",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest_path: Path) -> Tuple[bool, List[str]]:
    """
    Verifies all entries in manifest_path.
    Returns (success, error_list).
    """
    errors: List[str] = []
    if not manifest_path.exists():
        return False, [f"Manifest file not found: {manifest_path}"]

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    files_dict = data.get("files", {})
    if not files_dict:
        return False, [f"Manifest {manifest_path} contains no files entry."]

    for rel_path, entry in files_dict.items():
        # Handle relative to A1_2_DIR or REPO_ROOT
        target_path = A1_2_DIR / rel_path
        if not target_path.exists():
            target_path = REPO_ROOT / rel_path

        if not target_path.exists():
            errors.append(f"MISSING: {rel_path} does not exist at {target_path}")
            continue

        actual_size = target_path.stat().st_size
        expected_size = entry.get("size_bytes")
        if expected_size is not None and actual_size != expected_size:
            errors.append(
                f"SIZE MISMATCH for {rel_path}: expected {expected_size} bytes, got {actual_size} bytes"
            )

        actual_sha = sha256_file(target_path)
        expected_sha = entry.get("sha256")
        if actual_sha != expected_sha:
            errors.append(
                f"HASH MISMATCH for {rel_path}: expected {expected_sha}, got {actual_sha}"
            )

    return len(errors) == 0, errors


def verify_protected_baselines() -> Tuple[bool, List[str]]:
    """Verifies that none of the historical baseline manifests have been modified."""
    errors: List[str] = []
    for rel_path, expected_sha in PROTECTED_MANIFEST_DIGESTS.items():
        file_path = REPO_ROOT / rel_path
        if not file_path.exists():
            errors.append(f"MISSING PROTECTED FILE: {rel_path}")
            continue
        actual_sha = sha256_file(file_path)
        if actual_sha != expected_sha:
            errors.append(
                f"PROTECTED FILE COMPROMISED: {rel_path} modified! Expected {expected_sha}, got {actual_sha}"
            )
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Verify Stage 5B.4-A.1.2 Manifest Integrity")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=A1_2_DIR / "manifests" / "A1_2_MANIFEST.json",
        help="Path to A1_2_MANIFEST.json",
    )
    args = parser.parse_args()

    print(f"=== Verifying Protected Baseline Artifacts ===")
    p_ok, p_errors = verify_protected_baselines()
    if not p_ok:
        print("FAIL: Protected baseline manifests failed verification:")
        for err in p_errors:
            print(f"  - {err}")
        sys.exit(1)
    print(f"SUCCESS: All {len(PROTECTED_MANIFEST_DIGESTS)} protected baseline manifests intact.")

    print(f"\n=== Verifying Stage 5B.4-A.1.2 Manifest: {args.manifest} ===")
    m_ok, m_errors = verify_manifest(args.manifest)
    if not m_ok:
        print("FAIL: Manifest verification encountered errors:")
        for err in m_errors:
            print(f"  - {err}")
        sys.exit(1)

    with args.manifest.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"SUCCESS: All {len(data['files'])} manifest entries verified with valid SHA-256 and size.")


if __name__ == "__main__":
    main()
