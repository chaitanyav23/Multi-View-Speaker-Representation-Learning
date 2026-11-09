#!/usr/bin/env python3
"""
Manifest Verification Script for Stage 5B.4-A.1.3.1.
Verifies cryptographic SHA-256 checksums and file sizes for all files
registered in manifests/A1_3_1_MANIFEST.json, and checks 10 protected baseline manifests.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

from path_utils import get_repo_root, get_a1_3_1_dir

SCRIPT_DIR = Path(__file__).resolve().parent
A1_3_1_DIR = SCRIPT_DIR.parent
REPO_ROOT = get_repo_root().resolve()

# 10 Protected baseline manifest digests recorded at start of A.1.3.1
PROTECTED_MANIFEST_DIGESTS = {
    "research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_corrections/manifests/A1_3_MANIFEST.json": "752c90744b08f6b2325dab0136d2e7e7f50476b664821515aba9bb2fc0aa813e",
    "research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_2_corrections/manifests/A1_2_MANIFEST.json": "2c5b9807e86d0194ac6b478be497bee579aca138daac566f789114469e0b69fa",
    "research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_corrections/manifests/A1_1_MANIFEST.json": "2b6aeef078cee49ee426afd7d8ce0ba08fd08852597601979f963abacec99241",
    "research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/manifests/A1_MANIFEST.json": "4a80f20d9a598c88a675bace4fe07909f23e1451390f1033c37331b4d68c17f5",
    "research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4A_MANIFEST.json": "5216ff7066d84d440eb3911543f42580e4cae5ba0c3708d5ab128a1f7cf56b43",
    "research_stages/stage5b/STAGE5B_BASELINE_MANIFEST.json": "4ad738d7d67f9af9be2d4d9589fb293931847b05fce8c428f13b929791123a6c",
    "research_stages/stage5b/stage5b_1_vad_optimization/STAGE5B_1_MANIFEST.json": "3fe9becb9a72a24b99548bdfc7629d36309625abb0f164c047f3d564f214ffe9",
    "research_stages/stage5b/stage5b_2_overlap_optimization/STAGE5B_2_MANIFEST.json": "4ad3e03a6ac161836c0fe596e40977173f2f5f005683114ff10df7faccce7abf",
    "research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_2/V4_2_MANIFEST.json": "56ba622c83f81fb99b9359af80cdedfd042ced8b905bba09fa728e476d831919",
    "research_stages/stage1_data_prep/manifests/recording_manifest.json": "967ff3fb7aeb14543cd6bd5996724813bbd90669f54277daaf52c3736849be0b",
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
        # Target can be relative to A1_3_1_DIR or REPO_ROOT
        target_path = A1_3_1_DIR / rel_path
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
    """Verifies that none of the 10 historical baseline manifests have been modified."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Stage 5B.4-A.1.3.1 deliverables manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=A1_3_1_DIR / "manifests" / "A1_3_1_MANIFEST.json",
        help="Path to manifest file to verify",
    )
    parser.add_argument(
        "--check-baselines-only",
        action="store_true",
        help="Only check baseline manifests integrity",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 5B.4-A.1.3.1 MANIFEST & BASELINE INTEGRITY VERIFIER")
    print("=" * 70)

    # 1. Verify baseline manifests
    print("\n1. Verifying 10 Protected Baseline Manifests...")
    baselines_ok, baseline_errors = verify_protected_baselines()
    if not baselines_ok:
        print("[FAIL] Protected baseline manifest integrity check failed:")
        for err in baseline_errors:
            print(f"  - {err}")
        return 1
    print(f"[PASS] All {len(PROTECTED_MANIFEST_DIGESTS)} protected baseline manifests verified (100% untouched).")

    if args.check_baselines_only:
        return 0

    # 2. Verify A.1.3.1 Manifest
    manifest_path = args.manifest.resolve()
    print(f"\n2. Verifying Deliverables Manifest: {manifest_path}...")
    manifest_ok, manifest_errors = verify_manifest(manifest_path)
    if not manifest_ok:
        print(f"[FAIL] Deliverables manifest verification failed ({len(manifest_errors)} errors):")
        for err in manifest_errors:
            print(f"  - {err}")
        return 1

    with manifest_path.open("r", encoding="utf-8") as f:
        file_count = len(json.load(f).get("files", {}))
    print(f"[PASS] All {file_count} deliverables in manifest verified successfully.")
    print("=" * 70)
    print("VERDICT: PASS — All cryptographic digests and file sizes match.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
