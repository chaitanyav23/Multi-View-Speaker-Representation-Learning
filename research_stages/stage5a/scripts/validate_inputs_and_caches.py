#!/usr/bin/env python3
"""
Stage 5A Pre-Execution Validation Script:
Verifies cache compatibility, subsegment alignment, dimensions, finite values,
and Stage 4.1 safeguards between TitaNet-DFN and ResNet293-mono.
"""

import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning")
CONFIG_PATH = REPO_ROOT / "research_stages" / "stage5a" / "configs" / "stage5a_config.json"


def validate_all() -> bool:
    print("[Stage 5A Validation] Starting cache and protocol validation...")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    manifest_path = REPO_ROOT / cfg["paths"]["manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    recordings = cfg["recordings"]
    manifest_recs = [m["rec_id"] for m in manifest]
    assert set(recordings) == set(manifest_recs), f"Recordings mismatch: {recordings} vs {manifest_recs}"

    scales = [s["scale_idx"] for s in cfg["temporal_scales"]]
    total_subsegments_checked = 0

    for rec_id in recordings:
        for s in scales:
            dfn_dir = REPO_ROOT / f"research_stages/stage3_dual_encoder/cache/dfn/{rec_id}"
            mono_dir = REPO_ROOT / f"research_stages/stage3_dual_encoder/cache/16k_mono/{rec_id}"

            dfn_meta_p = dfn_dir / f"subsegments_metadata__scale{s}.json"
            mono_meta_p = mono_dir / f"subsegments_metadata__scale{s}.json"
            assert dfn_meta_p.exists(), f"Missing DFN metadata: {dfn_meta_p}"
            assert mono_meta_p.exists(), f"Missing Mono metadata: {mono_meta_p}"

            with open(dfn_meta_p, "r", encoding="utf-8") as f:
                meta_dfn = json.load(f)["subsegments"]
            with open(mono_meta_p, "r", encoding="utf-8") as f:
                meta_mono = json.load(f)["subsegments"]

            assert len(meta_dfn) == len(meta_mono), (
                f"Length mismatch for {rec_id} scale {s}: dfn={len(meta_dfn)} vs mono={len(meta_mono)}"
            )

            t_npz_p = dfn_dir / f"titanet_large__dfn__scale{s}.npz"
            r_npz_p = mono_dir / f"wespeaker_resnet293_lm__16k_mono__scale{s}.npz"
            assert t_npz_p.exists(), f"Missing TitaNet NPZ: {t_npz_p}"
            assert r_npz_p.exists(), f"Missing ResNet293 NPZ: {r_npz_p}"

            t_npz = np.load(t_npz_p)
            r_npz = np.load(r_npz_p)

            t_raw = t_npz["raw_embeddings"]
            r_raw = r_npz["raw_embeddings"]
            t_norm = t_npz["l2_normalized_embeddings"]
            r_norm = r_npz["l2_normalized_embeddings"]

            assert t_raw.shape == (len(meta_dfn), 192), f"TitaNet raw shape mismatch: {t_raw.shape}"
            assert r_raw.shape == (len(meta_mono), 256), f"ResNet293 raw shape mismatch: {r_raw.shape}"
            assert t_norm.shape == (len(meta_dfn), 192), f"TitaNet norm shape mismatch: {t_norm.shape}"
            assert r_norm.shape == (len(meta_mono), 256), f"ResNet293 norm shape mismatch: {r_norm.shape}"

            assert np.all(np.isfinite(t_raw)), f"Non-finite values in TitaNet raw: {rec_id} scale {s}"
            assert np.all(np.isfinite(r_raw)), f"Non-finite values in ResNet293 raw: {rec_id} scale {s}"
            assert np.all(np.isfinite(t_norm)), f"Non-finite values in TitaNet norm: {rec_id} scale {s}"
            assert np.all(np.isfinite(r_norm)), f"Non-finite values in ResNet293 norm: {rec_id} scale {s}"

            # Check L2 normalization: norms should be exactly 1.0 within 1e-4
            t_norms_calc = np.linalg.norm(t_norm, axis=-1)
            r_norms_calc = np.linalg.norm(r_norm, axis=-1)
            assert np.all(np.abs(t_norms_calc - 1.0) < 1e-4), f"TitaNet not L2-normalized: {rec_id} scale {s}"
            assert np.all(np.abs(r_norms_calc - 1.0) < 1e-4), f"ResNet293 not L2-normalized: {rec_id} scale {s}"

            # Check subsegment alignment
            for idx, (md, mm) in enumerate(zip(meta_dfn, meta_mono)):
                assert md["subseg_idx"] == mm["subseg_idx"], (
                    f"Subseg idx mismatch in {rec_id} scale {s} row {idx}: {md['subseg_idx']} vs {mm['subseg_idx']}"
                )
                assert abs(md["start"] - mm["start"]) < 1e-4, (
                    f"Start time mismatch in {rec_id} scale {s} row {idx}: {md['start']} vs {mm['start']}"
                )
                assert abs(md["end"] - mm["end"]) < 1e-4, (
                    f"End time mismatch in {rec_id} scale {s} row {idx}: {md['end']} vs {mm['end']}"
                )
                assert abs(md["duration"] - mm["duration"]) < 1e-4, (
                    f"Duration mismatch in {rec_id} scale {s} row {idx}: {md['duration']} vs {mm['duration']}"
                )

            total_subsegments_checked += len(meta_dfn)

    print(f"[Stage 5A Validation] SUCCESS! Validated {total_subsegments_checked} subsegments across all 8 recordings and 5 scales.")
    return True


if __name__ == "__main__":
    try:
        validate_all()
    except Exception as e:
        print(f"[Stage 5A Validation ERROR] {e}", file=sys.stderr)
        sys.exit(1)
