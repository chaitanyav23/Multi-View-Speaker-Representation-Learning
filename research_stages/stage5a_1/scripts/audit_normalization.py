#!/usr/bin/env python3
"""
Stage 5A.1 Normalization Audit:
Audits offdiag_zscore_affinity mathematical properties on synthetic matrices
and actual cached multiscale affinity matrices across all 8 recordings.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage4_complementarity"))

from run_stage4_complementarity import SCALES, load_cache  # noqa: E402
from nemo.collections.asr.parts.utils.offline_clustering import (  # noqa: E402
    ScalerMinMax,
    getMultiScaleCosAffinityMatrix,
)


def offdiag_zscore_affinity(A: torch.Tensor) -> torch.Tensor:
    """
    Z-score standardizes off-diagonal entries of an affinity matrix A and maps to [0, 1].
    Diagonal entries are fixed to 1.0.
    """
    n = A.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=A.device)
    off = A[mask]
    mu = torch.mean(off)
    sd = torch.clamp(torch.std(off), min=1e-6)
    Z = (A - mu) / sd
    Z_sig = torch.sigmoid(Z)
    Z_sig.fill_diagonal_(1.0)
    return ScalerMinMax(Z_sig)


def run_synthetic_tests() -> Dict[str, Any]:
    tests = {}
    device = torch.device("cpu")

    # Test 1: Standard symmetric matrix (4x4)
    A1 = torch.tensor([
        [1.0, 0.8, 0.2, 0.1],
        [0.8, 1.0, 0.3, 0.0],
        [0.2, 0.3, 1.0, 0.9],
        [0.1, 0.0, 0.9, 1.0],
    ], device=device)
    out1 = offdiag_zscore_affinity(A1)
    tests["test1_standard_symmetric"] = {
        "input_shape": list(A1.shape),
        "output_shape": list(out1.shape),
        "is_symmetric": bool(torch.allclose(out1, out1.T, atol=1e-6)),
        "diagonal_values": [round(float(v), 6) for v in torch.diagonal(out1)],
        "min_val": round(float(out1.min()), 6),
        "max_val": round(float(out1.max()), 6),
        "min_offdiag": round(float(out1[~torch.eye(4, dtype=torch.bool)].min()), 6),
        "max_offdiag": round(float(out1[~torch.eye(4, dtype=torch.bool)].max()), 6),
        "has_nan": bool(torch.isnan(out1).any()),
        "has_inf": bool(torch.isinf(out1).any()),
    }

    # Test 2: Constant off-diagonal matrix (all off-diag identical)
    A2 = torch.tensor([
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0],
    ], device=device)
    out2 = offdiag_zscore_affinity(A2)
    tests["test2_constant_offdiag"] = {
        "diagonal_values": [round(float(v), 6) for v in torch.diagonal(out2)],
        "offdiag_values": [round(float(v), 6) for v in out2[~torch.eye(3, dtype=torch.bool)]],
        "is_symmetric": bool(torch.allclose(out2, out2.T, atol=1e-6)),
        "has_nan": bool(torch.isnan(out2).any()),
    }

    # Test 3: Matrix with near-zero std off-diagonal (< 1e-6)
    A3 = torch.tensor([
        [1.0, 0.5 + 1e-8, 0.5],
        [0.5 + 1e-8, 1.0, 0.5 - 1e-8],
        [0.5, 0.5 - 1e-8, 1.0],
    ], device=device)
    out3 = offdiag_zscore_affinity(A3)
    tests["test3_near_zero_std"] = {
        "is_symmetric": bool(torch.allclose(out3, out3.T, atol=1e-6)),
        "has_nan": bool(torch.isnan(out3).any()),
        "max_val": round(float(out3.max()), 6),
        "min_val": round(float(out3.min()), 6),
    }

    # Test 4: Matrix with negative off-diagonal entries
    A4 = torch.tensor([
        [1.0, -0.7, 0.4],
        [-0.7, 1.0, -0.2],
        [0.4, -0.2, 1.0],
    ], device=device)
    out4 = offdiag_zscore_affinity(A4)
    tests["test4_negative_offdiag"] = {
        "is_symmetric": bool(torch.allclose(out4, out4.T, atol=1e-6)),
        "min_val": round(float(out4.min()), 6),
        "max_val": round(float(out4.max()), 6),
        "diagonal_values": [round(float(v), 6) for v in torch.diagonal(out4)],
    }

    # Test 5: Asymmetric input matrix (checking whether symmetrization occurs)
    A5 = torch.tensor([
        [1.0, 0.9, 0.2],
        [0.1, 1.0, 0.3],
        [0.4, 0.8, 1.0],
    ], device=device)
    out5 = offdiag_zscore_affinity(A5)
    tests["test5_asymmetric_input"] = {
        "input_is_symmetric": bool(torch.allclose(A5, A5.T, atol=1e-6)),
        "output_is_symmetric": bool(torch.allclose(out5, out5.T, atol=1e-6)),
        "conclusion": "Symmetrization is NOT performed; output preserves asymmetry if input is asymmetric.",
    }

    return tests


def run_cached_affinity_audit() -> List[Dict[str, Any]]:
    config_p = REPO_ROOT / "research_stages" / "stage5a" / "configs" / "stage5a_config.json"
    cfg = json.loads(config_p.read_text(encoding="utf-8"))
    manifest = json.loads((REPO_ROOT / cfg["paths"]["manifest"]).read_text(encoding="utf-8"))

    ms_weights = torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float32)
    results = []

    for entry in manifest:
        rec_id = entry["rec_id"]

        t_scales, r_scales, ts_scales = [], [], []
        for sk, _, _, _ in SCALES:
            t_d, _, _ = load_cache("dfn", rec_id, sk)
            _, r_d, _ = load_cache("16k_mono", rec_id, sk)
            t_scales.append(torch.from_numpy(t_d["l2_normalized_embeddings"]))
            r_scales.append(torch.from_numpy(r_d["l2_normalized_embeddings"]))
            ts_scales.append(torch.from_numpy(np.stack([t_d["starts"], t_d["ends"]], axis=-1).astype(np.float32)))

        A_tita = getMultiScaleCosAffinityMatrix(ms_weights, t_scales, ts_scales, torch.device("cpu"))
        A_res = getMultiScaleCosAffinityMatrix(ms_weights, r_scales, ts_scales, torch.device("cpu"))

        A_zt = offdiag_zscore_affinity(A_tita)
        A_zr = offdiag_zscore_affinity(A_res)

        n = A_tita.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)

        def mat_stats(mat: torch.Tensor, name: str) -> Dict[str, Any]:
            off = mat[mask].numpy()
            diag = torch.diagonal(mat).numpy()
            sym_diff = float(torch.max(torch.abs(mat - mat.T)))
            return {
                f"{name}_diag_mean": round(float(np.mean(diag)), 4),
                f"{name}_diag_min": round(float(np.min(diag)), 4),
                f"{name}_diag_max": round(float(np.max(diag)), 4),
                f"{name}_offdiag_mean": round(float(np.mean(off)), 4),
                f"{name}_offdiag_std": round(float(np.std(off)), 4),
                f"{name}_offdiag_min": round(float(np.min(off)), 4),
                f"{name}_offdiag_max": round(float(np.max(off)), 4),
                f"{name}_sym_max_diff": round(sym_diff, 8),
            }

        rec_stat = {
            "rec_id": rec_id,
            "n_subsegments": n,
            **mat_stats(A_tita, "A_tita_raw"),
            **mat_stats(A_res, "A_res_raw"),
            **mat_stats(A_zt, "A_zt_norm"),
            **mat_stats(A_zr, "A_zr_norm"),
        }
        results.append(rec_stat)

    return results


def main():
    print("[Normalization Audit] Running synthetic matrix tests...")
    syn_tests = run_synthetic_tests()

    print("[Normalization Audit] Auditing actual cached multiscale matrices...")
    cached_audit = run_cached_affinity_audit()

    out_data = {
        "synthetic_tests": syn_tests,
        "cached_affinity_audit": cached_audit,
        "protocol_classification": {
            "normalization_type": "Unsupervised, per-recording transductive normalization",
            "uses_ground_truth_labels": False,
            "uses_full_test_recording_affinity_distribution": True,
            "inductive_status": "Transductive at recording level; parameters (mean, std, min, max) estimated from test session off-diagonal similarities without label awareness.",
        }
    }

    out_dir = REPO_ROOT / "research_stages" / "stage5a_1" / "normalization_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "normalization_audit_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"[Normalization Audit] Completed! Saved report to {out_file}")


if __name__ == "__main__":
    main()
