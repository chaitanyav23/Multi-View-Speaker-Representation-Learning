#!/usr/bin/env python3
"""
Stage 5B.3: Parameter Sensitivity Analysis for R07 and R06.
Sweeps:
1. Candidate k from 1 to 10
2. Sparsification neighborhood parameter p across range
3. Random trials in k-means (1, 5, 10, 20)
4. Graph connectivity, degree distribution, and sparsity
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage2_baseline"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage3_dual_encoder"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage4_complementarity"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage5a" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage5b" / "stage5b_3_speaker_count" / "scripts"))

import common as p20_common
from run_stage2_baseline import compute_der_collar025_noovl
from run_stage5b_3_experiments import load_fused_affinity, evaluate_cluster_labels
from nemo.collections.asr.parts.utils.offline_clustering import (
    SpeakerClustering,
    getAffinityGraphMat,
    SpectralClustering,
    estimateNumofSpeakers,
    NMESC,
)

STAGE5B_3_DIR = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_3_speaker_count"
METRICS_DIR = STAGE5B_3_DIR / "metrics"
TMP_DIR = Path("/tmp/stage5b_3_sensitivity")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def analyze_graph_properties(A_mat: torch.Tensor, p_vals: List[int]) -> List[Dict[str, Any]]:
    props = []
    N = A_mat.shape[0]
    for p in p_vals:
        aff_bin = getAffinityGraphMat(A_mat, p)
        W = 0.5 * (aff_bin + aff_bin.T)
        W.fill_diagonal_(0.0)
        degrees = torch.sum(W > 0, dim=1).float().cpu().numpy()
        total_edges = int(torch.sum(W > 0).item() / 2)
        density = (2.0 * total_edges) / (N * (N - 1)) if N > 1 else 0.0
        isolated_nodes = int(np.sum(degrees == 0))
        props.append({
            "p": p,
            "density": density,
            "mean_degree": float(np.mean(degrees)),
            "min_degree": float(np.min(degrees)),
            "max_degree": float(np.max(degrees)),
            "isolated_nodes": isolated_nodes,
        })
    return props


def run_sensitivity_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sensitivity] Running on device: {device}")

    manifest_p = REPO_ROOT / "research_stages" / "stage1_data_prep" / "manifests" / "recording_manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    results = {}

    for rec_id in ["R07_bwc_cheap_shot", "R06_rich_karen"]:
        entry = [e for e in manifest if e["rec_id"] == rec_id][0]
        dfn_stem = Path(entry["dfn_wav"]).stem
        gt_rttm_path = REPO_ROOT / entry["rttm_uniform"]
        gt_rttm_segs = p20_common.rttm_to_segments(gt_rttm_path)
        ref_k = len(set(s["speaker"] for s in gt_rttm_segs))

        A_eq, base_ts, _ = load_fused_affinity(rec_id, device)
        base_ts_np = base_ts.cpu().numpy()
        N = A_eq.shape[0]

        print(f"\n--- Sensitivity Analysis for {rec_id} (N={N}, Ref k={ref_k}) ---")

        # 1. Sweep k from 1 to 10
        k_sweep = []
        nmesc = NMESC(
            A_eq.to(device).float(),
            max_num_speakers=15,
            max_rp_threshold=0.25,
            sparse_search=True,
            sparse_search_volume=30,
            nme_mat_size=512,
            cuda=True,
            device=device
        )
        _, p_hat_t = nmesc.forward()
        p_hat = int(p_hat_t.item())
        aff_graph = getAffinityGraphMat(A_eq, p_hat)

        for k in range(1, 11):
            torch.manual_seed(42)
            np.random.seed(42)
            if k == 1:
                c_lbls = np.zeros(N, dtype=np.int64)
            else:
                spectral = SpectralClustering(n_clusters=k, n_random_trials=5, cuda=True, device=device)
                c_lbls = spectral.forward(aff_graph).cpu().numpy()

            eval_res = evaluate_cluster_labels(
                c_labels=c_lbls,
                base_ts=base_ts_np,
                dfn_stem=dfn_stem,
                gt_rttm_path=gt_rttm_path,
                out_rttm_dir=TMP_DIR / f"{rec_id}_k_{k}"
            )
            k_sweep.append({
                "k": k,
                "der_c0": eval_res["der_c0"],
                "der_c025": eval_res["der_c025"],
                "jer": eval_res["jer"],
                "conf": eval_res["conf"],
                "cluster_sizes": eval_res["cluster_sizes"]
            })
            print(f"  k={k:2d}: DER(c=0)={eval_res['der_c0']*100:5.2f}% | DER(c=0.25)={eval_res['der_c025']*100:5.2f}% | Conf={eval_res['conf']*100:5.2f}% | sizes={eval_res['cluster_sizes']}")

        # 2. Graph Properties across p
        p_candidates = [2, 5, 10, 15, 25, 40, 60, 80, 100]
        p_candidates = [p for p in p_candidates if p < N // 2]
        graph_props = analyze_graph_properties(A_eq, p_candidates)

        # 3. K-means random trials stability (k=ref_k)
        trial_sweep = []
        for trials in [1, 5, 10, 20]:
            der_vals = []
            for trial_seed in [42, 100, 2024]:
                torch.manual_seed(trial_seed)
                np.random.seed(trial_seed)
                spectral = SpectralClustering(n_clusters=ref_k, n_random_trials=trials, cuda=True, device=device)
                c_lbls = spectral.forward(aff_graph).cpu().numpy()
                eval_res = evaluate_cluster_labels(
                    c_labels=c_lbls,
                    base_ts=base_ts_np,
                    dfn_stem=dfn_stem,
                    gt_rttm_path=gt_rttm_path,
                    out_rttm_dir=TMP_DIR / f"{rec_id}_trials_{trials}_{trial_seed}"
                )
                der_vals.append(eval_res["der_c025"])
            trial_sweep.append({
                "trials": trials,
                "mean_der_c025": float(np.mean(der_vals)),
                "std_der_c025": float(np.std(der_vals)),
            })
            print(f"  Trials={trials:2d}: mean DER(c=0.25)={np.mean(der_vals)*100:5.2f}% +/- {np.std(der_vals)*100:4.2f}%")

        results[rec_id] = {
            "p_hat": p_hat,
            "k_sweep": k_sweep,
            "graph_props": graph_props,
            "trial_sweep": trial_sweep,
        }

    out_json = METRICS_DIR / "stage5b_3_sensitivity_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Sensitivity] Saved sensitivity results to: {out_json}")


if __name__ == "__main__":
    run_sensitivity_sweep()
