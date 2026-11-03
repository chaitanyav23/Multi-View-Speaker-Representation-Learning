#!/usr/bin/env python3
"""
Stage 5B.3: Speaker-Count Estimation and Clustering Calibration.
Controlled experimental evaluation across all 8 benchmark recordings:
1. Baseline NME-SC (Automatic count from Condition J fused affinities)
2. Oracle ground-truth speaker count (Diagnostic upper-bound)
3. Conversational lower-bound (k >= 2)
4. Normalized Relative Eigengap selection
5. Silhouette-Guided Spectral selection
6. Agglomerative Hierarchical Clustering (AHC)
7. Parameter sensitivity and diagnostic sweeps for R07 and R06
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage2_baseline"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage3_dual_encoder"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage4_complementarity"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage5a" / "scripts"))

import common as p20_common
from run_stage2_baseline import compute_der_collar025_noovl
from run_stage4_complementarity import SCALES, load_cache
from fusion_models import offdiag_zscore_affinity

from nemo.collections.asr.parts.utils.offline_clustering import (
    SpeakerClustering,
    getMultiScaleCosAffinityMatrix,
    NMESC,
    estimateNumofSpeakers,
    getAffinityGraphMat,
    SpectralClustering,
)
from nemo.collections.asr.parts.utils.speaker_utils import (
    generate_cluster_labels,
    labels_to_rttmfile,
)

STAGE5B_3_DIR = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_3_speaker_count"
PRED_DIR = STAGE5B_3_DIR / "predictions"
METRICS_DIR = STAGE5B_3_DIR / "metrics"
CONFIGS_DIR = STAGE5B_3_DIR / "configs"

for d in [PRED_DIR, METRICS_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_fused_affinity(rec_id: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Loads multiscale caches and builds the frozen Condition J fused affinity matrix."""
    ms_weights_tensor = torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float32, device=device)
    t_scales, r_scales, ts_scales = [], [], []
    meta_s4 = []

    for scale_k, _, _, _ in SCALES:
        t_d, _, m_d = load_cache("dfn", rec_id, scale_k)
        _, r_d, _ = load_cache("16k_mono", rec_id, scale_k)
        t_scales.append(torch.from_numpy(t_d["l2_normalized_embeddings"]).to(device))
        r_scales.append(torch.from_numpy(r_d["l2_normalized_embeddings"]).to(device))
        ts_scales.append(torch.from_numpy(np.stack([t_d["starts"], t_d["ends"]], axis=-1).astype(np.float32)).to(device))
        if scale_k == 4:
            meta_s4 = m_d

    base_ts = ts_scales[-1]
    A_tita = getMultiScaleCosAffinityMatrix(ms_weights_tensor, t_scales, ts_scales, device)
    A_res = getMultiScaleCosAffinityMatrix(ms_weights_tensor, r_scales, ts_scales, device)
    A_zt = offdiag_zscore_affinity(A_tita)
    A_zr = offdiag_zscore_affinity(A_res)
    A_eq = 0.5 * A_zt + 0.5 * A_zr
    return A_eq, base_ts, meta_s4


def evaluate_cluster_labels(
    c_labels: np.ndarray,
    base_ts: np.ndarray,
    dfn_stem: str,
    gt_rttm_path: Path,
    out_rttm_dir: Path
) -> Dict[str, Any]:
    """Converts cluster labels to RTTM and evaluates standard DER/JER metrics."""
    out_rttm_dir.mkdir(parents=True, exist_ok=True)
    labels, _ = generate_cluster_labels(base_ts, c_labels)
    labels_to_rttmfile(labels, dfn_stem, str(out_rttm_dir))
    pred_rttm = out_rttm_dir / f"{dfn_stem}.rttm"

    diar_eval = p20_common.diarization_metrics(gt_rttm_path, pred_rttm)
    der_025 = compute_der_collar025_noovl(gt_rttm_path, pred_rttm)

    unique_lbls, counts = np.unique(c_labels, return_counts=True)
    cluster_sizes = [int(c) for c in counts]

    return {
        "est_k": len(unique_lbls),
        "cluster_sizes": cluster_sizes,
        "der_c0": float(diar_eval["der"]),
        "der_c025": float(der_025),
        "jer": float(diar_eval.get("jer") or 0.0),
        "fa": float(diar_eval["false_alarm"]),
        "miss": float(diar_eval["miss"]),
        "conf": float(diar_eval["confusion"]),
        "pred_rttm_path": str(pred_rttm),
    }


def estimate_k_relative_eigengap(affinity_mat: torch.Tensor, max_k: int = 10, cuda: bool = True) -> int:
    """Estimates k using relative eigengap: argmax_{k >= 2} (lambda_{k+1} - lambda_k) / lambda_k."""
    _, lambdas, gaps = estimateNumofSpeakers(affinity_mat, max_k, cuda=cuda)
    lam = lambdas.cpu().numpy()
    g = gaps.cpu().numpy()
    # evaluate candidate k in [2, max_k]
    # gap[k-1] corresponds to lambda_k - lambda_{k-1}
    # To find k, gap index idx = k-1 -> rel_gap = g[idx] / (lam[idx] + 1e-4)
    rel_gaps = []
    candidates = range(1, min(max_k, len(g)))  # indices 1..max_k-1 correspond to k=2..max_k
    for idx in candidates:
        rel_gaps.append(g[idx] / (lam[idx] + 1e-4))
    best_cand_idx = int(np.argmax(rel_gaps))
    best_k = candidates[best_cand_idx] + 1
    return best_k


def estimate_k_silhouette(
    affinity_mat: torch.Tensor,
    p_hat: int,
    k_range: range = range(2, 6),
    cuda: bool = True,
    device: torch.device = torch.device("cuda")
) -> int:
    """Estimates k by maximizing Silhouette score of spectral eigenvector projections."""
    aff_bin = getAffinityGraphMat(affinity_mat, p_hat)
    best_k = 2
    best_score = -1.0
    for k in k_range:
        try:
            spectral = SpectralClustering(n_clusters=k, n_random_trials=3, cuda=cuda, device=device)
            c_lbls = spectral.forward(aff_bin).cpu().numpy()
            if len(np.unique(c_lbls)) < 2:
                continue
            # use 1.0 - normalized affinity as distance matrix
            dist = 1.0 - torch.clamp(affinity_mat, 0.0, 1.0).cpu().numpy()
            np.fill_diagonal(dist, 0.0)
            score = silhouette_score(dist, c_lbls, metric="precomputed")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue
    return best_k


def run_all_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 5B.3] Initializing on device: {device}")

    manifest_p = REPO_ROOT / "research_stages" / "stage1_data_prep" / "manifests" / "recording_manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    clustering = SpeakerClustering(cuda=True if device.type == "cuda" else False)

    methods = [
        ("M1_nmesc_auto", "Original NME-SC (Automatic k)"),
        ("M2_oracle_k", "Oracle Ground-Truth Speaker Count (Diagnostic Upper-Bound)"),
        ("M3_floor_k2", "Conversational Lower-Bound (k >= 2)"),
        ("M4_rel_eigengap", "Normalized Relative Eigengap Estimation"),
        ("M5_silhouette", "Silhouette-Guided Spectral Clustering"),
        ("M6_ahc_oracle", "Agglomerative Hierarchical Clustering (Oracle k, Average Linkage)"),
    ]

    all_rows: List[Dict[str, Any]] = []

    for entry in manifest:
        rec_id = entry["rec_id"]
        dfn_stem = Path(entry["dfn_wav"]).stem
        dur_sec = float(entry["duration_sec"])
        gt_rttm_path = REPO_ROOT / entry["rttm_uniform"]
        gt_rttm_segs = p20_common.rttm_to_segments(gt_rttm_path)
        ref_k = len(set(s["speaker"] for s in gt_rttm_segs))

        print(f"\nProcessing {rec_id} (Ref k = {ref_k}, Duration = {dur_sec:.1f}s)...")
        A_eq, base_ts, meta_s4 = load_fused_affinity(rec_id, device)
        N = A_eq.shape[0]
        base_ts_np = base_ts.cpu().numpy()

        # Step 0: Run NMESC forward to obtain standard p_hat and raw est_k
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
        raw_est_k_t, p_hat_t = nmesc.forward()
        raw_est_k = int(raw_est_k_t.item())
        p_hat = int(p_hat_t.item())
        aff_graph = getAffinityGraphMat(A_eq, p_hat)

        for m_id, m_desc in methods:
            torch.manual_seed(42)
            np.random.seed(42)

            if m_id == "M1_nmesc_auto":
                # Standard forward_unit_infer
                c_lbls = clustering.forward_unit_infer(
                    mat=A_eq.to(device).float(),
                    oracle_num_speakers=-1,
                    max_num_speakers=15,
                    max_rp_threshold=0.25,
                    sparse_search_volume=30,
                    est_num_of_spk_enhanced=torch.tensor(-1),
                    fixed_thres=-1.0,
                    kmeans_random_trials=1,
                ).cpu().numpy()

            elif m_id == "M2_oracle_k":
                c_lbls = clustering.forward_unit_infer(
                    mat=A_eq.to(device).float(),
                    oracle_num_speakers=ref_k,
                    max_num_speakers=15,
                    max_rp_threshold=0.25,
                    sparse_search_volume=30,
                    est_num_of_spk_enhanced=torch.tensor(-1),
                    fixed_thres=-1.0,
                    kmeans_random_trials=1,
                ).cpu().numpy()

            elif m_id == "M3_floor_k2":
                # If raw_est_k == 1, floor to 2
                target_k = max(2, raw_est_k)
                spectral = SpectralClustering(n_clusters=target_k, n_random_trials=1, cuda=True, device=device)
                c_lbls = spectral.forward(aff_graph).cpu().numpy()

            elif m_id == "M4_rel_eigengap":
                target_k = estimate_k_relative_eigengap(aff_graph, max_k=15, cuda=True)
                spectral = SpectralClustering(n_clusters=target_k, n_random_trials=1, cuda=True, device=device)
                c_lbls = spectral.forward(aff_graph).cpu().numpy()

            elif m_id == "M5_silhouette":
                target_k = estimate_k_silhouette(A_eq, p_hat, k_range=range(2, 7), cuda=True, device=device)
                spectral = SpectralClustering(n_clusters=target_k, n_random_trials=3, cuda=True, device=device)
                c_lbls = spectral.forward(aff_graph).cpu().numpy()

            elif m_id == "M6_ahc_oracle":
                # Distance matrix from affinity
                A_np = np.clip(A_eq.cpu().numpy(), 0.0, 1.0)
                dist_np = 1.0 - A_np
                np.fill_diagonal(dist_np, 0.0)
                # Symmetrize
                dist_sym = 0.5 * (dist_np + dist_np.T)
                condensed = sch.distance.squareform(dist_sym, checks=False)
                link = sch.linkage(condensed, method="average")
                c_lbls = sch.fcluster(link, t=ref_k, criterion="maxclust") - 1

            eval_res = evaluate_cluster_labels(
                c_labels=c_lbls,
                base_ts=base_ts_np,
                dfn_stem=dfn_stem,
                gt_rttm_path=gt_rttm_path,
                out_rttm_dir=PRED_DIR / m_id / rec_id
            )

            row = {
                "rec_id": rec_id,
                "method_id": m_id,
                "method_name": m_desc,
                "duration_sec": dur_sec,
                "subsegments_N": N,
                "ref_k": ref_k,
                "est_k": eval_res["est_k"],
                "spk_count_err_abs": abs(eval_res["est_k"] - ref_k),
                "spk_count_signed_bias": eval_res["est_k"] - ref_k,
                "single_spk_collapse": eval_res["est_k"] == 1,
                "cluster_sizes": str(eval_res["cluster_sizes"]),
                "der_c0": eval_res["der_c0"],
                "der_c025": eval_res["der_c025"],
                "jer": eval_res["jer"],
                "false_alarm": eval_res["fa"],
                "miss": eval_res["miss"],
                "confusion": eval_res["conf"],
            }
            all_rows.append(row)
            print(f"  [{m_id}] est_k={eval_res['est_k']} (err={row['spk_count_signed_bias']:+d}) | DER(c=0)={eval_res['der_c0']*100:5.2f}% | DER(c=0.25)={eval_res['der_c025']*100:5.2f}% | Conf={eval_res['conf']*100:5.2f}% | sizes={eval_res['cluster_sizes']}")

    # Save detailed per-recording results CSV
    results_csv_p = STAGE5B_3_DIR / "STAGE5B_3_ALL_RECORDINGS_RESULTS.csv"
    with results_csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[Stage 5B.3] Saved per-recording results to: {results_csv_p}")

    # Generate Summary Table across methods
    summary_rows = []
    for m_id, m_desc in methods:
        m_subset = [r for r in all_rows if r["method_id"] == m_id]
        total_dur = sum(r["duration_sec"] for r in m_subset)
        summary_rows.append({
            "method_id": m_id,
            "method_name": m_desc,
            "macro_der_c0": float(np.mean([r["der_c0"] for r in m_subset])),
            "dur_weighted_der_c0": float(sum(r["der_c0"] * r["duration_sec"] for r in m_subset) / total_dur),
            "macro_der_c025": float(np.mean([r["der_c025"] for r in m_subset])),
            "dur_weighted_der_c025": float(sum(r["der_c025"] * r["duration_sec"] for r in m_subset) / total_dur),
            "macro_jer": float(np.mean([r["jer"] for r in m_subset])),
            "macro_fa": float(np.mean([r["false_alarm"] for r in m_subset])),
            "macro_miss": float(np.mean([r["miss"] for r in m_subset])),
            "macro_conf": float(np.mean([r["confusion"] for r in m_subset])),
            "mean_spk_count_mae": float(np.mean([r["spk_count_err_abs"] for r in m_subset])),
            "mean_spk_count_bias": float(np.mean([r["spk_count_signed_bias"] for r in m_subset])),
            "single_spk_collapse_count": sum(1 for r in m_subset if r["single_spk_collapse"]),
        })

    summary_csv_p = METRICS_DIR / "stage5b_3_summary_comparison.csv"
    with summary_csv_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[Stage 5B.3] Saved summary comparison to: {summary_csv_p}")

    # Print Summary Table
    print("\n" + "=" * 95)
    print("STAGE 5B.3 SUMMARY EVALUATION ACROSS ALL 8 RECORDINGS")
    print("=" * 95)
    print(f"{'Method ID':16s} | {'Macro DER(c=0)':14s} | {'Macro DER(c=0.25)':17s} | {'Macro Conf':10s} | {'Macro JER':10s} | {'Spk MAE':7s} | {'Collapse':8s}")
    print("-" * 95)
    for s in summary_rows:
        print(f"{s['method_id']:16s} | {s['macro_der_c0']*100:13.2f}% | {s['macro_der_c025']*100:16.2f}% | {s['macro_conf']*100:9.2f}% | {s['macro_jer']*100:9.2f}% | {s['mean_spk_count_mae']:7.2f} | {s['single_spk_collapse_count']:8d}")
    print("=" * 95)


if __name__ == "__main__":
    run_all_experiments()
