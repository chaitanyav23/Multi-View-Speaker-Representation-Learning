#!/usr/bin/env python3
"""
Stage 4 — Comprehensive Complementarity Verification & Mandatory Decision Gate Analysis.

Evaluates:
1. Linear CKA (standard & debiased) and cross-view alignment across all 5 scales and 3 audio conditions,
   plus cross-condition self-consistency (16k_mono vs dfn vs normalized) for each encoder.
2. Out-of-fold 8-fold LORO Orthogonal Procrustes alignment residuals (R^2_train vs R^2_test,
   Frobenius residual, and held-out cosine alignment) fitted strictly on training recordings.
3. Speaker discriminability metrics (within/between-speaker cosine similarity, EER, minDCF,
   FDR, Cosine Silhouette) evaluated BOTH on non-overlapping ground-truth turns (preventing
   sliding-window autocorrelation inflation) AND on per-recording high-purity subsegments.
4. Pairwise verification 2x2 error contingency tables, error overlap ratio, Yule's Q,
   and Oracle-Gate upper bound.
5. Stratified performance across segment duration/scale (1.5s..0.5s), overlap status,
   local SNR quartiles, DFN3 noise-reduction quartiles, and audio preprocessing conditions.
6. Predictive value (Spearman rho and error-prediction AUROC) of candidate acoustic & view
   reliability indicators for predicting per-encoder margin and differential encoder advantage.
7. Downstream diarization & word-attribution evaluation across all 8 recordings comparing:
   - Single-Encoder TitaNet-Large (192-D)
   - Single-Encoder WeSpeaker ResNet293-LM (256-D)
   - Zero-Training Fusion 1: Naive L2 Concatenation (448-D)
   - Zero-Training Fusion 2: Equal Multiscale Cosine Affinity Fusion (0.5 * A_tita + 0.5 * A_res)
   - Zero-Training Fusion 3: Z-Score Standardized Multiscale Cosine Affinity Fusion
   - Zero-Training Fusion 4: Heuristic Quality-Weighted Affinity Fusion (Norm + Cross-Scale Consistency)
   - LORO Feature Fusion: Out-of-fold Procrustes Aligned Average (192-D)
   - LORO Calibrated Gate Affinity Fusion: Out-of-fold ridge-calibrated reliability gate
"""

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["XDG_CACHE_HOME"] = "/home/priyanka/.cache"
os.environ["HF_HOME"] = "/home/priyanka/.cache/huggingface"
os.environ["TORCH_HOME"] = "/home/priyanka/.cache/torch"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import numpy as np
import scipy.linalg
import scipy.stats
import torch
from sklearn.metrics import roc_auc_score, silhouette_score

REPO_ROOT = Path("/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning")
STAGE1_DIR = REPO_ROOT / "research_stages" / "stage1_data_prep"
STAGE2_DIR = REPO_ROOT / "research_stages" / "stage2_baseline"
STAGE3_DIR = REPO_ROOT / "research_stages" / "stage3_dual_encoder"
STAGE4_DIR = REPO_ROOT / "research_stages" / "stage4_complementarity"

sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
sys.path.insert(0, str(STAGE2_DIR))

import common as p20_common  # noqa: E402
from run_stage2_baseline import compute_der_collar025_noovl, compute_macro_sacc  # noqa: E402
from nemo.collections.asr.parts.utils.offline_clustering import (  # noqa: E402
    ScalerMinMax,
    SpeakerClustering,
    getMultiScaleCosAffinityMatrix,
)
from nemo.collections.asr.parts.utils.speaker_utils import (  # noqa: E402
    generate_cluster_labels,
    labels_to_rttmfile,
)

SCALES = [
    (0, 1.50, 0.75, 1.0),
    (1, 1.20, 0.60, 2.0),
    (2, 1.00, 0.50, 3.0),
    (3, 0.80, 0.40, 3.0),
    (4, 0.50, 0.25, 3.0),
]
AUDIO_CONDITIONS = ["16k_mono", "dfn", "normalized"]


# ============================================================================
# 1. CKA, Procrustes, EER, minDCF, FDR, Silhouette Helpers
# ============================================================================

def linear_cka(X: np.ndarray, Y: np.ndarray, debiased: bool = False) -> float:
    """Computes standard or debiased Linear CKA between X [N, D1] and Y [N, D2]."""
    if X.shape[0] < 4:
        return float("nan")
    Xc = X - np.mean(X, axis=0, keepdims=True)
    Yc = Y - np.mean(Y, axis=0, keepdims=True)
    if not debiased:
        hsic_xy = float(np.sum((Xc.T @ Yc) ** 2))
        hsic_xx = float(np.sum((Xc.T @ Xc) ** 2))
        hsic_yy = float(np.sum((Yc.T @ Yc) ** 2))
        denom = math.sqrt(max(hsic_xx * hsic_yy, 1e-20))
        return hsic_xy / denom
    else:
        n = X.shape[0]
        K = Xc @ Xc.T
        L = Yc @ Yc.T
        np.fill_diagonal(K, 0.0)
        np.fill_diagonal(L, 0.0)

        def _hsic_unbiased(A: np.ndarray, B: np.ndarray) -> float:
            sum_A = np.sum(A, axis=1)
            sum_B = np.sum(B, axis=1)
            t1 = float(np.sum(A * B))
            t2 = float(np.sum(sum_A) * np.sum(sum_B)) / ((n - 1) * (n - 2))
            t3 = 2.0 * float(np.sum(sum_A * sum_B)) / (n - 2)
            return (t1 + t2 - t3) / (n * (n - 3))

        xy = _hsic_unbiased(K, L)
        xx = _hsic_unbiased(K, K)
        yy = _hsic_unbiased(L, L)
        if xx <= 0 or yy <= 0:
            return float("nan")
        return xy / math.sqrt(xx * yy)


def fit_orthogonal_procrustes(X_src: np.ndarray, Y_tgt: np.ndarray) -> np.ndarray:
    """
    Fits semi-orthogonal Procrustes matrix P in R^{d_src x d_tgt} (P^T P = I_{d_tgt})
    minimizing ||X_src @ P - Y_tgt||_F^2.
    For d_src=256, d_tgt=192: SVD(X_src^T @ Y_tgt) = U S V^T -> P = U[:, :192] @ V^T.
    """
    M = X_src.T @ Y_tgt  # [256, 192]
    U, _, Vt = scipy.linalg.svd(M, full_matrices=False)  # U: [256, 192], Vt: [192, 192]
    P = U @ Vt  # [256, 192], satisfies P.T @ P = I_192
    return P.astype(np.float32)


def procrustes_metrics(X_src: np.ndarray, Y_tgt: np.ndarray, P: np.ndarray) -> Dict[str, float]:
    proj = X_src @ P  # [N, 192]
    diff = proj - Y_tgt
    frob_sq = float(np.sum(diff ** 2))
    tgt_var = float(np.sum((Y_tgt - np.mean(Y_tgt, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - (frob_sq / max(tgt_var, 1e-12))
    proj_norm = proj / np.maximum(np.linalg.norm(proj, axis=-1, keepdims=True), 1e-12)
    tgt_norm = Y_tgt / np.maximum(np.linalg.norm(Y_tgt, axis=-1, keepdims=True), 1e-12)
    mean_cos = float(np.mean(np.sum(proj_norm * tgt_norm, axis=-1)))
    return {
        "r2_explained": round(r2, 4),
        "frobenius_rms_per_sample": round(math.sqrt(frob_sq / max(X_src.shape[0], 1)), 4),
        "mean_cosine_alignment": round(mean_cos, 4),
    }


def compute_eer_and_mindcf(
    scores: np.ndarray, labels: np.ndarray, p_target: float = 0.01, c_miss: float = 1.0, c_fa: float = 1.0
) -> Dict[str, Any]:
    """
    Computes exact Equal Error Rate (EER), EER threshold, and minDCF from pairwise
    similarity scores and binary labels (1=same speaker, 0=different speaker).
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return {"eer": None, "threshold": None, "mindcf": None, "n_pos": n_pos, "n_neg": n_neg}

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # At threshold t = sorted_scores[k], items < t are rejected (predicted 0)
    cum_pos_rejected = np.cumsum(sorted_labels == 1)
    cum_neg_rejected = np.cumsum(sorted_labels == 0)

    fnr = np.concatenate([[0.0], cum_pos_rejected / float(n_pos)])
    fpr = np.concatenate([[1.0], 1.0 - (cum_neg_rejected / float(n_neg))])
    thresholds = np.concatenate([[sorted_scores[0] - 1e-6], sorted_scores])

    diff = np.abs(fnr - fpr)
    best_idx = int(np.argmin(diff))
    eer = 0.5 * float(fnr[best_idx] + fpr[best_idx])
    eer_thr = float(thresholds[best_idx])

    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    min_dcf_raw = float(np.min(dcf))
    min_dcf_norm = min_dcf_raw / min(c_miss * p_target, c_fa * (1.0 - p_target))

    return {
        "eer": round(eer, 4),
        "threshold": round(eer_thr, 4),
        "mindcf_p001": round(min_dcf_norm, 4),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def compute_fdr_and_silhouette(embs_norm: np.ndarray, spk_labels: List[str]) -> Dict[str, Optional[float]]:
    unique_spks = sorted(set(spk_labels))
    counts = {s: spk_labels.count(s) for s in unique_spks}
    multi_spks = [s for s, c in counts.items() if c >= 2]
    if len(multi_spks) < 2:
        return {"fdr": None, "cosine_silhouette": None}

    mask = np.array([s in multi_spks for s in spk_labels], dtype=bool)
    X = embs_norm[mask]
    Y = [s for s in spk_labels if s in multi_spks]

    # Cosine matrix
    sim = np.clip(X @ X.T, -1.0, 1.0)
    n = X.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    pair_sims = sim[tri_i, tri_j]
    pair_same = np.array([1 if Y[i] == Y[j] else 0 for i, j in zip(tri_i, tri_j)], dtype=np.int32)
    pos_sims = pair_sims[pair_same == 1]
    neg_sims = pair_sims[pair_same == 0]
    if pos_sims.size == 0 or neg_sims.size == 0:
        return {"fdr": None, "cosine_silhouette": None}

    fdr = float(((np.mean(pos_sims) - np.mean(neg_sims)) ** 2) / (np.var(pos_sims) + np.var(neg_sims) + 1e-12))
    try:
        sil = float(silhouette_score(X, Y, metric="cosine"))
    except Exception:
        sil = None
    return {
        "fdr": round(fdr, 4),
        "cosine_silhouette": round(sil, 4) if sil is not None else None,
        "within_spk_cos_mean": round(float(np.mean(pos_sims)), 4),
        "within_spk_cos_std": round(float(np.std(pos_sims)), 4),
        "between_spk_cos_mean": round(float(np.mean(neg_sims)), 4),
        "between_spk_cos_std": round(float(np.std(neg_sims)), 4),
    }


def load_cache(cond: str, rec_id: str, scale_k: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    cond_dir = STAGE3_DIR / "cache" / cond / rec_id
    t_npz = np.load(str(cond_dir / f"titanet_large__{cond}__scale{scale_k}.npz"))
    r_npz = np.load(str(cond_dir / f"wespeaker_resnet293_lm__{cond}__scale{scale_k}.npz"))
    meta = json.loads((cond_dir / f"subsegments_metadata__scale{scale_k}.json").read_text(encoding="utf-8"))["subsegments"]
    return dict(t_npz), dict(r_npz), meta


def build_non_overlapping_turn_dataset(
    manifest: List[Dict[str, Any]], cond: str, scale_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Pools L2-normalized subsegment embeddings within each non-overlapping ground-truth turn
    (usable_for_clean_speaker_eval=True) so each ground-truth turn is represented by a single
    vector, completely eliminating sliding-window overlap autocorrelation between pairs!
    """
    turns: List[Dict[str, Any]] = []
    for entry in manifest:
        rec_id = entry["rec_id"]
        gt_segs = json.loads((REPO_ROOT / entry["json_uniform"]).read_text(encoding="utf-8"))["segments"]
        t_data, r_data, sub_meta = load_cache(cond, rec_id, scale_k)
        t_norm = t_data["l2_normalized_embeddings"]
        r_norm = r_data["l2_normalized_embeddings"]
        sub_st = np.array([m["start"] for m in sub_meta], dtype=np.float32)
        sub_en = np.array([m["end"] for m in sub_meta], dtype=np.float32)

        for g in gt_segs:
            if not g.get("usable_for_clean_speaker_eval", False):
                continue
            gst = float(g["start"])
            gen = float(g["end_clipped_to_audio"])
            if gen - gst < 0.5:
                continue
            # Find subsegments whose interval lies predominantly (>=75%) inside this GT turn
            ov = np.minimum(sub_en, gen) - np.maximum(sub_st, gst)
            sub_dur = np.maximum(sub_en - sub_st, 1e-6)
            idxs = np.where((ov / sub_dur) >= 0.75)[0]
            if idxs.size == 0:
                continue
            t_vec = np.mean(t_norm[idxs], axis=0)
            t_vec = (t_vec / max(float(np.linalg.norm(t_vec)), 1e-12)).astype(np.float32)
            r_vec = np.mean(r_norm[idxs], axis=0)
            r_vec = (r_vec / max(float(np.linalg.norm(r_vec)), 1e-12)).astype(np.float32)

            turns.append({
                "rec_id": rec_id,
                "seg_idx": g["seg_idx"],
                "speaker": g["speaker"],
                "global_speaker_id": g["global_speaker_id"],
                "start": gst,
                "end": gen,
                "duration": round(gen - gst, 3),
                "num_subsegments_pooled": int(idxs.size),
                "snr_local_db": float(np.mean([sub_meta[i]["snr_local_db"] for i in idxs])),
                "dfn_noise_reduction_db": float(np.mean([sub_meta[i]["dfn_noise_reduction_db"] for i in idxs])),
                "vad_conf_mean": float(np.mean([sub_meta[i]["vad_conf_mean"] for i in idxs])),
                "overlap_prob_mean": float(np.mean([sub_meta[i]["overlap_prob_mean"] for i in idxs])),
                "titanet_l2_norm": float(np.mean([sub_meta[i]["titanet_l2_norm"] for i in idxs])),
                "resnet293_l2_norm": float(np.mean([sub_meta[i]["resnet293_l2_norm"] for i in idxs])),
                "titanet_cross_scale_cos": float(np.mean([sub_meta[i]["titanet_cross_scale_cos"] for i in idxs])),
                "resnet293_cross_scale_cos": float(np.mean([sub_meta[i]["resnet293_cross_scale_cos"] for i in idxs])),
                "tita_emb": t_vec,
                "res_emb": r_vec,
            })
    return turns


def evaluate_turn_pairs_and_contingency(turns: List[Dict[str, Any]], within_recording_only: bool = True) -> Dict[str, Any]:
    """
    Evaluates pairwise speaker verification on non-overlapping ground-truth turns,
    computing EER, minDCF, 2x2 error contingency table, error overlap, Yule's Q,
    and Oracle-Gate upper bound.
    """
    n = len(turns)
    t_scores = []
    r_scores = []
    labels = []
    rec_ids = []

    for i in range(n):
        for j in range(i + 1, n):
            if within_recording_only and turns[i]["rec_id"] != turns[j]["rec_id"]:
                continue
            same = 1 if turns[i]["global_speaker_id"] == turns[j]["global_speaker_id"] else 0
            ts = float(np.dot(turns[i]["tita_emb"], turns[j]["tita_emb"]))
            rs = float(np.dot(turns[i]["res_emb"], turns[j]["res_emb"]))
            t_scores.append(ts)
            r_scores.append(rs)
            labels.append(same)
            rec_ids.append(turns[i]["rec_id"])

    t_arr = np.array(t_scores, dtype=np.float64)
    r_arr = np.array(r_scores, dtype=np.float64)
    y_arr = np.array(labels, dtype=np.int32)

    t_eval = compute_eer_and_mindcf(t_arr, y_arr)
    r_eval = compute_eer_and_mindcf(r_arr, y_arr)

    # Zero-training score fusions on turn pairs
    eq_arr = 0.5 * t_arr + 0.5 * r_arr
    t_z = (t_arr - np.mean(t_arr)) / max(float(np.std(t_arr)), 1e-12)
    r_z = (r_arr - np.mean(r_arr)) / max(float(np.std(r_arr)), 1e-12)
    z_arr = 0.5 * t_z + 0.5 * r_z

    eq_eval = compute_eer_and_mindcf(eq_arr, y_arr)
    z_eval = compute_eer_and_mindcf(z_arr, y_arr)

    # 2x2 Error Contingency Table at each encoder's EER threshold
    t_thr = t_eval["threshold"]
    r_thr = r_eval["threshold"]
    t_pred = (t_arr >= t_thr).astype(np.int32)
    r_pred = (r_arr >= r_thr).astype(np.int32)
    t_err = (t_pred != y_arr)
    r_err = (r_pred != y_arr)

    n00 = int(np.sum((~t_err) & (~r_err)))  # Both correct
    n10 = int(np.sum(t_err & (~r_err)))     # TitaNet wrong, ResNet293 right
    n01 = int(np.sum((~t_err) & r_err))     # TitaNet right, ResNet293 wrong
    n11 = int(np.sum(t_err & r_err))        # Both wrong
    total_pairs = len(y_arr)
    union_err = n11 + n10 + n01
    err_overlap = float(n11 / max(union_err, 1))
    yules_q = float((n00 * n11 - n10 * n01) / max(n00 * n11 + n10 * n01, 1))

    # Oracle gate: selects the encoder that has the larger signed margin toward the ground-truth class
    t_signed_margin = np.where(y_arr == 1, t_z - float((t_thr - np.mean(t_arr)) / np.std(t_arr)), float((t_thr - np.mean(t_arr)) / np.std(t_arr)) - t_z)
    r_signed_margin = np.where(y_arr == 1, r_z - float((r_thr - np.mean(r_arr)) / np.std(r_arr)), float((r_thr - np.mean(r_arr)) / np.std(r_arr)) - r_z)
    choose_res = r_signed_margin > t_signed_margin
    oracle_scores = np.where(choose_res, r_z, t_z)
    oracle_eval = compute_eer_and_mindcf(oracle_scores, y_arr)
    oracle_err_rate = round(float(n11 / max(total_pairs, 1)), 4)

    return {
        "num_clean_turns": n,
        "total_pairs": total_pairs,
        "positive_pairs": int(np.sum(y_arr == 1)),
        "negative_pairs": int(np.sum(y_arr == 0)),
        "titanet": t_eval,
        "resnet293": r_eval,
        "equal_cosine_fusion": eq_eval,
        "zscore_cosine_fusion": z_eval,
        "oracle_gate_upper_bound": {
            **oracle_eval,
            "oracle_error_rate_at_eer_thr": oracle_err_rate,
        },
        "contingency_2x2": {
            "n00_both_correct": n00,
            "n10_tita_err_res_ok": n10,
            "n01_tita_ok_res_err": n01,
            "n11_both_err": n11,
            "error_overlap_jaccard": round(err_overlap, 4),
            "complementary_error_fraction": round(1.0 - err_overlap, 4),
            "yules_q": round(yules_q, 4),
        },
    }


# ============================================================================
# 2. Downstream Multiscale Spectral Clustering Evaluation Helper
# ============================================================================

def run_nemo_clustering_from_affinity_or_embs(
    rec_id: str,
    dfn_stem: str,
    gt_json_path: Path,
    gt_rttm_path: Path,
    aligned_words_path: Path,
    duration_sec: float,
    timestamps_in_scales: List[torch.Tensor],
    multiscale_weights: torch.Tensor,
    affinity_mat: torch.Tensor,
    out_dir: Path,
    method_name: str,
) -> Dict[str, Any]:
    """
    Runs NeMo's exact SpeakerClustering.forward_unit_infer on a [N_base, N_base]
    affinity matrix, writes the predicted RTTM via generate_cluster_labels + labels_to_rttmfile,
    and computes all diarization and speaker-attribution metrics.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    out_dir.mkdir(parents=True, exist_ok=True)

    clustering = SpeakerClustering(cuda=True)
    base_ts = timestamps_in_scales[-1]
    n_base = base_ts.shape[0]

    # Run NeMo NMESC + SpectralClustering on the affinity matrix
    cluster_labels = clustering.forward_unit_infer(
        mat=affinity_mat.to(torch.device("cuda")).float(),
        oracle_num_speakers=-1,
        max_num_speakers=10,
        max_rp_threshold=0.25,
        sparse_search_volume=30,
        est_num_of_spk_enhanced=torch.tensor(-1),
        fixed_thres=-1.0,
        kmeans_random_trials=1,
    )
    cluster_labels_np = cluster_labels.cpu().numpy()
    labels, _ = generate_cluster_labels(base_ts.cpu().numpy(), cluster_labels_np)
    rttm_dir = out_dir / method_name
    rttm_dir.mkdir(parents=True, exist_ok=True)
    labels_to_rttmfile(labels, dfn_stem, str(rttm_dir))
    pred_rttm_path = rttm_dir / f"{dfn_stem}.rttm"

    gt_payload = p20_common.read_json(gt_json_path)
    gt_segments = gt_payload["segments"]
    gt_rttm_segs = p20_common.rttm_to_segments(gt_rttm_path)
    diar_segments = p20_common.rttm_to_segments(pred_rttm_path)

    diar_eval = p20_common.diarization_metrics(gt_rttm_path, pred_rttm_path)
    der_025 = compute_der_collar025_noovl(gt_rttm_path, pred_rttm_path)

    aligned_words = p20_common.read_json(aligned_words_path)
    words_attr = p20_common.attributed_words(aligned_words, diar_segments)
    segs_attr = p20_common.group_words(words_attr)
    attr_path = rttm_dir / f"{rec_id}_{method_name}_attributed.json"
    p20_common.write_json(
        attr_path,
        {"file": f"{rec_id}.wav", "method": method_name, "words": words_attr, "segments": segs_attr},
    )
    f_metrics = p20_common.fusion_metrics(gt_json_path, attr_path, duration_sec)
    macro_info = compute_macro_sacc(gt_segments, words_attr, f_metrics.get("speaker_mapping", {}))

    ref_spks_pos = len(set(s["speaker"] for s in gt_rttm_segs))
    pred_spks = len(set(s["speaker"] for s in diar_segments))

    return {
        "rec_id": rec_id,
        "method": method_name,
        "duration_sec": round(duration_sec, 3),
        "ref_speakers_pos": ref_spks_pos,
        "pred_speakers": pred_spks,
        "speaker_count_error_abs": abs(pred_spks - ref_spks_pos),
        "speaker_count_signed_bias": pred_spks - ref_spks_pos,
        "single_speaker_collapse": pred_spks == 1,
        "der_collar0_ovl": round(float(diar_eval["der"]), 4),
        "der_collar025_noovl": round(der_025, 4),
        "jer_collar0": round(float(diar_eval["jer"]), 4) if diar_eval["jer"] is not None else None,
        "false_alarm": round(float(diar_eval["false_alarm"]), 4),
        "miss": round(float(diar_eval["miss"]), 4),
        "confusion": round(float(diar_eval["confusion"]), 4),
        "cpwer": round(float(f_metrics["cpwer"]), 4),
        "sawer": round(float(f_metrics["sawer"]), 4),
        "micro_sacc": round(float(f_metrics["speaker_accuracy"]), 4),
        "micro_wae": round(float(f_metrics["word_attribution_error"]), 4),
        "timeline_accuracy": round(float(f_metrics["timeline_accuracy"]), 4),
        "macro_sacc": macro_info["macro_sacc"],
    }


def offdiag_zscore_affinity(A: torch.Tensor) -> torch.Tensor:
    """Z-score standardizes off-diagonal entries of an affinity matrix A and maps to [0, 1]."""
    n = A.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=A.device)
    off = A[mask]
    mu = torch.mean(off)
    sd = torch.clamp(torch.std(off), min=1e-6)
    Z = (A - mu) / sd
    Z_sig = torch.sigmoid(Z)
    Z_sig.fill_diagonal_(1.0)
    return ScalerMinMax(Z_sig)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    for sub in ["tables", "rttms", "reports"]:
        (STAGE4_DIR / sub).mkdir(parents=True, exist_ok=True)

    manifest = json.loads((STAGE1_DIR / "manifests" / "recording_manifest.json").read_text(encoding="utf-8"))

    # =========================================================================
    # SECTION 1: Linear CKA & Cross-View Alignment + Cross-Condition Self-CKA
    # =========================================================================
    print("[Stage 4.1] Computing Linear CKA across all 8 recordings x 3 conditions x 5 scales...")
    cka_rows: List[Dict[str, Any]] = []
    cross_cond_rows: List[Dict[str, Any]] = []

    for cond in AUDIO_CONDITIONS:
        for scale_k, win_sec, _, _ in SCALES:
            for entry in manifest:
                rec_id = entry["rec_id"]
                t_data, r_data, meta = load_cache(cond, rec_id, scale_k)
                t_norm = t_data["l2_normalized_embeddings"]
                r_norm = r_data["l2_normalized_embeddings"]
                cka_std = linear_cka(t_norm, r_norm, debiased=False)
                cka_deb = linear_cka(t_norm, r_norm, debiased=True)

                # Also compute CKA on clean single-speaker subsegments only
                clean_idx = [
                    i for i, m in enumerate(meta)
                    if not m["gt_is_overlap"]
                    and not str(m["gt_label"]).startswith(("NON_SPEECH", "PARTIAL", "OVERLAP", "AMBIGUOUS"))
                ]
                if len(clean_idx) >= 6:
                    cka_clean = linear_cka(t_norm[clean_idx], r_norm[clean_idx], debiased=False)
                else:
                    cka_clean = float("nan")

                cka_rows.append({
                    "audio_condition": cond,
                    "scale_idx": scale_k,
                    "window_sec": win_sec,
                    "rec_id": rec_id,
                    "num_subsegments": len(meta),
                    "num_clean_subsegments": len(clean_idx),
                    "linear_cka_all": round(cka_std, 4),
                    "debiased_cka_all": round(cka_deb, 4) if not math.isnan(cka_deb) else None,
                    "linear_cka_clean": round(cka_clean, 4) if not math.isnan(cka_clean) else None,
                })

    # Cross-condition self-consistency (16k_mono vs dfn, 16k_mono vs normalized) at Scale 4 (0.5s) and Scale 0 (1.5s)
    for scale_k in [0, 2, 4]:
        for entry in manifest:
            rec_id = entry["rec_id"]
            t_mono, r_mono, _ = load_cache("16k_mono", rec_id, scale_k)
            t_dfn, r_dfn, _ = load_cache("dfn", rec_id, scale_k)
            t_norm, r_norm, _ = load_cache("normalized", rec_id, scale_k)

            t_m, r_m = t_mono["l2_normalized_embeddings"], r_mono["l2_normalized_embeddings"]
            t_d, r_d = t_dfn["l2_normalized_embeddings"], r_dfn["l2_normalized_embeddings"]
            t_n, r_n = t_norm["l2_normalized_embeddings"], r_norm["l2_normalized_embeddings"]

            cross_cond_rows.append({
                "rec_id": rec_id,
                "scale_idx": scale_k,
                "tita_cos_mono_vs_dfn": round(float(np.mean(np.sum(t_m * t_d, axis=-1))), 4),
                "res_cos_mono_vs_dfn": round(float(np.mean(np.sum(r_m * r_d, axis=-1))), 4),
                "tita_cos_mono_vs_norm": round(float(np.mean(np.sum(t_m * t_n, axis=-1))), 4),
                "res_cos_mono_vs_norm": round(float(np.mean(np.sum(r_m * r_n, axis=-1))), 4),
                "tita_cka_mono_vs_dfn": round(linear_cka(t_m, t_d), 4),
                "res_cka_mono_vs_dfn": round(linear_cka(r_m, r_d), 4),
            })

    # =========================================================================
    # SECTION 2: 8-Fold LORO Orthogonal Procrustes Residuals (Training-Only Fitting)
    # =========================================================================
    print("[Stage 4.2] Evaluating 8-fold LORO Orthogonal Procrustes alignment...")
    loro_procrustes_rows: List[Dict[str, Any]] = []
    loro_procrustes_matrices: Dict[Tuple[str, int, str], np.ndarray] = {}

    for cond in ["16k_mono", "dfn"]:
        for scale_k, win_sec, _, _ in SCALES:
            rec_embs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for entry in manifest:
                rec_id = entry["rec_id"]
                t_data, r_data, _ = load_cache(cond, rec_id, scale_k)
                rec_embs[rec_id] = (t_data["l2_normalized_embeddings"], r_data["l2_normalized_embeddings"])

            for held_out in [e["rec_id"] for e in manifest]:
                train_X = np.concatenate([rec_embs[r][1] for r in rec_embs if r != held_out], axis=0)  # ResNet293 [N_tr, 256]
                train_Y = np.concatenate([rec_embs[r][0] for r in rec_embs if r != held_out], axis=0)  # TitaNet   [N_tr, 192]
                test_X = rec_embs[held_out][1]
                test_Y = rec_embs[held_out][0]

                P = fit_orthogonal_procrustes(train_X, train_Y)
                loro_procrustes_matrices[(cond, scale_k, held_out)] = P
                tr_m = procrustes_metrics(train_X, train_Y, P)
                te_m = procrustes_metrics(test_X, test_Y, P)

                loro_procrustes_rows.append({
                    "audio_condition": cond,
                    "scale_idx": scale_k,
                    "window_sec": win_sec,
                    "held_out_rec_id": held_out,
                    "train_samples": int(train_X.shape[0]),
                    "test_samples": int(test_X.shape[0]),
                    "train_r2": tr_m["r2_explained"],
                    "test_r2": te_m["r2_explained"],
                    "r2_generalization_drop": round(tr_m["r2_explained"] - te_m["r2_explained"], 4),
                    "train_mean_cos": tr_m["mean_cosine_alignment"],
                    "test_mean_cos": te_m["mean_cosine_alignment"],
                    "test_frobenius_rms": te_m["frobenius_rms_per_sample"],
                })

    # =========================================================================
    # SECTION 3 & 4: Turn-Level & Per-Recording Speaker Verification + Error Contingency
    # =========================================================================
    print("[Stage 4.3 & 4.4] Evaluating non-overlapping GT turn pairs & per-recording discriminability...")
    turn_analysis_by_cond: Dict[str, Any] = {}
    for cond in AUDIO_CONDITIONS:
        turns = build_non_overlapping_turn_dataset(manifest, cond, scale_k=4)
        within_rec = evaluate_turn_pairs_and_contingency(turns, within_recording_only=True)
        cross_rec = evaluate_turn_pairs_and_contingency(turns, within_recording_only=False)
        turn_analysis_by_cond[cond] = {
            "within_recording_pairs": within_rec,
            "pooled_all_pairs": cross_rec,
        }

    # Per-recording subsegment discriminability (Scale 4: 0.5s & Scale 0: 1.5s across all 3 conditions)
    per_rec_discrim_rows: List[Dict[str, Any]] = []
    for cond in AUDIO_CONDITIONS:
        for scale_k in [0, 2, 4]:
            for entry in manifest:
                rec_id = entry["rec_id"]
                t_data, r_data, meta = load_cache(cond, rec_id, scale_k)
                t_norm = t_data["l2_normalized_embeddings"]
                r_norm = r_data["l2_normalized_embeddings"]

                clean_idx = [
                    i for i, m in enumerate(meta)
                    if not m["gt_is_overlap"]
                    and not str(m["gt_label"]).startswith(("NON_SPEECH", "PARTIAL", "OVERLAP", "AMBIGUOUS"))
                ]
                if len(clean_idx) < 4:
                    continue
                t_c = t_norm[clean_idx]
                r_c = r_norm[clean_idx]
                spks = [str(meta[i]["gt_label"]) for i in clean_idx]

                t_fdr = compute_fdr_and_silhouette(t_c, spks)
                r_fdr = compute_fdr_and_silhouette(r_c, spks)

                # Pairwise EER inside recording (only non-overlapping subsegment pairs with |t_i - t_j| > 1.5s)
                mids = np.array([meta[i][" midpoint"] for i in clean_idx], dtype=np.float32)
                t_sc, r_sc, y_lb = [], [], []
                for a in range(len(clean_idx)):
                    for b in range(a + 1, len(clean_idx)):
                        # Exclude temporally overlapping sliding windows from same turn!
                        if abs(float(mids[a] - mids[b])) <= 1.5:
                            continue
                        t_sc.append(float(np.dot(t_c[a], t_c[b])))
                        r_sc.append(float(np.dot(r_c[a], r_c[b])))
                        y_lb.append(1 if spks[a] == spks[b] else 0)
                t_eer_info = compute_eer_and_mindcf(np.array(t_sc), np.array(y_lb)) if y_lb else {"eer": None, "mindcf_p001": None}
                r_eer_info = compute_eer_and_mindcf(np.array(r_sc), np.array(y_lb)) if y_lb else {"eer": None, "mindcf_p001": None}
                eq_eer_info = compute_eer_and_mindcf(0.5 * (np.array(t_sc) + np.array(r_sc)), np.array(y_lb)) if y_lb else {"eer": None, "mindcf_p001": None}

                per_rec_discrim_rows.append({
                    "audio_condition": cond,
                    "scale_idx": scale_k,
                    "rec_id": rec_id,
                    "clean_subsegments": len(clean_idx),
                    "clean_speakers": len(set(spks)),
                    "non_adj_pos_pairs": t_eer_info.get("n_pos", 0),
                    "non_adj_neg_pairs": t_eer_info.get("n_neg", 0),
                    "tita_eer": t_eer_info.get("eer"),
                    "res_eer": r_eer_info.get("eer"),
                    "eq_fusion_eer": eq_eer_info.get("eer"),
                    "tita_mindcf": t_eer_info.get("mindcf_p001"),
                    "res_mindcf": r_eer_info.get("mindcf_p001"),
                    "tita_fdr": t_fdr.get("fdr"),
                    "res_fdr": r_fdr.get("fdr"),
                    "tita_silhouette": t_fdr.get("cosine_silhouette"),
                    "res_silhouette": r_fdr.get("cosine_silhouette"),
                    "tita_within_cos": t_fdr.get("within_spk_cos_mean"),
                    "tita_between_cos": t_fdr.get("between_spk_cos_mean"),
                    "res_within_cos": r_fdr.get("within_spk_cos_mean"),
                    "res_between_cos": r_fdr.get("between_spk_cos_mean"),
                })

    # =========================================================================
    # SECTION 5 & 6: Stratified Analysis & Reliability Indicator Predictive Value
    # =========================================================================
    print("[Stage 4.5 & 4.6] Computing stratified performance and reliability indicator correlations...")
    # Compute subsegment-to-speaker-centroid margin m_i^(m) on all recordings with >=2 clean speakers
    subseg_margin_records: List[Dict[str, Any]] = []
    for cond in AUDIO_CONDITIONS:
        for scale_k, win_sec, _, _ in SCALES:
            for entry in manifest:
                rec_id = entry["rec_id"]
                t_data, r_data, meta = load_cache(cond, rec_id, scale_k)
                t_norm = t_data["l2_normalized_embeddings"]
                r_norm = r_data["l2_normalized_embeddings"]

                # Compute clean speaker centroids per recording
                spk_to_idxs: Dict[str, List[int]] = {}
                for i, m in enumerate(meta):
                    lbl = str(m["gt_label"])
                    if not m["gt_is_overlap"] and not lbl.startswith(("NON_SPEECH", "PARTIAL", "OVERLAP", "AMBIGUOUS")):
                        spk_to_idxs.setdefault(lbl, []).append(i)
                valid_spks = [s for s, ix in spk_to_idxs.items() if len(ix) >= 2]
                if len(valid_spks) < 2:
                    continue

                t_centroids = {}
                r_centroids = {}
                for s in valid_spks:
                    tc = np.mean(t_norm[spk_to_idxs[s]], axis=0)
                    t_centroids[s] = tc / max(float(np.linalg.norm(tc)), 1e-12)
                    rc = np.mean(r_norm[spk_to_idxs[s]], axis=0)
                    r_centroids[s] = rc / max(float(np.linalg.norm(rc)), 1e-12)

                for i, m in enumerate(meta):
                    raw_lbl = str(m["gt_label"])
                    base_spk = raw_lbl.split("::")[-1] if "::" in raw_lbl else raw_lbl
                    if base_spk not in valid_spks:
                        continue
                    other_spks = [s for s in valid_spks if s != base_spk]
                    t_sim_own = float(np.dot(t_norm[i], t_centroids[base_spk]))
                    t_sim_oth = max(float(np.dot(t_norm[i], t_centroids[s])) for s in other_spks)
                    t_margin = t_sim_own - t_sim_oth

                    r_sim_own = float(np.dot(r_norm[i], r_centroids[base_spk]))
                    r_sim_oth = max(float(np.dot(r_norm[i], r_centroids[s])) for s in other_spks)
                    r_margin = r_sim_own - r_sim_oth

                    subseg_margin_records.append({
                        "audio_condition": cond,
                        "scale_idx": scale_k,
                        "window_sec": win_sec,
                        "rec_id": rec_id,
                        "is_overlap": bool(m["gt_is_overlap"]),
                        "snr_local_db": float(m["snr_local_db"]),
                        "rms_dbfs": float(m["rms_dbfs"]),
                        "dfn_noise_reduction_db": float(m["dfn_noise_reduction_db"]),
                        "spectral_flatness": float(m["spectral_flatness"]),
                        "vad_conf_mean": float(m["vad_conf_mean"]),
                        "vad_conf_std": float(m["vad_conf_std"]),
                        "overlap_prob_mean": float(m["overlap_prob_mean"]),
                        "titanet_l2_norm": float(m["titanet_l2_norm"]),
                        "resnet293_l2_norm": float(m["resnet293_l2_norm"]),
                        "titanet_cross_scale_cos": float(m["titanet_cross_scale_cos"]),
                        "resnet293_cross_scale_cos": float(m["resnet293_cross_scale_cos"]),
                        "tita_margin": t_margin,
                        "res_margin": r_margin,
                        "delta_margin_res_minus_tita": r_margin - t_margin,
                        "tita_correct": int(t_margin > 0),
                        "res_correct": int(r_margin > 0),
                    })

    # Stratified Summary: by Scale, by Overlap, by SNR Quartile, by Audio Condition
    stratified_summary: Dict[str, Any] = {"by_scale": [], "by_overlap": [], "by_snr_quartile": [], "by_audio_condition": []}

    # 5a. By Audio Condition (clean subsegments at Scale 4)
    for cond in AUDIO_CONDITIONS:
        recs = [x for x in subseg_margin_records if x["audio_condition"] == cond and x["scale_idx"] == 4 and not x["is_overlap"]]
        stratified_summary["by_audio_condition"].append({
            "audio_condition": cond,
            "count": len(recs),
            "tita_centroid_acc": round(float(np.mean([x["tita_correct"] for x in recs])), 4),
            "res_centroid_acc": round(float(np.mean([x["res_correct"] for x in recs])), 4),
            "tita_margin_mean": round(float(np.mean([x["tita_margin"] for x in recs])), 4),
            "res_margin_mean": round(float(np.mean([x["res_margin"] for x in recs])), 4),
            "res_win_fraction": round(float(np.mean([x["res_margin"] > x["tita_margin"] for x in recs])), 4),
        })

    # 5b. By Scale (on 16k_mono and dfn)
    for cond in ["16k_mono", "dfn"]:
        for scale_k, win_sec, _, _ in SCALES:
            recs = [x for x in subseg_margin_records if x["audio_condition"] == cond and x["scale_idx"] == scale_k and not x["is_overlap"]]
            stratified_summary["by_scale"].append({
                "audio_condition": cond,
                "scale_idx": scale_k,
                "window_sec": win_sec,
                "count": len(recs),
                "tita_centroid_acc": round(float(np.mean([x["tita_correct"] for x in recs])), 4),
                "res_centroid_acc": round(float(np.mean([x["res_correct"] for x in recs])), 4),
                "tita_margin_mean": round(float(np.mean([x["tita_margin"] for x in recs])), 4),
                "res_margin_mean": round(float(np.mean([x["res_margin"] for x in recs])), 4),
                "res_win_fraction": round(float(np.mean([x["res_margin"] > x["tita_margin"] for x in recs])), 4),
            })

    # 5c. By Overlap Status (Scale 4 on 16k_mono and dfn)
    for cond in ["16k_mono", "dfn"]:
        for is_ovl in [False, True]:
            recs = [x for x in subseg_margin_records if x["audio_condition"] == cond and x["scale_idx"] == 4 and x["is_overlap"] == is_ovl]
            if not recs:
                continue
            stratified_summary["by_overlap"].append({
                "audio_condition": cond,
                "is_overlap": is_ovl,
                "count": len(recs),
                "tita_centroid_acc": round(float(np.mean([x["tita_correct"] for x in recs])), 4),
                "res_centroid_acc": round(float(np.mean([x["res_correct"] for x in recs])), 4),
                "tita_margin_mean": round(float(np.mean([x["tita_margin"] for x in recs])), 4),
                "res_margin_mean": round(float(np.mean([x["res_margin"] for x in recs])), 4),
                "res_win_fraction": round(float(np.mean([x["res_margin"] > x["tita_margin"] for x in recs])), 4),
            })

    # 5d. By SNR Quartile (Scale 4 on 16k_mono and dfn, clean subsegments)
    for cond in ["16k_mono", "dfn"]:
        recs = [x for x in subseg_margin_records if x["audio_condition"] == cond and x["scale_idx"] == 4 and not x["is_overlap"]]
        snrs = np.array([x["snr_local_db"] for x in recs])
        q_edges = np.quantile(snrs, [0.0, 0.25, 0.50, 0.75, 1.0])
        for q_idx in range(4):
            lo, hi = float(q_edges[q_idx]), float(q_edges[q_idx + 1])
            q_recs = [x for x in recs if (lo <= x["snr_local_db"] <= hi if q_idx == 3 else lo <= x["snr_local_db"] < hi)]
            stratified_summary["by_snr_quartile"].append({
                "audio_condition": cond,
                "snr_quartile": f"Q{q_idx+1} [{lo:.1f}, {hi:.1f}] dB",
                "count": len(q_recs),
                "tita_centroid_acc": round(float(np.mean([x["tita_correct"] for x in q_recs])), 4),
                "res_centroid_acc": round(float(np.mean([x["res_correct"] for x in q_recs])), 4),
                "tita_margin_mean": round(float(np.mean([x["tita_margin"] for x in q_recs])), 4),
                "res_margin_mean": round(float(np.mean([x["res_margin"] for x in q_recs])), 4),
                "res_win_fraction": round(float(np.mean([x["res_margin"] > x["tita_margin"] for x in q_recs])), 4),
            })

    # 6. Reliability Indicator Predictive Value Table (Scale 4 + pooled across scales on dfn & 16k_mono)
    indicator_names = [
        "snr_local_db",
        "rms_dbfs",
        "dfn_noise_reduction_db",
        "spectral_flatness",
        "vad_conf_mean",
        "vad_conf_std",
        "overlap_prob_mean",
        "titanet_l2_norm",
        "resnet293_l2_norm",
        "titanet_cross_scale_cos",
        "resnet293_cross_scale_cos",
    ]
    reliability_predictive_rows: List[Dict[str, Any]] = []
    for cond in ["16k_mono", "dfn"]:
        recs = [x for x in subseg_margin_records if x["audio_condition"] == cond and x["scale_idx"] == 4]
        t_m = np.array([x["tita_margin"] for x in recs])
        r_m = np.array([x["res_margin"] for x in recs])
        d_m = np.array([x["delta_margin_res_minus_tita"] for x in recs])
        t_ok = np.array([x["tita_correct"] for x in recs])
        r_ok = np.array([x["res_correct"] for x in recs])

        # Also add derived differential view indicators
        for r_item in recs:
            r_item["cross_scale_cos_diff_res_minus_tita"] = (
                r_item["resnet293_cross_scale_cos"] - r_item["titanet_cross_scale_cos"]
            )
        ext_indicators = indicator_names + ["cross_scale_cos_diff_res_minus_tita"]

        for ind in ext_indicators:
            vals = np.array([x[ind] for x in recs], dtype=np.float64)
            rho_t, _ = scipy.stats.spearmanr(vals, t_m)
            rho_r, _ = scipy.stats.spearmanr(vals, r_m)
            rho_d, _ = scipy.stats.spearmanr(vals, d_m)
            auroc_t = float(roc_auc_score(t_ok, vals)) if len(set(t_ok)) == 2 else 0.5
            auroc_r = float(roc_auc_score(r_ok, vals)) if len(set(r_ok)) == 2 else 0.5
            # Rectify AUROC to [0.5, 1.0] (discriminative power regardless of sign)
            auroc_t_abs = max(auroc_t, 1.0 - auroc_t)
            auroc_r_abs = max(auroc_r, 1.0 - auroc_r)

            reliability_predictive_rows.append({
                "audio_condition": cond,
                "indicator": ind,
                "spearman_rho_tita_margin": round(float(rho_t), 4),
                "spearman_rho_res_margin": round(float(rho_r), 4),
                "spearman_rho_delta_margin": round(float(rho_d), 4),
                "auroc_tita_correctness": round(auroc_t_abs, 4),
                "auroc_res_correctness": round(auroc_r_abs, 4),
            })

    # =========================================================================
    # SECTION 7: Downstream Diarization & Attribution Comparison (8 Recordings)
    # =========================================================================
    print("[Stage 4.7] Running downstream clustering & diarization evaluation across all 8 recordings...")
    diar_comparison_rows: List[Dict[str, Any]] = []

    # Pre-fit 8-fold LORO linear reliability gate weights on subsegment margin advantage (for Method 8)
    loro_gate_weights: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    gate_features = [
        "snr_local_db",
        "vad_conf_mean",
        "overlap_prob_mean",
        "titanet_l2_norm",
        "resnet293_l2_norm",
        "titanet_cross_scale_cos",
        "resnet293_cross_scale_cos",
    ]
    dfn_s4_recs = [x for x in subseg_margin_records if x["audio_condition"] == "dfn" and x["scale_idx"] == 4]
    for entry in manifest:
        held_rec = entry["rec_id"]
        tr_items = [x for x in dfn_s4_recs if x["rec_id"] != held_rec]
        X_tr = np.array([[x[f] for f in gate_features] for x in tr_items], dtype=np.float64)
        y_tr = np.array([x["delta_margin_res_minus_tita"] for x in tr_items], dtype=np.float64)
        mu_x = np.mean(X_tr, axis=0)
        sd_x = np.maximum(np.std(X_tr, axis=0), 1e-6)
        Xz = (X_tr - mu_x) / sd_x
        # Ridge regression (lambda = 100.0 for strong regularization on 7 recordings)
        w = np.linalg.solve(Xz.T @ Xz + 100.0 * np.eye(Xz.shape[1]), Xz.T @ y_tr)
        b = float(np.mean(y_tr))
        loro_gate_weights[held_rec] = (mu_x, sd_x, w, b)

    ms_weights_tensor = torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float32, device="cuda")

    for entry in manifest:
        rec_id = entry["rec_id"]
        dur = float(entry["duration_sec"])
        dfn_stem = Path(entry["dfn_wav"]).stem
        gt_json_uni = REPO_ROOT / entry["json_uniform"]
        gt_rttm_uni = REPO_ROOT / entry["rttm_uniform"]
        aligned_words_p = STAGE2_DIR / "whisperx" / f"{rec_id}_aligned.json"

        # Load all 5 scales on Path A ('dfn') and also on '16k_mono'
        t_scales_dfn: List[torch.Tensor] = []
        r_scales_dfn: List[torch.Tensor] = []
        concat_scales_dfn: List[torch.Tensor] = []
        proc_scales_dfn: List[torch.Tensor] = []
        ts_scales: List[torch.Tensor] = []
        meta_s4: List[Dict[str, Any]] = []

        for scale_k, _, _, _ in SCALES:
            t_d, r_d, meta_k = load_cache("dfn", rec_id, scale_k)
            t_nrm = t_d["l2_normalized_embeddings"]  # [N_k, 192]
            r_nrm = r_d["l2_normalized_embeddings"]  # [N_k, 256]

            # Naive L2 concatenation (448-D)
            cat_nrm = np.concatenate([t_nrm, r_nrm], axis=-1) / math.sqrt(2.0)

            # Out-of-fold LORO Procrustes aligned average (192-D)
            P_loro = loro_procrustes_matrices[("dfn", scale_k, rec_id)]
            r_proj = r_nrm @ P_loro
            r_proj = r_proj / np.maximum(np.linalg.norm(r_proj, axis=-1, keepdims=True), 1e-12)
            proc_avg = t_nrm + r_proj
            proc_avg = proc_avg / np.maximum(np.linalg.norm(proc_avg, axis=-1, keepdims=True), 1e-12)

            ts_arr = np.stack([t_d["starts"], t_d["ends"]], axis=-1).astype(np.float32)

            t_scales_dfn.append(torch.from_numpy(t_nrm).to("cuda"))
            r_scales_dfn.append(torch.from_numpy(r_nrm).to("cuda"))
            concat_scales_dfn.append(torch.from_numpy(cat_nrm.astype(np.float32)).to("cuda"))
            proc_scales_dfn.append(torch.from_numpy(proc_avg.astype(np.float32)).to("cuda"))
            ts_scales.append(torch.from_numpy(ts_arr).to("cuda"))
            if scale_k == 4:
                meta_s4 = meta_k

        # Compute NeMo multiscale cosine affinity matrices A_tita and A_res
        A_tita = getMultiScaleCosAffinityMatrix(
            multiscale_weights=ms_weights_tensor,
            embeddings_in_scales=t_scales_dfn,
            timestamps_in_scales=ts_scales,
            device=torch.device("cuda"),
        )
        A_res = getMultiScaleCosAffinityMatrix(
            multiscale_weights=ms_weights_tensor,
            embeddings_in_scales=r_scales_dfn,
            timestamps_in_scales=ts_scales,
            device=torch.device("cuda"),
        )
        A_concat = getMultiScaleCosAffinityMatrix(
            multiscale_weights=ms_weights_tensor,
            embeddings_in_scales=concat_scales_dfn,
            timestamps_in_scales=ts_scales,
            device=torch.device("cuda"),
        )
        A_proc = getMultiScaleCosAffinityMatrix(
            multiscale_weights=ms_weights_tensor,
            embeddings_in_scales=proc_scales_dfn,
            timestamps_in_scales=ts_scales,
            device=torch.device("cuda"),
        )

        # Equal affinity fusion (0.5 * A_tita + 0.5 * A_res)
        A_eq = 0.5 * A_tita + 0.5 * A_res

        # Z-score standardized affinity fusion
        A_z_tita = offdiag_zscore_affinity(A_tita)
        A_z_res = offdiag_zscore_affinity(A_res)
        A_z_eq = 0.5 * A_z_tita + 0.5 * A_z_res

        # Zero-training Heuristic Quality-Weighted Affinity Fusion (Norm + Cross-scale consistency rank)
        t_l2_s4 = np.array([m["titanet_l2_norm"] for m in meta_s4], dtype=np.float32)
        r_l2_s4 = np.array([m["resnet293_l2_norm"] for m in meta_s4], dtype=np.float32)
        t_cs_s4 = np.array([m["titanet_cross_scale_cos"] for m in meta_s4], dtype=np.float32)
        r_cs_s4 = np.array([m["resnet293_cross_scale_cos"] for m in meta_s4], dtype=np.float32)

        def _z(v: np.ndarray) -> np.ndarray:
            return (v - np.mean(v)) / max(float(np.std(v)), 1e-6)

        q_tita = 0.5 * _z(t_l2_s4) + 0.5 * _z(t_cs_s4)
        q_res = 0.5 * _z(r_l2_s4) + 0.5 * _z(r_cs_s4)
        w_res_i = 1.0 / (1.0 + np.exp(-0.5 * (q_res - q_tita)))  # in (0, 1)
        W_res_mat = torch.from_numpy(0.5 * (w_res_i[:, None] + w_res_i[None, :])).to("cuda").float()
        A_heur = (1.0 - W_res_mat) * A_z_tita + W_res_mat * A_z_res

        # LORO Calibrated Gate Affinity Fusion
        if rec_id in loro_gate_weights:
            mu_x, sd_x, w_g, b_g = loro_gate_weights[rec_id]
            X_te = np.array([[m[f] for f in gate_features] for m in meta_s4], dtype=np.float64)
            pred_delta = ((X_te - mu_x) / sd_x) @ w_g + b_g
            g_res_i = np.clip(0.5 + 2.0 * pred_delta, 0.15, 0.85).astype(np.float32)
        else:
            g_res_i = np.full(len(meta_s4), 0.5, dtype=np.float32)
        G_res_mat = torch.from_numpy(0.5 * (g_res_i[:, None] + g_res_i[None, :])).to("cuda").float()
        A_loro_gate = (1.0 - G_res_mat) * A_z_tita + G_res_mat * A_z_res

        methods = [
            ("1_titanet_nmesc", A_tita),
            ("2_resnet293_nmesc", A_res),
            ("3_naive_concat_448d", A_concat),
            ("4_equal_affinity_050", A_eq),
            ("5_zscore_affinity_050", A_z_eq),
            ("6_heuristic_quality_affinity", A_heur),
            ("7_loro_procrustes_192d", A_proc),
            ("8_loro_calibrated_gate_affinity", A_loro_gate),
        ]

        for m_name, A_mat in methods:
            res_m = run_nemo_clustering_from_affinity_or_embs(
                rec_id=rec_id,
                dfn_stem=dfn_stem,
                gt_json_path=gt_json_uni,
                gt_rttm_path=gt_rttm_uni,
                aligned_words_path=aligned_words_p,
                duration_sec=dur,
                timestamps_in_scales=ts_scales,
                multiscale_weights=ms_weights_tensor,
                affinity_mat=A_mat,
                out_dir=STAGE4_DIR / "rttms",
                method_name=m_name,
            )
            diar_comparison_rows.append(res_m)

        # Also evaluate single-encoder ResNet293-LM on 16k_mono (unenhanced) to test hypothesis that
        # ResNet293-LM prefers unenhanced audio over DFN3
        t_scales_mono, r_scales_mono, ts_mono = [], [], []
        for scale_k, _, _, _ in SCALES:
            tm_d, rm_d, _ = load_cache("16k_mono", rec_id, scale_k)
            t_scales_mono.append(torch.from_numpy(tm_d["l2_normalized_embeddings"]).to("cuda"))
            r_scales_mono.append(torch.from_numpy(rm_d["l2_normalized_embeddings"]).to("cuda"))
            ts_mono.append(torch.from_numpy(np.stack([tm_d["starts"], tm_d["ends"]], axis=-1).astype(np.float32)).to("cuda"))
        A_tita_mono = getMultiScaleCosAffinityMatrix(ms_weights_tensor, t_scales_mono, ts_mono, torch.device("cuda"))
        A_res_mono = getMultiScaleCosAffinityMatrix(ms_weights_tensor, r_scales_mono, ts_mono, torch.device("cuda"))
        A_hybrid_titaDFN_resMONO = 0.5 * offdiag_zscore_affinity(A_tita) + 0.5 * offdiag_zscore_affinity(A_res_mono)

        for m_name, A_mat in [
            ("9_titanet_mono_nmesc", A_tita_mono),
            ("10_resnet293_mono_nmesc", A_res_mono),
            ("11_hybrid_titaDFN_resMONO_zscore", A_hybrid_titaDFN_resMONO),
        ]:
            res_m = run_nemo_clustering_from_affinity_or_embs(
                rec_id=rec_id,
                dfn_stem=dfn_stem,
                gt_json_path=gt_json_uni,
                gt_rttm_path=gt_rttm_uni,
                aligned_words_path=aligned_words_p,
                duration_sec=dur,
                timestamps_in_scales=ts_scales,
                multiscale_weights=ms_weights_tensor,
                affinity_mat=A_mat,
                out_dir=STAGE4_DIR / "rttms",
                method_name=m_name,
            )
            diar_comparison_rows.append(res_m)

    # Aggregate diarization comparison across all 8 recordings per method
    method_names = sorted(set(r["method"] for r in diar_comparison_rows), key=lambda x: int(x.split("_")[0]))
    diar_method_summary: List[Dict[str, Any]] = []
    for m_name in method_names:
        m_rows = [r for r in diar_comparison_rows if r["method"] == m_name]
        tot_d = sum(r["duration_sec"] for r in m_rows)
        diar_method_summary.append({
            "method": m_name,
            "macro_der_collar0_ovl": round(float(np.mean([r["der_collar0_ovl"] for r in m_rows])), 4),
            "dur_weighted_der_collar0_ovl": round(float(sum(r["der_collar0_ovl"] * r["duration_sec"] for r in m_rows) / tot_d), 4),
            "macro_der_collar025_noovl": round(float(np.mean([r["der_collar025_noovl"] for r in m_rows])), 4),
            "macro_jer": round(float(np.mean([r["jer_collar0"] for r in m_rows if r["jer_collar0"] is not None])), 4),
            "macro_spk_count_err_abs": round(float(np.mean([r["speaker_count_error_abs"] for r in m_rows])), 4),
            "single_spk_collapse_count": int(sum(1 for r in m_rows if r["single_speaker_collapse"])),
            "macro_cpwer": round(float(np.mean([r["cpwer"] for r in m_rows])), 4),
            "macro_sawer": round(float(np.mean([r["sawer"] for r in m_rows])), 4),
            "macro_micro_sacc": round(float(np.mean([r["micro_sacc"] for r in m_rows])), 4),
            "macro_macro_sacc": round(float(np.mean([r["macro_sacc"] for r in m_rows])), 4),
        })

    # Write all CSV and JSON tables
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(STAGE4_DIR / "tables" / "stage4_linear_cka.csv", cka_rows)
    _write_csv(STAGE4_DIR / "tables" / "stage4_cross_condition_consistency.csv", cross_cond_rows)
    _write_csv(STAGE4_DIR / "tables" / "stage4_loro_procrustes.csv", loro_procrustes_rows)
    _write_csv(STAGE4_DIR / "tables" / "stage4_per_recording_discriminability.csv", per_rec_discrim_rows)
    _write_csv(STAGE4_DIR / "tables" / "stage4_reliability_predictive_value.csv", reliability_predictive_rows)
    _write_csv(STAGE4_DIR / "tables" / "stage4_diarization_per_recording.csv", diar_comparison_rows)
    _write_csv(STAGE4_DIR / "tables" / "stage4_diarization_method_summary.csv", diar_method_summary)

    master_json = {
        "stage": "Stage 4 — Complementarity Verification & Mandatory Decision Gate",
        "turn_pair_verification_and_contingency": turn_analysis_by_cond,
        "stratified_performance": stratified_summary,
        "diarization_method_summary": diar_method_summary,
        "reliability_predictive_value": reliability_predictive_rows,
        "loro_procrustes_summary": {
            cond: {
                f"scale_{sk}": {
                    "mean_train_r2": round(float(np.mean([r["train_r2"] for r in loro_procrustes_rows if r["audio_condition"] == cond and r["scale_idx"] == sk])), 4),
                    "mean_test_r2": round(float(np.mean([r["test_r2"] for r in loro_procrustes_rows if r["audio_condition"] == cond and r["scale_idx"] == sk])), 4),
                    "mean_test_cos": round(float(np.mean([r["test_mean_cos"] for r in loro_procrustes_rows if r["audio_condition"] == cond and r["scale_idx"] == sk])), 4),
                }
                for sk, _, _, _ in SCALES
            }
            for cond in ["16k_mono", "dfn"]
        },
        "cka_summary": {
            cond: {
                f"scale_{sk}": {
                    "mean_linear_cka_all": round(float(np.mean([r["linear_cka_all"] for r in cka_rows if r["audio_condition"] == cond and r["scale_idx"] == sk])), 4),
                    "mean_linear_cka_clean": round(float(np.nanmean([r["linear_cka_clean"] if r["linear_cka_clean"] is not None else float("nan") for r in cka_rows if r["audio_condition"] == cond and r["scale_idx"] == sk])), 4),
                }
                for sk, _, _, _ in SCALES
            }
            for cond in AUDIO_CONDITIONS
        },
    }
    (STAGE4_DIR / "tables" / "stage4_master_results.json").write_text(
        json.dumps(master_json, indent=2) + "\n", encoding="utf-8"
    )
    print("Stage 4 complementarity verification completed successfully!")


if __name__ == "__main__":
    main()
