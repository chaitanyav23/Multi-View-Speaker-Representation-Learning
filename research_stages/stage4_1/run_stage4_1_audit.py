#!/usr/bin/env python3
"""
Stage 4.1 — Statistical Validity, Leakage Audit & Evaluation Correction.

Strictly adheres to all Stage 4.1 constraints:
- Does NOT modify any Stage 1–4 files, scripts, caches, or outputs.
- Does NOT rerun TitaNet or ResNet293 embedding extraction.
- Does NOT train or fine-tune any neural network.
- Writes all new outputs exclusively inside research_stages/stage4_1/.
"""

import csv
import json
import math
import os
import sys
import wave
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
STAGE4_1_DIR = REPO_ROOT / "research_stages" / "stage4_1"

sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
sys.path.insert(0, str(STAGE2_DIR))
sys.path.insert(0, str(STAGE4_DIR))

import common as p20_common  # noqa: E402
from run_stage2_baseline import compute_der_collar025_noovl, compute_macro_sacc  # noqa: E402
from run_stage4_complementarity import (  # noqa: E402
    AUDIO_CONDITIONS,
    SCALES,
    compute_eer_and_mindcf,
    load_cache,
    offdiag_zscore_affinity,
)
from nemo.collections.asr.parts.utils.offline_clustering import (  # noqa: E402
    SpeakerClustering,
    getMultiScaleCosAffinityMatrix,
)
from nemo.collections.asr.parts.utils.speaker_utils import (  # noqa: E402
    generate_cluster_labels,
    labels_to_rttmfile,
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ============================================================================
# Helper: Match subsegment to exact Ground-Truth Turn Index (seg_idx)
# ============================================================================

def assign_subsegments_to_gt_turns(
    meta: List[Dict[str, Any]], gt_segments: List[Dict[str, Any]], min_purity: float = 0.75
) -> List[Optional[int]]:
    """
    For each subsegment in meta, returns the unique clean ground-truth turn seg_idx
    if the subsegment is clean (not gt_is_overlap and not NON_SPEECH/PARTIAL/OVERLAP/AMBIGUOUS)
    and overlaps a single usable_for_clean_speaker_eval GT turn by >= min_purity of its duration.
    Otherwise returns None.
    """
    clean_gt = [g for g in gt_segments if g.get("usable_for_clean_speaker_eval", False)]
    assigned: List[Optional[int]] = []
    for m in meta:
        lbl = str(m["gt_label"])
        if m["gt_is_overlap"] or lbl.startswith(("NON_SPEECH", "PARTIAL", "OVERLAP", "AMBIGUOUS")):
            assigned.append(None)
            continue
        st = float(m["start"])
        en = float(m["end"])
        dur = max(en - st, 1e-6)
        best_seg_idx = None
        best_ov = 0.0
        for g in clean_gt:
            gst = float(g["start"])
            gen = float(g["end_clipped_to_audio"])
            ov = min(en, gen) - max(st, gst)
            if ov > best_ov:
                best_ov = ov
                best_seg_idx = int(g["seg_idx"])
        if best_seg_idx is not None and (best_ov / dur) >= min_purity:
            assigned.append(best_seg_idx)
        else:
            assigned.append(None)
    return assigned


def build_turn_records_for_condition(
    manifest: List[Dict[str, Any]], cond: str, scale_k: int = 4
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      1. usable_turns: All 254 clean turns that have >=1 Scale-k subsegment with >=75% overlap
      2. all_clean_gt_turns: All 264 clean turns from Stage 1 annotations (including those with 0 subsegments or <0.5s)
    """
    usable_turns: List[Dict[str, Any]] = []
    all_clean_gt_turns: List[Dict[str, Any]] = []

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
            dur = gen - gst
            ov = np.minimum(sub_en, gen) - np.maximum(sub_st, gst)
            sub_dur = np.maximum(sub_en - sub_st, 1e-6)
            idxs = np.where((ov / sub_dur) >= 0.75)[0]

            rec_info = {
                "rec_id": rec_id,
                "seg_idx": int(g["seg_idx"]),
                "speaker": str(g["speaker"]),
                "global_speaker_id": str(g["global_speaker_id"]),
                "start": gst,
                "end": gen,
                "duration": round(dur, 3),
                "num_subsegments_pooled": int(idxs.size),
                "included_in_stage4_turn_pairs": bool(dur >= 0.5 and idxs.size > 0),
                "exclusion_reason": (
                    "INCLUDED"
                    if (dur >= 0.5 and idxs.size > 0)
                    else ("DURATION_LT_0.5S" if dur < 0.5 else "NO_VAD_SUBSEGMENT_WITH_75PCT_COVERAGE")
                ),
            }
            all_clean_gt_turns.append(rec_info)

            if dur >= 0.5 and idxs.size > 0:
                t_vec = np.mean(t_norm[idxs], axis=0)
                t_vec = (t_vec / max(float(np.linalg.norm(t_vec)), 1e-12)).astype(np.float32)
                r_vec = np.mean(r_norm[idxs], axis=0)
                r_vec = (r_vec / max(float(np.linalg.norm(r_vec)), 1e-12)).astype(np.float32)

                usable_turns.append({
                    **rec_info,
                    "snr_local_db": float(np.mean([sub_meta[i]["snr_local_db"] for i in idxs])),
                    "rms_dbfs": float(np.mean([sub_meta[i]["rms_dbfs"] for i in idxs])),
                    "dfn_noise_reduction_db": float(np.mean([sub_meta[i]["dfn_noise_reduction_db"] for i in idxs])),
                    "spectral_flatness": float(np.mean([sub_meta[i]["spectral_flatness"] for i in idxs])),
                    "vad_conf_mean": float(np.mean([sub_meta[i]["vad_conf_mean"] for i in idxs])),
                    "vad_conf_std": float(np.mean([sub_meta[i]["vad_conf_std"] for i in idxs])),
                    "overlap_prob_mean": float(np.mean([sub_meta[i]["overlap_prob_mean"] for i in idxs])),
                    "titanet_l2_norm": float(np.mean([sub_meta[i]["titanet_l2_norm"] for i in idxs])),
                    "resnet293_l2_norm": float(np.mean([sub_meta[i]["resnet293_l2_norm"] for i in idxs])),
                    "titanet_cross_scale_cos": float(np.mean([sub_meta[i]["titanet_cross_scale_cos"] for i in idxs])),
                    "resnet293_cross_scale_cos": float(np.mean([sub_meta[i]["resnet293_cross_scale_cos"] for i in idxs])),
                    "tita_emb": t_vec,
                    "res_emb": r_vec,
                })
    return usable_turns, all_clean_gt_turns


def eval_at_frozen_threshold(
    scores: np.ndarray, labels: np.ndarray, thr: float
) -> Dict[str, Any]:
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos == 0 or n_neg == 0:
        return {
            "far_at_frozen_thr": None,
            "frr_at_frozen_thr": None,
            "hter_at_frozen_thr": None,
            "act_dcf_p001": None,
            "act_dcf_p01": None,
            "roc_auc": None,
            "exploratory_in_sample_eer": None,
        }
    preds = (scores >= thr).astype(np.int32)
    far = float(np.mean(preds[neg_mask] == 1))
    frr = float(np.mean(preds[pos_mask] == 0))
    hter = 0.5 * (far + frr)

    # Normalized actual DCF at frozen threshold for p_target=0.01 and p_target=0.10
    act_dcf_001 = (0.01 * frr + 0.99 * far) / 0.01
    act_dcf_010 = (0.10 * frr + 0.90 * far) / 0.10

    auc = float(roc_auc_score(labels, scores))
    in_sample = compute_eer_and_mindcf(scores, labels)

    return {
        "far_at_frozen_thr": round(far, 4),
        "frr_at_frozen_thr": round(frr, 4),
        "hter_at_frozen_thr": round(hter, 4),
        "act_dcf_p001": round(act_dcf_001, 4),
        "act_dcf_p01": round(act_dcf_010, 4),
        "roc_auc": round(auc, 4),
        "exploratory_in_sample_eer": in_sample["eer"],
        "exploratory_in_sample_mindcf": in_sample["mindcf_p001"],
    }


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    STAGE4_1_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((STAGE1_DIR / "manifests" / "recording_manifest.json").read_text(encoding="utf-8"))
    rec_quality = list(csv.DictReader((STAGE1_DIR / "reports" / "stage1_recording_quality.csv").open(encoding="utf-8")))
    spk_quality = list(csv.DictReader((STAGE1_DIR / "reports" / "stage1_speaker_quality.csv").open(encoding="utf-8")))
    seg_ledger = list(csv.DictReader((STAGE1_DIR / "reports" / "stage1_segment_ledger.csv").open(encoding="utf-8")))
    audio_val = list(csv.DictReader((STAGE1_DIR / "reports" / "audio_validation_report.csv").open(encoding="utf-8")))
    stage3_val = list(csv.DictReader((STAGE3_DIR / "verification" / "stage3_cache_validation.csv").open(encoding="utf-8")))

    # =========================================================================
    # ISSUE A: Speaker Identity & Cross-Recording Verification Audit
    # =========================================================================
    print("[Stage 4.1 - Issue A] Auditing speaker identity and cross-recording pairs...")
    turns_mono, all_clean_turns_mono = build_turn_records_for_condition(manifest, "16k_mono", scale_k=4)
    turns_dfn, _ = build_turn_records_for_condition(manifest, "dfn", scale_k=4)
    turns_norm, _ = build_turn_records_for_condition(manifest, "normalized", scale_k=4)

    # Map speaker -> usable turn count (Scale 4)
    usable_turn_counts: Dict[str, int] = {}
    for t in turns_mono:
        usable_turn_counts[t["global_speaker_id"]] = usable_turn_counts.get(t["global_speaker_id"], 0) + 1

    speaker_identity_rows: List[Dict[str, Any]] = []
    for sq in spk_quality:
        gid = sq["global_speaker_id"]
        n_usable_s4 = usable_turn_counts.get(gid, 0)
        speaker_identity_rows.append({
            "rec_id": sq["rec_id"],
            "local_speaker_id": sq["speaker"],
            "global_speaker_id": gid,
            "has_audio": sq["has_audio"],
            "is_collective_label": sq["is_collective_speaker_all"],
            "named_person_metadata_in_gt": False,
            "verified_cross_recording_link": "NONE",
            "total_gt_segments": int(sq["total_segments"]),
            "positive_dur_segments": int(sq["positive_dur_segments"]),
            "overlapping_segments": int(sq["overlapping_segments"]),
            "clean_non_overlapping_segments": int(sq["clean_single_speaker_segments"]),
            "clean_turns_with_scale4_subseg": n_usable_s4,
            "within_rec_positive_turn_pairs": int(n_usable_s4 * (n_usable_s4 - 1) // 2),
            "eligible_for_loto_prototype": n_usable_s4 >= 2,
            "speaker_status": sq["speaker_status"],
        })
    write_csv(STAGE4_1_DIR / "speaker_identity_audit.csv", speaker_identity_rows)

    # Audit verification pairs per recording and cross-recording
    verification_pair_rows: List[Dict[str, Any]] = []
    tot_within_pos = 0
    tot_within_neg = 0
    for entry in manifest:
        rec_id = entry["rec_id"]
        r_turns = [t for t in turns_mono if t["rec_id"] == rec_id]
        n_t = len(r_turns)
        pos_p = 0
        neg_p = 0
        spks_present = sorted(set(t["speaker"] for t in r_turns))
        multi_turn_spks = [
            s for s in spks_present if sum(1 for t in r_turns if t["speaker"] == s) >= 2
        ]
        for i in range(n_t):
            for j in range(i + 1, n_t):
                if r_turns[i]["speaker"] == r_turns[j]["speaker"]:
                    pos_p += 1
                else:
                    neg_p += 1
        tot_within_pos += pos_p
        tot_within_neg += neg_p
        eligible = (pos_p > 0 and neg_p > 0)

        # Also check Scale-4 non-adjacent subsegment pairs (|t_i - t_j| > 1.5s) AND turn-disjoint subsegment pairs
        t_data, r_data, meta = load_cache("16k_mono", rec_id, 4)
        gt_segs = json.loads((REPO_ROOT / entry["json_uniform"]).read_text(encoding="utf-8"))["segments"]
        turn_assign = assign_subsegments_to_gt_turns(meta, gt_segs)
        clean_idx = [
            i for i, m in enumerate(meta)
            if not m["gt_is_overlap"]
            and not str(m["gt_label"]).startswith(("NON_SPEECH", "PARTIAL", "OVERLAP", "AMBIGUOUS"))
        ]
        mids = np.array([meta[i][" midpoint"] for i in clean_idx], dtype=np.float32)
        spks_sub = [str(meta[i]["gt_label"]) for i in clean_idx]
        turns_sub = [turn_assign[i] for i in clean_idx]

        sub_nonadj_pos, sub_nonadj_neg = 0, 0
        sub_turndisj_pos, sub_turndisj_neg = 0, 0
        for a in range(len(clean_idx)):
            for b in range(a + 1, len(clean_idx)):
                same_s = (spks_sub[a] == spks_sub[b])
                if abs(float(mids[a] - mids[b])) > 1.5:
                    if same_s:
                        sub_nonadj_pos += 1
                    else:
                        sub_nonadj_neg += 1
                if turns_sub[a] is not None and turns_sub[b] is not None and turns_sub[a] != turns_sub[b]:
                    if same_s:
                        sub_turndisj_pos += 1
                    else:
                        sub_turndisj_neg += 1

        verification_pair_rows.append({
            "scope": f"WITHIN_RECORDING::{rec_id}",
            "rec_id": rec_id,
            "clean_gt_turns_stage1": int(entry["clean_non_overlapping_segments"]),
            "usable_clean_turns_scale4": n_t,
            "clean_speakers_present": len(spks_present),
            "speakers_with_ge2_clean_turns": len(multi_turn_spks),
            "turn_pos_pairs": pos_p,
            "turn_neg_pairs": neg_p,
            "turn_total_pairs": pos_p + neg_p,
            "share_of_within_rec_pairs": round((pos_p + neg_p) / 9817.0, 4),
            "turn_eer_eligible": eligible,
            "ineligibility_reason": (
                "ELIGIBLE"
                if eligible
                else (
                    "ZERO_NEGATIVE_PAIRS_ONLY_1_CLEAN_SPEAKER"
                    if neg_p == 0
                    else "ZERO_POSITIVE_PAIRS_EACH_SPEAKER_HAS_1_CLEAN_TURN"
                )
            ),
            "scale4_nonadj_subseg_pos_pairs": sub_nonadj_pos,
            "scale4_nonadj_subseg_neg_pairs": sub_nonadj_neg,
            "scale4_turndisj_subseg_pos_pairs": sub_turndisj_pos,
            "scale4_turndisj_subseg_neg_pairs": sub_turndisj_neg,
        })

    # Add summary rows for pooled within-recording and cross-recording pairs
    total_all_pairs = len(turns_mono) * (len(turns_mono) - 1) // 2
    cross_rec_pairs = total_all_pairs - (tot_within_pos + tot_within_neg)
    verification_pair_rows.append({
        "scope": "SUMMARY::ALL_WITHIN_RECORDING_TURN_PAIRS",
        "rec_id": "ALL_8_RECORDINGS_WITHIN",
        "clean_gt_turns_stage1": 264,
        "usable_clean_turns_scale4": len(turns_mono),
        "clean_speakers_present": 32,
        "speakers_with_ge2_clean_turns": sum(1 for r in speaker_identity_rows if r["eligible_for_loto_prototype"]),
        "turn_pos_pairs": tot_within_pos,
        "turn_neg_pairs": tot_within_neg,
        "turn_total_pairs": tot_within_pos + tot_within_neg,
        "share_of_within_rec_pairs": 1.0,
        "turn_eer_eligible": True,
        "ineligibility_reason": "VALID_WITHIN_RECORDING_POPULATION (Note: R06 holds 76.43% of pairs)",
        "scale4_nonadj_subseg_pos_pairs": sum(r["scale4_nonadj_subseg_pos_pairs"] for r in verification_pair_rows),
        "scale4_nonadj_subseg_neg_pairs": sum(r["scale4_nonadj_subseg_neg_pairs"] for r in verification_pair_rows),
        "scale4_turndisj_subseg_pos_pairs": sum(r["scale4_turndisj_subseg_pos_pairs"] for r in verification_pair_rows),
        "scale4_turndisj_subseg_neg_pairs": sum(r["scale4_turndisj_subseg_neg_pairs"] for r in verification_pair_rows),
    })
    verification_pair_rows.append({
        "scope": "SUMMARY::CROSS_RECORDING_TURN_PAIRS_ONLY",
        "rec_id": "CROSS_RECORDING_ONLY",
        "clean_gt_turns_stage1": 264,
        "usable_clean_turns_scale4": len(turns_mono),
        "clean_speakers_present": 32,
        "speakers_with_ge2_clean_turns": 0,
        "turn_pos_pairs": 0,
        "turn_neg_pairs": cross_rec_pairs,
        "turn_total_pairs": cross_rec_pairs,
        "share_of_within_rec_pairs": 0.0,
        "turn_eer_eligible": False,
        "ineligibility_reason": "INVALID_FOR_VERIFICATION_EER: 0 positive cross-recording pairs exist (no cross-recording speaker IDs)",
        "scale4_nonadj_subseg_pos_pairs": 0,
        "scale4_nonadj_subseg_neg_pairs": 0,
        "scale4_turndisj_subseg_pos_pairs": 0,
        "scale4_turndisj_subseg_neg_pairs": 0,
    })
    write_csv(STAGE4_1_DIR / "verification_pair_audit.csv", verification_pair_rows)

    # =========================================================================
    # ISSUE B: LORO Pairwise Verification & Threshold Leakage Correction
    # =========================================================================
    print("[Stage 4.1 - Issue B] Running strict 8-fold LORO pairwise verification with frozen thresholds...")
    loro_ver_rows: List[Dict[str, Any]] = []
    loro_thr_rows: List[Dict[str, Any]] = []
    loro_fusion_comp_rows: List[Dict[str, Any]] = []

    for cond, turns_cond in [("16k_mono", turns_mono), ("dfn", turns_dfn), ("normalized", turns_norm)]:
        # Build within-recording pairs per recording
        rec_pairs: Dict[str, Dict[str, np.ndarray]] = {}
        for entry in manifest:
            rec_id = entry["rec_id"]
            r_turns = [t for t in turns_cond if t["rec_id"] == rec_id]
            ts, rs, ys = [], [], []
            # Also store pair-averaged reliability indicators for LORO reliability-gated score fusion
            r_l2_pair, dfn_nr_pair, vad_pair, cs_diff_pair = [], [], [], []
            for i in range(len(r_turns)):
                for j in range(i + 1, len(r_turns)):
                    ts.append(float(np.dot(r_turns[i]["tita_emb"], r_turns[j]["tita_emb"])))
                    rs.append(float(np.dot(r_turns[i]["res_emb"], r_turns[j]["res_emb"])))
                    ys.append(1 if r_turns[i]["speaker"] == r_turns[j]["speaker"] else 0)
                    r_l2_pair.append(0.5 * (r_turns[i]["resnet293_l2_norm"] + r_turns[j]["resnet293_l2_norm"]))
                    dfn_nr_pair.append(0.5 * (r_turns[i]["dfn_noise_reduction_db"] + r_turns[j]["dfn_noise_reduction_db"]))
                    vad_pair.append(0.5 * (r_turns[i]["vad_conf_mean"] + r_turns[j]["vad_conf_mean"]))
                    cs_d_i = r_turns[i]["resnet293_cross_scale_cos"] - r_turns[i]["titanet_cross_scale_cos"]
                    cs_d_j = r_turns[j]["resnet293_cross_scale_cos"] - r_turns[j]["titanet_cross_scale_cos"]
                    cs_diff_pair.append(0.5 * (cs_d_i + cs_d_j))
            rec_pairs[rec_id] = {
                "t": np.array(ts, dtype=np.float64),
                "r": np.array(rs, dtype=np.float64),
                "y": np.array(ys, dtype=np.int32),
                "res_l2": np.array(r_l2_pair, dtype=np.float64),
                "dfn_nr": np.array(dfn_nr_pair, dtype=np.float64),
                "vad": np.array(vad_pair, dtype=np.float64),
                "cs_diff": np.array(cs_diff_pair, dtype=np.float64),
            }

        # Aggregate out-of-fold predictions across all 8 recordings
        oof_scores: Dict[str, List[np.ndarray]] = {
            "titanet": [],
            "resnet293": [],
            "equal_fusion": [],
            "zscore_fusion_train_cal": [],
            "reliability_gate_loro": [],
        }
        oof_preds: Dict[str, List[np.ndarray]] = {k: [] for k in oof_scores}
        oof_labels: List[np.ndarray] = []
        oof_rec_ids: List[str] = []

        for entry in manifest:
            held_rec = entry["rec_id"]
            # Training set: within-recording pairs from the other 7 recordings
            tr_t = np.concatenate([rec_pairs[r]["t"] for r in rec_pairs if r != held_rec and rec_pairs[r]["t"].size > 0])
            tr_r = np.concatenate([rec_pairs[r]["r"] for r in rec_pairs if r != held_rec and rec_pairs[r]["r"].size > 0])
            tr_y = np.concatenate([rec_pairs[r]["y"] for r in rec_pairs if r != held_rec and rec_pairs[r]["y"].size > 0])
            tr_res_l2 = np.concatenate([rec_pairs[r]["res_l2"] for r in rec_pairs if r != held_rec and rec_pairs[r]["res_l2"].size > 0])
            tr_vad = np.concatenate([rec_pairs[r]["vad"] for r in rec_pairs if r != held_rec and rec_pairs[r]["vad"].size > 0])
            tr_cs_diff = np.concatenate([rec_pairs[r]["cs_diff"] for r in rec_pairs if r != held_rec and rec_pairs[r]["cs_diff"].size > 0])

            # Fit training score normalization parameters strictly on training pairs
            mu_t, sd_t = float(np.mean(tr_t)), max(float(np.std(tr_t)), 1e-12)
            mu_r, sd_r = float(np.mean(tr_r)), max(float(np.std(tr_r)), 1e-12)

            tr_eq = 0.5 * tr_t + 0.5 * tr_r
            tr_tz = (tr_t - mu_t) / sd_t
            tr_rz = (tr_r - mu_r) / sd_r
            tr_zf = 0.5 * tr_tz + 0.5 * tr_rz

            # Fit LORO reliability-gated score fusion on training pairs only
            # Gate weight w_res in [0.2, 0.8] based on training-standardized (res_l2, vad, cs_diff)
            tr_feats = np.stack([tr_res_l2, tr_vad, tr_cs_diff], axis=-1)
            f_mu = np.mean(tr_feats, axis=0)
            f_sd = np.maximum(np.std(tr_feats, axis=0), 1e-6)
            tr_fz = (tr_feats - f_mu) / f_sd
            # Target on training pairs: signed advantage of r_z over t_z toward true label (+1 for pos, -1 for neg)
            y_sign = np.where(tr_y == 1, 1.0, -1.0)
            tr_adv = y_sign * (tr_rz - tr_tz)
            w_gate = np.linalg.solve(tr_fz.T @ tr_fz + 100.0 * np.eye(3), tr_fz.T @ tr_adv)
            b_gate = float(np.mean(tr_adv))
            tr_alpha = 1.0 / (1.0 + np.exp(-np.clip(tr_fz @ w_gate + b_gate, -5.0, 5.0)))
            tr_gate_sc = (1.0 - tr_alpha) * tr_tz + tr_alpha * tr_rz

            # Estimate frozen EER thresholds strictly on training pairs
            t_tr_info = compute_eer_and_mindcf(tr_t, tr_y)
            r_tr_info = compute_eer_and_mindcf(tr_r, tr_y)
            eq_tr_info = compute_eer_and_mindcf(tr_eq, tr_y)
            zf_tr_info = compute_eer_and_mindcf(tr_zf, tr_y)
            gt_tr_info = compute_eer_and_mindcf(tr_gate_sc, tr_y)

            thr_t = float(t_tr_info["threshold"])
            thr_r = float(r_tr_info["threshold"])
            thr_eq = float(eq_tr_info["threshold"])
            thr_zf = float(zf_tr_info["threshold"])
            thr_gt = float(gt_tr_info["threshold"])

            loro_thr_rows.append({
                "audio_condition": cond,
                "held_out_rec_id": held_rec,
                "train_pairs": int(tr_y.size),
                "train_pos_pairs": int(np.sum(tr_y == 1)),
                "train_neg_pairs": int(np.sum(tr_y == 0)),
                "mu_tita_train": round(mu_t, 5),
                "sd_tita_train": round(sd_t, 5),
                "mu_res_train": round(mu_r, 5),
                "sd_res_train": round(sd_r, 5),
                "thr_tita_train": round(thr_t, 5),
                "thr_res_train": round(thr_r, 5),
                "thr_eq_train": round(thr_eq, 5),
                "thr_zscore_train": round(thr_zf, 5),
                "thr_gate_train": round(thr_gt, 5),
                "train_eer_tita": t_tr_info["eer"],
                "train_eer_res": r_tr_info["eer"],
                "train_eer_eq": eq_tr_info["eer"],
                "train_eer_zscore": zf_tr_info["eer"],
                "train_eer_gate": gt_tr_info["eer"],
            })

            # Evaluate held-out recording using strictly frozen parameters
            te_t = rec_pairs[held_rec]["t"]
            te_r = rec_pairs[held_rec]["r"]
            te_y = rec_pairs[held_rec]["y"]
            n_pos = int(np.sum(te_y == 1))
            n_neg = int(np.sum(te_y == 0))
            eligible = (n_pos > 0 and n_neg > 0)

            te_eq = 0.5 * te_t + 0.5 * te_r
            te_tz = (te_t - mu_t) / sd_t
            te_rz = (te_r - mu_r) / sd_r
            te_zf = 0.5 * te_tz + 0.5 * te_rz

            if te_y.size > 0:
                te_feats = np.stack(
                    [rec_pairs[held_rec]["res_l2"], rec_pairs[held_rec]["vad"], rec_pairs[held_rec]["cs_diff"]],
                    axis=-1,
                )
                te_fz = (te_feats - f_mu) / f_sd
                te_alpha = 1.0 / (1.0 + np.exp(-np.clip(te_fz @ w_gate + b_gate, -5.0, 5.0)))
                te_gate_sc = (1.0 - te_alpha) * te_tz + te_alpha * te_rz
            else:
                te_gate_sc = np.array([], dtype=np.float64)

            if te_y.size > 0:
                oof_scores["titanet"].append(te_t)
                oof_scores["resnet293"].append(te_r)
                oof_scores["equal_fusion"].append(te_eq)
                oof_scores["zscore_fusion_train_cal"].append(te_zf)
                oof_scores["reliability_gate_loro"].append(te_gate_sc)

                oof_preds["titanet"].append((te_t >= thr_t).astype(np.int32))
                oof_preds["resnet293"].append((te_r >= thr_r).astype(np.int32))
                oof_preds["equal_fusion"].append((te_eq >= thr_eq).astype(np.int32))
                oof_preds["zscore_fusion_train_cal"].append((te_zf >= thr_zf).astype(np.int32))
                oof_preds["reliability_gate_loro"].append((te_gate_sc >= thr_gt).astype(np.int32))
                oof_labels.append(te_y)
                oof_rec_ids.extend([held_rec] * int(te_y.size))

            for m_key, sc_arr, thr_val in [
                ("titanet", te_t, thr_t),
                ("resnet293", te_r, thr_r),
                ("equal_fusion", te_eq, thr_eq),
                ("zscore_fusion_train_cal", te_zf, thr_zf),
                ("reliability_gate_loro", te_gate_sc, thr_gt),
            ]:
                ev = eval_at_frozen_threshold(sc_arr, te_y, thr_val) if eligible else {
                    "far_at_frozen_thr": None,
                    "frr_at_frozen_thr": None,
                    "hter_at_frozen_thr": None,
                    "act_dcf_p001": None,
                    "act_dcf_p01": None,
                    "roc_auc": None,
                    "exploratory_in_sample_eer": None,
                    "exploratory_in_sample_mindcf": None,
                }
                loro_ver_rows.append({
                    "audio_condition": cond,
                    "held_out_rec_id": held_rec,
                    "method": m_key,
                    "test_pos_pairs": n_pos,
                    "test_neg_pairs": n_neg,
                    "eligible_for_eer_and_hter": eligible,
                    "frozen_threshold": round(thr_val, 5),
                    **ev,
                })

        # Compute pooled out-of-fold (strictly held-out predictions) AND macro-across-eligible-recordings summary
        all_y = np.concatenate(oof_labels)
        p_t_oof = np.concatenate(oof_preds["titanet"])
        p_r_oof = np.concatenate(oof_preds["resnet293"])
        t_err_oof = (p_t_oof != all_y)
        r_err_oof = (p_r_oof != all_y)
        n00_oof = int(np.sum((~t_err_oof) & (~r_err_oof)))
        n10_oof = int(np.sum(t_err_oof & (~r_err_oof)))
        n01_oof = int(np.sum((~t_err_oof) & r_err_oof))
        n11_oof = int(np.sum(t_err_oof & r_err_oof))
        jacc_oof = float(n11_oof / max(n10_oof + n01_oof + n11_oof, 1))

        for m_key in oof_scores:
            all_s = np.concatenate(oof_scores[m_key])
            all_p = np.concatenate(oof_preds[m_key])
            far_oof = float(np.mean(all_p[all_y == 0] == 1))
            frr_oof = float(np.mean(all_p[all_y == 1] == 0))
            hter_oof = 0.5 * (far_oof + frr_oof)
            act_dcf_001 = (0.01 * frr_oof + 0.99 * far_oof) / 0.01
            act_dcf_010 = (0.10 * frr_oof + 0.90 * far_oof) / 0.10
            auc_oof = float(roc_auc_score(all_y, all_s))
            exp_in = compute_eer_and_mindcf(all_s, all_y)

            # Macro-average across the 6 eligible recordings (R01, R03, R05, R06, R07, R08)
            elig_rows = [
                r for r in loro_ver_rows
                if r["audio_condition"] == cond and r["method"] == m_key and r["eligible_for_eer_and_hter"]
            ]
            macro_hter = float(np.mean([r["hter_at_frozen_thr"] for r in elig_rows]))
            macro_auc = float(np.mean([r["roc_auc"] for r in elig_rows]))
            macro_exp_eer = float(np.mean([r["exploratory_in_sample_eer"] for r in elig_rows]))

            loro_fusion_comp_rows.append({
                "audio_condition": cond,
                "method": m_key,
                "protocol": "HELD_OUT_LORO_FROZEN_VS_EXPLORATORY",
                "total_within_rec_pairs": int(all_y.size),
                "eligible_recordings_count": len(elig_rows),
                "oof_pooled_far_frozen": round(far_oof, 4),
                "oof_pooled_frr_frozen": round(frr_oof, 4),
                "oof_pooled_hter_frozen": round(hter_oof, 4),
                "oof_pooled_act_dcf_p001": round(act_dcf_001, 4),
                "oof_pooled_act_dcf_p01": round(act_dcf_010, 4),
                "oof_pooled_roc_auc": round(auc_oof, 4),
                "macro_6rec_hter_frozen": round(macro_hter, 4),
                "macro_6rec_roc_auc": round(macro_auc, 4),
                "macro_6rec_exploratory_eer": round(macro_exp_eer, 4),
                "exploratory_full_dataset_eer_leaked": exp_in["eer"],
                "exploratory_full_dataset_mindcf_leaked": exp_in["mindcf_p001"],
                "oof_n00_both_ok": n00_oof if m_key == "titanet" else None,
                "oof_n10_tita_err_res_ok": n10_oof if m_key == "titanet" else None,
                "oof_n01_tita_ok_res_err": n01_oof if m_key == "titanet" else None,
                "oof_n11_both_err": n11_oof if m_key == "titanet" else None,
                "oof_error_jaccard_frozen": round(jacc_oof, 4) if m_key == "titanet" else None,
                "oof_complementary_error_frac_frozen": round(1.0 - jacc_oof, 4) if m_key == "titanet" else None,
            })

    write_csv(STAGE4_1_DIR / "loro_verification_metrics.csv", loro_ver_rows)
    write_csv(STAGE4_1_DIR / "loro_thresholds.csv", loro_thr_rows)
    write_csv(STAGE4_1_DIR / "loro_fusion_comparison.csv", loro_fusion_comp_rows)

    # =========================================================================
    # ISSUE C: Centroid Leakage Audit & Turn-Disjoint Prototype Correction
    # =========================================================================
    print("[Stage 4.1 - Issue C] Auditing centroid self-inclusion leakage and running turn-disjoint LOTO & split enrollment...")
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
        "cross_scale_cos_diff_res_minus_tita",
    ]

    reliability_loro_rows: List[Dict[str, Any]] = []
    reliability_per_rec_rows: List[Dict[str, Any]] = []
    loto_subseg_records_by_cond: Dict[str, List[Dict[str, Any]]] = {}

    for cond in ["16k_mono", "dfn"]:
        leaked_all5290_recs: List[Dict[str, Any]] = []
        leaked_clean4234_recs: List[Dict[str, Any]] = []
        loto_clean_subseg_recs: List[Dict[str, Any]] = []
        loto_clean_turn_recs: List[Dict[str, Any]] = []
        split_enroll_turn_recs: List[Dict[str, Any]] = []

        for entry in manifest:
            rec_id = entry["rec_id"]
            gt_segs = json.loads((REPO_ROOT / entry["json_uniform"]).read_text(encoding="utf-8"))["segments"]
            t_data, r_data, meta = load_cache(cond, rec_id, 4)
            t_norm = t_data["l2_normalized_embeddings"]
            r_norm = r_data["l2_normalized_embeddings"]
            turn_assign = assign_subsegments_to_gt_turns(meta, gt_segs)

            # 1. Stage 4 original self-included centroids
            spk_to_idxs: Dict[str, List[int]] = {}
            spk_to_turn_idxs: Dict[str, Dict[int, List[int]]] = {}
            for i, m in enumerate(meta):
                lbl = str(m["gt_label"])
                if not m["gt_is_overlap"] and not lbl.startswith(("NON_SPEECH", "PARTIAL", "OVERLAP", "AMBIGUOUS")):
                    spk_to_idxs.setdefault(lbl, []).append(i)
                    t_id = turn_assign[i]
                    if t_id is not None:
                        spk_to_turn_idxs.setdefault(lbl, {}).setdefault(t_id, []).append(i)

            valid_spks_leaked = [s for s, ix in spk_to_idxs.items() if len(ix) >= 2]
            if len(valid_spks_leaked) >= 2:
                t_cent_leaked = {
                    s: np.mean(t_norm[spk_to_idxs[s]], axis=0) / max(float(np.linalg.norm(np.mean(t_norm[spk_to_idxs[s]], axis=0))), 1e-12)
                    for s in valid_spks_leaked
                }
                r_cent_leaked = {
                    s: np.mean(r_norm[spk_to_idxs[s]], axis=0) / max(float(np.linalg.norm(np.mean(r_norm[spk_to_idxs[s]], axis=0))), 1e-12)
                    for s in valid_spks_leaked
                }
                for i, m in enumerate(meta):
                    raw_lbl = str(m["gt_label"])
                    base_spk = raw_lbl.split("::")[-1] if "::" in raw_lbl else raw_lbl
                    if base_spk not in valid_spks_leaked:
                        continue
                    others = [s for s in valid_spks_leaked if s != base_spk]
                    tm = float(np.dot(t_norm[i], t_cent_leaked[base_spk])) - max(float(np.dot(t_norm[i], t_cent_leaked[s])) for s in others)
                    rm = float(np.dot(r_norm[i], r_cent_leaked[base_spk])) - max(float(np.dot(r_norm[i], r_cent_leaked[s])) for s in others)
                    item = {
                        "rec_id": rec_id,
                        "subseg_idx": i,
                        "speaker": base_spk,
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
                        "cross_scale_cos_diff_res_minus_tita": float(m["resnet293_cross_scale_cos"] - m["titanet_cross_scale_cos"]),
                        "tita_margin": tm,
                        "res_margin": rm,
                        "delta_margin": rm - tm,
                        "tita_correct": int(tm > 0),
                        "res_correct": int(rm > 0),
                        "res_wins": int(rm > tm),
                    }
                    leaked_all5290_recs.append(item)
                    if not m["gt_is_overlap"] and not raw_lbl.startswith(("NON_SPEECH", "PARTIAL", "OVERLAP", "AMBIGUOUS")):
                        leaked_clean4234_recs.append(item)

            # 2. Corrected Protocol A: Turn-Disjoint Leave-One-Turn-Out (LOTO) Prototypes
            # Reference prototype for speaker s when evaluating turn t is built strictly from turns t' != t of speaker s.
            # Requires speaker s to have >= 2 distinct clean turns in the recording.
            multi_turn_spks = [s for s, tdict in spk_to_turn_idxs.items() if len(tdict) >= 2]
            if len(multi_turn_spks) >= 2:
                # Full-speaker centroids from all clean turns (used only when s_other != base_spk)
                t_cent_full = {}
                r_cent_full = {}
                for s in multi_turn_spks:
                    all_ix = [idx for ix_list in spk_to_turn_idxs[s].values() for idx in ix_list]
                    tc = np.mean(t_norm[all_ix], axis=0)
                    t_cent_full[s] = tc / max(float(np.linalg.norm(tc)), 1e-12)
                    rc = np.mean(r_norm[all_ix], axis=0)
                    r_cent_full[s] = rc / max(float(np.linalg.norm(rc)), 1e-12)

                for s in multi_turn_spks:
                    others = [o for o in multi_turn_spks if o != s]
                    for t_id, cur_idxs in spk_to_turn_idxs[s].items():
                        other_turn_idxs = [
                            idx for tid_other, ix_list in spk_to_turn_idxs[s].items() if tid_other != t_id for idx in ix_list
                        ]
                        if not other_turn_idxs:
                            continue
                        tc_loto = np.mean(t_norm[other_turn_idxs], axis=0)
                        tc_loto = tc_loto / max(float(np.linalg.norm(tc_loto)), 1e-12)
                        rc_loto = np.mean(r_norm[other_turn_idxs], axis=0)
                        rc_loto = rc_loto / max(float(np.linalg.norm(rc_loto)), 1e-12)

                        # Turn-level pooled embedding evaluated against LOTO prototype
                        t_turn_emb = np.mean(t_norm[cur_idxs], axis=0)
                        t_turn_emb = t_turn_emb / max(float(np.linalg.norm(t_turn_emb)), 1e-12)
                        r_turn_emb = np.mean(r_norm[cur_idxs], axis=0)
                        r_turn_emb = r_turn_emb / max(float(np.linalg.norm(r_turn_emb)), 1e-12)

                        tm_turn = float(np.dot(t_turn_emb, tc_loto)) - max(float(np.dot(t_turn_emb, t_cent_full[o])) for o in others)
                        rm_turn = float(np.dot(r_turn_emb, rc_loto)) - max(float(np.dot(r_turn_emb, r_cent_full[o])) for o in others)

                        loto_clean_turn_recs.append({
                            "rec_id": rec_id,
                            "turn_id": t_id,
                            "speaker": s,
                            "snr_local_db": float(np.mean([meta[i]["snr_local_db"] for i in cur_idxs])),
                            "rms_dbfs": float(np.mean([meta[i]["rms_dbfs"] for i in cur_idxs])),
                            "dfn_noise_reduction_db": float(np.mean([meta[i]["dfn_noise_reduction_db"] for i in cur_idxs])),
                            "spectral_flatness": float(np.mean([meta[i]["spectral_flatness"] for i in cur_idxs])),
                            "vad_conf_mean": float(np.mean([meta[i]["vad_conf_mean"] for i in cur_idxs])),
                            "vad_conf_std": float(np.mean([meta[i]["vad_conf_std"] for i in cur_idxs])),
                            "overlap_prob_mean": float(np.mean([meta[i]["overlap_prob_mean"] for i in cur_idxs])),
                            "titanet_l2_norm": float(np.mean([meta[i]["titanet_l2_norm"] for i in cur_idxs])),
                            "resnet293_l2_norm": float(np.mean([meta[i]["resnet293_l2_norm"] for i in cur_idxs])),
                            "titanet_cross_scale_cos": float(np.mean([meta[i]["titanet_cross_scale_cos"] for i in cur_idxs])),
                            "resnet293_cross_scale_cos": float(np.mean([meta[i]["resnet293_cross_scale_cos"] for i in cur_idxs])),
                            "cross_scale_cos_diff_res_minus_tita": float(
                                np.mean([meta[i]["resnet293_cross_scale_cos"] - meta[i]["titanet_cross_scale_cos"] for i in cur_idxs])
                            ),
                            "tita_margin": tm_turn,
                            "res_margin": rm_turn,
                            "delta_margin": rm_turn - tm_turn,
                            "tita_correct": int(tm_turn > 0),
                            "res_correct": int(rm_turn > 0),
                            "res_wins": int(rm_turn > tm_turn),
                        })

                        # Subsegment-level evaluated against turn-disjoint LOTO prototype
                        for i in cur_idxs:
                            m = meta[i]
                            tm = float(np.dot(t_norm[i], tc_loto)) - max(float(np.dot(t_norm[i], t_cent_full[o])) for o in others)
                            rm = float(np.dot(r_norm[i], rc_loto)) - max(float(np.dot(r_norm[i], r_cent_full[o])) for o in others)
                            loto_clean_subseg_recs.append({
                                "rec_id": rec_id,
                                "subseg_idx": i,
                                "turn_id": t_id,
                                "speaker": s,
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
                                "cross_scale_cos_diff_res_minus_tita": float(m["resnet293_cross_scale_cos"] - m["titanet_cross_scale_cos"]),
                                "tita_margin": tm,
                                "res_margin": rm,
                                "delta_margin": rm - tm,
                                "tita_correct": int(tm > 0),
                                "res_correct": int(rm > 0),
                                "res_wins": int(rm > tm),
                            })

                # 3. Corrected Protocol B: Independent Chronological Split-Turn Enrollment vs Held-Out Evaluation Turns
                # First half of each speaker's turns -> Enrollment Prototype; Second half -> Evaluation Turns
                t_cent_enroll = {}
                r_cent_enroll = {}
                eval_turns_by_spk = {}
                for s in multi_turn_spks:
                    sorted_tids = sorted(spk_to_turn_idxs[s].keys())
                    n_enroll = max(1, len(sorted_tids) // 2)
                    enroll_tids = sorted_tids[:n_enroll]
                    eval_tids = sorted_tids[n_enroll:]
                    enroll_ix = [idx for tid in enroll_tids for idx in spk_to_turn_idxs[s][tid]]
                    tc_e = np.mean(t_norm[enroll_ix], axis=0)
                    t_cent_enroll[s] = tc_e / max(float(np.linalg.norm(tc_e)), 1e-12)
                    rc_e = np.mean(r_norm[enroll_ix], axis=0)
                    r_cent_enroll[s] = rc_e / max(float(np.linalg.norm(rc_e)), 1e-12)
                    eval_turns_by_spk[s] = eval_tids

                for s in multi_turn_spks:
                    others = [o for o in multi_turn_spks if o != s]
                    for t_id in eval_turns_by_spk[s]:
                        cur_idxs = spk_to_turn_idxs[s][t_id]
                        t_turn_emb = np.mean(t_norm[cur_idxs], axis=0)
                        t_turn_emb = t_turn_emb / max(float(np.linalg.norm(t_turn_emb)), 1e-12)
                        r_turn_emb = np.mean(r_norm[cur_idxs], axis=0)
                        r_turn_emb = r_turn_emb / max(float(np.linalg.norm(r_turn_emb)), 1e-12)

                        tm_sp = float(np.dot(t_turn_emb, t_cent_enroll[s])) - max(float(np.dot(t_turn_emb, t_cent_enroll[o])) for o in others)
                        rm_sp = float(np.dot(r_turn_emb, r_cent_enroll[s])) - max(float(np.dot(r_turn_emb, r_cent_enroll[o])) for o in others)

                        split_enroll_turn_recs.append({
                            "rec_id": rec_id,
                            "turn_id": t_id,
                            "speaker": s,
                            "snr_local_db": float(np.mean([meta[i]["snr_local_db"] for i in cur_idxs])),
                            "rms_dbfs": float(np.mean([meta[i]["rms_dbfs"] for i in cur_idxs])),
                            "dfn_noise_reduction_db": float(np.mean([meta[i]["dfn_noise_reduction_db"] for i in cur_idxs])),
                            "spectral_flatness": float(np.mean([meta[i]["spectral_flatness"] for i in cur_idxs])),
                            "vad_conf_mean": float(np.mean([meta[i]["vad_conf_mean"] for i in cur_idxs])),
                            "vad_conf_std": float(np.mean([meta[i]["vad_conf_std"] for i in cur_idxs])),
                            "overlap_prob_mean": float(np.mean([meta[i]["overlap_prob_mean"] for i in cur_idxs])),
                            "titanet_l2_norm": float(np.mean([meta[i]["titanet_l2_norm"] for i in cur_idxs])),
                            "resnet293_l2_norm": float(np.mean([meta[i]["resnet293_l2_norm"] for i in cur_idxs])),
                            "titanet_cross_scale_cos": float(np.mean([meta[i]["titanet_cross_scale_cos"] for i in cur_idxs])),
                            "resnet293_cross_scale_cos": float(np.mean([meta[i]["resnet293_cross_scale_cos"] for i in cur_idxs])),
                            "cross_scale_cos_diff_res_minus_tita": float(
                                np.mean([meta[i]["resnet293_cross_scale_cos"] - meta[i]["titanet_cross_scale_cos"] for i in cur_idxs])
                            ),
                            "tita_margin": tm_sp,
                            "res_margin": rm_sp,
                            "delta_margin": rm_sp - tm_sp,
                            "tita_correct": int(tm_sp > 0),
                            "res_correct": int(rm_sp > 0),
                            "res_wins": int(rm_sp > tm_sp),
                        })

        loto_subseg_records_by_cond[cond] = loto_clean_subseg_recs

        # Evaluate each indicator across all 4 protocols + strict LORO sign-calibrated held-out evaluation
        for proto_name, dataset in [
            ("1_stage4_leaked_all5290", leaked_all5290_recs),
            ("2_self_included_clean4234", leaked_clean4234_recs),
            ("3_corrected_LOTO_clean_subsegs", loto_clean_subseg_recs),
            ("4_corrected_LOTO_clean_turns", loto_clean_turn_recs),
            ("5_corrected_split_enrollment_turns", split_enroll_turn_recs),
        ]:
            t_m = np.array([x["tita_margin"] for x in dataset], dtype=np.float64)
            r_m = np.array([x["res_margin"] for x in dataset], dtype=np.float64)
            d_m = np.array([x["delta_margin"] for x in dataset], dtype=np.float64)
            t_ok = np.array([x["tita_correct"] for x in dataset], dtype=np.int32)
            r_ok = np.array([x["res_correct"] for x in dataset], dtype=np.int32)
            r_win = np.array([x["res_wins"] for x in dataset], dtype=np.int32)
            recs_present = sorted(set(x["rec_id"] for x in dataset))

            for ind in indicator_names:
                vals = np.array([x[ind] for x in dataset], dtype=np.float64)
                if np.std(vals) < 1e-12:
                    rho_t = rho_r = rho_d = float("nan")
                    auroc_t_raw = auroc_r_raw = auroc_win_raw = 0.5
                    loro_auroc_t = loro_auroc_r = loro_auroc_win = 0.5
                else:
                    rho_t = float(scipy.stats.spearmanr(vals, t_m)[0])
                    rho_r = float(scipy.stats.spearmanr(vals, r_m)[0])
                    rho_d = float(scipy.stats.spearmanr(vals, d_m)[0])
                    auroc_t_raw = float(roc_auc_score(t_ok, vals)) if len(set(t_ok)) == 2 else 0.5
                    auroc_r_raw = float(roc_auc_score(r_ok, vals)) if len(set(r_ok)) == 2 else 0.5
                    auroc_win_raw = float(roc_auc_score(r_win, vals)) if len(set(r_win)) == 2 else 0.5

                    # LORO held-out evaluation: fit direction (+1 or -1) and z-score scaling on training recordings only
                    oof_pred_t, oof_pred_r, oof_pred_w = [], [], []
                    oof_true_t, oof_true_r, oof_true_w = [], [], []
                    for h_rec in recs_present:
                        tr_sub = [x for x in dataset if x["rec_id"] != h_rec]
                        te_sub = [x for x in dataset if x["rec_id"] == h_rec]
                        tr_v = np.array([x[ind] for x in tr_sub], dtype=np.float64)
                        te_v = np.array([x[ind] for x in te_sub], dtype=np.float64)
                        mu_v, sd_v = float(np.mean(tr_v)), max(float(np.std(tr_v)), 1e-12)
                        te_vz = (te_v - mu_v) / sd_v

                        # Determine sign strictly on training recordings
                        sign_t = 1.0 if float(scipy.stats.spearmanr(tr_v, [x["tita_margin"] for x in tr_sub])[0]) >= 0 else -1.0
                        sign_r = 1.0 if float(scipy.stats.spearmanr(tr_v, [x["res_margin"] for x in tr_sub])[0]) >= 0 else -1.0
                        sign_w = 1.0 if float(scipy.stats.spearmanr(tr_v, [x["delta_margin"] for x in tr_sub])[0]) >= 0 else -1.0

                        oof_pred_t.append(sign_t * te_vz)
                        oof_pred_r.append(sign_r * te_vz)
                        oof_pred_w.append(sign_w * te_vz)
                        oof_true_t.append(np.array([x["tita_correct"] for x in te_sub], dtype=np.int32))
                        oof_true_r.append(np.array([x["res_correct"] for x in te_sub], dtype=np.int32))
                        oof_true_w.append(np.array([x["res_wins"] for x in te_sub], dtype=np.int32))

                    yt_all = np.concatenate(oof_true_t)
                    yr_all = np.concatenate(oof_true_r)
                    yw_all = np.concatenate(oof_true_w)
                    loro_auroc_t = float(roc_auc_score(yt_all, np.concatenate(oof_pred_t))) if len(set(yt_all)) == 2 else 0.5
                    loro_auroc_r = float(roc_auc_score(yr_all, np.concatenate(oof_pred_r))) if len(set(yr_all)) == 2 else 0.5
                    loro_auroc_win = float(roc_auc_score(yw_all, np.concatenate(oof_pred_w))) if len(set(yw_all)) == 2 else 0.5

                reliability_loro_rows.append({
                    "audio_condition": cond,
                    "protocol": proto_name,
                    "sample_count": len(dataset),
                    "recordings_count": len(recs_present),
                    "indicator": ind,
                    "spearman_rho_tita_margin": round(rho_t, 4) if not math.isnan(rho_t) else None,
                    "spearman_rho_res_margin": round(rho_r, 4) if not math.isnan(rho_r) else None,
                    "spearman_rho_delta_margin": round(rho_d, 4) if not math.isnan(rho_d) else None,
                    "in_sample_auroc_tita_rectified": round(max(auroc_t_raw, 1.0 - auroc_t_raw), 4),
                    "in_sample_auroc_res_rectified": round(max(auroc_r_raw, 1.0 - auroc_r_raw), 4),
                    "loro_held_out_auroc_tita_correctness": round(loro_auroc_t, 4),
                    "loro_held_out_auroc_res_correctness": round(loro_auroc_r, 4),
                    "loro_held_out_auroc_res_wins_margin": round(loro_auroc_win, 4),
                })

                # Also record per-recording breakdown for the primary corrected protocol (3_corrected_LOTO_clean_subsegs)
                if proto_name == "3_corrected_LOTO_clean_subsegs":
                    for h_rec in recs_present:
                        sub_r = [x for x in dataset if x["rec_id"] == h_rec]
                        vr = np.array([x[ind] for x in sub_r], dtype=np.float64)
                        tm_r = np.array([x["tita_margin"] for x in sub_r], dtype=np.float64)
                        rm_r = np.array([x["res_margin"] for x in sub_r], dtype=np.float64)
                        dm_r = np.array([x["delta_margin"] for x in sub_r], dtype=np.float64)
                        tok_r = np.array([x["tita_correct"] for x in sub_r], dtype=np.int32)
                        rok_r = np.array([x["res_correct"] for x in sub_r], dtype=np.int32)
                        rwin_r = np.array([x["res_wins"] for x in sub_r], dtype=np.int32)
                        if np.std(vr) < 1e-12:
                            r_t = r_r = r_d = None
                            a_t = a_r = a_w = 0.5
                        else:
                            r_t = round(float(scipy.stats.spearmanr(vr, tm_r)[0]), 4)
                            r_r = round(float(scipy.stats.spearmanr(vr, rm_r)[0]), 4)
                            r_d = round(float(scipy.stats.spearmanr(vr, dm_r)[0]), 4)
                            a_t = round(float(roc_auc_score(tok_r, vr)), 4) if len(set(tok_r)) == 2 else None
                            a_r = round(float(roc_auc_score(rok_r, vr)), 4) if len(set(rok_r)) == 2 else None
                            a_w = round(float(roc_auc_score(rwin_r, vr)), 4) if len(set(rwin_r)) == 2 else None
                        reliability_per_rec_rows.append({
                            "audio_condition": cond,
                            "rec_id": h_rec,
                            "protocol": proto_name,
                            "subsegment_count": len(sub_r),
                            "indicator": ind,
                            "spearman_rho_tita_margin": r_t,
                            "spearman_rho_res_margin": r_r,
                            "spearman_rho_delta_margin": r_d,
                            "auroc_tita_correctness_raw": a_t,
                            "auroc_res_correctness_raw": a_r,
                            "auroc_res_wins_raw": a_w,
                        })

    write_csv(STAGE4_1_DIR / "reliability_loro_metrics.csv", reliability_loro_rows)
    write_csv(STAGE4_1_DIR / "reliability_per_recording.csv", reliability_per_rec_rows)

    # =========================================================================
    # ISSUE E & SECTION 9: LORO Procrustes & LORO Gate Audit + Downstream Diarization
    # =========================================================================
    print("[Stage 4.1 - Issue E & Section 9] Auditing LORO Procrustes & LORO Gate and re-evaluating downstream diarization...")
    corrected_loro_rows: List[Dict[str, Any]] = []

    # 1. Test LORO Procrustes under 4 weighting/filtering configurations on 16k_mono and dfn (Scale 0 and Scale 4)
    for cond in ["16k_mono", "dfn"]:
        for scale_k in [0, 4]:
            win_sec = dict((s[0], s[1]) for s in SCALES)[scale_k]
            rec_all: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            rec_clean: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for entry in manifest:
                rec_id = entry["rec_id"]
                t_d, r_d, meta = load_cache(cond, rec_id, scale_k)
                tn = t_d["l2_normalized_embeddings"]
                rn = r_d["l2_normalized_embeddings"]
                rec_all[rec_id] = (tn, rn)
                c_idx = [
                    i for i, m in enumerate(meta)
                    if not m["gt_is_overlap"]
                    and not str(m["gt_label"]).startswith(("NON_SPEECH", "PARTIAL", "OVERLAP", "AMBIGUOUS"))
                ]
                rec_clean[rec_id] = (tn[c_idx], rn[c_idx])

            for held_rec in [e["rec_id"] for e in manifest]:
                test_Y, test_X = rec_all[held_rec]
                for variant_name, use_clean, equal_rec_weight, center_cols in [
                    ("procrustes_stage4_subseg_weighted_all", False, False, False),
                    ("procrustes_equal_rec_weighted_all", False, True, False),
                    ("procrustes_equal_rec_weighted_clean_only", True, True, False),
                    ("procrustes_equal_rec_weighted_centered", False, True, True),
                ]:
                    src_dict = rec_clean if use_clean else rec_all
                    M = np.zeros((256, 192), dtype=np.float64)
                    tr_X_list, tr_Y_list = [], []
                    for r_id, (Y_r, X_r) in src_dict.items():
                        if r_id == held_rec or X_r.shape[0] == 0:
                            continue
                        tr_X_list.append(X_r)
                        tr_Y_list.append(Y_r)
                    tr_X_cat = np.concatenate(tr_X_list, axis=0)
                    tr_Y_cat = np.concatenate(tr_Y_list, axis=0)
                    mu_x = np.mean(tr_X_cat, axis=0, keepdims=True) if center_cols else np.zeros((1, 256), dtype=np.float32)
                    mu_y = np.mean(tr_Y_cat, axis=0, keepdims=True) if center_cols else np.zeros((1, 192), dtype=np.float32)

                    for r_id, (Y_r, X_r) in src_dict.items():
                        if r_id == held_rec or X_r.shape[0] == 0:
                            continue
                        Xc = X_r - mu_x
                        Yc = Y_r - mu_y
                        w_r = (1.0 / float(X_r.shape[0])) if equal_rec_weight else 1.0
                        M += w_r * (Xc.T @ Yc)

                    U, _, Vt = scipy.linalg.svd(M, full_matrices=False)
                    P = (U @ Vt).astype(np.float32)

                    # Evaluate on training and held-out test
                    proj_tr = (tr_X_cat - mu_x) @ P + mu_y
                    tr_r2 = 1.0 - float(np.sum((proj_tr - tr_Y_cat) ** 2)) / max(
                        float(np.sum((tr_Y_cat - np.mean(tr_Y_cat, axis=0, keepdims=True)) ** 2)), 1e-12
                    )
                    proj_te = (test_X - mu_x) @ P + mu_y
                    te_r2 = 1.0 - float(np.sum((proj_te - test_Y) ** 2)) / max(
                        float(np.sum((test_Y - np.mean(test_Y, axis=0, keepdims=True)) ** 2)), 1e-12
                    )
                    proj_te_n = proj_te / np.maximum(np.linalg.norm(proj_te, axis=-1, keepdims=True), 1e-12)
                    te_cos = float(np.mean(np.sum(proj_te_n * test_Y, axis=-1)))

                    corrected_loro_rows.append({
                        "component": variant_name,
                        "audio_condition": cond,
                        "scale_idx": scale_k,
                        "window_sec": win_sec,
                        "held_out_rec_id": held_rec,
                        "train_r2": round(tr_r2, 4),
                        "test_r2": round(te_r2, 4),
                        "test_mean_cos": round(te_cos, 4),
                    })

    # 2. Fit Corrected LORO Calibrated Reliability Gate:
    #    - Uses strictly turn-disjoint LOTO targets (no self-inclusion centroid leakage)
    #    - Uses equal recording weighting across the eligible training recordings (so R06 does not dominate 36% of weight)
    #    - Uses only verified informative features (snr_local_db, dfn_noise_reduction_db, vad_conf_mean, resnet293_l2_norm,
    #      titanet_cross_scale_cos, resnet293_cross_scale_cos; drops constant overlap_prob_mean=0)
    corrected_gate_features = [
        "snr_local_db",
        "dfn_noise_reduction_db",
        "vad_conf_mean",
        "resnet293_l2_norm",
        "titanet_cross_scale_cos",
        "resnet293_cross_scale_cos",
    ]
    dfn_loto_s4 = loto_subseg_records_by_cond["dfn"]
    corrected_loro_gate_params: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    for entry in manifest:
        held_rec = entry["rec_id"]
        tr_items = [x for x in dfn_loto_s4 if x["rec_id"] != held_rec]
        tr_recs = sorted(set(x["rec_id"] for x in tr_items))
        rec_counts = {r: sum(1 for x in tr_items if x["rec_id"] == r) for r in tr_recs}
        # Equal recording weight per training recording
        weights = np.array([1.0 / (len(tr_recs) * rec_counts[x["rec_id"]]) for x in tr_items], dtype=np.float64)
        weights = weights / np.sum(weights)

        X_tr = np.array([[x[f] for f in corrected_gate_features] for x in tr_items], dtype=np.float64)
        y_tr = np.array([x["delta_margin"] for x in tr_items], dtype=np.float64)

        mu_x = np.sum(X_tr * weights[:, None], axis=0)
        var_x = np.sum(((X_tr - mu_x) ** 2) * weights[:, None], axis=0)
        sd_x = np.maximum(np.sqrt(var_x), 1e-6)
        Xz = (X_tr - mu_x) / sd_x
        b = float(np.sum(y_tr * weights))
        yc = y_tr - b

        # Weighted ridge regression (lambda_eff = 0.05 on normalized weights, equivalent to ~150 on unweighted N=2500)
        W_diag = weights[:, None]
        w = np.linalg.solve((Xz * W_diag).T @ Xz + 0.05 * np.eye(Xz.shape[1]), (Xz * W_diag).T @ yc)
        corrected_loro_gate_params[held_rec] = (mu_x, sd_x, w, b)

        # Evaluate gate prediction on held-out LOTO subsegments (if held_rec has >=2 multi-turn speakers)
        te_items = [x for x in dfn_loto_s4 if x["rec_id"] == held_rec]
        pred_tr = Xz @ w + b
        tr_rho = float(scipy.stats.spearmanr(pred_tr, y_tr)[0])
        if len(te_items) >= 5:
            X_te = np.array([[x[f] for f in corrected_gate_features] for x in te_items], dtype=np.float64)
            y_te = np.array([x["delta_margin"] for x in te_items], dtype=np.float64)
            pred_te = ((X_te - mu_x) / sd_x) @ w + b
            te_rho = float(scipy.stats.spearmanr(pred_te, y_te)[0])
            te_r2 = 1.0 - float(np.sum((pred_te - y_te) ** 2)) / max(float(np.sum((y_te - np.mean(y_te)) ** 2)), 1e-12)
        else:
            te_rho = float("nan")
            te_r2 = float("nan")

        corrected_loro_rows.append({
            "component": "loro_gate_corrected_loto_equal_rec_weight",
            "audio_condition": "dfn",
            "scale_idx": 4,
            "window_sec": 0.5,
            "held_out_rec_id": held_rec,
            "train_r2": round(tr_rho, 4),  # store train spearman rho in train_r2 col for gate
            "test_r2": round(te_r2, 4) if not math.isnan(te_r2) else None,
            "test_mean_cos": round(te_rho, 4) if not math.isnan(te_rho) else None,  # held-out spearman rho
        })

    write_csv(STAGE4_1_DIR / "corrected_loro_results.csv", corrected_loro_rows)

    # 3. Load existing Stage 4 diarization per-recording results and run:
    #    (a) 8b_corrected_loro_gate_loto_affinity (leak-free LOTO targets + equal recording weights)
    #    (b) 12_oracle_spk_count_equal_affinity (explicit diagnostic to test whether speaker-count estimation is the primary bottleneck)
    stage4_diar_rows = list(csv.DictReader((STAGE4_DIR / "tables" / "stage4_diarization_per_recording.csv").open(encoding="utf-8")))
    parsed_diar_rows: List[Dict[str, Any]] = []
    for r in stage4_diar_rows:
        parsed_diar_rows.append({
            "rec_id": r["rec_id"],
            "method": r["method"],
            "duration_sec": float(r["duration_sec"]),
            "ref_speakers_pos": int(r["ref_speakers_pos"]),
            "pred_speakers": int(r["pred_speakers"]),
            "speaker_count_error_abs": int(r["speaker_count_error_abs"]),
            "speaker_count_signed_bias": int(r["speaker_count_signed_bias"]),
            "single_speaker_collapse": r["single_speaker_collapse"] == "True",
            "der_collar0_ovl": float(r["der_collar0_ovl"]),
            "der_collar025_noovl": float(r["der_collar025_noovl"]),
            "jer_collar0": float(r["jer_collar0"]),
            "false_alarm": float(r["false_alarm"]),
            "miss": float(r["miss"]),
            "confusion": float(r["confusion"]),
            "cpwer": float(r["cpwer"]),
            "sawer": float(r["sawer"]),
            "micro_sacc": float(r["micro_sacc"]),
            "micro_wae": float(r["micro_wae"]),
            "timeline_accuracy": float(r["timeline_accuracy"]),
            "macro_sacc": float(r["macro_sacc"]),
        })

    ms_weights_tensor = torch.tensor([1.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float32, device="cuda")
    clustering = SpeakerClustering(cuda=True)

    for entry in manifest:
        rec_id = entry["rec_id"]
        dur = float(entry["duration_sec"])
        dfn_stem = Path(entry["dfn_wav"]).stem
        gt_json_uni = REPO_ROOT / entry["json_uniform"]
        gt_rttm_uni = REPO_ROOT / entry["rttm_uniform"]
        aligned_words_p = STAGE2_DIR / "whisperx" / f"{rec_id}_aligned.json"

        t_scales_dfn, r_scales_dfn, ts_scales = [], [], []
        meta_s4 = []
        for scale_k, _, _, _ in SCALES:
            t_d, r_d, meta_k = load_cache("dfn", rec_id, scale_k)
            t_scales_dfn.append(torch.from_numpy(t_d["l2_normalized_embeddings"]).to("cuda"))
            r_scales_dfn.append(torch.from_numpy(r_d["l2_normalized_embeddings"]).to("cuda"))
            ts_scales.append(torch.from_numpy(np.stack([t_d["starts"], t_d["ends"]], axis=-1).astype(np.float32)).to("cuda"))
            if scale_k == 4:
                meta_s4 = meta_k

        A_tita = getMultiScaleCosAffinityMatrix(ms_weights_tensor, t_scales_dfn, ts_scales, torch.device("cuda"))
        A_res = getMultiScaleCosAffinityMatrix(ms_weights_tensor, r_scales_dfn, ts_scales, torch.device("cuda"))
        A_z_tita = offdiag_zscore_affinity(A_tita)
        A_z_res = offdiag_zscore_affinity(A_res)
        A_eq = 0.5 * A_tita + 0.5 * A_res

        # Corrected LORO Gate Affinity (Method 8b)
        mu_x, sd_x, w_g, b_g = corrected_loro_gate_params[rec_id]
        X_te = np.array([[m[f] for f in corrected_gate_features] for m in meta_s4], dtype=np.float64)
        pred_delta = ((X_te - mu_x) / sd_x) @ w_g + b_g
        g_res_i = np.clip(0.5 + 2.0 * pred_delta, 0.15, 0.85).astype(np.float32)
        G_res_mat = torch.from_numpy(0.5 * (g_res_i[:, None] + g_res_i[None, :])).to("cuda").float()
        A_loro_corr = (1.0 - G_res_mat) * A_z_tita + G_res_mat * A_z_res

        gt_payload = p20_common.read_json(gt_json_uni)
        gt_segments = gt_payload["segments"]
        gt_rttm_segs = p20_common.rttm_to_segments(gt_rttm_uni)
        ref_spks_pos = len(set(s["speaker"] for s in gt_rttm_segs))
        aligned_words = p20_common.read_json(aligned_words_p)
        base_ts = ts_scales[-1]

        for m_name, A_mat, oracle_k in [
            ("8b_loro_gate_corrected_loto_affinity", A_loro_corr, -1),
            ("12_diagnostic_oracle_spkcount_equal_affinity", A_eq, ref_spks_pos),
        ]:
            torch.manual_seed(42)
            np.random.seed(42)
            c_labels = clustering.forward_unit_infer(
                mat=A_mat.to(torch.device("cuda")).float(),
                oracle_num_speakers=oracle_k,
                max_num_speakers=10,
                max_rp_threshold=0.25,
                sparse_search_volume=30,
                est_num_of_spk_enhanced=torch.tensor(-1),
                fixed_thres=-1.0,
                kmeans_random_trials=1,
            )
            labels, _ = generate_cluster_labels(base_ts.cpu().numpy(), c_labels.cpu().numpy())
            rttm_dir = STAGE4_1_DIR / "rttms" / m_name
            rttm_dir.mkdir(parents=True, exist_ok=True)
            labels_to_rttmfile(labels, dfn_stem, str(rttm_dir))
            pred_rttm_path = rttm_dir / f"{dfn_stem}.rttm"

            diar_segments = p20_common.rttm_to_segments(pred_rttm_path)
            diar_eval = p20_common.diarization_metrics(gt_rttm_uni, pred_rttm_path)
            der_025 = compute_der_collar025_noovl(gt_rttm_uni, pred_rttm_path)
            words_attr = p20_common.attributed_words(aligned_words, diar_segments)
            segs_attr = p20_common.group_words(words_attr)
            attr_path = rttm_dir / f"{rec_id}_{m_name}_attributed.json"
            p20_common.write_json(
                attr_path,
                {"file": f"{rec_id}.wav", "method": m_name, "words": words_attr, "segments": segs_attr},
            )
            f_metrics = p20_common.fusion_metrics(gt_json_uni, attr_path, dur)
            macro_info = compute_macro_sacc(gt_segments, words_attr, f_metrics.get("speaker_mapping", {}))
            pred_spks = len(set(s["speaker"] for s in diar_segments))

            parsed_diar_rows.append({
                "rec_id": rec_id,
                "method": m_name,
                "duration_sec": round(dur, 3),
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
            })

    method_order = [
        "1_titanet_nmesc",
        "2_resnet293_nmesc",
        "3_naive_concat_448d",
        "4_equal_affinity_050",
        "5_zscore_affinity_050",
        "6_heuristic_quality_affinity",
        "7_loro_procrustes_192d",
        "8_loro_calibrated_gate_affinity",
        "8b_loro_gate_corrected_loto_affinity",
        "9_titanet_mono_nmesc",
        "10_resnet293_mono_nmesc",
        "11_hybrid_titaDFN_resMONO_zscore",
        "12_diagnostic_oracle_spkcount_equal_affinity",
    ]
    parsed_diar_rows.sort(key=lambda x: (x["rec_id"], method_order.index(x["method"]) if x["method"] in method_order else 99))
    write_csv(STAGE4_1_DIR / "corrected_diarization_per_recording.csv", parsed_diar_rows)

    diar_summary_rows: List[Dict[str, Any]] = []
    for m_name in method_order:
        m_rows = [r for r in parsed_diar_rows if r["method"] == m_name]
        if not m_rows:
            continue
        tot_d = sum(r["duration_sec"] for r in m_rows)
        diar_summary_rows.append({
            "method": m_name,
            "macro_der_collar0_ovl": round(float(np.mean([r["der_collar0_ovl"] for r in m_rows])), 4),
            "dur_weighted_der_collar0_ovl": round(float(sum(r["der_collar0_ovl"] * r["duration_sec"] for r in m_rows) / tot_d), 4),
            "macro_der_collar025_noovl": round(float(np.mean([r["der_collar025_noovl"] for r in m_rows])), 4),
            "dur_weighted_der_collar025_noovl": round(float(sum(r["der_collar025_noovl"] * r["duration_sec"] for r in m_rows) / tot_d), 4),
            "macro_jer": round(float(np.mean([r["jer_collar0"] for r in m_rows])), 4),
            "spk_count_mae": round(float(np.mean([r["speaker_count_error_abs"] for r in m_rows])), 4),
            "spk_count_signed_bias": round(float(np.mean([r["speaker_count_signed_bias"] for r in m_rows])), 4),
            "single_spk_collapse_count": int(sum(1 for r in m_rows if r["single_speaker_collapse"])),
            "macro_cpwer": round(float(np.mean([r["cpwer"] for r in m_rows])), 4),
            "macro_sawer": round(float(np.mean([r["sawer"] for r in m_rows])), 4),
            "macro_micro_sacc": round(float(np.mean([r["micro_sacc"] for r in m_rows])), 4),
            "macro_macro_sacc": round(float(np.mean([r["macro_sacc"] for r in m_rows])), 4),
        })
    write_csv(STAGE4_1_DIR / "corrected_diarization_summary.csv", diar_summary_rows)

    # =========================================================================
    # ISSUE D: Clustered Bootstrap Uncertainty & Paired Method Differences
    # =========================================================================
    print("[Stage 4.1 - Issue D] Running recording-level, turn-clustered, and hierarchical bootstraps (B=2000, seed=42)...")
    rng = np.random.default_rng(42)
    B = 2000

    bootstrap_metric_rows: List[Dict[str, Any]] = []
    paired_diff_rows: List[Dict[str, Any]] = []

    # 1. Recording-Level Cluster Bootstrap for Downstream Diarization (8 recordings, B=2000)
    rec_ids_ordered = [e["rec_id"] for e in manifest]
    n_rec = len(rec_ids_ordered)
    boot_rec_indices = rng.integers(0, n_rec, size=(B, n_rec))

    diar_by_method_rec: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in parsed_diar_rows:
        diar_by_method_rec.setdefault(r["method"], {})[r["rec_id"]] = r

    for m_name in method_order:
        for metric_key in ["der_collar0_ovl", "der_collar025_noovl", "jer_collar0", "cpwer", "sawer", "micro_sacc", "macro_sacc", "speaker_count_error_abs"]:
            vals = np.array([diar_by_method_rec[m_name][rid][metric_key] for rid in rec_ids_ordered], dtype=np.float64)
            pt = float(np.mean(vals))
            b_means = np.mean(vals[boot_rec_indices], axis=1)
            ci_lo, ci_hi = float(np.percentile(b_means, 2.5)), float(np.percentile(b_means, 97.5))
            bootstrap_metric_rows.append({
                "domain": "downstream_diarization",
                "audio_condition": "dfn" if m_name not in ("9_titanet_mono_nmesc", "10_resnet293_mono_nmesc", "11_hybrid_titaDFN_resMONO_zscore") else "mono_or_hybrid",
                "method": m_name,
                "metric": f"macro_{metric_key}",
                "bootstrap_scheme": "recording_cluster_bootstrap",
                "num_replicates": B,
                "random_seed": 42,
                "num_recordings": 8,
                "num_turns": 264,
                "num_pos_pairs": 0,
                "num_neg_pairs": 0,
                "point_estimate": round(pt, 4),
                "ci95_low": round(ci_lo, 4),
                "ci95_high": round(ci_hi, 4),
                "ci95_width": round(ci_hi - ci_lo, 4),
            })

    # Paired diarization differences against 1_titanet_nmesc on the exact same recording bootstrap samples
    diar_comparisons = [
        ("2_resnet293_nmesc", "1_titanet_nmesc"),
        ("4_equal_affinity_050", "1_titanet_nmesc"),
        ("6_heuristic_quality_affinity", "1_titanet_nmesc"),
        ("7_loro_procrustes_192d", "1_titanet_nmesc"),
        ("8_loro_calibrated_gate_affinity", "1_titanet_nmesc"),
        ("8b_loro_gate_corrected_loto_affinity", "1_titanet_nmesc"),
        ("8b_loro_gate_corrected_loto_affinity", "4_equal_affinity_050"),
        ("10_resnet293_mono_nmesc", "1_titanet_nmesc"),
        ("10_resnet293_mono_nmesc", "2_resnet293_nmesc"),
        ("11_hybrid_titaDFN_resMONO_zscore", "1_titanet_nmesc"),
        ("12_diagnostic_oracle_spkcount_equal_affinity", "4_equal_affinity_050"),
    ]
    for m_a, m_b in diar_comparisons:
        for metric_key in ["der_collar0_ovl", "der_collar025_noovl", "cpwer", "macro_sacc", "speaker_count_error_abs"]:
            va = np.array([diar_by_method_rec[m_a][rid][metric_key] for rid in rec_ids_ordered], dtype=np.float64)
            vb = np.array([diar_by_method_rec[m_b][rid][metric_key] for rid in rec_ids_ordered], dtype=np.float64)
            diff_vec = va - vb
            pt_diff = float(np.mean(diff_vec))
            b_diffs = np.mean(diff_vec[boot_rec_indices], axis=1)
            ci_lo, ci_hi = float(np.percentile(b_diffs, 2.5)), float(np.percentile(b_diffs, 97.5))
            wins_a = int(np.sum(va < vb)) if metric_key != "macro_sacc" else int(np.sum(va > vb))
            wins_b = int(np.sum(va > vb)) if metric_key != "macro_sacc" else int(np.sum(va < vb))
            ties = 8 - wins_a - wins_b
            paired_diff_rows.append({
                "domain": "downstream_diarization",
                "audio_condition": "dfn_vs_mono",
                "method_a": m_a,
                "method_b": m_b,
                "metric": f"delta_macro_{metric_key} (A - B)",
                "bootstrap_scheme": "paired_recording_cluster_bootstrap",
                "num_replicates": B,
                "random_seed": 42,
                "num_recordings": 8,
                "point_diff_a_minus_b": round(pt_diff, 4),
                "ci95_low": round(ci_lo, 4),
                "ci95_high": round(ci_hi, 4),
                "includes_zero": bool(ci_lo <= 0.0 <= ci_hi),
                "rec_wins_a_better": wins_a,
                "rec_wins_b_better": wins_b,
                "rec_ties": ties,
            })

    # 2. Turn-Clustered & Hierarchical Bootstrap for Within-Recording Verification (16k_mono & dfn)
    for cond, turns_cond in [("16k_mono", turns_mono), ("dfn", turns_dfn)]:
        # Group turns by recording
        turns_by_rec: Dict[str, List[Dict[str, Any]]] = {}
        for t in turns_cond:
            turns_by_rec.setdefault(t["rec_id"], []).append(t)

        # Precompute within-recording pairwise matrices per recording
        rec_pair_mats: Dict[str, Dict[str, np.ndarray]] = {}
        for rid, r_turns in turns_by_rec.items():
            n_t = len(r_turns)
            T_emb = np.stack([x["tita_emb"] for x in r_turns], axis=0)
            R_emb = np.stack([x["res_emb"] for x in r_turns], axis=0)
            spks = [x["speaker"] for x in r_turns]
            Y_mat = np.array([[1 if spks[i] == spks[j] else 0 for j in range(n_t)] for i in range(n_t)], dtype=np.int32)
            rec_pair_mats[rid] = {
                "T_sim": T_emb @ T_emb.T,
                "R_sim": R_emb @ R_emb.T,
                "Y": Y_mat,
                "n_turns": n_t,
            }

        # Point estimates on original within-recording pairs
        orig_t, orig_r, orig_y = [], [], []
        for rid in rec_ids_ordered:
            pm = rec_pair_mats[rid]
            iu, ju = np.triu_indices(pm["n_turns"], k=1)
            orig_t.append(pm["T_sim"][iu, ju])
            orig_r.append(pm["R_sim"][iu, ju])
            orig_y.append(pm["Y"][iu, ju])
        full_t = np.concatenate(orig_t)
        full_r = np.concatenate(orig_r)
        full_eq = 0.5 * full_t + 0.5 * full_r
        full_y = np.concatenate(orig_y)

        pt_eer_t = compute_eer_and_mindcf(full_t, full_y)["eer"]
        pt_eer_r = compute_eer_and_mindcf(full_r, full_y)["eer"]
        pt_eer_eq = compute_eer_and_mindcf(full_eq, full_y)["eer"]

        # Scheme A: Turn-clustered bootstrap stratified by recording (resamples turns within each recording)
        # Scheme B: Hierarchical bootstrap (resamples recordings with replacement, then resamples turns within each recording)
        for scheme_name, resample_recs in [
            ("turn_clustered_bootstrap_within_rec", False),
            ("hierarchical_recording_then_turn_bootstrap", True),
        ]:
            b_eer_t, b_eer_r, b_eer_eq = [], [], []
            for b_idx in range(B):
                sampled_rids = [rec_ids_ordered[i] for i in rng.integers(0, n_rec, size=n_rec)] if resample_recs else rec_ids_ordered
                bt_list, br_list, by_list = [], [], []
                for rid in sampled_rids:
                    pm = rec_pair_mats[rid]
                    nt = pm["n_turns"]
                    if nt < 2:
                        continue
                    t_idx = rng.integers(0, nt, size=nt)
                    # Form pairs only between distinct original turns (t_idx[a] != t_idx[b]) to avoid artificial self-pairs
                    iu, ju = np.triu_indices(nt, k=1)
                    orig_a, orig_b = t_idx[iu], t_idx[ju]
                    valid_m = orig_a != orig_b
                    if not np.any(valid_m):
                        continue
                    oa, ob = orig_a[valid_m], orig_b[valid_m]
                    bt_list.append(pm["T_sim"][oa, ob])
                    br_list.append(pm["R_sim"][oa, ob])
                    by_list.append(pm["Y"][oa, ob])
                if not by_list:
                    continue
                by_cat = np.concatenate(by_list)
                if np.sum(by_cat == 1) == 0 or np.sum(by_cat == 0) == 0:
                    continue
                bt_cat = np.concatenate(bt_list)
                br_cat = np.concatenate(br_list)
                beq_cat = 0.5 * bt_cat + 0.5 * br_cat
                b_eer_t.append(compute_eer_and_mindcf(bt_cat, by_cat)["eer"])
                b_eer_r.append(compute_eer_and_mindcf(br_cat, by_cat)["eer"])
                b_eer_eq.append(compute_eer_and_mindcf(beq_cat, by_cat)["eer"])

            arr_t = np.array(b_eer_t, dtype=np.float64)
            arr_r = np.array(b_eer_r, dtype=np.float64)
            arr_eq = np.array(b_eer_eq, dtype=np.float64)

            for m_label, pt_val, b_arr in [
                ("titanet", pt_eer_t, arr_t),
                ("resnet293", pt_eer_r, arr_r),
                ("equal_cosine_fusion", pt_eer_eq, arr_eq),
            ]:
                lo, hi = float(np.percentile(b_arr, 2.5)), float(np.percentile(b_arr, 97.5))
                bootstrap_metric_rows.append({
                    "domain": "within_recording_turn_verification",
                    "audio_condition": cond,
                    "method": m_label,
                    "metric": "within_rec_turn_eer",
                    "bootstrap_scheme": scheme_name,
                    "num_replicates": len(b_arr),
                    "random_seed": 42,
                    "num_recordings": 8,
                    "num_turns": len(turns_cond),
                    "num_pos_pairs": int(np.sum(full_y == 1)),
                    "num_neg_pairs": int(np.sum(full_y == 0)),
                    "point_estimate": round(pt_val, 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                    "ci95_width": round(hi - lo, 4),
                })

            for m_a, m_b, arr_a, arr_b, pt_a, pt_b in [
                ("resnet293", "titanet", arr_r, arr_t, pt_eer_r, pt_eer_t),
                ("equal_cosine_fusion", "titanet", arr_eq, arr_t, pt_eer_eq, pt_eer_t),
                ("equal_cosine_fusion", "resnet293", arr_eq, arr_r, pt_eer_eq, pt_eer_r),
            ]:
                d_arr = arr_a - arr_b
                lo, hi = float(np.percentile(d_arr, 2.5)), float(np.percentile(d_arr, 97.5))
                paired_diff_rows.append({
                    "domain": "within_recording_turn_verification",
                    "audio_condition": cond,
                    "method_a": m_a,
                    "method_b": m_b,
                    "metric": "delta_within_rec_turn_eer (A - B)",
                    "bootstrap_scheme": scheme_name,
                    "num_replicates": len(d_arr),
                    "random_seed": 42,
                    "num_recordings": 8,
                    "point_diff_a_minus_b": round(pt_a - pt_b, 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                    "includes_zero": bool(lo <= 0.0 <= hi),
                    "rec_wins_a_better": None,
                    "rec_wins_b_better": None,
                    "rec_ties": None,
                })

    write_csv(STAGE4_1_DIR / "clustered_bootstrap_metrics.csv", bootstrap_metric_rows)
    write_csv(STAGE4_1_DIR / "paired_method_differences.csv", paired_diff_rows)

    # =========================================================================
    # ISSUE F: Reconcile Dataset, Population Flow, Cache Accounting & Audio Normalization
    # =========================================================================
    print("[Stage 4.1 - Issue F] Reconciling dataset populations, cache counts, and audio normalization...")
    pop_flow_rows = [
        {
            "step_order": 1,
            "population_name": "All parsed ground-truth segments (12 files: R01..R08 + M01..M04)",
            "unit": "annotated_segments",
            "recordings_included": 12,
            "speakers_included": 88,
            "count_per_condition": 1362,
            "total_across_conditions": 1362,
            "filter_rule": "All header blocks matched by HEADER_STAGE1 in ground_truth/*.txt",
            "excluded_from_prev_step": 0,
            "exclusion_reason": "None (full non-destructive ledger)",
            "used_by_analyses": "Stage 1 segment ledger & anomaly audit",
        },
        {
            "step_order": 2,
            "population_name": "Audio-available ground-truth segments (R01..R08)",
            "unit": "annotated_segments",
            "recordings_included": 8,
            "speakers_included": 41,
            "count_per_condition": 432,
            "total_across_conditions": 432,
            "filter_rule": "has_audio == True (excludes M01..M04 which have no MP3 in raw_audio/)",
            "excluded_from_prev_step": 930,
            "exclusion_reason": "930 segments belong to M01..M04 (missing source audio files)",
            "used_by_analyses": "Stage 2 & Stage 4 ASR WER, cpWER, saWER, Micro-SACC, Macro-SACC",
        },
        {
            "step_order": 3,
            "population_name": "Positive-duration RTTM reference segments (R01..R08)",
            "unit": "rttm_segments",
            "recordings_included": 8,
            "speakers_included": 36,
            "count_per_condition": 312,
            "total_across_conditions": 312,
            "filter_rule": "duration > 0.0s (excludes 120 zero-duration 1s-rounded segments)",
            "excluded_from_prev_step": 120,
            "exclusion_reason": "120 segments have start == end (0.0s duration due to 1s timestamp rounding)",
            "used_by_analyses": "Stage 2 & Stage 4 DER (c=0.0 & c=0.25), JER, FA, Miss, Confusion, Speaker Count Error",
        },
        {
            "step_order": 4,
            "population_name": "Clean single-speaker positive-duration GT turns (R01..R08)",
            "unit": "clean_gt_turns",
            "recordings_included": 8,
            "speakers_included": 32,
            "count_per_condition": 264,
            "total_across_conditions": 264,
            "filter_rule": "usable_for_clean_speaker_eval == True (duration > 0, has_gt_overlap == False, speaker != Speaker_all)",
            "excluded_from_prev_step": 48,
            "exclusion_reason": "47 multi-speaker overlapping segments + 1 collective Speaker_all segment (R08 seg_idx 0)",
            "used_by_analyses": "Candidate clean turns for speaker verification",
        },
        {
            "step_order": 5,
            "population_name": "Usable clean GT turns with >=1 Scale-4 (0.5s) VAD subsegment (R01..R08)",
            "unit": "verified_clean_turns",
            "recordings_included": 8,
            "speakers_included": 30,
            "count_per_condition": 254,
            "total_across_conditions": 762,
            "filter_rule": "duration >= 0.5s AND >=1 Scale-4 subsegment with >=75% overlap inside turn",
            "excluded_from_prev_step": 10,
            "exclusion_reason": "10 clean turns have 0 VAD subsegments with >=75% coverage (missed by Path A VAD or boundary misalignment)",
            "used_by_analyses": "Stage 4 & Stage 4.1 turn-level pairwise verification (9,817 within-rec pairs; 32,131 all pairs)",
        },
        {
            "step_order": 6,
            "population_name": "Clean turns belonging to speakers with >=2 clean turns in same recording (LOTO eligible)",
            "unit": "loto_clean_turns",
            "recordings_included": 6,
            "speakers_included": 18,
            "count_per_condition": 240,
            "total_across_conditions": 720,
            "filter_rule": "Speaker has >=2 usable clean turns in recording AND recording has >=2 such speakers",
            "excluded_from_prev_step": 14,
            "exclusion_reason": "12 single-turn speakers + 2 turns of R02::Speaker_0 (only clean speaker in R02)",
            "used_by_analyses": "Stage 4.1 leak-free Leave-One-Turn-Out (LOTO) turn & subsegment reliability evaluation",
        },
        {
            "step_order": 7,
            "population_name": "All multiscale VAD subsegments across Scales 0..4 (R01..R08)",
            "unit": "subsegments",
            "recordings_included": 8,
            "speakers_included": 36,
            "count_per_condition": 15994,
            "total_across_conditions": 47982,
            "filter_rule": "All NeMo Path A VAD subsegments across Scales 0(1840), 1(2297), 2(2761), 3(3474), 4(5622)",
            "excluded_from_prev_step": 0,
            "exclusion_reason": "Full multiscale VAD subsegment population (Note: earlier summary text '40,386' was an obsolete pre-fix draft number)",
            "used_by_analyses": "Stage 3 caches, Stage 4 Linear CKA (all), Stage 4 LORO Procrustes, Stage 4 NME-SC clustering",
        },
        {
            "step_order": 8,
            "population_name": "Scale-4 (0.5s) base subsegments (all)",
            "unit": "scale4_subsegments",
            "recordings_included": 8,
            "speakers_included": 36,
            "count_per_condition": 5622,
            "total_across_conditions": 16866,
            "filter_rule": "scale_idx == 4 (0.5s window, 0.25s shift)",
            "excluded_from_prev_step": 10372,
            "exclusion_reason": "Excludes Scales 0..3 (1,840 + 2,297 + 2,761 + 3,474 = 10,372)",
            "used_by_analyses": "Base scale for NME-SC affinity matrix size [N_s4 x N_s4]",
        },
        {
            "step_order": 9,
            "population_name": "Scale-4 subsegments matched to a multi-subsegment speaker (Stage 4 leaked reliability set)",
            "unit": "scale4_matched_subsegments",
            "recordings_included": 8,
            "speakers_included": 31,
            "count_per_condition": 5290,
            "total_across_conditions": 15870,
            "filter_rule": "base_spk in valid_spks (includes 4,234 clean + 703 OVERLAP:: + 353 PARTIAL_BOUNDARY:: subsegments)",
            "excluded_from_prev_step": 332,
            "exclusion_reason": "331 NON_SPEECH_OR_ROUNDING_GAP subsegments + 1 single-subsegment speaker clip",
            "used_by_analyses": "Stage 4 original reliability correlation/AUROC table & Stage 4 original LORO gate training",
        },
        {
            "step_order": 10,
            "population_name": "Scale-4 clean single-speaker subsegments (>=75% purity, no overlap)",
            "unit": "scale4_clean_subsegments",
            "recordings_included": 8,
            "speakers_included": 31,
            "count_per_condition": 4234,
            "total_across_conditions": 12702,
            "filter_rule": "not gt_is_overlap and not gt_label.startswith((NON_SPEECH, PARTIAL, OVERLAP, AMBIGUOUS))",
            "excluded_from_prev_step": 1056,
            "exclusion_reason": "703 OVERLAP:: subsegments + 353 PARTIAL_BOUNDARY:: subsegments",
            "used_by_analyses": "Stage 4 CKA (clean), Stage 4 stratified SNR/scale/condition tables, per-rec discriminability",
        },
        {
            "step_order": 11,
            "population_name": "Scale-4 clean subsegments eligible for leak-free Turn-Disjoint LOTO evaluation",
            "unit": "scale4_loto_clean_subsegments",
            "recordings_included": 6,
            "speakers_included": 18,
            "count_per_condition": 3437,
            "total_across_conditions": 10311,
            "filter_rule": "Assigned to a clean turn of a speaker with >=2 clean turns in a recording with >=2 such speakers",
            "excluded_from_prev_step": 797,
            "exclusion_reason": "323 subsegments in R02 (only 1 clean speaker) + 331 in R04 (each speaker has 1 clean turn) + 143 from 1-turn speakers",
            "used_by_analyses": "Stage 4.1 leak-free LOTO reliability evaluation & corrected LORO gate training",
        },
    ]
    write_csv(STAGE4_1_DIR / "population_flow.csv", pop_flow_rows)

    # Cache accounting CSV
    cache_acct_rows: List[Dict[str, Any]] = []
    for r in stage3_val:
        cond = r["audio_condition"]
        rec_id = r["rec_id"]
        sk = int(r["scale_idx"])
        n_sub = int(r["num_subsegments"])
        c_dir = STAGE3_DIR / "cache" / cond / rec_id
        t_p = c_dir / f"titanet_large__{cond}__scale{sk}.npz"
        r_p = c_dir / f"wespeaker_resnet293_lm__{cond}__scale{sk}.npz"
        m_p = c_dir / f"subsegments_metadata__scale{sk}.json"
        meta = json.loads(m_p.read_text(encoding="utf-8"))["subsegments"]
        n_clean = sum(
            1 for m in meta
            if not m["gt_is_overlap"] and not str(m["gt_label"]).startswith(("NON_SPEECH", "PARTIAL", "OVERLAP", "AMBIGUOUS"))
        )
        n_ovl = sum(1 for m in meta if m["gt_is_overlap"])
        n_part = sum(1 for m in meta if str(m["gt_label"]).startswith("PARTIAL_BOUNDARY"))
        n_nonspeech = sum(1 for m in meta if str(m["gt_label"]).startswith("NON_SPEECH"))

        cache_acct_rows.append({
            "rec_id": rec_id,
            "audio_condition": cond,
            "scale_idx": sk,
            "window_sec": float(r["window_sec"]),
            "num_subsegments": n_sub,
            "clean_subsegments": n_clean,
            "overlap_subsegments": n_ovl,
            "partial_boundary_subsegments": n_part,
            "non_speech_subsegments": n_nonspeech,
            "titanet_npz_exists": t_p.exists(),
            "resnet293_npz_exists": r_p.exists(),
            "metadata_json_exists": m_p.exists(),
            "encoders_stored_in_separate_npz": True,
            "titanet_vectors": n_sub,
            "resnet293_vectors": n_sub,
            "total_embedding_vectors_in_pair": 2 * n_sub,
        })
    write_csv(STAGE4_1_DIR / "cache_accounting.csv", cache_acct_rows)

    # Audio condition audit CSV (verifying actual loudness/gain applied to each WAV file)
    audio_audit_rows: List[Dict[str, Any]] = []
    for av in audio_val:
        wav_p = REPO_ROOT / av["wav_path"]
        with wave.open(str(wav_p), "rb") as wf:
            raw_b = wf.readframes(wf.getnframes())
        s_f32 = np.frombuffer(raw_b, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(s_f32 ** 2)))
        rms_db = float(20.0 * math.log10(max(rms, 1e-12)))
        peak_db = float(20.0 * math.log10(max(float(np.max(np.abs(s_f32))), 1e-12)))
        # Active speech frame RMS (frames > -45 dBFS)
        frames = np.lib.stride_tricks.sliding_window_view(s_f32, 1600)[::800]
        f_rms_db = 10.0 * np.log10(np.mean(frames ** 2, axis=-1) + 1e-12)
        active_rms_db = float(np.mean(f_rms_db[f_rms_db > -45.0])) if np.any(f_rms_db > -45.0) else rms_db

        variant = av["audio_variant"]
        audio_audit_rows.append({
            "rec_id": av["rec_id"],
            "audio_variant": variant,
            "wav_path": av["wav_path"],
            "sha256": av["sha256"],
            "duration_sec": float(av["duration_sec"]),
            "rms_dbfs": round(rms_db, 3),
            "active_frame_mean_dbfs": round(active_rms_db, 3),
            "peak_amplitude": float(av["peak_amplitude"]),
            "peak_dbfs": round(peak_db, 3),
            "actual_processing_command_in_code": (
                "ffmpeg -vn -acodec pcm_s16le -ar 16000 -ac 1 (run_stage1_prep.py L458)"
                if variant == "16k_mono"
                else (
                    "deepFilter <mono_wav> --no-suffix (run_stage1b_dfn.py L36)"
                    if variant == "dfn"
                    else "ffmpeg -af loudnorm=I=-16:TP=-1.5:LRA=11 -ar 16000 -ac 1 -acodec pcm_s16le (run_stage1_prep.py L471)"
                )
            ),
            "discrepancy_note": (
                "NONE"
                if variant != "normalized"
                else "Code executed loudnorm=I=-16:TP=-1.5:LRA=11 (resulting in -15.8 to -17.7 dBFS RMS), whereas Stage 1/2/3 report text mistakenly cited phase20's -20 LUFS label."
            ),
        })
    write_csv(STAGE4_1_DIR / "audio_condition_audit.csv", audio_audit_rows)

    # Save master summary JSON for easy reference
    master_summary = {
        "stage": "Stage 4.1 — Statistical Validity, Leakage Audit & Evaluation Correction",
        "issue_A_summary": {
            "total_within_rec_pairs": tot_within_pos + tot_within_neg,
            "within_rec_pos_pairs": tot_within_pos,
            "within_rec_neg_pairs": tot_within_neg,
            "cross_rec_pos_pairs": 0,
            "cross_rec_neg_pairs": cross_rec_pairs,
            "r06_pair_share": round(7503 / 9817.0, 4),
            "eligible_recordings_for_turn_eer": 6,
            "ineligible_recordings": ["R02_dennis_skinner (0 neg pairs)", "R04_piers_robinson (0 pos pairs)"],
        },
        "issue_B_loro_fusion_comparison": loro_fusion_comp_rows,
        "issue_C_reliability_summary": [
            r for r in reliability_loro_rows
            if r["audio_condition"] == "16k_mono"
            and r["indicator"] in ("resnet293_l2_norm", "dfn_noise_reduction_db", "vad_conf_mean", "resnet293_cross_scale_cos", "titanet_l2_norm")
        ],
        "section_9_diarization_summary": diar_summary_rows,
    }
    (STAGE4_1_DIR / "stage4_1_master_audit.json").write_text(
        json.dumps(master_summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Stage 4.1 audit script completed successfully!")


if __name__ == "__main__":
    main()
