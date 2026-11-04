#!/usr/bin/env python3
"""
Stage 5B.2 Corrective Audit & Implementation Repair Suite.
Executes mathematically rigorous, track-aware, and locally bounded overlap optimization
experiments across all 8 canonical benchmark recordings.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "phase20_standalone" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage2_baseline"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage3_dual_encoder"))
sys.path.insert(0, str(REPO_ROOT / "research_stages" / "stage5a" / "scripts"))

import common as p20_common
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

AUDIT_DIR = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_2_overlap_optimization" / "audit_corrections"
CONFIGS_DIR = AUDIT_DIR / "configs"
METRICS_DIR = AUDIT_DIR / "metrics"
PLOTS_DIR = AUDIT_DIR / "STAGE5B_2_CORRECTED_PLOTS"
PRED_DIR = AUDIT_DIR / "predictions"
CACHE_DIR = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_2_overlap_optimization" / "overlap_cache"

for d in [CONFIGS_DIR, METRICS_DIR, PLOTS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Track-Aware Pyannote Loading & Diarization Metrics
# -----------------------------------------------------------------------------

def load_track_aware_annotation(path: Path) -> Annotation:
    """Loads an RTTM into a pyannote Annotation with unique track IDs to prevent track overwriting."""
    ann = Annotation()
    for i, seg in enumerate(p20_common.rttm_to_segments(path)):
        ann[Segment(seg["start"], seg["end"]), f"track_{i}"] = seg["speaker"]
    return ann


def compute_track_aware_diarization_metrics(
    ref_annot: Annotation,
    hyp_annot: Annotation,
) -> Dict[str, float]:
    """Computes track-aware DER (collar=0.0s), collar=0.25s, JER, and component decompositions."""
    der_metric_c0 = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    details_c0 = der_metric_c0(ref_annot, hyp_annot, detailed=True)
    total_ref = details_c0.get("total", 0.0) or 1e-9

    der_metric_c25 = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    details_c25 = der_metric_c25(ref_annot, hyp_annot, detailed=True)
    total_c25 = details_c25.get("total", 0.0) or 1e-9

    try:
        jer = float(JaccardErrorRate()(ref_annot, hyp_annot))
    except Exception:
        jer = 0.0

    return {
        "der_c0": float(details_c0.get("diarization error rate", 0.0)),
        "der_c25": float(details_c25.get("diarization error rate", 0.0)),
        "diar_miss": float(details_c0.get("missed detection", 0.0)) / total_ref,
        "diar_fa": float(details_c0.get("false alarm", 0.0)) / total_ref,
        "diar_confusion": float(details_c0.get("confusion", 0.0)) / total_ref,
        "jer": jer,
        "total_reference_sec": float(total_ref),
        "miss_sec": float(details_c0.get("missed detection", 0.0)),
        "fa_sec": float(details_c0.get("false alarm", 0.0)),
        "conf_sec": float(details_c0.get("confusion", 0.0)),
    }


# -----------------------------------------------------------------------------
# 2. Overlap Posterior Binarization with Physical Window Support
# -----------------------------------------------------------------------------

def load_overlap_posteriors(rec_id: str, stream: str = "mono") -> Tuple[np.ndarray, np.ndarray]:
    """Loads frame_times and p_overlap for the specified stream from frozen cache."""
    npz_p = CACHE_DIR / f"{rec_id}_{stream}_overlap.npz"
    if not npz_p.exists():
        raise FileNotFoundError(f"Overlap cache not found: {npz_p}")
    data = np.load(npz_p)
    return data["frame_times"], data["p_overlap"]


def binarize_overlap_corrected(
    p_ovl: np.ndarray,
    duration_sec: float,
    threshold: float = 0.40,
    min_duration_on: float = 0.10,
    min_duration_off: float = 0.15,
) -> List[Dict[str, float]]:
    """
    Binarizes overlap scores using exact physical window temporal support:
    Receptive field = 0.0619375 s, Hop/Step = 0.016875 s (270 samples at 16kHz).
    """
    step = 0.016875
    win_dur = 0.0619375
    n_frames = len(p_ovl)

    # 1. Extract contiguous active index runs
    is_active = p_ovl >= threshold
    raw_runs: List[Tuple[int, int]] = []
    in_run = False
    start_idx = 0

    for i in range(n_frames):
        if not in_run and is_active[i]:
            in_run = True
            start_idx = i
        elif in_run and not is_active[i]:
            in_run = False
            raw_runs.append((start_idx, i - 1))
    if in_run:
        raw_runs.append((start_idx, n_frames - 1))

    if not raw_runs:
        return []

    # 2. Convert index runs to continuous physical time intervals
    raw_segs: List[Dict[str, float]] = []
    for s_i, e_i in raw_runs:
        st_sec = max(0.0, s_i * step)
        en_sec = min(duration_sec, e_i * step + win_dur)
        raw_segs.append({"start": st_sec, "end": en_sec})

    # 3. Gap merging (fill inactive regions shorter than min_duration_off)
    merged = [raw_segs[0]]
    for seg in raw_segs[1:]:
        gap = seg["start"] - merged[-1]["end"]
        if gap < min_duration_off:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)

    # 4. Remove active regions shorter than min_duration_on
    filtered = []
    for s in merged:
        dur = s["end"] - s["start"]
        if dur >= min_duration_on:
            filtered.append({"start": round(s["start"], 3), "end": round(s["end"], 3)})

    return filtered


# -----------------------------------------------------------------------------
# 3. Prototype Affinity Extraction (M2-R1 and Ablations)
# -----------------------------------------------------------------------------

def load_dual_encoder_prototypes(
    base_segs: List[Dict[str, Any]],
    rec_id: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Builds speaker prototypes using frozen Stage 3 scale 0 embeddings:
    Returns dict with keys: 'titanet', 'wespeaker', 'concat'
    """
    emb_dir = REPO_ROOT / "research_stages" / "stage3_dual_encoder" / "cache" / "dfn" / rec_id
    t_npz = emb_dir / "titanet_large__dfn__scale0.npz"
    w_npz = emb_dir / "wespeaker_resnet293_lm__dfn__scale0.npz"
    meta_json = emb_dir / "subsegments_metadata__scale0.json"

    if not (t_npz.exists() and w_npz.exists() and meta_json.exists()):
        return {"titanet": {}, "wespeaker": {}, "concat": {}, "subsegs": [], "t_embs": np.array([]), "w_embs": np.array([]), "c_embs": np.array([])}

    d_t = np.load(t_npz)
    d_w = np.load(w_npz)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))["subsegments"]

    t_embs = d_t["l2_normalized_embeddings"]  # (N, 192)
    w_embs = d_w["l2_normalized_embeddings"]  # (N, 256)

    # Joint concatenated representation (448d) L2-normalized
    c_embs = np.hstack([t_embs, w_embs])
    c_norms = np.linalg.norm(c_embs, axis=1, keepdims=True)
    c_embs = c_embs / np.maximum(c_norms, 1e-8)

    spk_t: Dict[str, List[np.ndarray]] = {}
    spk_w: Dict[str, List[np.ndarray]] = {}
    spk_c: Dict[str, List[np.ndarray]] = {}

    for subseg, et, ew, ec in zip(meta, t_embs, w_embs, c_embs):
        mid = (subseg["start"] + subseg["end"]) / 2.0
        matched_spk = None
        for s in base_segs:
            if s["start"] <= mid <= s["end"]:
                matched_spk = s["speaker"]
                break
        if matched_spk is not None:
            spk_t.setdefault(matched_spk, []).append(et)
            spk_w.setdefault(matched_spk, []).append(ew)
            spk_c.setdefault(matched_spk, []).append(ec)

    def centroid(embs_list):
        m = np.mean(embs_list, axis=0)
        norm = np.linalg.norm(m)
        return m / max(norm, 1e-8)

    proto_t = {k: centroid(v) for k, v in spk_t.items()}
    proto_w = {k: centroid(v) for k, v in spk_w.items()}
    proto_c = {k: centroid(v) for k, v in spk_c.items()}

    return {
        "titanet": proto_t,
        "wespeaker": proto_w,
        "concat": proto_c,
        "subsegs": meta,
        "t_embs": t_embs,
        "w_embs": w_embs,
        "c_embs": c_embs,
    }


# -----------------------------------------------------------------------------
# 4. Hungarian Speaker Mapping for Oracle Diagnostics (M5)
# -----------------------------------------------------------------------------

def compute_hungarian_mapping(
    gt_segs: List[Dict[str, Any]],
    base_segs: List[Dict[str, Any]],
    duration_sec: float,
    resolution: float = 0.01,
) -> Dict[str, str]:
    """Computes optimal Hungarian mapping between ground-truth speakers and hypothesis clusters."""
    ref_spks = sorted(list(set(s["speaker"] for s in gt_segs)))
    hyp_spks = sorted(list(set(s["speaker"] for s in base_segs)))

    if not ref_spks or not hyp_spks:
        return {}

    n_frames = int(round(duration_sec / resolution)) + 1
    ref_frames = [None] * n_frames
    hyp_frames = [None] * n_frames

    for s in gt_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        for i in range(st_i, en_i):
            ref_frames[i] = s["speaker"]

    for s in base_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        for i in range(st_i, en_i):
            hyp_frames[i] = s["speaker"]

    cost_matrix = np.zeros((len(ref_spks), len(hyp_spks)))
    for r_idx, r_spk in enumerate(ref_spks):
        for h_idx, h_spk in enumerate(hyp_spks):
            cooccur = sum(1 for i in range(n_frames) if ref_frames[i] == r_spk and hyp_frames[i] == h_spk)
            cost_matrix[r_idx, h_idx] = -cooccur

    r_ind, h_ind = linear_sum_assignment(cost_matrix)
    mapping = {ref_spks[r]: hyp_spks[h] for r, h in zip(r_ind, h_ind)}
    return mapping


# -----------------------------------------------------------------------------
# 5. Corrected Secondary Speaker Attribution Strategies
# -----------------------------------------------------------------------------

def attribute_secondary_speakers_corrected(
    base_segs: List[Dict[str, Any]],
    ovl_segs: List[Dict[str, Any]],
    attribution_method: str,
    duration_sec: float,
    proto_data: Dict[str, Any],
    gt_segs: Optional[List[Dict[str, Any]]] = None,
    hungarian_map: Optional[Dict[str, str]] = None,
    resolution: float = 0.01,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Applies corrected secondary speaker attribution strategies with strict abstention.
    Returns: (secondary_segments, total_detected_overlaps, total_attributed_overlaps)
    """
    if not ovl_segs:
        return [], 0, 0

    n_frames = int(round(duration_sec / resolution)) + 1
    frame_spk = [None] * n_frames
    for s in base_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        for i in range(st_i, en_i):
            frame_spk[i] = s["speaker"]

    all_hyp_speakers = sorted(list(set(s["speaker"] for s in base_segs)))
    if len(all_hyp_speakers) < 2 and not attribution_method.startswith("M5"):
        # Single-speaker baseline; cannot attribute secondary speaker
        return [], len(ovl_segs), 0

    sec_segments: List[Dict[str, Any]] = []
    attributed_count = 0

    for ovl in ovl_segs:
        st = ovl["start"]
        en = ovl["end"]
        mid = (st + en) / 2.0
        mid_idx = min(n_frames - 1, max(0, int(round(mid / resolution))))
        primary_spk = frame_spk[mid_idx]

        if primary_spk is None:
            # Match closest segment in base_segs
            closest = min(base_segs, key=lambda s: abs((s["start"] + s["end"]) / 2.0 - mid))
            primary_spk = closest["speaker"]

        sec_spk: Optional[str] = None

        if attribution_method == "M0_no_overlap":
            sec_spk = None

        elif attribution_method == "M1_temporal_context":
            # Search context window [mid - 3.0, mid + 3.0]
            w_ctx = 3.0
            min_evidence_frames = int(round(0.20 / resolution))  # 0.20s threshold
            ctx_st = max(0, int(round((mid - w_ctx) / resolution)))
            ctx_en = min(n_frames, int(round((mid + w_ctx) / resolution)))

            counts: Dict[str, int] = {}
            for s in frame_spk[ctx_st:ctx_en]:
                if s is not None and s != primary_spk:
                    counts[s] = counts.get(s, 0) + 1

            # Filter candidates meeting minimum evidence threshold
            valid_cands = [(spk, cnt) for spk, cnt in counts.items() if cnt >= min_evidence_frames]
            if valid_cands:
                # Select candidate with max frames; tie break by temporal proximity
                valid_cands.sort(key=lambda x: x[1], reverse=True)
                sec_spk = valid_cands[0][0]
            else:
                # Strict abstention (NO arbitrary global fallback)
                sec_spk = None

        elif attribution_method.startswith("M2_prototype"):
            # Subsegment-based overlap representation & prototype affinity
            subsegs = proto_data.get("subsegs", [])
            weights: List[float] = []
            matching_indices: List[int] = []

            for idx, s in enumerate(subsegs):
                inter = max(0.0, min(en, s["end"]) - max(st, s["start"]))
                if inter > 0.0:
                    weights.append(inter)
                    matching_indices.append(idx)

            if not matching_indices or sum(weights) <= 0.0:
                sec_spk = None
            else:
                weights_arr = np.array(weights)
                weights_arr = weights_arr / np.sum(weights_arr)

                if attribution_method == "M2_prototype_concat":
                    embs = proto_data["c_embs"][matching_indices]
                    protos = proto_data["concat"]
                    e_ovl = np.sum(embs * weights_arr[:, None], axis=0)
                    e_ovl = e_ovl / max(np.linalg.norm(e_ovl), 1e-8)
                    alt_spks = [s for s in protos if s != primary_spk]
                    if alt_spks:
                        sims = {s: float(np.dot(e_ovl, protos[s])) for s in alt_spks}
                        sec_spk = max(sims.items(), key=lambda x: x[1])[0]

                elif attribution_method == "M2_prototype_titanet":
                    embs = proto_data["t_embs"][matching_indices]
                    protos = proto_data["titanet"]
                    e_ovl = np.sum(embs * weights_arr[:, None], axis=0)
                    e_ovl = e_ovl / max(np.linalg.norm(e_ovl), 1e-8)
                    alt_spks = [s for s in protos if s != primary_spk]
                    if alt_spks:
                        sims = {s: float(np.dot(e_ovl, protos[s])) for s in alt_spks}
                        sec_spk = max(sims.items(), key=lambda x: x[1])[0]

                elif attribution_method == "M2_prototype_wespeaker":
                    embs = proto_data["w_embs"][matching_indices]
                    protos = proto_data["wespeaker"]
                    e_ovl = np.sum(embs * weights_arr[:, None], axis=0)
                    e_ovl = e_ovl / max(np.linalg.norm(e_ovl), 1e-8)
                    alt_spks = [s for s in protos if s != primary_spk]
                    if alt_spks:
                        sims = {s: float(np.dot(e_ovl, protos[s])) for s in alt_spks}
                        sec_spk = max(sims.items(), key=lambda x: x[1])[0]

                elif attribution_method == "M2_prototype_normfusion":
                    embs_t = proto_data["t_embs"][matching_indices]
                    protos_t = proto_data["titanet"]
                    e_t = np.sum(embs_t * weights_arr[:, None], axis=0)
                    e_t = e_t / max(np.linalg.norm(e_t), 1e-8)

                    embs_w = proto_data["w_embs"][matching_indices]
                    protos_w = proto_data["wespeaker"]
                    e_w = np.sum(embs_w * weights_arr[:, None], axis=0)
                    e_w = e_w / max(np.linalg.norm(e_w), 1e-8)

                    alt_spks = [s for s in protos_t if s != primary_spk and s in protos_w]
                    if alt_spks:
                        sims = {s: 0.5 * float(np.dot(e_t, protos_t[s])) + 0.5 * float(np.dot(e_w, protos_w[s])) for s in alt_spks}
                        sec_spk = max(sims.items(), key=lambda x: x[1])[0]

        elif attribution_method == "M3_turn_alternation":
            # Search within bounded local context: max gap 5.0s, min turn duration 0.20s
            max_gap_frames = int(round(5.0 / resolution))
            min_turn_frames = int(round(0.20 / resolution))

            # Backward search for preceding turn P
            p_cand = None
            search_start = max(0, mid_idx - max_gap_frames)
            # Find contiguous runs of alternate speakers
            curr_spk = None
            curr_len = 0
            for i in range(mid_idx - 1, search_start - 1, -1):
                s = frame_spk[i]
                if s is not None and s != primary_spk:
                    if curr_spk is None:
                        curr_spk = s
                        curr_len = 1
                    elif s == curr_spk:
                        curr_len += 1
                    else:
                        break
                    if curr_len >= min_turn_frames:
                        p_cand = curr_spk
                        break
                elif s == primary_spk and curr_len > 0:
                    break

            # Forward search for subsequent turn N
            n_cand = None
            search_end = min(n_frames, mid_idx + max_gap_frames)
            curr_spk = None
            curr_len = 0
            for i in range(mid_idx + 1, search_end):
                s = frame_spk[i]
                if s is not None and s != primary_spk:
                    if curr_spk is None:
                        curr_spk = s
                        curr_len = 1
                    elif s == curr_spk:
                        curr_len += 1
                    else:
                        break
                    if curr_len >= min_turn_frames:
                        n_cand = curr_spk
                        break
                elif s == primary_spk and curr_len > 0:
                    break

            if p_cand is not None:
                sec_spk = p_cand
            elif n_cand is not None:
                sec_spk = n_cand
            else:
                sec_spk = None

        elif attribution_method == "M4_conservative_abstention":
            # Requires strict bidirectional agreement between P and N within 5.0s
            max_gap_frames = int(round(5.0 / resolution))
            min_turn_frames = int(round(0.20 / resolution))

            # Backward search
            p_cand = None
            search_start = max(0, mid_idx - max_gap_frames)
            curr_spk = None
            curr_len = 0
            for i in range(mid_idx - 1, search_start - 1, -1):
                s = frame_spk[i]
                if s is not None and s != primary_spk:
                    if curr_spk is None:
                        curr_spk = s
                        curr_len = 1
                    elif s == curr_spk:
                        curr_len += 1
                    else:
                        break
                    if curr_len >= min_turn_frames:
                        p_cand = curr_spk
                        break
                elif s == primary_spk and curr_len > 0:
                    break

            # Forward search
            n_cand = None
            search_end = min(n_frames, mid_idx + max_gap_frames)
            curr_spk = None
            curr_len = 0
            for i in range(mid_idx + 1, search_end):
                s = frame_spk[i]
                if s is not None and s != primary_spk:
                    if curr_spk is None:
                        curr_spk = s
                        curr_len = 1
                    elif s == curr_spk:
                        curr_len += 1
                    else:
                        break
                    if curr_len >= min_turn_frames:
                        n_cand = curr_spk
                        break
                elif s == primary_spk and curr_len > 0:
                    break

            if p_cand is not None and n_cand is not None and p_cand == n_cand:
                sec_spk = p_cand
            else:
                sec_spk = None

        elif attribution_method == "M5B_predicted_ovl_oracle_attr":
            # Predicted Overlap + Hungarian Oracle Attribution
            if gt_segs and hungarian_map:
                active_gt = [s["speaker"] for s in gt_segs if s["start"] <= mid <= s["end"]]
                # Map active GT speakers to hypothesis space
                active_mapped = [hungarian_map[g] for g in active_gt if g in hungarian_map]
                alts = [h for h in active_mapped if h != primary_spk]
                sec_spk = alts[0] if alts else None

        elif attribution_method == "M5C_full_oracle":
            # Handled directly at configuration level via GT overlap segments
            if gt_segs and hungarian_map:
                active_gt = [s["speaker"] for s in gt_segs if s["start"] <= mid <= s["end"]]
                active_mapped = [hungarian_map[g] for g in active_gt if g in hungarian_map]
                alts = [h for h in active_mapped if h != primary_spk]
                sec_spk = alts[0] if alts else None

        if sec_spk is not None:
            sec_segments.append({"start": st, "end": en, "speaker": sec_spk})
            attributed_count += 1

    return sec_segments, len(ovl_segs), attributed_count


def generate_multilabel_rttm(
    base_segs: List[Dict[str, Any]],
    sec_segs: List[Dict[str, Any]],
    out_rttm_path: Path,
    dfn_stem: str,
):
    """Writes a track-clean multi-speaker RTTM."""
    out_rttm_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for s in base_segs:
        dur = max(0.01, s["end"] - s["start"])
        lines.append(f"SPEAKER {dfn_stem} 1 {s['start']:.3f} {dur:.3f} <NA> <NA> {s['speaker']} <NA> <NA>")
    for s in sec_segs:
        dur = max(0.01, s["end"] - s["start"])
        lines.append(f"SPEAKER {dfn_stem} 1 {s['start']:.3f} {dur:.3f} <NA> <NA> {s['speaker']} <NA> <NA>")
    lines.sort(key=lambda l: float(l.split()[3]))
    out_rttm_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_overlap_detection_frame_level(
    gt_segs: List[Dict[str, Any]],
    ovl_segs: List[Dict[str, Any]],
    duration_sec: float,
    resolution: float = 0.01,
) -> Dict[str, float]:
    """Computes exact frame-level overlap detection metrics (Precision, Recall, F1, Durations)."""
    n_frames = int(round(duration_sec / resolution)) + 1
    gt_spk_counts = np.zeros(n_frames, dtype=int)
    for s in gt_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        gt_spk_counts[st_i:en_i] += 1

    gt_ovl_mask = gt_spk_counts >= 2
    pred_ovl_mask = np.zeros(n_frames, dtype=bool)
    for s in ovl_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        pred_ovl_mask[st_i:en_i] = True

    tp = np.sum(gt_ovl_mask & pred_ovl_mask)
    fp = np.sum((~gt_ovl_mask) & pred_ovl_mask)
    fn = np.sum(gt_ovl_mask & (~pred_ovl_mask))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if np.sum(pred_ovl_mask) == 0 else 0.0)
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if np.sum(gt_ovl_mask) == 0 else 0.0)
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "gt_overlap_sec": round(float(np.sum(gt_ovl_mask) * resolution), 3),
        "pred_overlap_sec": round(float(np.sum(pred_ovl_mask) * resolution), 3),
        "tp_overlap_sec": round(float(tp * resolution), 3),
        "fp_overlap_sec": round(float(fp * resolution), 3),
        "fn_overlap_sec": round(float(fn * resolution), 3),
        "overlap_precision": round(prec, 4),
        "overlap_recall": round(rec, 4),
        "overlap_f1": round(f1, 4),
    }


# -----------------------------------------------------------------------------
# 6. Main Execution Engine
# -----------------------------------------------------------------------------

def run_stage5b_2_corrective_suite():
    t_start_suite = time.time()
    manifest_p = REPO_ROOT / "research_stages" / "stage1_data_prep" / "manifests" / "recording_manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    # Define Complete Suite of Corrected Configurations
    # Format: (config_id, name, vad_backbone, ovl_stream, threshold, attribution_method)
    configurations = [
        # Baseline Single-Label (No Overlap)
        ("O00", "Baseline Single-Label (C00 VAD, No Overlap)", "C00", "none", 0.0, "M0_no_overlap"),
        ("O01", "Baseline Single-Label (C50 VAD, No Overlap)", "C50", "none", 0.0, "M0_no_overlap"),
        ("O02", "Baseline Single-Label (C62 VAD, No Overlap)", "C62", "none", 0.0, "M0_no_overlap"),

        # Threshold Sweeps with Physical Support Binarization (C00 + Mono Overlap + M1-R1)
        ("O10-R1", "Overlap Sweep T=0.30 (Mono, M1-R1)", "C00", "mono", 0.30, "M1_temporal_context"),
        ("O11-R1", "Overlap Sweep T=0.40 (Mono, M1-R1)", "C00", "mono", 0.40, "M1_temporal_context"),
        ("O12-R1", "Overlap Sweep T=0.50 (Mono, M1-R1)", "C00", "mono", 0.50, "M1_temporal_context"),
        ("O13-R1", "Overlap Sweep T=0.60 (Mono, M1-R1)", "C00", "mono", 0.60, "M1_temporal_context"),
        ("O14-R1", "Overlap Sweep T=0.70 (Mono, M1-R1)", "C00", "mono", 0.70, "M1_temporal_context"),

        # Stream Comparison (DFN vs Mono at T=0.40)
        ("O20-R1", "Overlap Stream DFN (T=0.40, M1-R1)", "C00", "dfn", 0.40, "M1_temporal_context"),
        ("O21-R1", "Overlap Stream Mono (T=0.40, M1-R1)", "C00", "mono", 0.40, "M1_temporal_context"),

        # Attribution Strategy Comparison (C00, Mono, T=0.40)
        ("O30-R1", "Attribution: M1-R1 Local Temporal Context", "C00", "mono", 0.40, "M1_temporal_context"),
        ("O31-R1", "Attribution: M2-R1 Prototype Affinity (Concat 448d)", "C00", "mono", 0.40, "M2_prototype_concat"),
        ("O31-T", "Attribution: M2-R1 TitaNet-Only (192d)", "C00", "mono", 0.40, "M2_prototype_titanet"),
        ("O31-W", "Attribution: M2-R1 WeSpeaker-Only (256d)", "C00", "mono", 0.40, "M2_prototype_wespeaker"),
        ("O31-NF", "Attribution: M2-R1 Normalized Cosine Average", "C00", "mono", 0.40, "M2_prototype_normfusion"),
        ("O32-R1", "Attribution: M3-R1 Local Turn Alternation", "C00", "mono", 0.40, "M3_turn_alternation"),
        ("O33-R1", "Attribution: M4-R1 Conservative Abstention", "C00", "mono", 0.40, "M4_conservative_abstention"),

        # Hungarian-Aligned Oracle Diagnostics
        ("O34-R1", "Oracle Diag: M5B Predicted Overlap + Hungarian Oracle Attr", "C00", "mono", 0.40, "M5B_predicted_ovl_oracle_attr"),
        ("O35-R1", "Oracle Diag: M5C Full Oracle Overlap + Hungarian Oracle Attr", "C00", "oracle_gt", 0.0, "M5C_full_oracle"),

        # Integrated VAD + Overlap
        ("O40-R1", "Integrated: C50 Mono VAD + Mono Overlap (M1-R1)", "C50", "mono", 0.40, "M1_temporal_context"),
        ("O41-R1", "Integrated: C62 Consensus VAD + Mono Overlap (M1-R1)", "C62", "mono", 0.40, "M1_temporal_context"),
        ("O42-R1", "Integrated: C62 Consensus VAD + Conservative Overlap (M4-R1)", "C62", "mono", 0.40, "M4_conservative_abstention"),
    ]

    # Save configuration manifest
    config_payload = {
        "stage": "Stage 5B.2 Corrective Audit",
        "description": "Corrected Multi-Speaker Overlap Optimization Suite",
        "configurations": [
            {"id": c[0], "name": c[1], "vad_backbone": c[2], "ovl_stream": c[3], "threshold": c[4], "attribution_method": c[5]}
            for c in configurations
        ]
    }
    with open(CONFIGS_DIR / "stage5b_2_corrected_config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    # Pre-load ground truth and recording metadata
    recording_data = {}
    for entry in manifest:
        rec_id = entry["rec_id"]
        dur = float(entry["duration_sec"])
        dfn_stem = Path(entry["dfn_wav"]).stem
        gt_rttm_p = REPO_ROOT / entry["rttm_uniform"]
        gt_segs = p20_common.rttm_to_segments(gt_rttm_p)
        gt_annot = load_track_aware_annotation(gt_rttm_p)
        aligned_words_p = REPO_ROOT / f"research_stages/stage2_baseline/whisperx/{rec_id}_aligned.json"

        # Ground truth overlap intervals
        resolution = 0.01
        n_frames = int(round(dur / resolution)) + 1
        gt_spk_counts = np.zeros(n_frames, dtype=int)
        for s in gt_segs:
            st_i = max(0, int(round(s["start"] / resolution)))
            en_i = min(n_frames, int(round(s["end"] / resolution)))
            gt_spk_counts[st_i:en_i] += 1
        gt_ovl_runs = []
        in_run = False
        start_idx = 0
        for i in range(n_frames):
            if not in_run and gt_spk_counts[i] >= 2:
                in_run = True
                start_idx = i
            elif in_run and gt_spk_counts[i] < 2:
                in_run = False
                gt_ovl_runs.append({"start": round(start_idx * resolution, 3), "end": round((i - 1) * resolution, 3)})
        if in_run:
            gt_ovl_runs.append({"start": round(start_idx * resolution, 3), "end": round((n_frames - 1) * resolution, 3)})

        # VAD backbone paths
        vad_rttm_paths = {
            "C00": REPO_ROOT / f"research_stages/stage5b/stage5b_1_vad_optimization/predictions/C00_baseline/{dfn_stem}.rttm",
            "C50": REPO_ROOT / f"research_stages/stage5b/stage5b_1_vad_optimization/predictions/C50_stream_mono/{dfn_stem}.rttm",
            "C62": REPO_ROOT / f"research_stages/stage5b/stage5b_1_vad_optimization/predictions/C62_comb_consensus_hyst_gap/{dfn_stem}.rttm",
        }

        # Load posteriors
        ovl_streams = {}
        for s in ["mono", "dfn"]:
            ovl_streams[s] = load_overlap_posteriors(rec_id, s)

        # Baseline segments for C00, C50, C62
        base_segs_map = {k: p20_common.rttm_to_segments(vad_rttm_paths[k]) for k in vad_rttm_paths}

        # Prototypes for C00
        proto_data = load_dual_encoder_prototypes(base_segs_map["C00"], rec_id)

        # Hungarian mappings
        hungarian_maps = {
            k: compute_hungarian_mapping(gt_segs, base_segs_map[k], dur) for k in base_segs_map
        }

        recording_data[rec_id] = {
            "entry": entry,
            "duration": dur,
            "dfn_stem": dfn_stem,
            "gt_rttm_p": gt_rttm_p,
            "gt_segs": gt_segs,
            "gt_annot": gt_annot,
            "gt_ovl_segs": gt_ovl_runs,
            "aligned_words_p": aligned_words_p,
            "vad_rttm_paths": vad_rttm_paths,
            "base_segs_map": base_segs_map,
            "ovl_streams": ovl_streams,
            "proto_data": proto_data,
            "hungarian_maps": hungarian_maps,
        }

    all_per_rec_rows: List[Dict[str, Any]] = []

    print(f"Beginning Stage 5B.2 Corrective Re-evaluation ({len(configurations)} configurations x 8 recordings = {len(configurations)*8} runs)...")

    for c_id, c_name, vad_bb, ovl_str, thresh, attr_method in configurations:
        for entry in manifest:
            rec_id = entry["rec_id"]
            rdata = recording_data[rec_id]
            dur = rdata["duration"]
            dfn_stem = rdata["dfn_stem"]
            gt_annot = rdata["gt_annot"]
            gt_segs = rdata["gt_segs"]
            base_segs = rdata["base_segs_map"][vad_bb]

            # 1. Obtain Overlap Segments
            if ovl_str == "oracle_gt":
                ovl_segs = rdata["gt_ovl_segs"]
            elif ovl_str in ["mono", "dfn"]:
                frame_times, p_ovl = rdata["ovl_streams"][ovl_str]
                ovl_segs = binarize_overlap_corrected(p_ovl, dur, threshold=thresh)
            else:
                ovl_segs = []

            # 2. Attribute Secondary Speakers
            hungarian_map = rdata["hungarian_maps"][vad_bb]
            sec_segs, tot_ovl, attr_ovl = attribute_secondary_speakers_corrected(
                base_segs=base_segs,
                ovl_segs=ovl_segs,
                attribution_method=attr_method,
                duration_sec=dur,
                proto_data=rdata["proto_data"],
                gt_segs=gt_segs,
                hungarian_map=hungarian_map,
            )

            # 3. Generate Multi-Label RTTM
            out_rttm_p = PRED_DIR / c_id / f"{dfn_stem}.rttm"
            generate_multilabel_rttm(base_segs, sec_segs, out_rttm_p, dfn_stem)
            hyp_annot = load_track_aware_annotation(out_rttm_p)

            # 4. Compute Diarization Metrics
            diar_metrics = compute_track_aware_diarization_metrics(gt_annot, hyp_annot)

            # 5. Compute Overlap Detection Metrics
            ovl_metrics = evaluate_overlap_detection_frame_level(gt_segs, ovl_segs, dur)

            # 6. Compute Word Attribution Metrics (cpWER & saWER)
            cpwer = None
            sawer = None
            if rdata["aligned_words_p"].exists():
                try:
                    hyp_segs_list = p20_common.rttm_to_segments(out_rttm_p)
                    aligned_words = p20_common.read_json(rdata["aligned_words_p"])
                    gt_json_p = REPO_ROOT / rdata["entry"]["json_uniform"]
                    words_attr = p20_common.attributed_words(aligned_words, hyp_segs_list)
                    segs_attr = p20_common.group_words(words_attr)
                    tmp_attr_p = PRED_DIR / c_id / f"{rec_id}_attr.json"
                    p20_common.write_json(tmp_attr_p, {"segments": segs_attr})
                    f_metrics = p20_common.fusion_metrics(gt_json_p, tmp_attr_p, dur)
                    if "cpwer" in f_metrics and f_metrics["cpwer"] is not None:
                        cpwer = float(f_metrics["cpwer"])
                    if "sawer" in f_metrics and f_metrics["sawer"] is not None:
                        sawer = float(f_metrics["sawer"])
                except Exception as e:
                    print(f"Word metric failed for {rec_id} {c_id}: {e}")

            sec_dur_added = round(sum(s["end"] - s["start"] for s in sec_segs), 3)
            abstain_rate = (1.0 - (attr_ovl / tot_ovl)) if tot_ovl > 0 else 0.0

            row = {
                "rec_id": rec_id,
                "config_id": c_id,
                "config_name": c_name,
                "vad_backbone": vad_bb,
                "ovl_stream": ovl_str,
                "ovl_threshold": thresh,
                "attr_method": attr_method,
                "duration_sec": dur,
                **diar_metrics,
                **ovl_metrics,
                "detected_overlap_count": tot_ovl,
                "attributed_overlap_count": attr_ovl,
                "abstention_rate": round(abstain_rate, 4),
                "secondary_segments_added": len(sec_segs),
                "secondary_duration_added_sec": sec_dur_added,
                "cpwer": cpwer,
                "sawer": sawer,
            }
            all_per_rec_rows.append(row)

    # Save per-recording CSV
    per_rec_csv = METRICS_DIR / "STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv"
    with per_rec_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_per_rec_rows[0].keys()))
        w.writeheader()
        w.writerows(all_per_rec_rows)

    # -------------------------------------------------------------------------
    # 7. Summary Table: Macro and Pooled Corpus Aggregations
    # -------------------------------------------------------------------------
    summary_rows = []
    overlap_metric_rows = []
    word_metric_rows = []

    for c_id, c_name, vad_bb, ovl_str, thresh, attr_method in configurations:
        sub = [r for r in all_per_rec_rows if r["config_id"] == c_id]
        cpwers = [r["cpwer"] for r in sub if r["cpwer"] is not None]
        sawers = [r["sawer"] for r in sub if r["sawer"] is not None]

        r02_row = next(r for r in sub if "R02" in r["rec_id"])
        r04_row = next(r for r in sub if "R04" in r["rec_id"])

        # Pooled metrics
        tot_ref_sec = sum(r["total_reference_sec"] for r in sub)
        tot_miss_sec = sum(r["miss_sec"] for r in sub)
        tot_fa_sec = sum(r["fa_sec"] for r in sub)
        tot_conf_sec = sum(r["conf_sec"] for r in sub)
        pooled_der_c0 = (tot_miss_sec + tot_fa_sec + tot_conf_sec) / tot_ref_sec
        pooled_miss = tot_miss_sec / tot_ref_sec
        pooled_fa = tot_fa_sec / tot_ref_sec
        pooled_conf = tot_conf_sec / tot_ref_sec

        # Overlap pooled metrics
        tot_tp = sum(r["tp_overlap_sec"] for r in sub)
        tot_fp = sum(r["fp_overlap_sec"] for r in sub)
        tot_fn = sum(r["fn_overlap_sec"] for r in sub)
        pooled_ovl_prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        pooled_ovl_rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        pooled_ovl_f1 = (2 * pooled_ovl_prec * pooled_ovl_rec / (pooled_ovl_prec + pooled_ovl_rec)) if (pooled_ovl_prec + pooled_ovl_rec) > 0 else 0.0

        s_row = {
            "config_id": c_id,
            "config_name": c_name,
            "vad_backbone": vad_bb,
            "ovl_stream": ovl_str,
            "ovl_threshold": thresh,
            "attr_method": attr_method,
            # Macro Diarization
            "macro_der_c0": float(np.mean([r["der_c0"] for r in sub])),
            "macro_der_c25": float(np.mean([r["der_c25"] for r in sub])),
            "macro_diar_miss": float(np.mean([r["diar_miss"] for r in sub])),
            "macro_diar_fa": float(np.mean([r["diar_fa"] for r in sub])),
            "macro_diar_conf": float(np.mean([r["diar_confusion"] for r in sub])),
            "macro_jer": float(np.mean([r["jer"] for r in sub])),
            # Pooled Diarization
            "pooled_der_c0": float(pooled_der_c0),
            "pooled_diar_miss": float(pooled_miss),
            "pooled_diar_fa": float(pooled_fa),
            "pooled_diar_conf": float(pooled_conf),
            # Key files
            "r04_der_c0": float(r04_row["der_c0"]),
            "r04_der_c25": float(r04_row["der_c25"]),
            "r04_miss": float(r04_row["diar_miss"]),
            "r04_fa": float(r04_row["diar_fa"]),
            "r04_conf": float(r04_row["diar_confusion"]),
            "r02_der_c0": float(r02_row["der_c0"]),
            "r02_miss": float(r02_row["diar_miss"]),
            # Word metrics
            "macro_cpwer": float(np.mean(cpwers)) if cpwers else None,
            "macro_sawer": float(np.mean(sawers)) if sawers else None,
            # Attribution stats
            "macro_abstention_rate": float(np.mean([r["abstention_rate"] for r in sub])),
            "total_secondary_sec_added": round(sum(r["secondary_duration_added_sec"] for r in sub), 3),
        }
        summary_rows.append(s_row)

        ovl_row = {
            "config_id": c_id,
            "config_name": c_name,
            "macro_ovl_precision": float(np.mean([r["overlap_precision"] for r in sub])),
            "macro_ovl_recall": float(np.mean([r["overlap_recall"] for r in sub])),
            "macro_ovl_f1": float(np.mean([r["overlap_f1"] for r in sub])),
            "pooled_ovl_precision": float(pooled_ovl_prec),
            "pooled_ovl_recall": float(pooled_ovl_rec),
            "pooled_ovl_f1": float(pooled_ovl_f1),
            "total_gt_overlap_sec": round(sum(r["gt_overlap_sec"] for r in sub), 3),
            "total_pred_overlap_sec": round(sum(r["pred_overlap_sec"] for r in sub), 3),
            "total_tp_overlap_sec": round(tot_tp, 3),
            "total_fp_overlap_sec": round(tot_fp, 3),
            "total_fn_overlap_sec": round(tot_fn, 3),
        }
        overlap_metric_rows.append(ovl_row)

        w_row = {
            "config_id": c_id,
            "config_name": c_name,
            "macro_cpwer": float(np.mean(cpwers)) if cpwers else None,
            "macro_sawer": float(np.mean(sawers)) if sawers else None,
            "r04_cpwer": float(r04_row["cpwer"]) if r04_row["cpwer"] is not None else None,
            "r02_cpwer": float(r02_row["cpwer"]) if r02_row["cpwer"] is not None else None,
        }
        word_metric_rows.append(w_row)

    # Save summary CSVs
    summary_csv = METRICS_DIR / "STAGE5B_2_CORRECTED_RESULTS.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    ovl_csv = METRICS_DIR / "STAGE5B_2_CORRECTED_OVERLAP_METRICS.csv"
    with ovl_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(overlap_metric_rows[0].keys()))
        w.writeheader()
        w.writerows(overlap_metric_rows)

    word_csv = METRICS_DIR / "STAGE5B_2_CORRECTED_WORD_METRICS.csv"
    with word_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(word_metric_rows[0].keys()))
        w.writeheader()
        w.writerows(word_metric_rows)

    # -------------------------------------------------------------------------
    # 8. Statistical Bootstrap & Hypothesis Tests
    # -------------------------------------------------------------------------
    print("\n--- STATISTICAL SIGNIFICANCE TESTS (Recording Cluster Unit, N=8) ---")
    base_rows = {r["rec_id"]: r for r in all_per_rec_rows if r["config_id"] == "O00"}
    comparisons = [
        ("O11-R1", "O00", "M1-R1 vs O00 Baseline"),
        ("O31-R1", "O00", "M2-R1 Concat vs O00 Baseline"),
        ("O32-R1", "O00", "M3-R1 Turn Alternation vs O00 Baseline"),
        ("O33-R1", "O00", "M4-R1 Conservative Abstain vs O00 Baseline"),
        ("O42-R1", "O02", "O42-R1 Integrated vs O02 Baseline"),
    ]

    stat_records = []
    np.random.seed(42)
    rec_keys = sorted(list(base_rows.keys()))

    for cand_id, ref_id, comp_name in comparisons:
        cand_map = {r["rec_id"]: r for r in all_per_rec_rows if r["config_id"] == cand_id}
        ref_map = {r["rec_id"]: r for r in all_per_rec_rows if r["config_id"] == ref_id}

        deltas = [cand_map[k]["der_c0"] - ref_map[k]["der_c0"] for k in rec_keys]
        macro_delta = float(np.mean(deltas))

        # 10,000 cluster bootstrap
        boot_means = []
        for _ in range(10000):
            sample_idx = np.random.choice(len(deltas), size=len(deltas), replace=True)
            boot_means.append(np.mean([deltas[i] for i in sample_idx]))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        # Wilcoxon signed rank test
        try:
            w_res = wilcoxon(deltas)
            w_stat, p_val = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, p_val = 0.0, 1.0

        r04_delta = cand_map["R04_piers_robinson"]["der_c0"] - ref_map["R04_piers_robinson"]["der_c0"]
        stat_records.append({
            "comparison": comp_name,
            "macro_delta_der": macro_delta,
            "boot_ci_95": f"[{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]",
            "wilcoxon_stat": w_stat,
            "wilcoxon_pval": p_val,
            "r04_delta_der": r04_delta,
        })
        print(f"{comp_name:35s} | Macro Delta: {macro_delta*100:+.2f}% | 95% CI: [{ci_lower*100:+.2f}%, {ci_upper*100:+.2f}%] | W={w_stat:.1f}, p={p_val:.4f} | R04 Delta: {r04_delta*100:+.2f}%")

    # -------------------------------------------------------------------------
    # 9. Corrected Plot Generation
    # -------------------------------------------------------------------------
    generate_corrected_plots(all_per_rec_rows, summary_rows)

    print(f"\nStage 5B.2 Corrective Re-evaluation completed in {time.time()-t_start_suite:.2f}s!")


# -----------------------------------------------------------------------------
# 10. Analytical Plotting Suite
# -----------------------------------------------------------------------------

def generate_corrected_plots(all_rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]]):
    # Plot 1: Macro DER Comparison (collar 0.0s and collar 0.25s)
    sel_configs = ["O00", "O30-R1", "O31-R1", "O31-T", "O31-W", "O31-NF", "O32-R1", "O33-R1", "O34-R1", "O35-R1", "O42-R1"]
    sub_summary = [s for s in summary_rows if s["config_id"] in sel_configs]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sub_summary))
    w = 0.35
    der0 = [s["macro_der_c0"] * 100 for s in sub_summary]
    der25 = [s["macro_der_c25"] * 100 for s in sub_summary]
    ax.bar(x - w/2, der0, w, label="Macro DER (c=0.0s)", color="#1f77b4")
    ax.bar(x + w/2, der25, w, label="Macro DER (c=0.25s)", color="#aec7e8")
    ax.set_xticks(x)
    ax.set_xticklabels([s["config_id"] for s in sub_summary], rotation=30, ha="right")
    ax.set_ylabel("Macro DER (%)")
    ax.set_title("Stage 5B.2 Corrected Macro DER Across Overlap & Attribution Strategies")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "macro_der_comparison.png", dpi=150)
    plt.close()

    # Plot 2: Per-Recording DER Change relative to O00 (for O32-R1, O33-R1, O42-R1)
    base_map = {r["rec_id"]: r["der_c0"] for r in all_rows if r["config_id"] == "O00"}
    recs = sorted(list(base_map.keys()))
    comp_configs = ["O30-R1", "O31-R1", "O32-R1", "O33-R1", "O42-R1"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(recs))
    w = 0.16
    for i, cid in enumerate(comp_configs):
        c_map = {r["rec_id"]: r["der_c0"] for r in all_rows if r["config_id"] == cid}
        deltas = [(c_map[r] - base_map[r]) * 100 for r in recs]
        ax.bar(x + (i - 2) * w, deltas, w, label=cid)

    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(recs, rotation=30, ha="right")
    ax.set_ylabel("Absolute DER Delta vs O00 Baseline (%) [Negative = Improved]")
    ax.set_title("Per-Recording DER Delta Across Attribution Strategies")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_recording_der_delta.png", dpi=150)
    plt.close()

    # Plot 3: R02 and R04 Overlap Threshold Sensitivity
    t_configs = ["O00", "O10-R1", "O11-R1", "O12-R1", "O13-R1", "O14-R1"]
    thresholds = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70]
    r04_sub = [r for r in all_rows if r["config_id"] in t_configs and "R04" in r["rec_id"]]
    r02_sub = [r for r in all_rows if r["config_id"] in t_configs and "R02" in r["rec_id"]]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(thresholds, [r["der_c0"]*100 for r in r04_sub], marker="o", label="DER (c=0)")
    ax[0].plot(thresholds, [r["diar_miss"]*100 for r in r04_sub], marker="s", linestyle="--", label="Miss Rate")
    ax[0].plot(thresholds, [r["diar_fa"]*100 for r in r04_sub], marker="^", linestyle=":", label="False Alarm Rate")
    ax[0].set_title("R04 (Piers Robinson) vs Overlap Threshold (Corrected)")
    ax[0].set_xlabel("Overlap Threshold (0.0 = Single-Label Baseline)")
    ax[0].set_ylabel("Error Rate (%)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(thresholds, [r["der_c0"]*100 for r in r02_sub], marker="o", color="orange", label="DER (c=0)")
    ax[1].plot(thresholds, [r["diar_miss"]*100 for r in r02_sub], marker="s", color="red", linestyle="--", label="Miss Rate")
    ax[1].plot(thresholds, [r["diar_fa"]*100 for r in r02_sub], marker="^", color="green", linestyle=":", label="False Alarm Rate")
    ax[1].set_title("R02 (Dennis Skinner) vs Overlap Threshold (Corrected)")
    ax[1].set_xlabel("Overlap Threshold (0.0 = Single-Label Baseline)")
    ax[1].set_ylabel("Error Rate (%)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overlap_threshold_sensitivity_r02_r04_corrected.png", dpi=150)
    plt.close()

    # Plot 4: Overlap Precision-Recall Comparison across configurations
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in summary_rows:
        ovl_p = s.get("pooled_ovl_precision") or 0.0
        ovl_r = s.get("pooled_ovl_recall") or 0.0
        if s["ovl_stream"] != "none":
            ax.scatter(ovl_r * 100, ovl_p * 100, s=80, label=s["config_id"])
    ax.set_xlabel("Pooled Overlap Recall (%)")
    ax.set_ylabel("Pooled Overlap Precision (%)")
    ax.set_title("Pooled Overlap Detection Precision vs Recall")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overlap_precision_recall_comparison.png", dpi=150)
    plt.close()

    # Plot 5: Miss / FA / Confusion Decomposition across configurations
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sub_summary))
    w = 0.55
    misses = [s["macro_diar_miss"] * 100 for s in sub_summary]
    fas = [s["macro_diar_fa"] * 100 for s in sub_summary]
    confs = [s["macro_diar_conf"] * 100 for s in sub_summary]

    ax.bar(x, misses, w, label="Missed Speech", color="#d62728")
    ax.bar(x, confs, w, bottom=misses, label="Speaker Confusion", color="#ff7f0e")
    ax.bar(x, fas, w, bottom=np.array(misses)+np.array(confs), label="False Alarm", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels([s["config_id"] for s in sub_summary], rotation=30, ha="right")
    ax.set_ylabel("Macro Error Rate (%)")
    ax.set_title("DER Component Decomposition (Miss / Confusion / False Alarm)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "der_component_decomposition.png", dpi=150)
    plt.close()

    # Plot 6: cpWER comparison across configurations
    cpwer_sub = [s for s in sub_summary if s["macro_cpwer"] is not None]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(cpwer_sub))
    ax.bar(x, [s["macro_cpwer"] * 100 for s in cpwer_sub], width=0.45, color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels([s["config_id"] for s in cpwer_sub], rotation=30, ha="right")
    ax.set_ylabel("Macro cpWER (%)")
    ax.set_title("Downstream Conversational cpWER Across Strategies")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "macro_cpwer_comparison.png", dpi=150)
    plt.close()

    # Plot 7: Secondary Attribution Coverage and Abstention Rate
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(sub_summary))
    w = 0.35
    abstains = [s["macro_abstention_rate"] * 100 for s in sub_summary]
    durs = [s["total_secondary_sec_added"] for s in sub_summary]
    ax.bar(x - w/2, abstains, w, label="Abstention Rate (%)", color="#7f7f7f")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, durs, w, label="Total Secondary Sec Added (s)", color="#17becf")
    ax.set_xticks(x)
    ax.set_xticklabels([s["config_id"] for s in sub_summary], rotation=30, ha="right")
    ax.set_ylabel("Abstention Rate (%)")
    ax2.set_ylabel("Total Secondary Duration Added (s)")
    ax.set_title("Attribution Abstention Rate vs Secondary Duration Added")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "attribution_coverage_and_abstention.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_stage5b_2_corrective_suite()
