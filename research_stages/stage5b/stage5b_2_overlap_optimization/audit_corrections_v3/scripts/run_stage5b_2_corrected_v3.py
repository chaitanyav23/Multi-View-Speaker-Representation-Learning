#!/usr/bin/env python3
"""
Stage 5B.2 Corrective Audit v2 Execution Engine.
Applies verified correctness patches:
- Correction A: M1 Temporal Context Attribution (evidence, distance tie-break, deterministic order)
- Correction B: M3/M4 Contiguous Local Turn Extraction (boundary search, gap termination, min duration)
- Correction C: M2 Embedding Purity and Validity (overlap fraction filtering, prototype purity, abstention)
- Correction D: Overlap Timestamp Alignment (frame bin support [start, end), receptive field decoupling)
- Correction E: Multi-Label Hungarian Speaker Mapping (simultaneous active speakers preserved)
- Correction F: Strict Metric Error Handling (no zero fallbacks, explicit None, denominator validation)
"""

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

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

AUDIT_V3_DIR = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_2_overlap_optimization" / "audit_corrections_v3"
CONFIGS_DIR = AUDIT_V3_DIR / "configs"
METRICS_DIR = AUDIT_V3_DIR / "metrics"
PLOTS_DIR = AUDIT_V3_DIR / "plots"
PRED_DIR = AUDIT_V3_DIR / "predictions"
CACHE_DIR = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_2_overlap_optimization" / "overlap_cache"

for d in [CONFIGS_DIR, METRICS_DIR, PLOTS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Correction F: Strict Metric Error Handling & Track-Aware Annotation
# -----------------------------------------------------------------------------

def load_track_aware_annotation(path: Path) -> Annotation:
    """Loads an RTTM into a pyannote Annotation with unique track IDs to prevent track overwriting."""
    ann = Annotation()
    for i, seg in enumerate(p20_common.rttm_to_segments(path)):
        ann[Segment(seg["start"], seg["end"]), f"track_{i}"] = seg["speaker"]
    return ann


def safe_diarization_metrics(
    ref_annot: Annotation,
    hyp_annot: Annotation,
    collar_c0: float = 0.0,
    collar_c25: float = 0.25,
) -> Dict[str, Any]:
    """Computes track-aware DER without zero fallbacks. Distinguishes failure from valid zero."""
    ref_chart = ref_annot.get_timeline().duration()
    if ref_chart <= 0.0:
        raise ValueError("Empty reference annotation: total speech duration is zero.")

    # Collar 0.0s
    der_metric_c0 = DiarizationErrorRate(collar=collar_c0, skip_overlap=False)
    details_c0 = der_metric_c0(ref_annot, hyp_annot, detailed=True)
    total_ref = details_c0.get("total")
    if total_ref is None or total_ref <= 0.0:
        raise ValueError(f"DiarizationErrorRate failed to return positive reference duration: {total_ref}")

    miss_sec = details_c0.get("missed detection")
    fa_sec = details_c0.get("false alarm")
    conf_sec = details_c0.get("confusion")
    der_c0 = details_c0.get("diarization error rate")

    if any(x is None for x in [miss_sec, fa_sec, conf_sec, der_c0]):
        raise KeyError(f"Missing required detail key in DiarizationErrorRate results: {details_c0}")

    # Collar 0.25s
    der_metric_c25 = DiarizationErrorRate(collar=collar_c25, skip_overlap=False)
    details_c25 = der_metric_c25(ref_annot, hyp_annot, detailed=True)
    der_c25 = details_c25.get("diarization error rate")
    if der_c25 is None:
        raise KeyError(f"Missing collar 0.25 DER in {details_c25}")

    # JER
    jer_val = None
    jer_error = None
    try:
        jer_val = float(JaccardErrorRate()(ref_annot, hyp_annot))
    except Exception as e:
        jer_val = None
        jer_error = str(e)

    return {
        "der_c0": float(der_c0),
        "der_c25": float(der_c25),
        "diar_miss": float(miss_sec) / total_ref,
        "diar_fa": float(fa_sec) / total_ref,
        "diar_confusion": float(conf_sec) / total_ref,
        "jer": jer_val,
        "jer_error": jer_error,
        "total_reference_sec": float(total_ref),
        "miss_sec": float(miss_sec),
        "fa_sec": float(fa_sec),
        "conf_sec": float(conf_sec),
        "status": "SUCCESS",
    }


# -----------------------------------------------------------------------------
# Correction D: Overlap Timestamp Alignment (Frame-Bin Support)
# -----------------------------------------------------------------------------

def load_overlap_posteriors(rec_id: str, stream: str = "mono") -> Tuple[np.ndarray, np.ndarray]:
    """Loads frame_times and p_overlap for the specified stream from frozen cache."""
    npz_p = CACHE_DIR / f"{rec_id}_{stream}_overlap.npz"
    if not npz_p.exists():
        raise FileNotFoundError(f"Overlap cache not found: {npz_p}")
    data = np.load(npz_p)
    return data["frame_times"], data["p_overlap"]


def binarize_overlap_frame_bin_support(
    frame_times: np.ndarray,
    p_ovl: np.ndarray,
    duration_sec: float,
    threshold: float = 0.40,
    step_sec: float = 0.016875,
    min_duration_on: float = 0.10,
    min_duration_off: float = 0.15,
) -> List[Dict[str, float]]:
    """
    Correction D: Frame-bin temporal support binarization.
    Each frame i represents a time bin centered at frame_times[i] with width step_sec.
    """
    n_frames = len(p_ovl)
    is_active = p_ovl >= threshold

    runs: List[Tuple[int, int]] = []
    in_run = False
    start_idx = 0
    for i in range(n_frames):
        if not in_run and is_active[i]:
            in_run = True
            start_idx = i
        elif in_run and not is_active[i]:
            in_run = False
            runs.append((start_idx, i - 1))
    if in_run:
        runs.append((start_idx, n_frames - 1))

    if not runs:
        return []

    raw_segs: List[Dict[str, float]] = []
    for s_i, e_i in runs:
        st = max(0.0, float(frame_times[s_i] - step_sec / 2.0))
        en = min(duration_sec, float(frame_times[e_i] + step_sec / 2.0))
        raw_segs.append({"start": st, "end": en})

    # Gap merging (fill inactive gaps shorter than min_duration_off)
    merged = [raw_segs[0]]
    for seg in raw_segs[1:]:
        gap = seg["start"] - merged[-1]["end"]
        if gap < min_duration_off:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)

    # Prune bursts shorter than min_duration_on
    filtered = []
    for s in merged:
        if (s["end"] - s["start"]) >= min_duration_on:
            filtered.append({"start": round(s["start"], 3), "end": round(s["end"], 3)})

    return filtered


# -----------------------------------------------------------------------------
# Correction C: M2 Embedding Purity and Validity
# -----------------------------------------------------------------------------

def load_dual_encoder_prototypes_pure(
    base_segs: List[Dict[str, Any]],
    rec_id: str,
    min_prototype_purity: float = 0.80,
) -> Dict[str, Any]:
    """
    Builds speaker prototypes using only clean, high-purity baseline segments.
    Excludes subsegments containing competing-speaker activity.
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
        s_st, s_en = subseg["start"], subseg["end"]
        s_dur = subseg["duration"]

        # Calculate speaker duration breakdown within subsegment window
        spk_coverage: Dict[str, float] = {}
        for b in base_segs:
            inter = max(0.0, min(s_en, b["end"]) - max(s_st, b["start"]))
            if inter > 0:
                spk = b["speaker"]
                spk_coverage[spk] = spk_coverage.get(spk, 0.0) + inter

        # Purity check: must have a dominant speaker covering >= min_prototype_purity
        if spk_coverage:
            best_spk, best_dur = max(spk_coverage.items(), key=lambda x: x[1])
            competing_dur = sum(v for k, v in spk_coverage.items() if k != best_spk)
            if (best_dur / s_dur) >= min_prototype_purity and competing_dur == 0.0:
                spk_t.setdefault(best_spk, []).append(et)
                spk_w.setdefault(best_spk, []).append(ew)
                spk_c.setdefault(best_spk, []).append(ec)

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


def attribute_m2_pure(
    subsegments: List[Dict[str, Any]],
    proto_data: Dict[str, Any],
    overlap_interval: Tuple[float, float],
    primary_spk: Optional[str],
    method: str = "concat",  # 'concat', 'titanet', 'wespeaker', 'normfusion'
    min_overlap_purity: float = 0.40,
) -> Dict[str, Any]:
    """Correction C: M2 Embedding Purity and Validity."""
    if primary_spk is None:
        return {"speaker": None, "reason": "unresolved_primary", "retained_subsegs": 0}

    st_sec, en_sec = overlap_interval

    # Identify subsegments satisfying min_overlap_purity
    valid_indices = []
    weights = []
    for idx, s in enumerate(subsegments):
        inter = max(0.0, min(en_sec, s["end"]) - max(st_sec, s["start"]))
        frac_of_window = inter / s["duration"]
        if frac_of_window >= min_overlap_purity:
            valid_indices.append(idx)
            weights.append(inter)

    if not valid_indices:
        return {"speaker": None, "reason": "insufficient_pure_embeddings", "retained_subsegs": 0}

    weights_arr = np.array(weights) / sum(weights)

    if method == "concat":
        protos = proto_data["concat"]
        alt_spks = sorted([s for s in protos if s != primary_spk])
        if not alt_spks:
            return {"speaker": None, "reason": "no_alternate_prototypes", "retained_subsegs": len(valid_indices)}
        embs = proto_data["c_embs"][valid_indices]
        e_ovl = np.sum(embs * weights_arr[:, None], axis=0)
        e_ovl = e_ovl / max(np.linalg.norm(e_ovl), 1e-8)
        sims = {s: float(np.dot(e_ovl, protos[s])) for s in alt_spks}
        best_spk = max(alt_spks, key=lambda s: (sims[s], -alt_spks.index(s)))
        return {"speaker": best_spk, "reason": "selected", "retained_subsegs": len(valid_indices)}

    elif method == "titanet":
        protos = proto_data["titanet"]
        alt_spks = sorted([s for s in protos if s != primary_spk])
        if not alt_spks:
            return {"speaker": None, "reason": "no_alternate_prototypes", "retained_subsegs": len(valid_indices)}
        embs = proto_data["t_embs"][valid_indices]
        e_ovl = np.sum(embs * weights_arr[:, None], axis=0)
        e_ovl = e_ovl / max(np.linalg.norm(e_ovl), 1e-8)
        sims = {s: float(np.dot(e_ovl, protos[s])) for s in alt_spks}
        best_spk = max(alt_spks, key=lambda s: (sims[s], -alt_spks.index(s)))
        return {"speaker": best_spk, "reason": "selected", "retained_subsegs": len(valid_indices)}

    elif method == "wespeaker":
        protos = proto_data["wespeaker"]
        alt_spks = sorted([s for s in protos if s != primary_spk])
        if not alt_spks:
            return {"speaker": None, "reason": "no_alternate_prototypes", "retained_subsegs": len(valid_indices)}
        embs = proto_data["w_embs"][valid_indices]
        e_ovl = np.sum(embs * weights_arr[:, None], axis=0)
        e_ovl = e_ovl / max(np.linalg.norm(e_ovl), 1e-8)
        sims = {s: float(np.dot(e_ovl, protos[s])) for s in alt_spks}
        best_spk = max(alt_spks, key=lambda s: (sims[s], -alt_spks.index(s)))
        return {"speaker": best_spk, "reason": "selected", "retained_subsegs": len(valid_indices)}

    elif method == "normfusion":
        protos_t = proto_data["titanet"]
        protos_w = proto_data["wespeaker"]
        alt_spks = sorted([s for s in protos_t if s != primary_spk and s in protos_w])
        if not alt_spks:
            return {"speaker": None, "reason": "no_alternate_prototypes", "retained_subsegs": len(valid_indices)}
        e_t = np.sum(proto_data["t_embs"][valid_indices] * weights_arr[:, None], axis=0)
        e_t = e_t / max(np.linalg.norm(e_t), 1e-8)
        e_w = np.sum(proto_data["w_embs"][valid_indices] * weights_arr[:, None], axis=0)
        e_w = e_w / max(np.linalg.norm(e_w), 1e-8)
        sims = {s: 0.5 * float(np.dot(e_t, protos_t[s])) + 0.5 * float(np.dot(e_w, protos_w[s])) for s in alt_spks}
        best_spk = max(alt_spks, key=lambda s: (sims[s], -alt_spks.index(s)))
        return {"speaker": best_spk, "reason": "selected", "retained_subsegs": len(valid_indices)}

    raise ValueError(f"Unknown M2 method: {method}")


# -----------------------------------------------------------------------------
# Correction E: Multi-Label Hungarian Speaker Mapping
# -----------------------------------------------------------------------------

def compute_multilabel_hungarian_mapping(
    ref_segs: List[Dict[str, Any]],
    hyp_segs: List[Dict[str, Any]],
    duration_sec: float,
    resolution: float = 0.01,
) -> Tuple[Dict[str, str], List[str], List[str], np.ndarray, List[str], List[str]]:
    """Correction E: Multi-label co-occurrence Hungarian matching."""
    ref_spks = sorted(list(set(s["speaker"] for s in ref_segs)))
    hyp_spks = sorted(list(set(s["speaker"] for s in hyp_segs)))

    if not ref_spks or not hyp_spks:
        return {}, ref_spks, hyp_spks, np.zeros((len(ref_spks), len(hyp_spks))), ref_spks, hyp_spks

    n_frames = int(math.ceil(duration_sec / resolution)) + 1
    ref_active: List[Set[str]] = [set() for _ in range(n_frames)]
    hyp_active: List[Set[str]] = [set() for _ in range(n_frames)]

    for s in ref_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        for i in range(st_i, en_i):
            ref_active[i].add(s["speaker"])

    for s in hyp_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        for i in range(st_i, en_i):
            hyp_active[i].add(s["speaker"])

    cost_matrix = np.zeros((len(ref_spks), len(hyp_spks)))
    for i in range(n_frames):
        r_set = ref_active[i]
        h_set = hyp_active[i]
        if r_set and h_set:
            for r in r_set:
                r_idx = ref_spks.index(r)
                for h in h_set:
                    h_idx = hyp_spks.index(h)
                    cost_matrix[r_idx, h_idx] -= 1

    r_ind, h_ind = linear_sum_assignment(cost_matrix)
    mapping = {}
    matched_refs = set()
    matched_hyps = set()
    for r, h in zip(r_ind, h_ind):
        mapping[ref_spks[r]] = hyp_spks[h]
        matched_refs.add(ref_spks[r])
        matched_hyps.add(hyp_spks[h])

    unmatched_refs = sorted(list(set(ref_spks) - matched_refs))
    unmatched_hyps = sorted(list(set(hyp_spks) - matched_hyps))
    return mapping, unmatched_refs, unmatched_hyps, cost_matrix, ref_spks, hyp_spks


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Correction A (V3): Deterministic Multi-Label Oracle Speaker Attribution
# -----------------------------------------------------------------------------

def attribute_secondary_oracle_multilabel(
    overlap_interval: Tuple[float, float],
    primary_spk: Optional[str],
    gt_segs: List[Dict[str, Any]],
    hungarian_map: Dict[str, str],
    cost_matrix: Optional[np.ndarray] = None,
    ref_spk_list: Optional[List[str]] = None,
    hyp_spk_list: Optional[List[str]] = None,
    resolution: float = 0.01,
) -> Dict[str, Any]:
    """
    Deterministic multi-label oracle secondary speaker attribution.
    Resolves V2 order-dependence and arbitrary first-alternate selection.
    
    1. Computes exact temporal intersection of all reference segments in [st, en).
    2. Maps active reference speakers to hypothesis speakers using hungarian_map.
    3. Handles single-reference false alarms, unresolved primary, 2-speaker, and >2 speaker overlaps.
    4. Deterministic ranking on:
       - local overlap duration in interval (descending)
       - global co-occurrence in recording (descending)
       - lexicographical speaker ID (ascending)
    5. Invariant to input ordering of gt_segs, hungarian_map, and candidates.
    """
    st_sec, en_sec = overlap_interval
    if en_sec <= st_sec:
        return {
            "speaker": None,
            "reason": "empty_overlap_interval",
            "all_active_gt": {},
            "all_active_mapped": {},
            "discarded_speakers": [],
            "is_truncated_multispeaker": False,
            "confidence": 0.0,
        }

    # 1. Compute duration of intersection for every reference segment in overlap interval
    gt_durations: Dict[str, float] = {}
    for s in gt_segs:
        inter_st = max(st_sec, s["start"])
        inter_en = min(en_sec, s["end"])
        if inter_en > inter_st:
            spk = s["speaker"]
            gt_durations[spk] = gt_durations.get(spk, 0.0) + (inter_en - inter_st)

    gt_durations = {k: round(v, 6) for k, v in gt_durations.items() if v > 1e-6}

    if not gt_durations:
        return {
            "speaker": None,
            "reason": "no_active_reference_speakers",
            "all_active_gt": {},
            "all_active_mapped": {},
            "discarded_speakers": [],
            "is_truncated_multispeaker": False,
            "confidence": 0.0,
        }

    # 2. Map active reference speakers to hypothesis speakers
    hyp_durations: Dict[str, float] = {}
    ref_to_hyp: Dict[str, str] = {}
    unmapped_ref: List[str] = []

    for r_spk, dur in sorted(gt_durations.items()):
        if r_spk in hungarian_map:
            h_spk = hungarian_map[r_spk]
            ref_to_hyp[r_spk] = h_spk
            hyp_durations[h_spk] = round(hyp_durations.get(h_spk, 0.0) + dur, 6)
        else:
            unmapped_ref.append(r_spk)

    if not hyp_durations:
        return {
            "speaker": None,
            "reason": "no_valid_mapped_hypothesis_speaker",
            "all_active_gt": gt_durations,
            "all_active_mapped": {},
            "discarded_speakers": [],
            "is_truncated_multispeaker": False,
            "confidence": 0.0,
        }

    # 3. Single reference speaker is NOT true overlap
    if len(gt_durations) == 1:
        return {
            "speaker": None,
            "reason": "single_reference_speaker_no_overlap",
            "all_active_gt": gt_durations,
            "all_active_mapped": hyp_durations,
            "discarded_speakers": [],
            "is_truncated_multispeaker": False,
            "confidence": 0.0,
        }

    def get_global_cooc(h: str) -> float:
        if cost_matrix is None or ref_spk_list is None or hyp_spk_list is None:
            return 0.0
        if h not in hyp_spk_list:
            return 0.0
        h_idx = hyp_spk_list.index(h)
        tot_cooc = 0.0
        for r_spk in gt_durations:
            if r_spk in ref_spk_list:
                r_idx = ref_spk_list.index(r_spk)
                tot_cooc += max(0.0, -float(cost_matrix[r_idx, h_idx]))
        return tot_cooc

    # 4. If primary_spk is None, rank all mapped speakers
    if primary_spk is None:
        ranked_all = sorted(
            hyp_durations.keys(),
            key=lambda h: (-round(hyp_durations[h], 6), -round(get_global_cooc(h), 6), str(h)),
        )
        if len(ranked_all) < 2:
            return {
                "speaker": None,
                "reason": "unresolved_primary_insufficient_alternates",
                "all_active_gt": gt_durations,
                "all_active_mapped": hyp_durations,
                "discarded_speakers": [],
                "is_truncated_multispeaker": False,
                "confidence": 0.0,
            }
        sec_spk = ranked_all[1]
        discarded = ranked_all[2:]
        return {
            "speaker": sec_spk,
            "reason": "attributed_unresolved_primary_fallback",
            "all_active_gt": gt_durations,
            "all_active_mapped": hyp_durations,
            "discarded_speakers": discarded,
            "is_truncated_multispeaker": len(discarded) > 0,
            "confidence": round(hyp_durations[sec_spk] / (en_sec - st_sec), 4),
        }

    candidates = [h for h in hyp_durations if h != primary_spk]
    if not candidates:
        return {
            "speaker": None,
            "reason": "no_alternate_mapped_speaker",
            "all_active_gt": gt_durations,
            "all_active_mapped": hyp_durations,
            "discarded_speakers": [],
            "is_truncated_multispeaker": False,
            "confidence": 0.0,
        }

    ranked_candidates = sorted(
        candidates,
        key=lambda h: (-round(hyp_durations[h], 6), -round(get_global_cooc(h), 6), str(h)),
    )

    sec_spk = ranked_candidates[0]
    discarded = ranked_candidates[1:]
    is_truncated = (len(ranked_candidates) > 1)

    return {
        "speaker": sec_spk,
        "reason": "attributed_active_gt",
        "all_active_gt": gt_durations,
        "all_active_mapped": hyp_durations,
        "discarded_speakers": discarded,
        "is_truncated_multispeaker": is_truncated,
        "confidence": round(hyp_durations[sec_spk] / (en_sec - st_sec), 4),
    }

# Correction A & B: Local Temporal Context & Contiguous Turns Attribution
# -----------------------------------------------------------------------------

def attribute_m1_corrected(
    frame_spk: List[Optional[str]],
    overlap_interval: Tuple[float, float],
    primary_spk: Optional[str],
    resolution: float = 0.01,
    w_ctx: float = 3.0,
    min_evidence_sec: float = 0.20,
) -> Dict[str, Any]:
    """
    Correction A: M1 Temporal Attribution.
    Primary: evidence duration >= 0.20s descending.
    Secondary: minimum temporal distance from overlap ascending.
    Tertiary: speaker ID ascending.
    """
    st_sec, en_sec = overlap_interval
    if primary_spk is None:
        return {"speaker": None, "reason": "unresolved_primary", "evidence_sec": 0.0, "distance_sec": math.inf}

    n_frames = len(frame_spk)
    mid_sec = (st_sec + en_sec) / 2.0
    ctx_st_idx = max(0, int(math.floor((mid_sec - w_ctx) / resolution)))
    ctx_en_idx = min(n_frames, int(math.ceil((mid_sec + w_ctx) / resolution)))
    min_evidence_frames = int(round(min_evidence_sec / resolution))

    candidate_frames: Dict[str, int] = {}
    candidate_min_dist: Dict[str, float] = {}

    for idx in range(ctx_st_idx, ctx_en_idx):
        spk = frame_spk[idx]
        if spk is not None and spk != primary_spk:
            candidate_frames[spk] = candidate_frames.get(spk, 0) + 1
            frame_t = idx * resolution
            if frame_t < st_sec:
                d = st_sec - (frame_t + resolution)
            elif frame_t >= en_sec:
                d = frame_t - en_sec
            else:
                d = 0.0
            d = round(max(0.0, d), 6)
            if spk not in candidate_min_dist or d < candidate_min_dist[spk]:
                candidate_min_dist[spk] = d

    eligible = [spk for spk, cnt in candidate_frames.items() if cnt >= min_evidence_frames]
    if not eligible:
        return {"speaker": None, "reason": "insufficient_evidence", "evidence_sec": 0.0, "distance_sec": math.inf}

    eligible.sort(key=lambda s: (-candidate_frames[s], candidate_min_dist[s], s))
    chosen = eligible[0]
    return {
        "speaker": chosen,
        "reason": "selected",
        "evidence_sec": candidate_frames[chosen] * resolution,
        "distance_sec": candidate_min_dist[chosen],
    }


def extract_contiguous_turns(
    frame_spk: List[Optional[str]],
    search_start_idx: int,
    search_end_idx: int,
    direction: str,
    resolution: float = 0.01,
) -> List[Dict[str, Any]]:
    """Extracts genuinely contiguous speaker runs without bridging None gaps."""
    runs = []
    curr_spk = None
    curr_st = 0
    curr_en = 0

    indices = range(search_start_idx, search_end_idx - 1, -1) if direction == 'backward' else range(search_start_idx, search_end_idx)

    for i in indices:
        s = frame_spk[i]
        if s is not None:
            if curr_spk is None:
                curr_spk = s
                curr_st = i
                curr_en = i + 1
            elif s == curr_spk:
                if direction == 'backward':
                    curr_st = i
                else:
                    curr_en = i + 1
            else:
                runs.append({
                    "speaker": curr_spk,
                    "start_sec": curr_st * resolution,
                    "end_sec": curr_en * resolution,
                    "duration_sec": (curr_en - curr_st) * resolution,
                })
                curr_spk = s
                curr_st = i
                curr_en = i + 1
        else:
            if curr_spk is not None:
                runs.append({
                    "speaker": curr_spk,
                    "start_sec": curr_st * resolution,
                    "end_sec": curr_en * resolution,
                    "duration_sec": (curr_en - curr_st) * resolution,
                })
                curr_spk = None

    if curr_spk is not None:
        runs.append({
            "speaker": curr_spk,
            "start_sec": curr_st * resolution,
            "end_sec": curr_en * resolution,
            "duration_sec": (curr_en - curr_st) * resolution,
        })
    return runs


def attribute_m3_m4_corrected(
    frame_spk: List[Optional[str]],
    overlap_interval: Tuple[float, float],
    primary_spk: Optional[str],
    method: str = "M3",
    resolution: float = 0.01,
    max_gap_sec: float = 5.0,
    min_turn_dur_sec: float = 0.20,
) -> Dict[str, Any]:
    """Correction B: Contiguous Local Turn Extraction for M3 and M4."""
    st_sec, en_sec = overlap_interval
    if primary_spk is None:
        return {"speaker": None, "reason": "unresolved_primary", "p_cand": None, "n_cand": None}

    n_frames = len(frame_spk)
    st_idx = max(0, int(round(st_sec / resolution)))
    en_idx = min(n_frames, int(round(en_sec / resolution)))
    max_gap_frames = int(round(max_gap_sec / resolution))

    # Preceding turns: search backwards from st_idx - 1
    p_runs = extract_contiguous_turns(
        frame_spk,
        search_start_idx=st_idx - 1,
        search_end_idx=max(0, st_idx - max_gap_frames),
        direction="backward",
        resolution=resolution,
    )

    # Subsequent turns: search forwards from en_idx
    n_runs = extract_contiguous_turns(
        frame_spk,
        search_start_idx=en_idx,
        search_end_idx=min(n_frames, en_idx + max_gap_frames),
        direction="forward",
        resolution=resolution,
    )

    eligible_p = [
        r for r in p_runs
        if r["speaker"] != primary_spk and r["duration_sec"] >= min_turn_dur_sec and (st_sec - r["end_sec"]) <= max_gap_sec
    ]
    p_cand = eligible_p[0]["speaker"] if eligible_p else None

    eligible_n = [
        r for r in n_runs
        if r["speaker"] != primary_spk and r["duration_sec"] >= min_turn_dur_sec and (r["start_sec"] - en_sec) <= max_gap_sec
    ]
    n_cand = eligible_n[0]["speaker"] if eligible_n else None

    if method == "M3":
        if p_cand is not None:
            return {"speaker": p_cand, "reason": "preceding_turn", "p_cand": p_cand, "n_cand": n_cand}
        elif n_cand is not None:
            return {"speaker": n_cand, "reason": "subsequent_turn", "p_cand": p_cand, "n_cand": n_cand}
        else:
            return {"speaker": None, "reason": "no_valid_turn", "p_cand": None, "n_cand": None}

    elif method == "M4":
        if p_cand is not None and n_cand is not None and p_cand == n_cand:
            return {"speaker": p_cand, "reason": "bidirectional_agreement", "p_cand": p_cand, "n_cand": n_cand}
        else:
            return {"speaker": None, "reason": "abstain_disagreement_or_missing", "p_cand": p_cand, "n_cand": n_cand}

    raise ValueError(f"Unknown method {method}")


# -----------------------------------------------------------------------------
# Secondary Attribution Dispatcher
# -----------------------------------------------------------------------------

def attribute_secondary_speakers_v3(
    base_segs: List[Dict[str, Any]],
    ovl_segs: List[Dict[str, Any]],
    attribution_method: str,
    duration_sec: float,
    proto_data: Dict[str, Any],
    gt_segs: Optional[List[Dict[str, Any]]] = None,
    hungarian_map: Optional[Dict[str, str]] = None,
    hungarian_cost_matrix: Optional[np.ndarray] = None,
    hungarian_ref_spks: Optional[List[str]] = None,
    hungarian_hyp_spks: Optional[List[str]] = None,
    resolution: float = 0.01,
) -> Tuple[List[Dict[str, Any]], int, int, float]:
    """
    Applies the verified attribution methods.
    Returns: (secondary_segments, total_detected_overlaps, attributed_overlaps, total_overlap_dur_sec)
    """
    if not ovl_segs:
        return [], 0, 0, 0.0

    n_frames = int(round(duration_sec / resolution)) + 1
    frame_spk: List[Optional[str]] = [None] * n_frames
    for s in base_segs:
        st_i = max(0, int(round(s["start"] / resolution)))
        en_i = min(n_frames, int(round(s["end"] / resolution)))
        for i in range(st_i, en_i):
            frame_spk[i] = s["speaker"]

    all_hyp_speakers = sorted(list(set(s["speaker"] for s in base_segs)))
    if len(all_hyp_speakers) < 2 and not attribution_method.startswith("M5"):
        return [], len(ovl_segs), 0, sum(s["end"] - s["start"] for s in ovl_segs)

    sec_segments: List[Dict[str, Any]] = []
    attributed_count = 0
    total_ovl_dur = sum(s["end"] - s["start"] for s in ovl_segs)

    for ovl in ovl_segs:
        st = ovl["start"]
        en = ovl["end"]
        mid = (st + en) / 2.0
        st_i = max(0, int(round(st / resolution)))
        en_i = min(n_frames, int(round(en / resolution)))

        # Determine primary speaker during interval without global nearest search
        spk_counts: Dict[str, int] = {}
        for i in range(st_i, en_i):
            s = frame_spk[i]
            if s is not None:
                spk_counts[s] = spk_counts.get(s, 0) + 1

        primary_spk: Optional[str] = None
        if spk_counts:
            primary_spk = max(spk_counts.items(), key=lambda x: x[1])[0]

        sec_spk: Optional[str] = None

        if attribution_method == "M0_no_overlap":
            sec_spk = None

        elif attribution_method == "M1_temporal_context":
            res_m1 = attribute_m1_corrected(
                frame_spk=frame_spk,
                overlap_interval=(st, en),
                primary_spk=primary_spk,
                resolution=resolution,
            )
            sec_spk = res_m1["speaker"]

        elif attribution_method == "M2_prototype_concat":
            res_m2 = attribute_m2_pure(
                subsegments=proto_data.get("subsegs", []),
                proto_data=proto_data,
                overlap_interval=(st, en),
                primary_spk=primary_spk,
                method="concat",
            )
            sec_spk = res_m2["speaker"]

        elif attribution_method == "M2_prototype_titanet":
            res_m2 = attribute_m2_pure(
                subsegments=proto_data.get("subsegs", []),
                proto_data=proto_data,
                overlap_interval=(st, en),
                primary_spk=primary_spk,
                method="titanet",
            )
            sec_spk = res_m2["speaker"]

        elif attribution_method == "M2_prototype_wespeaker":
            res_m2 = attribute_m2_pure(
                subsegments=proto_data.get("subsegs", []),
                proto_data=proto_data,
                overlap_interval=(st, en),
                primary_spk=primary_spk,
                method="wespeaker",
            )
            sec_spk = res_m2["speaker"]

        elif attribution_method == "M2_prototype_normfusion":
            res_m2 = attribute_m2_pure(
                subsegments=proto_data.get("subsegs", []),
                proto_data=proto_data,
                overlap_interval=(st, en),
                primary_spk=primary_spk,
                method="normfusion",
            )
            sec_spk = res_m2["speaker"]

        elif attribution_method == "M3_turn_alternation":
            res_m3 = attribute_m3_m4_corrected(
                frame_spk=frame_spk,
                overlap_interval=(st, en),
                primary_spk=primary_spk,
                method="M3",
                resolution=resolution,
            )
            sec_spk = res_m3["speaker"]

        elif attribution_method == "M4_conservative_abstention":
            res_m4 = attribute_m3_m4_corrected(
                frame_spk=frame_spk,
                overlap_interval=(st, en),
                primary_spk=primary_spk,
                method="M4",
                resolution=resolution,
            )
            sec_spk = res_m4["speaker"]

        elif attribution_method in ["M5B_predicted_ovl_oracle_attr", "M5C_full_oracle"]:
            res_oracle = attribute_secondary_oracle_multilabel(
                overlap_interval=(st, en),
                primary_spk=primary_spk,
                gt_segs=gt_segs or [],
                hungarian_map=hungarian_map or {},
                cost_matrix=hungarian_cost_matrix,
                ref_spk_list=hungarian_ref_spks,
                hyp_spk_list=hungarian_hyp_spks,
                resolution=resolution,
            )
            sec_spk = res_oracle["speaker"]
        if sec_spk is not None:
            sec_segments.append({"start": st, "end": en, "speaker": sec_spk})
            attributed_count += 1

    return sec_segments, len(ovl_segs), attributed_count, total_ovl_dur


attribute_secondary_speakers_v2 = attribute_secondary_speakers_v3

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
    """Computes exact frame-level overlap detection metrics."""
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
# Main Rerun Suite
# -----------------------------------------------------------------------------

def run_stage5b_2_corrected_v3_suite(target_configs: Optional[List[str]] = None, seed: int = 42):
    t_start = time.time()
    manifest_p = REPO_ROOT / "research_stages" / "stage1_data_prep" / "manifests" / "recording_manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    # Focused Rerun Matrix:
    # 1. Baselines (O00, O01, O02)
    # 2. Threshold Sweeps (O10-R2, O11-R2, O12-R2, O13-R2, O14-R2)
    # 3. Stream Comparison (O20-R2 vs O21-R2)
    # 4. Attribution Strategies (O30-R2, O31-R2, O31-T-R2, O31-W-R2, O31-NF-R2, O32-R2, O33-R2)
    # 5. Hungarian Oracle Diagnostics (O34-R2, O35-R2)
    # 6. Integrated Pipelines (O40-R2, O41-R2, O42-R2)
    configurations = [
        # Baseline Single-Label (No Overlap)
        ("O00", "Baseline Single-Label (C00 VAD, No Overlap)", "C00", "none", 0.0, "M0_no_overlap"),
        ("O01", "Baseline Single-Label (C50 VAD, No Overlap)", "C50", "none", 0.0, "M0_no_overlap"),
        ("O02", "Baseline Single-Label (C62 VAD, No Overlap)", "C62", "none", 0.0, "M0_no_overlap"),

        # Threshold Sweeps (Mono Stream, M1-R2 with proximity tie-break)
        ("O10-R2", "Overlap Sweep T=0.30 (Mono, M1-R2)", "C00", "mono", 0.30, "M1_temporal_context"),
        ("O11-R2", "Overlap Sweep T=0.40 (Mono, M1-R2)", "C00", "mono", 0.40, "M1_temporal_context"),
        ("O12-R2", "Overlap Sweep T=0.50 (Mono, M1-R2)", "C00", "mono", 0.50, "M1_temporal_context"),
        ("O13-R2", "Overlap Sweep T=0.60 (Mono, M1-R2)", "C00", "mono", 0.60, "M1_temporal_context"),
        ("O14-R2", "Overlap Sweep T=0.70 (Mono, M1-R2)", "C00", "mono", 0.70, "M1_temporal_context"),

        # Stream Comparison (DFN vs Mono at T=0.40, M1-R2)
        ("O20-R2", "Overlap Stream DFN (T=0.40, M1-R2)", "C00", "dfn", 0.40, "M1_temporal_context"),
        ("O21-R2", "Overlap Stream Mono (T=0.40, M1-R2)", "C00", "mono", 0.40, "M1_temporal_context"),

        # Attribution Strategies Comparison (C00, Mono, T=0.40)
        ("O30-R2", "Attribution: M1-R2 Proximity Tie-Break", "C00", "mono", 0.40, "M1_temporal_context"),
        ("O31-R2", "Attribution: M2-R2 Pure Prototype Concat (448d)", "C00", "mono", 0.40, "M2_prototype_concat"),
        ("O31-T-R2", "Attribution: M2-R2 Pure TitaNet (192d)", "C00", "mono", 0.40, "M2_prototype_titanet"),
        ("O31-W-R2", "Attribution: M2-R2 Pure WeSpeaker (256d)", "C00", "mono", 0.40, "M2_prototype_wespeaker"),
        ("O31-NF-R2", "Attribution: M2-R2 Pure Normalized Cosine Avg", "C00", "mono", 0.40, "M2_prototype_normfusion"),
        ("O32-R2", "Attribution: M3-R2 Contiguous Turn Alternation", "C00", "mono", 0.40, "M3_turn_alternation"),
        ("O33-R2", "Attribution: M4-R2 Conservative Contiguous Abstain", "C00", "mono", 0.40, "M4_conservative_abstention"),

        # Hungarian Multi-Label Oracle Diagnostics
        ("O34-R2", "Oracle Diag: M5B-R2 Pred Ovl + Hungarian Multi-Label Attr", "C00", "mono", 0.40, "M5B_predicted_ovl_oracle_attr"),
        ("O35-R2", "Oracle Diag: M5C-R2 Full Oracle Ovl + Hungarian Multi-Label Attr", "C00", "oracle_gt", 0.0, "M5C_full_oracle"),

        # Integrated Upstream Pipelines
        ("O40-R2", "Integrated: C50 Mono VAD + Mono Overlap (M1-R2)", "C50", "mono", 0.40, "M1_temporal_context"),
        ("O41-R2", "Integrated: C62 Consensus VAD + Mono Overlap (M1-R2)", "C62", "mono", 0.40, "M1_temporal_context"),
        ("O42-R2", "Integrated: C62 Consensus VAD + Conservative Overlap (M4-R2)", "C62", "mono", 0.40, "M4_conservative_abstention"),
    ]

    all_configurations = list(configurations)
    if target_configs is not None:
        configurations = [c for c in all_configurations if c[0] in target_configs]
        print(f"Executing minimal rerun matrix ({len(configurations)} configs: {[c[0] for c in configurations]}) across {len(manifest)} recordings...")
    else:
        print(f"Executing full matrix ({len(configurations)} configs) across {len(manifest)} recordings...")

    # Preload recording data
    recording_data = {}
    for entry in manifest:
        rec_id = entry["rec_id"]
        dur = float(entry["duration_sec"])
        dfn_stem = Path(entry["dfn_wav"]).stem
        gt_rttm_p = REPO_ROOT / entry["rttm_uniform"]
        gt_segs = p20_common.rttm_to_segments(gt_rttm_p)
        gt_annot = load_track_aware_annotation(gt_rttm_p)
        aligned_words_p = REPO_ROOT / f"research_stages/stage2_baseline/whisperx/{rec_id}_aligned.json"

        # Extract GT overlap runs
        resolution = 0.01
        n_frames = int(round(dur / resolution)) + 1
        gt_spk_counts = np.zeros(n_frames, dtype=int)
        for s in gt_segs:
            st_i = max(0, int(round(s["start"] / resolution)))
            en_i = min(n_frames, int(round(s["end"] / resolution)))
            gt_spk_counts[st_i:en_i] += 1
        gt_ovl_runs = []
        in_run = False
        s_idx = 0
        for i in range(n_frames):
            if not in_run and gt_spk_counts[i] >= 2:
                in_run = True
                s_idx = i
            elif in_run and gt_spk_counts[i] < 2:
                in_run = False
                gt_ovl_runs.append({"start": round(s_idx * resolution, 3), "end": round((i - 1) * resolution, 3)})
        if in_run:
            gt_ovl_runs.append({"start": round(s_idx * resolution, 3), "end": round((n_frames - 1) * resolution, 3)})

        vad_rttm_paths = {
            "C00": REPO_ROOT / f"research_stages/stage5b/stage5b_1_vad_optimization/predictions/C00_baseline/{dfn_stem}.rttm",
            "C50": REPO_ROOT / f"research_stages/stage5b/stage5b_1_vad_optimization/predictions/C50_stream_mono/{dfn_stem}.rttm",
            "C62": REPO_ROOT / f"research_stages/stage5b/stage5b_1_vad_optimization/predictions/C62_comb_consensus_hyst_gap/{dfn_stem}.rttm",
        }

        ovl_streams = {}
        for s in ["mono", "dfn"]:
            ovl_streams[s] = load_overlap_posteriors(rec_id, s)

        base_segs_map = {k: p20_common.rttm_to_segments(vad_rttm_paths[k]) for k in vad_rttm_paths}
        proto_data = load_dual_encoder_prototypes_pure(base_segs_map["C00"], rec_id, min_prototype_purity=0.80)

        # Multi-label Hungarian mapping
        hungarian_maps = {}
        hungarian_cost_matrices = {}
        hungarian_ref_spks = {}
        hungarian_hyp_spks = {}
        for k in base_segs_map:
            m, _, _, c_mat, r_spks, h_spks = compute_multilabel_hungarian_mapping(gt_segs, base_segs_map[k], dur)
            hungarian_maps[k] = m
            hungarian_cost_matrices[k] = c_mat
            hungarian_ref_spks[k] = r_spks
            hungarian_hyp_spks[k] = h_spks

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
            "hungarian_cost_matrices": hungarian_cost_matrices,
            "hungarian_ref_spks": hungarian_ref_spks,
            "hungarian_hyp_spks": hungarian_hyp_spks,
        }
    all_per_rec_rows: List[Dict[str, Any]] = []
    metric_errors: List[Dict[str, Any]] = []

    print(f"Beginning Stage 5B.2 Corrective v2 Execution ({len(configurations)} configs x 8 recordings = {len(configurations)*8} runs)...")

    for c_id, c_name, vad_bb, ovl_str, thresh, attr_method in configurations:
        for entry in manifest:
            rec_id = entry["rec_id"]
            rdata = recording_data[rec_id]
            dur = rdata["duration"]
            dfn_stem = rdata["dfn_stem"]
            gt_annot = rdata["gt_annot"]
            gt_segs = rdata["gt_segs"]
            base_segs = rdata["base_segs_map"][vad_bb]

            # 1. Overlap Binarization using frame-bin support
            if ovl_str == "oracle_gt":
                ovl_segs = rdata["gt_ovl_segs"]
            elif ovl_str in ["mono", "dfn"]:
                frame_times, p_ovl = rdata["ovl_streams"][ovl_str]
                ovl_segs = binarize_overlap_frame_bin_support(frame_times, p_ovl, dur, threshold=thresh)
            else:
                ovl_segs = []

            # 2. Secondary Speaker Attribution
            hungarian_map = rdata["hungarian_maps"][vad_bb]
            c_mat = rdata["hungarian_cost_matrices"][vad_bb]
            r_spks = rdata["hungarian_ref_spks"][vad_bb]
            h_spks = rdata["hungarian_hyp_spks"][vad_bb]
            sec_segs, tot_ovl, attr_ovl, tot_ovl_dur = attribute_secondary_speakers_v3(
                base_segs=base_segs,
                ovl_segs=ovl_segs,
                attribution_method=attr_method,
                duration_sec=dur,
                proto_data=rdata["proto_data"],
                gt_segs=gt_segs,
                hungarian_map=hungarian_map,
                hungarian_cost_matrix=c_mat,
                hungarian_ref_spks=r_spks,
                hungarian_hyp_spks=h_spks,
            )
            # 3. Multi-label RTTM Generation
            out_rttm_p = PRED_DIR / c_id / f"{dfn_stem}.rttm"
            generate_multilabel_rttm(base_segs, sec_segs, out_rttm_p, dfn_stem)
            hyp_annot = load_track_aware_annotation(out_rttm_p)

            # 4. Strict Diarization Metrics
            try:
                diar_metrics = safe_diarization_metrics(gt_annot, hyp_annot)
            except Exception as e:
                metric_errors.append({"rec_id": rec_id, "config_id": c_id, "metric": "diarization", "error": str(e)})
                raise e

            # 5. Overlap Detection Metrics
            ovl_metrics = evaluate_overlap_detection_frame_level(gt_segs, ovl_segs, dur)

            # 6. Word Metrics (cpWER & saWER)
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
                    metric_errors.append({"rec_id": rec_id, "config_id": c_id, "metric": "cpwer", "error": str(e)})

            sec_dur_added = round(sum(s["end"] - s["start"] for s in sec_segs), 3)
            # Event-weighted abstention rate vs time-weighted abstention rate
            event_abstain_rate = (1.0 - (attr_ovl / tot_ovl)) if tot_ovl > 0 else 0.0
            time_abstain_rate = (1.0 - (sec_dur_added / tot_ovl_dur)) if tot_ovl_dur > 0 else 0.0

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
                "event_abstention_rate": round(event_abstain_rate, 4),
                "time_abstention_rate": round(time_abstain_rate, 4),
                "secondary_segments_added": len(sec_segs),
                "secondary_duration_added_sec": sec_dur_added,
                "cpwer": cpwer,
                "sawer": sawer,
            }
            all_per_rec_rows.append(row)

    # If target_configs was specified, merge unchanged configurations from V2
    executed_c_ids = set(r["config_id"] for r in all_per_rec_rows)
    all_target_c_ids = [c[0] for c in all_configurations]
    missing_c_ids = [c_id for c_id in all_target_c_ids if c_id not in executed_c_ids]
    if missing_c_ids:
        v2_per_rec_p = REPO_ROOT / "research_stages" / "stage5b" / "stage5b_2_overlap_optimization" / "audit_corrections_v2" / "STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv"
        if v2_per_rec_p.exists():
            print(f"Merging {len(missing_c_ids)} unchanged configurations from frozen V2 results: {missing_c_ids}")
            with v2_per_rec_p.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r["config_id"] in missing_c_ids:
                        # Convert numeric fields
                        for num_k in ["duration_sec", "der_c0", "der_c25", "diar_miss", "diar_fa", "diar_confusion",
                                      "total_reference_sec", "miss_sec", "fa_sec", "conf_sec", "gt_overlap_sec",
                                      "pred_overlap_sec", "tp_overlap_sec", "fp_overlap_sec", "fn_overlap_sec",
                                      "overlap_precision", "overlap_recall", "overlap_f1", "event_abstention_rate",
                                      "time_abstention_rate", "secondary_duration_added_sec"]:
                            if num_k in r and r[num_k] != "" and r[num_k] is not None:
                                r[num_k] = float(r[num_k])
                        for int_k in ["detected_overlap_count", "attributed_overlap_count", "secondary_segments_added"]:
                            if int_k in r and r[int_k] != "" and r[int_k] is not None:
                                r[int_k] = int(r[int_k])
                        for opt_k in ["jer", "cpwer", "sawer"]:
                            if opt_k in r and r[opt_k] != "" and r[opt_k] is not None:
                                r[opt_k] = float(r[opt_k])
                            else:
                                r[opt_k] = None
                        all_per_rec_rows.append(r)

    # Sort all_per_rec_rows by configuration order and rec_id
    config_order = {c[0]: idx for idx, c in enumerate(all_configurations)}
    all_per_rec_rows.sort(key=lambda r: (config_order.get(r["config_id"], 999), r["rec_id"]))

    # Iterate over all_configurations for summary metrics
    configurations = all_configurations

    # Save per-recording CSV
    per_rec_csv = METRICS_DIR / "STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv"
    per_rec_root_csv = AUDIT_V3_DIR / "STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv"
    for target_csv in [per_rec_csv, per_rec_root_csv]:
        with target_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_per_rec_rows[0].keys()))
            w.writeheader()
            w.writerows(all_per_rec_rows)
    # Summary Table Aggregations (Macro & Pooled)
    # -------------------------------------------------------------------------
    summary_rows = []
    overlap_metric_rows = []
    word_metric_rows = []

    for c_id, c_name, vad_bb, ovl_str, thresh, attr_method in configurations:
        sub = [r for r in all_per_rec_rows if r["config_id"] == c_id]
        cpwers = [r["cpwer"] for r in sub if r["cpwer"] is not None]
        sawers = [r["sawer"] for r in sub if r["sawer"] is not None]
        jers = [r["jer"] for r in sub if r["jer"] is not None]

        r02_row = next(r for r in sub if "R02" in r["rec_id"])
        r04_row = next(r for r in sub if "R04" in r["rec_id"])

        tot_ref_sec = sum(r["total_reference_sec"] for r in sub)
        tot_miss_sec = sum(r["miss_sec"] for r in sub)
        tot_fa_sec = sum(r["fa_sec"] for r in sub)
        tot_conf_sec = sum(r["conf_sec"] for r in sub)
        pooled_der_c0 = (tot_miss_sec + tot_fa_sec + tot_conf_sec) / tot_ref_sec
        pooled_miss = tot_miss_sec / tot_ref_sec
        pooled_fa = tot_fa_sec / tot_ref_sec
        pooled_conf = tot_conf_sec / tot_ref_sec

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
            "macro_jer": float(np.mean(jers)) if jers else None,
            "jer_valid_count": len(jers),
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
            "cpwer_valid_count": len(cpwers),
            "macro_sawer": float(np.mean(sawers)) if sawers else None,
            # Attribution stats
            "macro_event_abstention_rate": float(np.mean([r["event_abstention_rate"] for r in sub])),
            "macro_time_abstention_rate": float(np.mean([r["time_abstention_rate"] for r in sub])),
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
    summary_root_csv = AUDIT_V3_DIR / "STAGE5B_2_CORRECTED_RESULTS.csv"
    for target_csv in [summary_csv, summary_root_csv]:
        with target_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    ovl_csv = METRICS_DIR / "STAGE5B_2_CORRECTED_OVERLAP_METRICS.csv"
    ovl_root_csv = AUDIT_V3_DIR / "STAGE5B_2_CORRECTED_OVERLAP_METRICS.csv"
    for target_csv in [ovl_csv, ovl_root_csv]:
        with target_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(overlap_metric_rows[0].keys()))
            w.writeheader()
            w.writerows(overlap_metric_rows)

    word_csv = METRICS_DIR / "STAGE5B_2_CORRECTED_WORD_METRICS.csv"
    word_root_csv = AUDIT_V3_DIR / "STAGE5B_2_CORRECTED_WORD_METRICS.csv"
    for target_csv in [word_csv, word_root_csv]:
        with target_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(word_metric_rows[0].keys()))
            w.writeheader()
            w.writerows(word_metric_rows)
    # Statistical Analysis (Cluster Bootstrap N=8, Fixed Seed 42, 5000 resamples)
    # -------------------------------------------------------------------------
    print("\n--- STATISTICAL SIGNIFICANCE TESTS (Recording Cluster Unit, N=8) ---")
    base_rows = {r["rec_id"]: r for r in all_per_rec_rows if r["config_id"] == "O00"}
    rec_keys = sorted(list(base_rows.keys()))
    np.random.seed(42)

    comparisons = [
        ("O11-R2", "O00", "M1-R2 vs O00 Baseline"),
        ("O31-R2", "O00", "M2-R2 Pure Concat vs O00 Baseline"),
        ("O32-R2", "O00", "M3-R2 Contiguous Turn Alternation vs O00 Baseline"),
        ("O33-R2", "O00", "M4-R2 Conservative Abstain vs O00 Baseline"),
        ("O42-R2", "O02", "O42-R2 Integrated vs O02 Baseline"),
    ]

    for cand_id, ref_id, comp_name in comparisons:
        cand_map = {r["rec_id"]: r for r in all_per_rec_rows if r["config_id"] == cand_id}
        ref_map = {r["rec_id"]: r for r in all_per_rec_rows if r["config_id"] == ref_id}

        deltas = [cand_map[k]["der_c0"] - ref_map[k]["der_c0"] for k in rec_keys]
        macro_delta = float(np.mean(deltas))

        # 5,000 cluster bootstrap
        boot_means = []
        for _ in range(5000):
            sample_idx = np.random.choice(len(deltas), size=len(deltas), replace=True)
            boot_means.append(np.mean([deltas[i] for i in sample_idx]))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        try:
            w_res = wilcoxon(deltas)
            w_stat, p_val = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, p_val = 0.0, 1.0

        r04_delta = cand_map["R04_piers_robinson"]["der_c0"] - ref_map["R04_piers_robinson"]["der_c0"]
        print(f"{comp_name:40s} | Macro Delta: {macro_delta*100:+.2f}% | 95% CI: [{ci_lower*100:+.2f}%, {ci_upper*100:+.2f}%] | W={w_stat:.1f}, p={p_val:.4f} | R04 Delta: {r04_delta*100:+.2f}%")

    # Generate Analytical Plots
    generate_corrected_v2_plots(all_per_rec_rows, summary_rows)

    print(f"\nStage 5B.2 Corrective v2 Suite completed in {time.time()-t_start:.2f}s!")


# -----------------------------------------------------------------------------
# Diagnostic Plotting Suite
# -----------------------------------------------------------------------------

def generate_corrected_v2_plots(all_rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]]):
    sel_configs = ["O00", "O30-R2", "O31-R2", "O31-T-R2", "O31-W-R2", "O31-NF-R2", "O32-R2", "O33-R2", "O34-R2", "O35-R2", "O42-R2"]
    sub_summary = [s for s in summary_rows if s["config_id"] in sel_configs]

    # Plot 1: Macro DER Comparison
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
    ax.set_title("Stage 5B.2 Corrected v2 Macro DER Across Overlap & Attribution Strategies")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "macro_der_comparison.png", dpi=150)
    plt.close()

    # Plot 2: Per-Recording DER Delta relative to O00
    base_map = {r["rec_id"]: r["der_c0"] for r in all_rows if r["config_id"] == "O00"}
    recs = sorted(list(base_map.keys()))
    comp_configs = ["O30-R2", "O31-R2", "O32-R2", "O33-R2", "O42-R2"]

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
    ax.set_title("Per-Recording DER Delta Across Attribution Strategies (v2)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_recording_der_delta.png", dpi=150)
    plt.close()

    # Plot 3: R02 and R04 Threshold Sensitivity
    t_configs = ["O00", "O10-R2", "O11-R2", "O12-R2", "O13-R2", "O14-R2"]
    thresholds = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70]
    r04_sub = [r for r in all_rows if r["config_id"] in t_configs and "R04" in r["rec_id"]]
    r02_sub = [r for r in all_rows if r["config_id"] in t_configs and "R02" in r["rec_id"]]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(thresholds, [r["der_c0"]*100 for r in r04_sub], marker="o", label="DER (c=0)")
    ax[0].plot(thresholds, [r["diar_miss"]*100 for r in r04_sub], marker="s", linestyle="--", label="Miss Rate")
    ax[0].plot(thresholds, [r["diar_fa"]*100 for r in r04_sub], marker="^", linestyle=":", label="False Alarm Rate")
    ax[0].set_title("R04 (Piers Robinson) vs Overlap Threshold (v2)")
    ax[0].set_xlabel("Overlap Threshold (0.0 = Baseline)")
    ax[0].set_ylabel("Error Rate (%)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(thresholds, [r["der_c0"]*100 for r in r02_sub], marker="o", color="orange", label="DER (c=0)")
    ax[1].plot(thresholds, [r["diar_miss"]*100 for r in r02_sub], marker="s", color="red", linestyle="--", label="Miss Rate")
    ax[1].plot(thresholds, [r["diar_fa"]*100 for r in r02_sub], marker="^", color="green", linestyle=":", label="False Alarm Rate")
    ax[1].set_title("R02 (Dennis Skinner) vs Overlap Threshold (v2)")
    ax[1].set_xlabel("Overlap Threshold (0.0 = Baseline)")
    ax[1].set_ylabel("Error Rate (%)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overlap_threshold_sensitivity_r02_r04_corrected.png", dpi=150)
    plt.close()

    # Plot 4: Overlap Precision-Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in summary_rows:
        ovl_p = s.get("pooled_ovl_precision") or 0.0
        ovl_r = s.get("pooled_ovl_recall") or 0.0
        if s["ovl_stream"] != "none":
            ax.scatter(ovl_r * 100, ovl_p * 100, s=80, label=s["config_id"])
    ax.set_xlabel("Pooled Overlap Recall (%)")
    ax.set_ylabel("Pooled Overlap Precision (%)")
    ax.set_title("Pooled Overlap Detection Precision vs Recall (v2)")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overlap_precision_recall_comparison.png", dpi=150)
    plt.close()

    # Plot 5: Miss / FA / Confusion Decomposition
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

    # Plot 6: cpWER comparison
    cpwer_sub = [s for s in sub_summary if s["macro_cpwer"] is not None]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(cpwer_sub))
    ax.bar(x, [s["macro_cpwer"] * 100 for s in cpwer_sub], width=0.45, color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels([s["config_id"] for s in cpwer_sub], rotation=30, ha="right")
    ax.set_ylabel("Macro cpWER (%)")
    ax.set_title("Downstream Conversational cpWER Across Strategies (v2)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "macro_cpwer_comparison.png", dpi=150)
    plt.close()

    # Plot 7: Attribution Coverage and Abstention
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(sub_summary))
    w = 0.35
    abstains = [s["macro_event_abstention_rate"] * 100 for s in sub_summary]
    durs = [s["total_secondary_sec_added"] for s in sub_summary]
    ax.bar(x - w/2, abstains, w, label="Event Abstention Rate (%)", color="#7f7f7f")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, durs, w, label="Total Secondary Sec Added (s)", color="#17becf")
    ax.set_xticks(x)
    ax.set_xticklabels([s["config_id"] for s in sub_summary], rotation=30, ha="right")
    ax.set_ylabel("Event Abstention Rate (%)")
    ax2.set_ylabel("Total Secondary Duration Added (s)")
    ax.set_title("Attribution Abstention Rate vs Secondary Duration Added (v2)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "attribution_coverage_and_abstention.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 5B.2 V3 Corrective Audit Runner")
    parser.add_argument("--configs", type=str, default="minimal",
                        help="Comma-separated config IDs, 'minimal' (oracle + matched baselines), or 'all'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    args = parser.parse_args()

    if args.configs == "all":
        target = None
    elif args.configs == "minimal":
        # Minimal rerun set: oracle configurations affected by correction + matched comparisons
        target = ["O00", "O02", "O11-R2", "O34-R2", "O35-R2", "O42-R2"]
    else:
        target = [c.strip() for c in args.configs.split(",")]

    run_stage5b_2_corrected_v3_suite(target_configs=target, seed=args.seed)
