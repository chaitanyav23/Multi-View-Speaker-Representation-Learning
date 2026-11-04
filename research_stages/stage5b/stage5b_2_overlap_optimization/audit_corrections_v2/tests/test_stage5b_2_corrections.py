#!/usr/bin/env python3
"""
Unit Test Suite for Stage 5B.2 Corrective Patch v2.
Validates all six corrected modules:
- Test A: M1 Temporal Context Attribution (evidence, distance tie-break, deterministic ties)
- Test B: M3/M4 Contiguous Local Turn Extraction (boundary search, gap termination, min duration)
- Test C: M2 Embedding Purity and Validity (overlap fraction, prototype purity, abstention)
- Test D: Overlap Timestamp Alignment (frame bin support, boundary clipping, half-open intervals)
- Test E: Multi-Label Hungarian Speaker Mapping (simultaneous speakers, order independence)
- Test F: Strict Metric Error Handling (no zero fallbacks, explicit None, denominator validation)
"""

import math
import unittest
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pyannote.core import Annotation, Segment


# =============================================================================
# MODULE IMPLEMENTATIONS UNDER TEST
# =============================================================================

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
    Primary: valid alternate evidence duration in context window >= min_evidence_sec (maximize duration).
    Secondary: minimum temporal distance from overlap interval (minimize distance).
    Tertiary: deterministic speaker ID (lexicographical order).
    """
    st_sec, en_sec = overlap_interval
    if primary_spk is None:
        return {"speaker": None, "reason": "unresolved_primary", "evidence_sec": 0.0, "distance_sec": math.inf}

    n_frames = len(frame_spk)
    mid_sec = (st_sec + en_sec) / 2.0
    ctx_st_idx = max(0, int(math.floor((mid_sec - w_ctx) / resolution)))
    ctx_en_idx = min(n_frames, int(math.ceil((mid_sec + w_ctx) / resolution)))

    min_evidence_frames = int(round(min_evidence_sec / resolution))

    # Tally candidate frames and calculate minimum distance to overlap interval [st_sec, en_sec)
    candidate_frames: Dict[str, int] = {}
    candidate_min_dist: Dict[str, float] = {}

    for idx in range(ctx_st_idx, ctx_en_idx):
        spk = frame_spk[idx]
        if spk is not None and spk != primary_spk:
            candidate_frames[spk] = candidate_frames.get(spk, 0) + 1
            frame_t = idx * resolution
            # Distance to interval [st_sec, en_sec)
            if frame_t < st_sec:
                d = st_sec - (frame_t + resolution)
            elif frame_t >= en_sec:
                d = frame_t - en_sec
            else:
                d = 0.0
            d = max(0.0, d)
            d = round(d, 6)
            if spk not in candidate_min_dist or d < candidate_min_dist[spk]:
                candidate_min_dist[spk] = d

    # Filter by minimum evidence
    eligible = [
        spk for spk, cnt in candidate_frames.items()
        if cnt >= min_evidence_frames
    ]

    if not eligible:
        return {"speaker": None, "reason": "insufficient_evidence", "evidence_sec": 0.0, "distance_sec": math.inf}

    # Deterministic sorting rule:
    # 1. Evidence duration descending (-cnt)
    # 2. Distance ascending (+min_dist)
    # 3. Speaker ID ascending (spk)
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
    direction: str,  # 'backward' or 'forward'
    resolution: float = 0.01,
) -> List[Dict[str, Any]]:
    """Extracts genuinely contiguous speaker runs without bridging gaps."""
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
                # Different speaker terminates run
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
            # None terminates run (no gap bridging!)
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
    method: str = "M3",  # 'M3' or 'M4'
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

    # Preceding turns: search backwards from st_idx - 1 down to st_idx - max_gap_frames
    p_runs = extract_contiguous_turns(
        frame_spk,
        search_start_idx=st_idx - 1,
        search_end_idx=max(0, st_idx - max_gap_frames),
        direction="backward",
        resolution=resolution,
    )

    # Subsequent turns: search forwards from en_idx up to en_idx + max_gap_frames
    n_runs = extract_contiguous_turns(
        frame_spk,
        search_start_idx=en_idx,
        search_end_idx=min(n_frames, en_idx + max_gap_frames),
        direction="forward",
        resolution=resolution,
    )

    # Filter eligible alternate turns
    eligible_p = [
        r for r in p_runs
        if r["speaker"] != primary_spk and r["duration_sec"] >= min_turn_dur_sec and (st_sec - r["end_sec"]) <= max_gap_sec
    ]
    # Nearest preceding turn is the first in p_runs that meets criteria
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


def evaluate_m2_purity_and_attribute(
    subsegments: List[Dict[str, Any]],
    embeddings: np.ndarray,  # (N, D)
    prototypes: Dict[str, np.ndarray],
    overlap_interval: Tuple[float, float],
    primary_spk: Optional[str],
    min_overlap_purity: float = 0.40,
) -> Dict[str, Any]:
    """Correction C: M2 Embedding Purity and Validity."""
    if not prototypes:
        return {"speaker": None, "reason": "empty_prototypes", "retained_subsegs": 0}
    if primary_spk is None:
        return {"speaker": None, "reason": "unresolved_primary", "retained_subsegs": 0}

    st_sec, en_sec = overlap_interval
    ovl_dur = en_sec - st_sec

    # Evaluate overlap fraction of each subsegment
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
    embs_selected = embeddings[valid_indices]
    e_ovl = np.sum(embs_selected * weights_arr[:, None], axis=0)
    norm = np.linalg.norm(e_ovl)
    if norm < 1e-8:
        return {"speaker": None, "reason": "zero_norm_embedding", "retained_subsegs": 0}
    e_ovl = e_ovl / norm

    # Evaluate cosine similarity against candidate prototypes
    alt_spks = sorted([s for s in prototypes if s != primary_spk])
    if not alt_spks:
        return {"speaker": None, "reason": "no_alternate_prototypes", "retained_subsegs": len(valid_indices)}

    sims = {s: float(np.dot(e_ovl, prototypes[s])) for s in alt_spks}
    # Deterministic tie-break
    best_spk = max(alt_spks, key=lambda s: (sims[s], -alt_spks.index(s)))
    return {
        "speaker": best_spk,
        "reason": "selected_by_cosine_affinity",
        "retained_subsegs": len(valid_indices),
        "cosine_similarity": sims[best_spk],
    }


def binarize_overlap_frame_bin_support(
    frame_times: np.ndarray,
    p_ovl: np.ndarray,
    duration_sec: float,
    threshold: float = 0.40,
    step_sec: float = 0.016875,
    min_duration_on: float = 0.10,
    min_duration_off: float = 0.15,
) -> List[Dict[str, float]]:
    """Correction D: Frame-bin temporal support binarization [start, end)."""
    n_frames = len(p_ovl)
    is_active = p_ovl >= threshold

    # 1. Contiguous active frame runs
    runs = []
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

    # 2. Map runs to half-open physical frame-bin intervals [st, en)
    raw_segs = []
    for s_i, e_i in runs:
        # Each frame i is centered at frame_times[i] with bin width step_sec
        st = max(0.0, float(frame_times[s_i] - step_sec / 2.0))
        en = min(duration_sec, float(frame_times[e_i] + step_sec / 2.0))
        raw_segs.append({"start": st, "end": en})

    # 3. Gap merging (fill inactive gaps shorter than min_duration_off)
    merged = [raw_segs[0]]
    for seg in raw_segs[1:]:
        gap = seg["start"] - merged[-1]["end"]
        if gap < min_duration_off:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)

    # 4. Prune bursts shorter than min_duration_on
    filtered = []
    for s in merged:
        if (s["end"] - s["start"]) >= min_duration_on:
            filtered.append({"start": round(s["start"], 3), "end": round(s["end"], 3)})

    return filtered


def compute_multilabel_hungarian_mapping(
    ref_segs: List[Dict[str, Any]],
    hyp_segs: List[Dict[str, Any]],
    duration_sec: float,
    resolution: float = 0.01,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """Correction E: Multi-label co-occurrence Hungarian matching."""
    from scipy.optimize import linear_sum_assignment

    ref_spks = sorted(list(set(s["speaker"] for s in ref_segs)))
    hyp_spks = sorted(list(set(s["speaker"] for s in hyp_segs)))

    if not ref_spks or not hyp_spks:
        return {}, ref_spks, hyp_spks

    n_frames = int(math.ceil(duration_sec / resolution)) + 1
    ref_active = [set() for _ in range(n_frames)]
    hyp_active = [set() for _ in range(n_frames)]

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

    # Build co-occurrence matrix preserving all simultaneously active speakers
    cost_matrix = np.zeros((len(ref_spks), len(hyp_spks)))
    for i in range(n_frames):
        r_set = ref_active[i]
        h_set = hyp_active[i]
        if r_set and h_set:
            for r in r_set:
                r_idx = ref_spks.index(r)
                for h in h_set:
                    h_idx = hyp_spks.index(h)
                    cost_matrix[r_idx, h_idx] -= 1  # negative for maximization

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
    return mapping, unmatched_refs, unmatched_hyps


def safe_diarization_metrics(
    ref_annot: Annotation,
    hyp_annot: Annotation,
) -> Dict[str, Any]:
    """Correction F: Strict metric computation without zero fallbacks."""
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    ref_chart = ref_annot.get_timeline().duration()
    if ref_chart <= 0.0:
        raise ValueError("Empty reference annotation: total speech duration is zero.")

    der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    details = der_metric(ref_annot, hyp_annot, detailed=True)

    total_ref = details.get("total")
    if total_ref is None or total_ref <= 0.0:
        raise ValueError(f"DiarizationErrorRate failed to return positive reference duration: {total_ref}")

    miss_sec = details.get("missed detection")
    fa_sec = details.get("false alarm")
    conf_sec = details.get("confusion")
    der_val = details.get("diarization error rate")

    if any(x is None for x in [miss_sec, fa_sec, conf_sec, der_val]):
        raise KeyError(f"Missing required DiarizationErrorRate detail key in {details}")

    # Explicit JER computation
    jer_val = None
    jer_error = None
    try:
        jer_val = float(JaccardErrorRate()(ref_annot, hyp_annot))
    except Exception as e:
        jer_val = None
        jer_error = str(e)

    return {
        "der_c0": float(der_val),
        "miss": float(miss_sec) / total_ref,
        "false_alarm": float(fa_sec) / total_ref,
        "confusion": float(conf_sec) / total_ref,
        "jer": jer_val,
        "jer_error": jer_error,
        "total_ref_sec": float(total_ref),
        "status": "SUCCESS",
    }


# =============================================================================
# UNIT TEST CASES
# =============================================================================

class TestStage5B2Corrections(unittest.TestCase):

    # -------------------------------------------------------------------------
    # Correction A Tests: M1 Temporal Attribution
    # -------------------------------------------------------------------------
    def test_m1_equal_evidence_different_distances(self):
        # Spk_A and Spk_B both have 30 frames (0.30s evidence)
        # Spk_A is at dist 0.50s, Spk_B is at dist 0.10s -> Spk_B must win!
        frame_spk = [None] * 500
        # Overlap interval: [2.00, 2.50)
        # Primary speaker: P
        for i in range(200, 250):
            frame_spk[i] = "P"
        # Spk_A: frames [120, 150) -> distance: 2.00 - 1.50 = 0.50s
        for i in range(120, 150):
            frame_spk[i] = "Spk_A"
        # Spk_B: frames [160, 190) -> distance: 2.00 - 1.90 = 0.10s
        for i in range(160, 190):
            frame_spk[i] = "Spk_B"

        res = attribute_m1_corrected(frame_spk, (2.00, 2.50), primary_spk="P")
        self.assertEqual(res["speaker"], "Spk_B")
        self.assertAlmostEqual(res["distance_sec"], 0.10, places=2)

    def test_m1_nearby_insufficient_evidence(self):
        # Spk_Near is very close (0.05s) but only 10 frames (0.10s < 0.20s threshold)
        # Spk_Valid is 1.0s away but has 25 frames (0.25s >= 0.20s) -> Spk_Valid must win!
        frame_spk = [None] * 500
        for i in range(200, 250): frame_spk[i] = "P"
        for i in range(185, 195): frame_spk[i] = "Spk_Near" # 10 frames = 0.10s
        for i in range(50, 75): frame_spk[i] = "Spk_Valid" # 25 frames = 0.25s

        res = attribute_m1_corrected(frame_spk, (2.00, 2.50), primary_spk="P")
        self.assertEqual(res["speaker"], "Spk_Valid")

    def test_m1_distant_speaker_greater_activity(self):
        # Spk_Major has 50 frames (0.50s) at 1.0s dist.
        # Spk_Minor has 25 frames (0.25s) at 0.2s dist.
        # Primary criterion is evidence duration -> Spk_Major must win!
        frame_spk = [None] * 500
        for i in range(200, 250): frame_spk[i] = "P"
        for i in range(50, 100): frame_spk[i] = "Spk_Major"
        for i in range(155, 180): frame_spk[i] = "Spk_Minor"

        res = attribute_m1_corrected(frame_spk, (2.00, 2.50), primary_spk="P")
        self.assertEqual(res["speaker"], "Spk_Major")

    def test_m1_no_valid_alternate(self):
        # Only primary speaker active
        frame_spk = ["P"] * 500
        res = attribute_m1_corrected(frame_spk, (2.00, 2.50), primary_spk="P")
        self.assertIsNone(res["speaker"])
        self.assertEqual(res["reason"], "insufficient_evidence")

    def test_m1_deterministic_tie(self):
        # Spk_B and Spk_A have identical evidence and identical distance -> Spk_A wins deterministically
        frame_spk = [None] * 500
        for i in range(200, 250): frame_spk[i] = "P"
        for i in range(150, 180): frame_spk[i] = "Spk_B" # 30 frames, dist 0.20s
        for i in range(270, 300): frame_spk[i] = "Spk_A" # 30 frames, dist 0.20s

        res = attribute_m1_corrected(frame_spk, (2.00, 2.50), primary_spk="P")
        self.assertEqual(res["speaker"], "Spk_A")

    # -------------------------------------------------------------------------
    # Correction B Tests: M3/M4 Contiguous Local Turn Extraction
    # -------------------------------------------------------------------------
    def test_m3_contiguous_alternate_turns(self):
        frame_spk = [None] * 500
        # Preceding turn Spk_A: [1.50, 1.90) = 40 frames = 0.40s
        for i in range(150, 190): frame_spk[i] = "Spk_A"
        # Overlap: [2.00, 2.50) with primary P
        for i in range(200, 250): frame_spk[i] = "P"
        # Subsequent turn Spk_B: [2.60, 3.00) = 40 frames = 0.40s
        for i in range(260, 300): frame_spk[i] = "Spk_B"

        res_m3 = attribute_m3_m4_corrected(frame_spk, (2.00, 2.50), primary_spk="P", method="M3")
        self.assertEqual(res_m3["speaker"], "Spk_A") # M3 prefers nearest preceding turn

    def test_m3_none_gap_terminates_turn(self):
        # Two Spk_A bursts of 0.15s separated by None gap of 0.10s.
        # Neither burst alone reaches 0.20s. They must NOT be bridged!
        frame_spk = [None] * 500
        for i in range(140, 155): frame_spk[i] = "Spk_A" # 15 frames
        # [155, 165) is None
        for i in range(165, 180): frame_spk[i] = "Spk_A" # 15 frames
        for i in range(200, 250): frame_spk[i] = "P"

        res_m3 = attribute_m3_m4_corrected(frame_spk, (2.00, 2.50), primary_spk="P", method="M3")
        self.assertIsNone(res_m3["speaker"]) # Must abstain because gap prevented bridging

    def test_m3_turn_outside_max_gap(self):
        # Spk_A is 6.0s before overlap start (max_gap is 5.0s) -> must abstain!
        frame_spk = [None] * 1200
        for i in range(100, 150): frame_spk[i] = "Spk_A" # 1.0s to 1.5s
        for i in range(800, 850): frame_spk[i] = "P"     # 8.0s to 8.5s (gap is 6.5s)

        res_m3 = attribute_m3_m4_corrected(frame_spk, (8.00, 8.50), primary_spk="P", method="M3")
        self.assertIsNone(res_m3["speaker"])

    def test_m3_unresolved_primary(self):
        # If primary speaker is None, must mark unresolved and abstain
        frame_spk = [None] * 500
        for i in range(150, 190): frame_spk[i] = "Spk_A"
        res_m3 = attribute_m3_m4_corrected(frame_spk, (2.00, 2.50), primary_spk=None, method="M3")
        self.assertIsNone(res_m3["speaker"])
        self.assertEqual(res_m3["reason"], "unresolved_primary")

    def test_m4_bidirectional_agreement_and_disagreement(self):
        frame_spk = [None] * 500
        for i in range(150, 190): frame_spk[i] = "Spk_A"
        for i in range(200, 250): frame_spk[i] = "P"
        for i in range(260, 300): frame_spk[i] = "Spk_A" # Agreement!

        res_agree = attribute_m3_m4_corrected(frame_spk, (2.00, 2.50), primary_spk="P", method="M4")
        self.assertEqual(res_agree["speaker"], "Spk_A")

        # Now change subsequent turn to Spk_B -> Disagreement!
        for i in range(260, 300): frame_spk[i] = "Spk_B"
        res_disagree = attribute_m3_m4_corrected(frame_spk, (2.00, 2.50), primary_spk="P", method="M4")
        self.assertIsNone(res_disagree["speaker"])
        self.assertEqual(res_disagree["reason"], "abstain_disagreement_or_missing")

    # -------------------------------------------------------------------------
    # Correction C Tests: M2 Embedding Purity and Validity
    # -------------------------------------------------------------------------
    def test_m2_embedding_purity_filtering(self):
        # Subseg 0: [0.0, 1.5], overlap is [1.0, 2.0] -> inter = 0.5s -> frac = 0.5/1.5 = 33.3% < 40% (rejected)
        # Subseg 1: [0.8, 2.3], overlap is [1.0, 2.0] -> inter = 1.0s -> frac = 1.0/1.5 = 66.7% >= 40% (retained)
        subsegs = [
            {"start": 0.0, "end": 1.5, "duration": 1.5},
            {"start": 0.8, "end": 2.3, "duration": 1.5},
        ]
        embs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        protos = {
            "Spk_A": np.array([0.0, 1.0]),
            "Spk_B": np.array([1.0, 0.0]),
        }
        res = evaluate_m2_purity_and_attribute(
            subsegs, embs, protos, (1.0, 2.0), primary_spk="P", min_overlap_purity=0.40
        )
        self.assertEqual(res["retained_subsegs"], 1)
        self.assertEqual(res["speaker"], "Spk_A")

    def test_m2_insufficient_pure_embeddings_abstains(self):
        # All embeddings have < 40% overlap intersection -> must abstain!
        subsegs = [{"start": 0.0, "end": 1.5, "duration": 1.5}]
        embs = np.array([[1.0, 0.0]])
        protos = {"Spk_A": np.array([1.0, 0.0])}
        res = evaluate_m2_purity_and_attribute(
            subsegs, embs, protos, (1.3, 1.5), primary_spk="P", min_overlap_purity=0.40
        )
        self.assertIsNone(res["speaker"])
        self.assertEqual(res["reason"], "insufficient_pure_embeddings")

    def test_m2_empty_prototypes_abstains(self):
        res = evaluate_m2_purity_and_attribute([], np.array([]), {}, (1.0, 2.0), primary_spk="P")
        self.assertIsNone(res["speaker"])
        self.assertEqual(res["reason"], "empty_prototypes")

    # -------------------------------------------------------------------------
    # Correction D Tests: Overlap Timestamp Alignment
    # -------------------------------------------------------------------------
    def test_timestamp_single_positive_frame(self):
        # 1 positive frame at index 2 (center = 0.03096875 + 2 * 0.016875 = 0.06471875)
        # Expected bin: [center - step/2, center + step/2) = [0.05628125, 0.07315625), duration = 0.016875s
        # Since duration < min_duration_on (0.10s), single frame must be filtered out!
        times = np.array([0.03096875 + i * 0.016875 for i in range(10)])
        p_ovl = np.zeros(10)
        p_ovl[2] = 0.8
        segs = binarize_overlap_frame_bin_support(times, p_ovl, duration_sec=1.0, threshold=0.5, min_duration_on=0.10)
        self.assertEqual(len(segs), 0)

    def test_timestamp_consecutive_positive_frames(self):
        # 8 consecutive positive frames = 8 * 0.016875 = 0.135s >= 0.10s -> retained!
        times = np.array([0.03096875 + i * 0.016875 for i in range(20)])
        p_ovl = np.zeros(20)
        p_ovl[5:13] = 0.8 # indices 5 through 12
        segs = binarize_overlap_frame_bin_support(times, p_ovl, duration_sec=1.0, threshold=0.5, min_duration_on=0.10)
        self.assertEqual(len(segs), 1)
        expected_dur = 8 * 0.016875
        self.assertAlmostEqual(segs[0]["end"] - segs[0]["start"], expected_dur, places=3)

    def test_timestamp_boundary_clipping(self):
        # Active frame at the very beginning and very end
        times = np.array([0.03096875 + i * 0.016875 for i in range(20)])
        p_ovl = np.ones(20) # entire file active
        segs = binarize_overlap_frame_bin_support(times, p_ovl, duration_sec=0.30, threshold=0.5, min_duration_on=0.10)
        self.assertEqual(len(segs), 1)
        self.assertGreaterEqual(segs[0]["start"], 0.0)
        self.assertLessEqual(segs[0]["end"], 0.30)

    # -------------------------------------------------------------------------
    # Correction E Tests: Multi-Label Hungarian Speaker Mapping
    # -------------------------------------------------------------------------
    def test_hungarian_multi_label_simultaneous_speakers(self):
        # Ref has R1 and R2 overlapping in [0.0, 1.0]
        # Hyp has H1 and H2 overlapping in [0.0, 1.0]
        # Multi-label matrix must capture co-occurrence for BOTH pairs regardless of order!
        ref_segs = [
            {"start": 0.0, "end": 1.0, "speaker": "R1"},
            {"start": 0.0, "end": 1.0, "speaker": "R2"},
        ]
        # Extra single-speaker evidence to differentiate H1->R1 and H2->R2
        ref_segs.append({"start": 1.0, "end": 2.0, "speaker": "R1"})
        hyp_segs = [
            {"start": 0.0, "end": 1.0, "speaker": "H2"},
            {"start": 0.0, "end": 1.0, "speaker": "H1"},
            {"start": 1.0, "end": 2.0, "speaker": "H1"},
        ]

        mapping, unmatched_r, unmatched_h = compute_multilabel_hungarian_mapping(ref_segs, hyp_segs, duration_sec=2.0)
        self.assertEqual(mapping["R1"], "H1")
        self.assertEqual(mapping["R2"], "H2")
        self.assertEqual(len(unmatched_r), 0)
        self.assertEqual(len(unmatched_h), 0)

    def test_hungarian_input_order_invariance(self):
        # Permuting segment order must produce identical mapping
        ref_segs_1 = [{"start": 0.0, "end": 1.0, "speaker": "R1"}, {"start": 0.0, "end": 1.0, "speaker": "R2"}]
        ref_segs_2 = [{"start": 0.0, "end": 1.0, "speaker": "R2"}, {"start": 0.0, "end": 1.0, "speaker": "R1"}]
        hyp_segs = [{"start": 0.0, "end": 1.0, "speaker": "H1"}]

        m1, _, _ = compute_multilabel_hungarian_mapping(ref_segs_1, hyp_segs, duration_sec=1.0)
        m2, _, _ = compute_multilabel_hungarian_mapping(ref_segs_2, hyp_segs, duration_sec=1.0)
        self.assertEqual(m1, m2)

    def test_hungarian_unequal_speaker_counts(self):
        ref_segs = [{"start": 0.0, "end": 1.0, "speaker": "R1"}]
        hyp_segs = [{"start": 0.0, "end": 1.0, "speaker": "H1"}, {"start": 1.0, "end": 2.0, "speaker": "H2"}]
        mapping, unmatched_r, unmatched_h = compute_multilabel_hungarian_mapping(ref_segs, hyp_segs, duration_sec=2.0)
        self.assertEqual(mapping["R1"], "H1")
        self.assertEqual(len(unmatched_r), 0)
        self.assertIn("H2", unmatched_h)

    # -------------------------------------------------------------------------
    # Correction F Tests: Strict Metric Error Handling
    # -------------------------------------------------------------------------
    def test_metric_empty_reference_raises_exception(self):
        ref_annot = Annotation()
        hyp_annot = Annotation()
        hyp_annot[Segment(0.0, 1.0), "t0"] = "H1"
        with self.assertRaises(ValueError):
            safe_diarization_metrics(ref_annot, hyp_annot)

    def test_metric_distinguishes_zero_from_none(self):
        # Perfect diarization should yield DER = 0.0, Miss = 0.0, FA = 0.0, Conf = 0.0
        ref_annot = Annotation()
        ref_annot[Segment(0.0, 1.0), "t0"] = "S1"
        hyp_annot = Annotation()
        hyp_annot[Segment(0.0, 1.0), "t0"] = "S1"

        res = safe_diarization_metrics(ref_annot, hyp_annot)
        self.assertEqual(res["der_c0"], 0.0)
        self.assertEqual(res["miss"], 0.0)
        self.assertEqual(res["false_alarm"], 0.0)
        self.assertEqual(res["confusion"], 0.0)
        self.assertIsNotNone(res["der_c0"])
        self.assertEqual(res["status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
