#!/usr/bin/env python3
"""
Stage 5B.4 Overlap Attribution Adapter
Adapts audited M0 (single-label baseline) and M4-R2 (conservative turn-alternation with abstention)
for fresh diarization hypotheses. Operates strictly on arbitrary categorical cluster labels
without reference-identity leakage.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from rttm_adapter import rasterize_segments_to_frame_spk


def extract_contiguous_turns(
    frame_spk: List[Optional[str]],
    search_start_idx: int,
    search_end_idx: int,
    direction: str,
    resolution: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Extracts genuinely contiguous speaker runs without bridging None gaps.
    Audited implementation from Stage 5B.2 V4.1.
    """
    runs: List[Dict[str, Any]] = []
    curr_spk: Optional[str] = None
    curr_st: int = 0
    curr_en: int = 0

    indices = (
        range(search_start_idx, search_end_idx - 1, -1)
        if direction == "backward"
        else range(search_start_idx, search_end_idx)
    )

    for i in indices:
        s = frame_spk[i]
        if s is not None:
            if curr_spk is None:
                curr_spk = s
                curr_st = i
                curr_en = i + 1
            elif s == curr_spk:
                if direction == "backward":
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


def attribute_m4_conservative(
    frame_spk: List[Optional[str]],
    overlap_interval: Tuple[float, float],
    primary_spk: Optional[str],
    resolution: float = 0.01,
    max_gap_sec: float = 5.0,
    min_turn_dur_sec: float = 0.20,
) -> Dict[str, Any]:
    """
    Conservative Contiguous Turn Alternation with Abstention (M4-R2).
    Audited implementation from Stage 5B.2 V4.1.
    """
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

    p_cand = None
    for r in p_runs:
        if r["duration_sec"] >= min_turn_dur_sec:
            p_cand = r["speaker"]
            break

    n_cand = None
    for r in n_runs:
        if r["duration_sec"] >= min_turn_dur_sec:
            n_cand = r["speaker"]
            break

    # Conservative abstention rules:
    # 1. Both neighbors must exist and be valid
    # 2. Neighbors must be different speakers (turn transition)
    # 3. One neighbor must match primary speaker, the other becomes secondary
    if p_cand is not None and n_cand is not None:
        if p_cand != n_cand:
            if p_cand == primary_spk:
                return {"speaker": n_cand, "reason": "turn_transition_forward", "p_cand": p_cand, "n_cand": n_cand}
            elif n_cand == primary_spk:
                return {"speaker": p_cand, "reason": "turn_transition_backward", "p_cand": p_cand, "n_cand": n_cand}
            else:
                # Neither matches primary: abstain to prevent hallucination
                return {"speaker": None, "reason": "neither_matches_primary", "p_cand": p_cand, "n_cand": n_cand}
        else:
            # Same speaker before and after (pause / hesitation): abstain
            return {"speaker": None, "reason": "same_speaker_adjacent", "p_cand": p_cand, "n_cand": n_cand}

    return {"speaker": None, "reason": "insufficient_context", "p_cand": p_cand, "n_cand": n_cand}


def detect_overlap_intervals(
    p_overlap: np.ndarray,
    frame_times: np.ndarray,
    tau_overlap: float = 0.40,
    min_overlap_dur: float = 0.10,
) -> List[Tuple[float, float]]:
    """
    Extracts contiguous overlap candidate intervals from frame-level overlap posteriors.
    """
    n_frames = len(p_overlap)
    if n_frames == 0 or len(frame_times) != n_frames:
        return []

    frame_dur = float(np.median(np.diff(frame_times))) if n_frames > 1 else 0.0270625

    active = False
    st_t = 0.0
    candidates: List[Tuple[float, float]] = []

    for i in range(n_frames):
        p = float(p_overlap[i])
        t = float(frame_times[i])
        if not active:
            if p >= tau_overlap:
                active = True
                st_t = max(0.0, t - frame_dur / 2.0)
        else:
            if p < tau_overlap:
                active = False
                en_t = t + frame_dur / 2.0
                if (en_t - st_t) >= min_overlap_dur:
                    candidates.append((round(st_t, 4), round(en_t, 4)))

    if active:
        en_t = float(frame_times[-1]) + frame_dur / 2.0
        if (en_t - st_t) >= min_overlap_dur:
            candidates.append((round(st_t, 4), round(en_t, 4)))

    return candidates


def apply_overlap_attribution(
    single_speaker_segments: List[Dict[str, Any]],
    p_overlap: Optional[np.ndarray],
    frame_times: Optional[np.ndarray],
    method_id: str = "M0_no_overlap",
    tau_overlap: float = 0.40,
    min_overlap_dur: float = 0.10,
    max_gap_sec: float = 5.0,
    min_turn_dur_sec: float = 0.20,
    resolution: float = 0.01,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Applies overlap attribution to fresh single-speaker hypothesis segments.
    Returns: (final_segments, attribution_stats)
    """
    if method_id == "M0_no_overlap":
        # Pass single-speaker hypothesis untouched
        stats = {
            "method": "M0_no_overlap",
            "overlap_candidates": 0,
            "secondary_attributed": 0,
            "abstentions": 0,
        }
        return list(single_speaker_segments), stats

    if method_id != "M4_conservative_abstention":
        raise ValueError(f"Unsupported overlap method: {method_id}")

    if p_overlap is None or frame_times is None:
        raise ValueError("M4 requires p_overlap and frame_times")

    # Detect candidate overlap regions
    ovl_intervals = detect_overlap_intervals(
        p_overlap=p_overlap,
        frame_times=frame_times,
        tau_overlap=tau_overlap,
        min_overlap_dur=min_overlap_dur,
    )

    if not single_speaker_segments or not ovl_intervals:
        stats = {
            "method": "M4_conservative_abstention",
            "overlap_candidates": len(ovl_intervals),
            "secondary_attributed": 0,
            "abstentions": len(ovl_intervals),
        }
        return list(single_speaker_segments), stats

    # Rasterize hypothesis segments
    max_time = max(
        max(float(s["end"]) for s in single_speaker_segments),
        float(frame_times[-1]),
    )
    frame_spk = rasterize_segments_to_frame_spk(
        single_speaker_segments,
        total_duration_sec=max_time,
        resolution=resolution,
    )

    secondary_segments: List[Dict[str, Any]] = []
    attributed_count = 0
    abstain_count = 0

    for st_sec, en_sec in ovl_intervals:
        st_idx = max(0, int(round(st_sec / resolution)))
        en_idx = min(len(frame_spk), int(round(en_sec / resolution)))

        # Find primary speaker within the overlap window
        window_spks = [s for s in frame_spk[st_idx:en_idx] if s is not None]
        if not window_spks:
            primary_spk = None
        else:
            # Deterministic majority: most common speaker, break ties alphabetically
            counts: Dict[str, int] = {}
            for s in window_spks:
                counts[s] = counts.get(s, 0) + 1
            sorted_spks = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            primary_spk = sorted_spks[0][0]

        res = attribute_m4_conservative(
            frame_spk=frame_spk,
            overlap_interval=(st_sec, en_sec),
            primary_spk=primary_spk,
            resolution=resolution,
            max_gap_sec=max_gap_sec,
            min_turn_dur_sec=min_turn_dur_sec,
        )

        sec_spk = res["speaker"]
        if sec_spk is not None:
            secondary_segments.append({
                "start": round(st_sec, 4),
                "end": round(en_sec, 4),
                "duration": round(en_sec - st_sec, 4),
                "speaker": sec_spk,
            })
            attributed_count += 1
        else:
            abstain_count += 1

    # Merge primary and secondary segments
    final_segments = list(single_speaker_segments) + secondary_segments
    final_segments.sort(key=lambda s: (float(s["start"]), float(s["end"])))

    stats = {
        "method": "M4_conservative_abstention",
        "overlap_candidates": len(ovl_intervals),
        "secondary_attributed": attributed_count,
        "abstentions": abstain_count,
    }
    return final_segments, stats
