#!/usr/bin/env python3
"""
Stage 5B.4 Dynamic Multi-Scale Subsegment Generator
Generates multi-scale subsegments from candidate VAD intervals (C00, C50, C62)
using exact NeMo speaker_utils get_subsegments() windowing and shift logic.
Preserves parent speech interval indices and recording-local timestamps.
"""

from typing import Any, Dict, List, Optional, Tuple


# Standard NeMo MSDD / Stage 3 multi-scale configuration
# (scale_k, window_sec, shift_sec, multi_scale_weight)
SCALES: List[Tuple[int, float, float, float]] = [
    (0, 1.75, 0.50, 0.225),
    (1, 1.50, 0.50, 0.225),
    (2, 1.25, 0.50, 0.225),
    (3, 1.00, 0.50, 0.225),
    (4, 0.50, 0.25, 0.100),  # Base scale (scale 4)
]

SCALE_DICT: Dict[int, Tuple[float, float, float]] = {
    s[0]: (s[1], s[2], s[3]) for s in SCALES
}


def generate_subsegments_for_interval(
    parent_idx: int,
    start: float,
    end: float,
    window: float,
    shift: float,
) -> List[Dict[str, Any]]:
    """
    Subsegments a single continuous speech interval [start, end]
    using exact NeMo get_subsegments sliding window semantics.
    """
    subsegs: List[Dict[str, Any]] = []
    offset = float(start)
    duration = float(end) - offset

    if duration <= 0.0:
        return subsegs

    if duration < window:
        subsegs.append({
            "parent_interval_idx": parent_idx,
            "start": round(offset, 4),
            "end": round(offset + duration, 4),
            "duration": round(duration, 4),
        })
    else:
        slice_st = offset
        end_time = offset + duration
        while slice_st + window <= end_time + 1e-6:
            subsegs.append({
                "parent_interval_idx": parent_idx,
                "start": round(slice_st, 4),
                "end": round(slice_st + window, 4),
                "duration": round(window, 4),
            })
            slice_st += shift

        # Handle trailing remainder if significant (>= 50ms) per NeMo convention
        if slice_st < end_time - 1e-4 and (end_time - slice_st) >= 0.05:
            st_clamped = round(end_time - window, 4) if (end_time - offset) >= window else round(slice_st, 4)
            subsegs.append({
                "parent_interval_idx": parent_idx,
                "start": st_clamped,
                "end": round(end_time, 4),
                "duration": round(min(window, end_time - offset), 4),
            })

    return subsegs


def generate_multiscale_subsegments(
    vad_intervals: List[Dict[str, float]],
    scale_k: int,
) -> List[Dict[str, Any]]:
    """
    Generates subsegments for a full list of VAD intervals at a specific scale.
    """
    if scale_k not in SCALE_DICT:
        raise ValueError(f"Invalid scale index {scale_k}. Must be in 0..4.")

    window, shift, _ = SCALE_DICT[scale_k]
    all_subsegs: List[Dict[str, Any]] = []

    for idx, seg in enumerate(vad_intervals):
        s = float(seg["start"])
        e = float(seg["end"])
        subsegs = generate_subsegments_for_interval(
            parent_idx=idx,
            start=s,
            end=e,
            window=window,
            shift=shift,
        )
        all_subsegs.extend(subsegs)

    # Assign sequential subsegment indices for deterministic tracking
    for subseg_idx, sub in enumerate(all_subsegs):
        sub["subsegment_idx"] = subseg_idx
        sub["scale"] = scale_k

    return all_subsegs


def generate_all_scales(
    vad_intervals: List[Dict[str, float]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Generates subsegments across all 5 scales (0..4) for a set of VAD intervals.
    Returns mapping: scale_k -> list of subsegment dicts.
    """
    return {
        scale_k: generate_multiscale_subsegments(vad_intervals, scale_k)
        for scale_k in range(5)
    }
