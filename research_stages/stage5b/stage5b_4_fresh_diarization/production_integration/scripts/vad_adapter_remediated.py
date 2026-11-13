#!/usr/bin/env python3
"""
Stage 5B.4-A.1 Remediation: Audited VAD Interval Adapter
Replicates the audited Stage 5B.1 binarize_posteriors implementation.
Guarantees verified boundary behavior, exact timestamp conventions,
and deterministic canonical interval hashing.
Zero model inference.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from condition_registry import VAD_REGISTRY, VADConfig


from path_utils import get_stage2_posteriors_dir

STAGE2_POSTERIORS_DIR = get_stage2_posteriors_dir()


def canonical_interval_hash(rec_id: str, vad_id: str, intervals: List[Dict[str, float]]) -> str:
    """
    Computes a deterministic canonical SHA-256 hash of ordered speech intervals.
    Explicit rounding policy: 4 decimal places.
    """
    elements = [
        f"{float(it['start']):.4f},{float(it['end']):.4f}"
        for it in sorted(intervals, key=lambda x: (x["start"], x["end"]))
    ]
    canonical_str = f"{rec_id}|{vad_id}|v1|" + ";".join(elements)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def binarize_posteriors_audited_reference(
    times: np.ndarray,
    probs: np.ndarray,
    onset: float = 0.481,
    offset: float = 0.481,
    min_duration_on: float = 0.10,
    min_duration_off: float = 0.15,
) -> List[Dict[str, float]]:
    """
    Binarizes continuous probabilities with hysteresis thresholds,
    minimum turn duration, and gap filling.
    Bit-exact replica of Stage 5B.1 run_stage5b_1_experiments.py lines 59-101.
    """
    if len(times) == 0 or len(probs) != len(times):
        return []

    is_speech = False
    segments: List[Dict[str, float]] = []
    seg_start = 0.0

    for t, p in zip(times, probs):
        t_val = float(t)
        p_val = float(p)
        if not is_speech and p_val >= onset:
            is_speech = True
            seg_start = t_val
        elif is_speech and p_val < offset:
            is_speech = False
            dur = t_val - seg_start
            if dur >= min_duration_on - 1e-6:
                segments.append({"start": round(seg_start, 4), "end": round(t_val, 4)})

    if is_speech:
        t_last = float(times[-1])
        dur = t_last - seg_start
        if dur >= min_duration_on - 1e-6:
            segments.append({"start": round(seg_start, 4), "end": round(t_last, 4)})

    # Gap-filling: merge segments separated by less than min_duration_off
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        gap = seg["start"] - merged[-1]["end"]
        if gap < min_duration_off - 1e-6:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(seg)

    # Post-filtering by minimum duration after merging
    filtered = [
        {
            "start": round(float(s["start"]), 4),
            "end": round(float(s["end"]), 4),
            "duration": round(float(s["end"] - s["start"]), 4),
        }
        for s in merged
        if (s["end"] - s["start"]) >= min_duration_on - 1e-6
    ]
    return filtered


class VADAdapterRemediated:
    """
    Remediated VAD adapter reading existing posterior arrays read-only
    and applying the audited binarize_posteriors logic.
    """

    def __init__(self, posteriors_dir: Optional[Path] = None):
        self.posteriors_dir = posteriors_dir or STAGE2_POSTERIORS_DIR

    def load_posteriors(self, rec_id: str, condition: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads cached Pyannote posteriors from stage2_baseline read-only."""
        npz_path = self.posteriors_dir / f"{rec_id}_{condition}_posteriors.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Posterior file not found: {npz_path}")

        data = np.load(str(npz_path))
        if "frame_times" not in data or "p_speech" not in data:
            raise KeyError(f"NPZ file {npz_path} missing 'frame_times' or 'p_speech'")

        return data["frame_times"].astype(np.float64), data["p_speech"].astype(np.float32)

    def generate_vad_intervals(self, rec_id: str, vad_id: str) -> Tuple[List[Dict[str, float]], str]:
        """Generates deterministic speech intervals for the given VAD config."""
        if vad_id not in VAD_REGISTRY:
            raise ValueError(f"Unknown VAD config ID: {vad_id}")

        cfg: VADConfig = VAD_REGISTRY[vad_id]

        if vad_id == "C00":
            f_times, p_speech = self.load_posteriors(rec_id, "dfn")
        elif vad_id == "C50":
            f_times, p_speech = self.load_posteriors(rec_id, "16k_mono")
        elif vad_id == "C62":
            f_times_dfn, p_dfn = self.load_posteriors(rec_id, "dfn")
            f_times_mono, p_mono = self.load_posteriors(rec_id, "16k_mono")

            # Strict validation: verify exact timestamp alignment
            if len(f_times_dfn) != len(f_times_mono):
                raise ValueError(
                    f"Timestamp array length mismatch for {rec_id}: "
                    f"DFN={len(f_times_dfn)}, Mono={len(f_times_mono)}"
                )
            if not np.allclose(f_times_dfn, f_times_mono, atol=1e-5):
                max_diff = float(np.max(np.abs(f_times_dfn - f_times_mono)))
                raise ValueError(
                    f"Timestamp alignment error for {rec_id}: max_diff={max_diff:.6f}s"
                )

            f_times = f_times_dfn
            p_speech = 0.5 * p_dfn + 0.5 * p_mono
        else:
            raise NotImplementedError(f"Unsupported VAD ID: {vad_id}")

        intervals = binarize_posteriors_audited_reference(
            times=f_times,
            probs=p_speech,
            onset=cfg.onset,
            offset=cfg.offset,
            min_duration_on=cfg.min_duration_on,
            min_duration_off=cfg.min_duration_off,
        )

        h = canonical_interval_hash(rec_id, vad_id, intervals)
        return intervals, h
