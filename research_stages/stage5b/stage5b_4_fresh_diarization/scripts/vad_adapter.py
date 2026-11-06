#!/usr/bin/env python3
"""
Stage 5B.4 VAD Interval Adapter
Reads existing posterior arrays read-only, verifies exact timestamp alignment,
applies deterministic binarization and hysteresis filtering for C00, C50, and C62,
and computes cryptographic interval hashes.
ZERO model inference or neural checkpoint loading.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from condition_registry import VADConfig, VAD_REGISTRY


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
STAGE2_POSTERIORS_DIR = REPO_ROOT / "research_stages" / "stage2_baseline" / "posteriors"


def canonical_interval_hash(rec_id: str, vad_id: str, intervals: List[Dict[str, float]]) -> str:
    """
    Computes a deterministic canonical SHA-256 hash of ordered speech intervals.
    Format: rec_id|vad_id|v1|start_0,end_0;start_1,end_1;...
    Explicit rounding policy: 4 decimal places.
    """
    elements = [
        f"{float(it['start']):.4f},{float(it['end']):.4f}"
        for it in sorted(intervals, key=lambda x: (x["start"], x["end"]))
    ]
    canonical_str = f"{rec_id}|{vad_id}|v1|" + ";".join(elements)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def binarize_posteriors_with_hysteresis(
    posteriors: np.ndarray,
    frame_times: np.ndarray,
    onset: float,
    offset: float,
    min_duration_on: float,
    min_duration_off: float,
) -> List[Dict[str, float]]:
    """
    Converts frame-level speech posteriors into speech intervals using hysteresis
    and duration filtering. Exact replication of pyannote Binarize logic.
    """
    n_frames = len(posteriors)
    if n_frames == 0 or len(frame_times) != n_frames:
        raise ValueError(f"Posteriors ({n_frames}) and frame_times ({len(frame_times)}) mismatch or empty")

    if n_frames == 1:
        frame_dur = 0.0270625
    else:
        frame_dur = float(np.median(np.diff(frame_times)))

    # Frame-level active state tracking
    is_active = False
    raw_segments: List[Tuple[float, float]] = []
    seg_start = 0.0

    for i in range(n_frames):
        p = float(posteriors[i])
        t = float(frame_times[i])

        if not is_active:
            if p >= onset:
                is_active = True
                seg_start = max(0.0, t - frame_dur / 2.0)
        else:
            if p < offset:
                is_active = False
                seg_end = t + frame_dur / 2.0
                raw_segments.append((seg_start, seg_end))

    if is_active:
        seg_end = float(frame_times[-1]) + frame_dur / 2.0
        raw_segments.append((seg_start, seg_end))

    if not raw_segments:
        return []

    # 1. Filter out speech bursts shorter than min_duration_on
    filtered_on = [
        (s, e) for (s, e) in raw_segments
        if (e - s) >= min_duration_on - 1e-6
    ]
    if not filtered_on:
        return []

    # 2. Bridge non-speech gaps shorter than min_duration_off
    bridged: List[Tuple[float, float]] = [filtered_on[0]]
    for s, e in filtered_on[1:]:
        prev_s, prev_e = bridged[-1]
        gap = s - prev_e
        if gap < min_duration_off:
            bridged[-1] = (prev_s, e)
        else:
            bridged.append((s, e))

    return [
        {
            "start": round(float(s), 4),
            "end": round(float(e), 4),
            "duration": round(float(e - s), 4),
        }
        for (s, e) in bridged
    ]


class VADAdapter:
    """
    Adapter for loading posterior arrays and generating speech intervals
    for C00, C50, and C62 without running any model inference.
    """

    def __init__(self, posteriors_dir: Optional[Path] = None):
        self.posteriors_dir = posteriors_dir or STAGE2_POSTERIORS_DIR

    def load_posteriors(self, rec_id: str, condition: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads cached Pyannote posteriors from stage2_baseline read-only.
        Returns: (frame_times, p_speech)
        """
        npz_path = self.posteriors_dir / f"{rec_id}_{condition}_posteriors.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Posterior file not found: {npz_path}")

        data = np.load(str(npz_path))
        if "frame_times" not in data or "p_speech" not in data:
            raise KeyError(f"NPZ file {npz_path} missing 'frame_times' or 'p_speech'")

        return data["frame_times"].astype(np.float64), data["p_speech"].astype(np.float32)

    def generate_vad_intervals(self, rec_id: str, vad_id: str) -> Tuple[List[Dict[str, float]], str]:
        """
        Generates deterministic speech intervals for the given VAD config.
        Returns: (intervals, canonical_sha256_hash)
        """
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

        intervals = binarize_posteriors_with_hysteresis(
            posteriors=p_speech,
            frame_times=f_times,
            onset=cfg.onset,
            offset=cfg.offset,
            min_duration_on=cfg.min_duration_on,
            min_duration_off=cfg.min_duration_off,
        )

        h = canonical_interval_hash(rec_id, vad_id, intervals)
        return intervals, h
