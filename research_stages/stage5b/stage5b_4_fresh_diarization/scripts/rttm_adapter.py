#!/usr/bin/env python3
"""
Stage 5B.4 RTTM Adapter
Converts subsegment cluster labels into canonical RTTMs, parses RTTMs into segment lists,
and rasterizes hypothesis segments into frame-level arrays for overlap processing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def subsegments_to_single_speaker_segments(
    subsegments: List[Dict[str, float]],
    speaker_labels: List[str],
) -> List[Dict[str, Any]]:
    """
    Combines base subsegments with their assigned speaker labels,
    merging adjacent contiguous subsegments of the same speaker.
    """
    if len(subsegments) != len(speaker_labels):
        raise ValueError(
            f"Length mismatch: {len(subsegments)} subsegments vs {len(speaker_labels)} labels"
        )

    if not subsegments:
        return []

    # Sort by start time
    sorted_pairs = sorted(
        zip(subsegments, speaker_labels),
        key=lambda pair: (float(pair[0]["start"]), float(pair[0]["end"])),
    )

    merged: List[Dict[str, Any]] = []
    curr_st = float(sorted_pairs[0][0]["start"])
    curr_en = float(sorted_pairs[0][0]["end"])
    curr_spk = str(sorted_pairs[0][1])

    for sub, spk in sorted_pairs[1:]:
        st = float(sub["start"])
        en = float(sub["end"])
        spk = str(spk)

        # Merge if same speaker and overlapping or seamlessly adjacent (gap < 0.05s)
        if spk == curr_spk and st <= curr_en + 0.05:
            curr_en = max(curr_en, en)
        else:
            merged.append({
                "start": round(curr_st, 4),
                "end": round(curr_en, 4),
                "duration": round(curr_en - curr_st, 4),
                "speaker": curr_spk,
            })
            curr_st = st
            curr_en = en
            curr_spk = spk

    merged.append({
        "start": round(curr_st, 4),
        "end": round(curr_en, 4),
        "duration": round(curr_en - curr_st, 4),
        "speaker": curr_spk,
    })

    return merged


def write_rttm_file(
    segments: List[Dict[str, Any]],
    rec_id: str,
    output_path: Path,
) -> None:
    """
    Writes standard NIST RTTM file.
    Format: SPEAKER <rec_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_segs = sorted(segments, key=lambda s: (float(s["start"]), float(s["end"])))
    lines = []
    for s in sorted_segs:
        st = float(s["start"])
        dur = float(s.get("duration", float(s["end"]) - st))
        spk = str(s["speaker"])
        lines.append(f"SPEAKER {rec_id} 1 {st:.4f} {dur:.4f} <NA> <NA> {spk} <NA> <NA>\n")

    output_path.write_text("".join(lines), encoding="utf-8")


def read_rttm_file(rttm_path: Path) -> List[Dict[str, Any]]:
    """
    Reads an RTTM file into a list of segment dictionaries.
    """
    rttm_path = Path(rttm_path)
    if not rttm_path.exists():
        raise FileNotFoundError(f"RTTM file not found: {rttm_path}")

    segments: List[Dict[str, Any]] = []
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        st = float(parts[3])
        dur = float(parts[4])
        spk = parts[7]
        segments.append({
            "start": round(st, 4),
            "end": round(st + dur, 4),
            "duration": round(dur, 4),
            "speaker": spk,
        })
    return segments


def rasterize_segments_to_frame_spk(
    segments: List[Dict[str, Any]],
    total_duration_sec: float,
    resolution: float = 0.01,
) -> List[Optional[str]]:
    """
    Rasterizes single-speaker hypothesis segments into a discrete frame array
    at resolution (default 10ms = 0.01s).
    """
    n_frames = max(1, int(np.ceil(total_duration_sec / resolution)))
    frame_spk: List[Optional[str]] = [None] * n_frames

    for seg in sorted(segments, key=lambda s: float(s["start"])):
        st_idx = max(0, int(round(float(seg["start"]) / resolution)))
        en_idx = min(n_frames, int(round(float(seg["end"]) / resolution)))
        spk = str(seg["speaker"])
        for idx in range(st_idx, en_idx):
            frame_spk[idx] = spk

    return frame_spk
