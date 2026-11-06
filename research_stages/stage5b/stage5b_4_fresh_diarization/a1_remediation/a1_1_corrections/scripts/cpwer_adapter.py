#!/usr/bin/env python3
"""
Stage 5B.4-A.1.1 Remediation: Production cpWER Adapter
Integrates audited Phase 20 cpWER calculation and word attribution logic.
Adheres strictly to the canonical {"segments": [...]} envelope schema
and delegates to phase20_standalone/scripts/cpwer.py for Hungarian-matched cpWER.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from path_utils import get_phase20_scripts_dir

# Ensure phase20_standalone/scripts is available for importing cpwer
_p20_scripts = get_phase20_scripts_dir()
if str(_p20_scripts) not in sys.path:
    sys.path.insert(0, str(_p20_scripts))

try:
    from cpwer import cpwer as phase20_cpwer, cpwer_stats
except ImportError as e:
    raise ImportError(f"Could not import cpwer from {_p20_scripts}: {e}")


def rttm_to_segments(rttm_path: Path) -> List[Dict[str, Any]]:
    """Parses an RTTM file into a list of segment dictionaries."""
    p = Path(rttm_path)
    if not p.exists():
        raise FileNotFoundError(f"RTTM file not found: {p}")

    segments = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            st = float(parts[3])
            dur = float(parts[4])
            if dur <= 0:
                continue
            spk = parts[7]
            segments.append({
                "start": round(st, 4),
                "end": round(st + dur, 4),
                "speaker": spk,
            })
    # Sort segments by start time
    segments.sort(key=lambda s: (s["start"], s["end"]))
    return segments


def max_overlap_speaker(
    word: Dict[str, Any],
    diar_segments: List[Dict[str, Any]],
    tolerance: float = 0.5,
) -> str:
    """
    Assigns a speaker to a word using maximum interval overlap.
    Falls back to nearest segment if within tolerance; otherwise 'UNK'.
    Matches phase20_standalone/scripts/common.py:max_overlap_speaker.
    """
    if word.get("start") is None or word.get("end") is None:
        return "UNK"

    w_start = float(word["start"])
    w_end = float(word["end"])

    overlaps = []
    for seg in diar_segments:
        overlap = min(w_end, seg["end"]) - max(w_start, seg["start"])
        if overlap > 0:
            overlaps.append((overlap, seg["speaker"]))

    if overlaps:
        # Choose speaker with greatest temporal overlap
        return max(overlaps, key=lambda x: x[0])[1]

    # Midpoint distance fallback
    midpoint = (w_start + w_end) / 2.0
    nearest = None
    best = float("inf")
    for seg in diar_segments:
        dist = min(abs(midpoint - seg["start"]), abs(midpoint - seg["end"]))
        if dist < best:
            best = dist
            nearest = seg["speaker"]

    return nearest if (nearest and best <= tolerance) else "UNK"


def attribute_words_to_segments(
    words: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
    tolerance: float = 0.5,
) -> List[Dict[str, Any]]:
    """Attributes each aligned word to a diarization segment speaker."""
    attributed = []
    for word in words:
        w_text = word.get("word") or word.get("text") or ""
        item = {
            "word": str(w_text).strip(),
            "start": float(word["start"]) if word.get("start") is not None else None,
            "end": float(word["end"]) if word.get("end") is not None else None,
            "confidence": word.get("confidence", word.get("score")),
            "speaker": max_overlap_speaker(word, diar_segments, tolerance=tolerance),
        }
        attributed.append(item)
    return attributed


def group_words_by_speaker(
    attributed_words: List[Dict[str, Any]],
    max_gap: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Groups adjacent attributed words by speaker into coherent speech segments.
    Matches phase20_standalone/scripts/common.py:group_words.
    """
    grouped: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for w in attributed_words:
        w_text = w.get("word", "").strip()
        if not w_text:
            continue

        spk = w.get("speaker", "UNK")
        st = w.get("start")
        en = w.get("end")

        if current is None:
            current = {
                "speaker": spk,
                "start": st,
                "end": en,
                "text": w_text,
                "words": [w],
            }
            continue

        gap = 0.0 if (st is None or current.get("end") is None) else (st - current["end"])
        if spk == current.get("speaker") and gap <= max_gap:
            current["end"] = en
            current["text"] += " " + w_text
            current["words"].append(w)
        else:
            grouped.append(current)
            current = {
                "speaker": spk,
                "start": st,
                "end": en,
                "text": w_text,
                "words": [w],
            }

    if current is not None:
        grouped.append(current)

    return grouped


def load_words_from_json(words_path: Path) -> List[Dict[str, Any]]:
    """Loads aligned words from a JSON file, accepting either a list or {'words': [...]'."""
    p = Path(words_path)
    if not p.exists():
        raise FileNotFoundError(f"Aligned words JSON not found: {p}")

    with p.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "words" in data and isinstance(data["words"], list):
            return data["words"]
        elif "segments" in data and isinstance(data["segments"], list):
            # Flatten segment words if present
            words = []
            for seg in data["segments"]:
                for w in seg.get("words", []):
                    words.append(w)
            if words:
                return words
        raise ValueError(f"JSON in {p} missing 'words' list.")
    else:
        raise ValueError(f"Unexpected JSON format in {p}: expected list or dict.")


def load_reference_segments_from_json(transcript_path: Path) -> List[Dict[str, Any]]:
    """
    Loads reference transcripts from a JSON file.
    Accepts either {'segments': [...]} envelope or a list of segment dicts.
    Each segment must contain 'speaker' (or 'speaker_id') and 'text'.
    """
    p = Path(transcript_path)
    if not p.exists():
        raise FileNotFoundError(f"Reference transcript JSON not found: {p}")

    with p.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw_segs = data.get("segments", [])
    elif isinstance(data, list):
        raw_segs = data
    else:
        raise ValueError(f"Unexpected JSON format in {p}: expected list or dict.")

    segments = []
    for s in raw_segs:
        spk = s.get("speaker") or s.get("speaker_id") or "SPEAKER_00"
        txt = s.get("text", "")
        segments.append({
            "speaker": str(spk),
            "text": str(txt).strip(),
            "start": s.get("start"),
            "end": s.get("end"),
        })
    return segments


def evaluate_cpwer(
    hypothesis_rttm: Path,
    aligned_words_json: Path,
    reference_transcript_json: Path,
) -> Dict[str, Any]:
    """
    Computes audited cpWER by attributing ASR words to hypothesis RTTM segments,
    packaging segments into canonical envelope, and running Hungarian-matched cpWER.
    """
    hyp_path = Path(hypothesis_rttm)
    words_path = Path(aligned_words_json)
    ref_path = Path(reference_transcript_json)

    if not hyp_path.exists():
        raise FileNotFoundError(f"Hypothesis RTTM not found: {hyp_path}")
    if not words_path.exists():
        raise FileNotFoundError(f"Aligned words JSON not found: {words_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference transcript JSON not found: {ref_path}")

    # 1. Parse hypothesis RTTM
    hyp_rttm_segs = rttm_to_segments(hyp_path)

    # 2. Load aligned words
    words = load_words_from_json(words_path)

    # 3. Attribute words and group into hypothesis segments
    attributed = attribute_words_to_segments(words, hyp_rttm_segs)
    hyp_grouped_segments = group_words_by_speaker(attributed)

    # Canonical envelope packaging
    hyp_envelope = {
        "segments": [
            {
                "speaker": s["speaker"],
                "text": s["text"],
                "start": s["start"],
                "end": s["end"],
            }
            for s in hyp_grouped_segments
        ],
        "words": attributed,
    }

    # 4. Load ground truth segments
    ref_segments = load_reference_segments_from_json(ref_path)

    # 5. Compute official cpWER stats via audited Hungarian solver
    stats = cpwer_stats(ref_segments, hyp_envelope["segments"])
    score = float(stats["cpwer"])

    return {
        "cpwer": round(score, 4),
        "insertions": stats.get("insertions", 0),
        "deletions": stats.get("deletions", 0),
        "substitutions": stats.get("substitutions", 0),
        "reference_words": stats.get("reference_words", 0),
        "hypothesis_words": stats.get("hypothesis_words", 0),
        "assignment": stats.get("assignment", []),
        "hyp_segment_count": len(hyp_grouped_segments),
        "ref_segment_count": len(ref_segments),
    }
