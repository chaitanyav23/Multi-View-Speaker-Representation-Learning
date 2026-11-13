#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3 Remediation: Production cpWER Adapter (Phase 20 Parity)
Imports and reuses the authoritative Phase 20 word-attribution and grouping pipeline
from phase20_standalone/scripts/common.py and Hungarian cpWER from cpwer.py.
Guarantees 100% bit-exact parity on word timestamps, equal-overlap ties,
midpoint distance fallbacks, and segment envelope formatting.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from path_utils import get_phase20_scripts_dir

# Add phase20_standalone/scripts to sys.path cleanly
_p20_scripts = get_phase20_scripts_dir().resolve()
if str(_p20_scripts) not in sys.path:
    sys.path.insert(0, str(_p20_scripts))

# Authoritative imports directly from Phase 20
try:
    from common import (
        attributed_words as phase20_attributed_words,
        group_words as phase20_group_words,
        rttm_to_segments as phase20_rttm_to_segments,
        max_overlap_speaker as phase20_max_overlap_speaker,
    )
    from cpwer import (
        cpwer as phase20_cpwer,
        cpwer_stats as phase20_cpwer_stats,
    )
except ImportError as e:
    raise ImportError(f"Could not import authoritative Phase 20 modules from {_p20_scripts}: {e}")


def rttm_to_segments(rttm_path: Path) -> List[Dict[str, Any]]:
    """Delegates to authoritative Phase 20 rttm_to_segments."""
    return phase20_rttm_to_segments(Path(rttm_path))


def max_overlap_speaker(
    word: Dict[str, Any],
    diar_segments: List[Dict[str, Any]],
    tolerance: float = 0.5,
) -> str:
    """Delegates to authoritative Phase 20 max_overlap_speaker."""
    return phase20_max_overlap_speaker(word, diar_segments, tolerance=tolerance)


def attribute_words_to_segments(
    words: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Delegates to authoritative Phase 20 attributed_words."""
    return phase20_attributed_words(words, diar_segments)


def group_words_by_speaker(
    attributed_words_list: List[Dict[str, Any]],
    max_gap: float = 2.0,
) -> List[Dict[str, Any]]:
    """Delegates to authoritative Phase 20 group_words."""
    return phase20_group_words(attributed_words_list, max_gap=max_gap)


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
    Computes audited cpWER with verified Phase 20 parity.
    Attributes aligned ASR words to hypothesis RTTM segments,
    packages into canonical Phase 20 {"segments": [...]} envelope,
    and runs Hungarian cpwer_stats solver.
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

    # 1. Parse hypothesis RTTM using authoritative Phase 20 parser
    hyp_rttm_segs = rttm_to_segments(hyp_path)

    # 2. Load aligned words
    words = load_words_from_json(words_path)

    # 3. Attribute and group using authoritative Phase 20 functions
    attributed = attribute_words_to_segments(words, hyp_rttm_segs)
    hyp_grouped_segments = group_words_by_speaker(attributed)

    # Packaging into canonical Phase 20 envelope
    hyp_envelope = {
        "segments": [
            {
                "speaker": s["speaker"],
                "text": s["text"],
                "start": s.get("start"),
                "end": s.get("end"),
            }
            for s in hyp_grouped_segments
        ],
        "words": attributed,
    }

    # 4. Load reference segments
    ref_segments = load_reference_segments_from_json(ref_path)

    # 5. Compute official cpWER stats via Hungarian assignment
    stats = phase20_cpwer_stats(ref_segments, hyp_envelope["segments"])
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
