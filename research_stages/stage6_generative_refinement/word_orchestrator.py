"""
Stage 6 — Step A: Word-Level Orchestration
Associates ASR word-level transcripts with diarization RTTM segments using
timestamp overlap while explicitly tracking missing speaker assignments ("UNK"),
midpoint fallbacks, and multi-speaker overlapping speech.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schemas import AttributionStatus, ConversationalTurnSummary, OrchestratedWord


def parse_rttm_file(rttm_path: Path) -> List[Dict[str, Any]]:
    """
    Parses a standard NIST RTTM file into ordered segment dictionaries.
    Compatible with Phase 20 common.rttm_to_segments.
    """
    p = Path(rttm_path)
    if not p.exists():
        raise FileNotFoundError(f"Diarization RTTM file not found: {p}")

    segments: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            parts = line.split()
            if parts[0] != "SPEAKER":
                continue
            if len(parts) < 8:
                raise ValueError(f"Malformed RTTM line {line_num} in {p}: {raw_line!r}")
            start = float(parts[3])
            dur = float(parts[4])
            end = start + dur
            speaker = str(parts[7]).strip()
            if not speaker:
                raise ValueError(f"Empty speaker ID on line {line_num} in {p}")
            segments.append({
                "start": start,
                "end": end,
                "duration": dur,
                "speaker": speaker,
            })
    return segments


def load_asr_words(words_json_path: Path) -> List[Dict[str, Any]]:
    """
    Loads word-level ASR transcript from JSON.
    Supports top-level list of word dicts, {'words': [...]}, or {'segments': [{'words': [...]}]}.
    """
    p = Path(words_json_path)
    if not p.exists():
        raise FileNotFoundError(f"ASR aligned words JSON not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        raw_words = payload
    elif isinstance(payload, dict):
        if "words" in payload and isinstance(payload["words"], list):
            raw_words = payload["words"]
        elif "segments" in payload and isinstance(payload["segments"], list):
            raw_words = []
            for seg in payload["segments"]:
                for w in seg.get("words", []):
                    raw_words.append(w)
        else:
            raise ValueError(f"Unsupported ASR JSON dictionary structure in {p}")
    else:
        raise ValueError(f"Unexpected ASR JSON root type in {p}: {type(payload).__name__}")

    return raw_words


def extract_available_speakers(diar_segments: Sequence[Dict[str, Any]]) -> Tuple[str, ...]:
    """
    Returns sorted tuple of unique valid speaker IDs present in diarization segments,
    strictly excluding 'UNK' or empty strings.
    """
    speakers = sorted({
        str(seg.get("speaker", "")).strip()
        for seg in diar_segments
        if seg.get("speaker") and str(seg.get("speaker")).strip() not in ("", "UNK")
    })
    return tuple(speakers)


def associate_word_with_diarization(
    word_dict: Dict[str, Any],
    word_index: int,
    diar_segments: Sequence[Dict[str, Any]],
    tolerance: float = 0.5,
) -> OrchestratedWord:
    """
    Associates a single ASR word with diarization segments via temporal overlap.
    Preserves exact Phase 20 max_overlap_speaker parity while recording detailed
    overlap and fallback provenance.
    """
    raw_text = word_dict.get("word")
    if raw_text is None:
        raw_text = word_dict.get("text", "")
    word_str = str(raw_text)

    start_raw = word_dict.get("start")
    end_raw = word_dict.get("end")
    conf_raw = word_dict.get("confidence", word_dict.get("score"))
    conf_val = float(conf_raw) if conf_raw is not None else None

    # Handle missing or invalid timestamps explicitly without inventing assignments
    if start_raw is None or end_raw is None:
        return OrchestratedWord(
            word_index=word_index,
            word=word_str,
            start=float(start_raw) if start_raw is not None else None,
            end=float(end_raw) if end_raw is not None else None,
            asr_confidence=conf_val,
            original_speaker_id="UNK",
            attribution_status=AttributionStatus.MISSING_TIMESTAMPS.value,
            is_overlapping_speech=False,
            overlapping_speakers=(),
            overlap_durations={},
            nearest_segment_distance=None,
        )

    w_start = float(start_raw)
    w_end = float(end_raw)

    overlaps: List[Tuple[float, str]] = []
    speaker_overlap_totals: Dict[str, float] = {}

    for seg in diar_segments:
        s_start = float(seg["start"])
        s_end = float(seg["end"])
        spk = str(seg["speaker"])
        ovl = min(w_end, s_end) - max(w_start, s_start)
        if ovl > 0.0:
            overlaps.append((ovl, spk))
            speaker_overlap_totals[spk] = round(speaker_overlap_totals.get(spk, 0.0) + ovl, 6)

    if overlaps:
        # Phase 20 parity: max(overlaps)[1] selects maximum single-segment overlap
        # (with lexicographical speaker tie-break if identical)
        chosen_speaker = max(overlaps)[1]
        unique_overlapping_speakers = tuple(sorted(speaker_overlap_totals.keys()))
        is_multi_speaker = len(unique_overlapping_speakers) > 1
        status = (
            AttributionStatus.MULTI_SPEAKER_OVERLAP.value
            if is_multi_speaker
            else AttributionStatus.SINGLE_SPEAKER_OVERLAP.value
        )
        return OrchestratedWord(
            word_index=word_index,
            word=word_str,
            start=w_start,
            end=w_end,
            asr_confidence=conf_val,
            original_speaker_id=chosen_speaker,
            attribution_status=status,
            is_overlapping_speech=is_multi_speaker,
            overlapping_speakers=unique_overlapping_speakers,
            overlap_durations=speaker_overlap_totals,
            nearest_segment_distance=0.0,
        )

    # No direct overlap: check midpoint distance fallback within tolerance
    midpoint = (w_start + w_end) / 2.0
    nearest_spk: Optional[str] = None
    best_dist = float("inf")
    for seg in diar_segments:
        s_start = float(seg["start"])
        s_end = float(seg["end"])
        dist = min(abs(midpoint - s_start), abs(midpoint - s_end))
        if dist < best_dist:
            best_dist = dist
            nearest_spk = str(seg["speaker"])

    if nearest_spk is not None and best_dist <= tolerance:
        return OrchestratedWord(
            word_index=word_index,
            word=word_str,
            start=w_start,
            end=w_end,
            asr_confidence=conf_val,
            original_speaker_id=nearest_spk,
            attribution_status=AttributionStatus.MIDPOINT_TOLERANCE_FALLBACK.value,
            is_overlapping_speech=False,
            overlapping_speakers=(nearest_spk,),
            overlap_durations={},
            nearest_segment_distance=round(best_dist, 6),
        )

    # Insufficient evidence: assign UNK explicitly
    return OrchestratedWord(
        word_index=word_index,
        word=word_str,
        start=w_start,
        end=w_end,
        asr_confidence=conf_val,
        original_speaker_id="UNK",
        attribution_status=AttributionStatus.UNASSIGNED_UNK.value,
        is_overlapping_speech=False,
        overlapping_speakers=(),
        overlap_durations={},
        nearest_segment_distance=round(best_dist, 6) if best_dist != float("inf") else None,
    )


def orchestrate_words(
    raw_words: Sequence[Dict[str, Any]],
    diar_segments: Sequence[Dict[str, Any]],
    tolerance: float = 0.5,
) -> List[OrchestratedWord]:
    """
    Associates an entire sequence of ASR words with diarization segments.
    Preserves exact word order, text, and timestamps.
    """
    orchestrated: List[OrchestratedWord] = []
    for idx, w in enumerate(raw_words):
        orchestrated.append(
            associate_word_with_diarization(
                word_dict=w,
                word_index=idx,
                diar_segments=diar_segments,
                tolerance=tolerance,
            )
        )
    return orchestrated


def build_conversational_turns(
    words: Sequence[OrchestratedWord],
    max_gap: float = 2.0,
) -> List[ConversationalTurnSummary]:
    """
    Groups consecutive words assigned to the same speaker (with gap <= max_gap)
    into conversational turns. Mirrors Phase 20 common.group_words.
    """
    turns: List[ConversationalTurnSummary] = []
    cur_speaker: Optional[str] = None
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_words: List[str] = []
    cur_has_ovl: bool = False

    for w in words:
        if not w.word:
            continue
        if cur_speaker is None:
            cur_speaker = w.original_speaker_id
            cur_start = w.start
            cur_end = w.end
            cur_words = [w.word]
            cur_has_ovl = w.is_overlapping_speech
            continue

        gap = 0.0
        if w.start is not None and cur_end is not None:
            gap = w.start - cur_end

        if w.original_speaker_id == cur_speaker and gap <= max_gap:
            cur_end = w.end
            cur_words.append(w.word)
            cur_has_ovl = cur_has_ovl or w.is_overlapping_speech
        else:
            turns.append(
                ConversationalTurnSummary(
                    turn_index=len(turns),
                    speaker_id=cur_speaker,
                    start=cur_start,
                    end=cur_end,
                    text=" ".join(cur_words),
                    word_count=len(cur_words),
                    has_overlap=cur_has_ovl,
                )
            )
            cur_speaker = w.original_speaker_id
            cur_start = w.start
            cur_end = w.end
            cur_words = [w.word]
            cur_has_ovl = w.is_overlapping_speech

    if cur_speaker is not None and cur_words:
        turns.append(
            ConversationalTurnSummary(
                turn_index=len(turns),
                speaker_id=cur_speaker,
                start=cur_start,
                end=cur_end,
                text=" ".join(cur_words),
                word_count=len(cur_words),
                has_overlap=cur_has_ovl,
            )
        )

    return turns
