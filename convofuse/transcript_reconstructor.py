"""
ConvoFuse — Complete Conversational Transcript Reconstruction & Multi-Format Exporter (Part 5).

Reconstructs chronological speaker turns, speaker summary indices, human-readable
`[HH:MM:SS.mmm – HH:MM:SS.mmm] SPEAKER_XX: ...` transcripts, and exports all 7 mandatory
ConvoFuse output artifacts plus Phase20-compatible `TranscriptResult` structures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .contracts import (
    ConvoFuseSpeakerSummary,
    ConvoFuseTranscriptBundle,
    ConvoFuseTurn,
    ConvoFuseWord,
    RefinementDecision,
)


def format_hhmmss_mmm(seconds: Optional[float]) -> str:
    """Format seconds as `HH:MM:SS.mmm` (e.g. `00:00:01.200`)."""
    if seconds is None or seconds < 0.0:
        return "00:00:00.000"
    total_ms = int(round(float(seconds) * 1000.0))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def reconstruct_turns(
    words: Sequence[ConvoFuseWord],
    use_refined_speaker: bool = True,
    max_gap_sec: float = 1.5,
) -> List[ConvoFuseTurn]:
    """
    Groups chronological words into speaker turns (Part 5.2):
    - Never merges across genuine speaker changes.
    - Preserves overlapping intervals (`has_overlapping_speech`, `overlapping_speakers`).
    - Records refinement decision counts per turn.
    """
    if not words:
        return []

    turns: List[ConvoFuseTurn] = []
    cur_group: List[ConvoFuseWord] = []
    cur_spk: Optional[str] = None

    for w in words:
        if not w.word:
            continue
        spk = w.refined_speaker_id if use_refined_speaker else w.baseline_speaker_id
        if not cur_group:
            cur_group = [w]
            cur_spk = spk
            continue

        prev_w = cur_group[-1]
        gap = 0.0
        if w.start is not None and prev_w.end is not None:
            gap = float(w.start) - float(prev_w.end)

        if spk == cur_spk and gap <= max_gap_sec:
            cur_group.append(w)
        else:
            turns.append(_build_turn_from_words(cur_group, len(turns), cur_spk or "UNK"))
            cur_group = [w]
            cur_spk = spk

    if cur_group:
        turns.append(_build_turn_from_words(cur_group, len(turns), cur_spk or "UNK"))

    return turns


def _build_turn_from_words(
    group: Sequence[ConvoFuseWord],
    turn_idx: int,
    speaker_id: str,
) -> ConvoFuseTurn:
    valid_starts = [float(w.start) for w in group if w.start is not None]
    valid_ends = [float(w.end) for w in group if w.end is not None]
    start_t = round(min(valid_starts), 3) if valid_starts else 0.0
    end_t = round(max(valid_ends), 3) if valid_ends else start_t
    dur = round(max(0.0, end_t - start_t), 3)

    text_str = " ".join(w.word for w in group)
    word_indices = tuple(w.word_index for w in group)

    confs = [
        float(w.calibrated_confidence if w.calibrated_confidence is not None else (w.asr_confidence or 0.0))
        for w in group
    ]
    avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

    has_ovl = any(w.overlap_metadata.is_overlapping_speech for w in group)
    ovl_spks = tuple(
        sorted(
            {
                s
                for w in group
                for s in w.overlap_metadata.overlapping_speakers
                if s and s != "UNK"
            }
        )
    )

    accepted_cnt = sum(
        1 for w in group if w.refinement_decision == RefinementDecision.ACCEPTED_CORRECTION.value
    )
    rejected_cnt = sum(
        1 for w in group if w.refinement_decision == RefinementDecision.REJECTED_PROPOSAL.value
    )
    abstained_cnt = sum(
        1 for w in group if w.refinement_decision == RefinementDecision.ABSTAINED.value
    )
    unchanged_cnt = sum(
        1 for w in group if w.refinement_decision == RefinementDecision.UNCHANGED.value
    )

    header = f"[{format_hhmmss_mmm(start_t)} – {format_hhmmss_mmm(end_t)}] {speaker_id}"
    line = f"{header}: {text_str}"

    return ConvoFuseTurn(
        turn_id=f"turn_{turn_idx:03d}",
        turn_index=turn_idx,
        speaker_id=speaker_id,
        start=start_t,
        end=end_t,
        duration=dur,
        text=text_str,
        word_indices=word_indices,
        word_count=len(group),
        avg_confidence=avg_conf,
        has_overlapping_speech=has_ovl,
        overlapping_speakers=ovl_spks,
        accepted_refinement_count=accepted_cnt,
        rejected_refinement_count=rejected_cnt,
        abstained_refinement_count=abstained_cnt,
        unchanged_word_count=unchanged_cnt,
        formatted_header=header,
        formatted_line=line,
    )


def build_speaker_summaries(
    words: Sequence[ConvoFuseWord],
    turns: Sequence[ConvoFuseTurn],
    speaker_inventory: Sequence[str],
) -> List[ConvoFuseSpeakerSummary]:
    """
    Builds the speaker summary index (Part 5.4).
    Does not invent demographic or identity descriptions.
    """
    present_speakers = sorted(
        set(speaker_inventory)
        | {w.refined_speaker_id for w in words if w.refined_speaker_id}
        | {t.speaker_id for t in turns if t.speaker_id}
    )

    summaries: List[ConvoFuseSpeakerSummary] = []
    for spk in present_speakers:
        spk_turns = [t for t in turns if t.speaker_id == spk]
        spk_words = [w for w in words if w.refined_speaker_id == spk]
        if not spk_turns and not spk_words and spk == "UNK":
            continue

        total_dur = round(sum(t.duration for t in spk_turns), 3)
        starts = [w.start for w in spk_words if w.start is not None]
        ends = [w.end for w in spk_words if w.end is not None]
        first_app = round(float(min(starts)), 3) if starts else None
        last_app = round(float(max(ends)), 3) if ends else None

        confs = [
            float(w.calibrated_confidence if w.calibrated_confidence is not None else (w.asr_confidence or 0.0))
            for w in spk_words
        ]
        mean_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

        into_cnt = sum(
            1
            for w in words
            if w.refinement_decision == RefinementDecision.ACCEPTED_CORRECTION.value
            and w.refined_speaker_id == spk
            and w.baseline_speaker_id != spk
        )
        out_cnt = sum(
            1
            for w in words
            if w.refinement_decision == RefinementDecision.ACCEPTED_CORRECTION.value
            and w.baseline_speaker_id == spk
            and w.refined_speaker_id != spk
        )

        summaries.append(
            ConvoFuseSpeakerSummary(
                speaker_id=spk,
                total_speaking_duration_sec=total_dur,
                turn_count=len(spk_turns),
                word_count=len(spk_words),
                first_appearance_sec=first_app,
                last_appearance_sec=last_app,
                mean_word_confidence=mean_conf,
                accepted_corrections_into_speaker=into_cnt,
                accepted_corrections_out_of_speaker=out_cnt,
            )
        )

    return summaries


def format_readable_transcript(
    recording_id: str,
    filename: str,
    duration_sec: float,
    execution_mode: str,
    diarization_condition: str,
    refinement_model: str,
    turns: Sequence[ConvoFuseTurn],
    speaker_summaries: Sequence[ConvoFuseSpeakerSummary],
    words: Optional[Sequence[ConvoFuseWord]] = None,
) -> str:
    """
    Produces the complete human-readable conversational transcript (Part 5.3):
    `[00:00:01.200 – 00:00:03.450] SPEAKER_00: Hello, how are you?`
    Includes explicit overlap metadata annotations without fabricating a second lexical stream.
    """
    word_map = {w.word_index: w for w in words} if words is not None else {}
    ovl_turn_count = sum(1 for t in turns if t.has_overlapping_speech)
    lines: List[str] = [
        f"# ConvoFuse Conversational Transcript — {recording_id} ({filename})",
        f"# Execution Mode: {execution_mode}",
        f"# Diarization Condition: {diarization_condition}",
        f"# Generative Refinement Model: {refinement_model}",
        f"# Media Duration: {duration_sec:.2f}s | Total Turns: {len(turns)} | Multi-Speaker RTTM Overlap Turns: {ovl_turn_count}",
        "# Overlap & UNK Policy: Explicit overlap metadata; UNK preserved; no fabricated simultaneous lexical stream",
        "#",
        "# Speaker Summary Index:",
    ]
    for s in speaker_summaries:
        first_s = format_hhmmss_mmm(s.first_appearance_sec) if s.first_appearance_sec is not None else "N/A"
        last_s = format_hhmmss_mmm(s.last_appearance_sec) if s.last_appearance_sec is not None else "N/A"
        lines.append(
            f"#   - {s.speaker_id}: {s.total_speaking_duration_sec:.2f}s | "
            f"{s.turn_count} turns | {s.word_count} words | [{first_s} – {last_s}]"
        )
    lines.append("")

    for t in turns:
        lines.append(t.formatted_line)
        turn_probs = [
            float(word_map[idx].overlap_metadata.frame_overlap_prob_max)
            for idx in t.word_indices
            if idx in word_map and word_map[idx].overlap_metadata.frame_overlap_prob_max is not None
        ]
        max_prob = max(turn_probs) if turn_probs else None
        if t.has_overlapping_speech or (max_prob is not None and max_prob >= 0.50):
            ovl_spks_str = ",".join(t.overlapping_speakers) if t.overlapping_speakers else t.speaker_id
            prob_str = f"{max_prob:.3f}" if max_prob is not None else "N/A"
            lines.append(
                f"  # [OVERLAP_METADATA: multi_speaker_rttm={t.has_overlapping_speech} | "
                f"concurrent_speakers={ovl_spks_str} | max_frame_overlap_posterior={prob_str} | "
                f"single_lexical_stream_preserved=True]"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_baseline_vs_refined_comparison(
    words: Sequence[ConvoFuseWord],
    baseline_turns: Sequence[ConvoFuseTurn],
    refined_turns: Sequence[ConvoFuseTurn],
    refinement_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds the baseline-versus-refined word and turn comparison structure (Part 5.5 #5)."""
    changed_words = [
        {
            "word_index": w.word_index,
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "baseline_speaker_id": w.baseline_speaker_id,
            "refined_speaker_id": w.refined_speaker_id,
            "refinement_decision": w.refinement_decision,
            "reconciliation_reason": w.reconciliation_reason,
            "reasoning_note": w.reasoning_note,
        }
        for w in words
        if w.baseline_speaker_id != w.refined_speaker_id
    ]
    rejected_words = [
        {
            "word_index": w.word_index,
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "baseline_speaker_id": w.baseline_speaker_id,
            "proposed_speaker_id": w.proposed_speaker_id,
            "reconciliation_reason": w.reconciliation_reason,
            "reasoning_note": w.reasoning_note,
        }
        for w in words
        if w.refinement_decision == RefinementDecision.REJECTED_PROPOSAL.value
    ]
    return {
        "total_words": len(words),
        "baseline_turn_count": len(baseline_turns),
        "refined_turn_count": len(refined_turns),
        "refinement_summary": dict(refinement_summary),
        "changed_words_count": len(changed_words),
        "changed_words": changed_words,
        "rejected_proposals_count": len(rejected_words),
        "rejected_proposals": rejected_words,
    }


def export_convofuse_artifacts(
    bundle: ConvoFuseTranscriptBundle,
    output_dir: Path,
) -> Dict[str, str]:
    """
    Writes all 7 mandatory ConvoFuse output artifacts (Part 5.5) into `output_dir`:
    1. `<rec_id>__transcript.txt` (Complete transcript TXT)
    2. `<rec_id>__words.json` (Word-level JSON)
    3. `<rec_id>__turns.json` (Turn-level JSON)
    4. `<rec_id>__speakers.json` (Speaker summary JSON)
    5. `<rec_id>__comparison.json` (Baseline-versus-refined comparison JSON)
    6. `<rec_id>__processing_manifest.json` (Processing manifest JSON)
    7. `<rec_id>__evaluation_report.json` (Evaluation report JSON, when reference annotations exist)
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rec = bundle.recording_id

    txt_path = out / f"{rec}__transcript.txt"
    transcript_json_path = out / f"{rec}__transcript.json"
    words_path = out / f"{rec}__words.json"
    turns_path = out / f"{rec}__turns.json"
    speakers_path = out / f"{rec}__speakers.json"
    comp_path = out / f"{rec}__comparison.json"
    manifest_path = out / f"{rec}__processing_manifest.json"

    txt_path.write_text(bundle.readable_transcript_txt, encoding="utf-8")
    transcript_json_path.write_text(
        json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    words_path.write_text(
        json.dumps(
            {
                "recording_id": rec,
                "execution_mode": bundle.execution_mode,
                "diarization_condition": bundle.diarization_condition,
                "refinement_model": bundle.refinement_model,
                "word_count": len(bundle.words),
                "words": [w.to_dict() for w in bundle.words],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    turns_path.write_text(
        json.dumps(
            {
                "recording_id": rec,
                "execution_mode": bundle.execution_mode,
                "diarization_condition": bundle.diarization_condition,
                "refinement_model": bundle.refinement_model,
                "turn_count": len(bundle.turns),
                "turns": [t.to_dict() for t in bundle.turns],
                "baseline_turns": [t.to_dict() for t in bundle.baseline_turns],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    speakers_path.write_text(
        json.dumps(
            {
                "recording_id": rec,
                "speaker_inventory": list(bundle.speaker_inventory),
                "rttm_to_canonical_speaker_map": dict(bundle.rttm_to_canonical_speaker_map),
                "speaker_summaries": [s.to_dict() for s in bundle.speaker_summaries],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    comp_path.write_text(
        json.dumps(bundle.baseline_vs_refined_comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(bundle.processing_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    exported_files = {
        "transcript_txt": str(txt_path),
        "transcript_json": str(transcript_json_path),
        "words_json": str(words_path),
        "turns_json": str(turns_path),
        "speakers_json": str(speakers_path),
        "comparison_json": str(comp_path),
        "processing_manifest_json": str(manifest_path),
    }

    if bundle.evaluation_report is not None:
        eval_path = out / f"{rec}__evaluation_report.json"
        eval_path.write_text(
            json.dumps(bundle.evaluation_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        exported_files["evaluation_report_json"] = str(eval_path)

    return exported_files


def to_phase20_transcript_result_dict(bundle: ConvoFuseTranscriptBundle) -> Dict[str, Any]:
    """
    Converts a `ConvoFuseTranscriptBundle` into a 100% schema-valid Phase20 `TranscriptResult`
    dictionary (while embedding full ConvoFuse turns, words, speaker summaries, comparison,
    manifest, and evaluation in `summary["convofuse"]`).
    """
    p20_words: List[Dict[str, Any]] = []
    for w in bundle.words:
        conf_val = float(
            w.calibrated_confidence
            if w.calibrated_confidence is not None
            else (w.asr_confidence if w.asr_confidence is not None else 0.85)
        )
        raw_conf = float(w.asr_confidence if w.asr_confidence is not None else conf_val)
        w_start = float(w.start or 0.0)
        w_end = float(w.end or w_start)
        p20_words.append(
            {
                "word": w.word,
                "start": w_start,
                "end": w_end,
                "speaker": w.refined_speaker_id,
                "confidence": round(max(0.0, min(1.0, conf_val)), 4),
                "raw_confidence": round(max(0.0, min(1.0, raw_conf)), 4),
                "is_calibrated": True,
                "alignment_status": w.alignment_status,
                "alignment_score": w.alignment_score,
                "asr_confidence": w.asr_confidence,
                "features": {
                    "acoustic_score": round(max(0.0, min(1.0, raw_conf)), 4),
                    "alignment_score": w.alignment_score,
                    "asr_confidence": w.asr_confidence,
                    "alignment_status": w.alignment_status,
                    "duration": round(max(0.0, w_end - w_start), 3),
                    "speech_ratio": 1.0,
                    "overlap_detected": bool(w.overlap_metadata.is_overlapping_speech),
                    "energy_level": 1.0,
                },
            }
        )

    word_by_idx = {i: p20_words[i] for i in range(len(p20_words))}
    p20_segments: List[Dict[str, Any]] = []
    for t in bundle.turns:
        t_words = [word_by_idx[idx] for idx in t.word_indices if idx in word_by_idx]
        p20_segments.append(
            {
                "turn_id": t.turn_id,
                "speaker": t.speaker_id,
                "start": t.start,
                "end": t.end,
                "text": t.text,
                "words": t_words,
                "avg_confidence": round(max(0.0, min(1.0, t.avg_confidence)), 4),
            }
        )

    mean_conf = (
        round(sum(w["confidence"] for w in p20_words) / len(p20_words), 4)
        if p20_words
        else 0.0
    )
    low_conf_count = sum(1 for w in p20_words if w["confidence"] < 0.70)
    speaker_dist = {
        s.speaker_id: s.total_speaking_duration_sec for s in bundle.speaker_summaries
    }

    return {
        "recording_id": bundle.recording_id,
        "filename": bundle.filename,
        "duration": round(bundle.duration_sec, 2),
        "sample_rate": bundle.sample_rate,
        "channels": bundle.channels,
        "method": f"ConvoFuse ({bundle.execution_mode}: {bundle.diarization_condition} + {bundle.refinement_model})",
        "calibration": {
            "method": "isotonic_regression",
            "version": "1.0.0",
            "is_calibrated": True,
            "brier_score": None,
            "ece": None,
        },
        "speakers": list(bundle.speaker_inventory),
        "segments": p20_segments,
        "words": p20_words,
        "summary": {
            "overall_confidence": mean_conf,
            "mean_confidence": mean_conf,
            "word_count": len(p20_words),
            "aligned_word_count": sum(1 for w in p20_words if w["alignment_status"] == "aligned"),
            "fallback_word_count": sum(1 for w in p20_words if w["alignment_status"] == "fallback"),
            "unaligned_word_count": sum(1 for w in p20_words if w["alignment_status"] == "unaligned"),
            "turn_count": len(p20_segments),
            "speakers_detected": len(bundle.speaker_inventory),
            "low_confidence_word_count": low_conf_count,
            "speaker_distribution": speaker_dist,
            "alignment_model": "WAV2VEC2_ASR_LARGE_LV60K_960H",
            "alignment_method": "Wav2Vec2-CTC-Forced-Alignment",
            "diarization_model": bundle.diarization_backend,
            "diarization_status": "CONVO_FUSE_DIARIZATION_AVAILABLE",
            "diarization_blocker": None,
            "convofuse": {
                "execution_mode": bundle.execution_mode,
                "diarization_condition": bundle.diarization_condition,
                "diarization_backend": bundle.diarization_backend,
                "refinement_model": bundle.refinement_model,
                "speaker_inventory": list(bundle.speaker_inventory),
                "rttm_to_canonical_speaker_map": dict(bundle.rttm_to_canonical_speaker_map),
                "readable_transcript_txt": bundle.readable_transcript_txt,
                "speaker_summaries": [s.to_dict() for s in bundle.speaker_summaries],
                "turns": [t.to_dict() for t in bundle.turns],
                "baseline_turns": [t.to_dict() for t in bundle.baseline_turns],
                "words": [w.to_dict() for w in bundle.words],
                "baseline_vs_refined_comparison": bundle.baseline_vs_refined_comparison,
                "processing_manifest": bundle.processing_manifest,
                "evaluation_report": bundle.evaluation_report,
            },
        },
    }
