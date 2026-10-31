#!/usr/bin/env python3
"""
Stage 1 — Non-Destructive Data, Audio, and Annotation Preparation.

Strictly preserves all original files in raw_audio/ and ground_truth/.
Generates validated 16 kHz mono and loudness-normalized WAVs, versioned
annotations (full ledger + RTTM + JSON), and recording/speaker/segment quality reports
inside research_stages/stage1_data_prep/.
"""

import csv
import hashlib
import json
import math
import re
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path("/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning")
STAGE1_DIR = REPO_ROOT / "research_stages" / "stage1_data_prep"

CANONICAL_RECORDINGS = [
    ("R01_sara_sharif", "Bodycam footage show arrests of Sara Sharif’s alleged killers"),
    ("R02_dennis_skinner", "Dennis Skinner kicked out of Commons"),
    ("R03_disabled_parking", "Karen Has A Problem With Disabled Parking"),
    ("R04_piers_robinson", "Piers Confronts Tommy Robinson"),
    ("R05_southwest_pilot", "Police bodycam shows arrest of Southwest Airlines pilot"),
    ("R06_rich_karen", "Rich Karen Messes Around & Finds Out"),
    ("R07_bwc_cheap_shot", "That was a bit of a cheap shot weren’t it mate"),
    ("R08_uk_parliament", "UK Parliament Debate"),
]

MISSING_AUDIO_RECORDINGS = [
    ("M01_drunk_karen", "Drunk Karen Gets a Huge Reality Check"),
    ("M02_entitled_restaurant", "Entitled Karen Refuses to Pay Restaurant Bill Then Attacks the Manager"),
    ("M03_entitled_girl", "When an Entitled Girl Gets Instantly Humbled"),
    ("M04_traffic_stop", "Woman's Attitude Makes Traffic Stop 10x Worse"),
]

# Strict Phase 20 regex (from phase20_standalone/scripts/step00_ground_truth.py L23-24)
HEADER_STRICT = re.compile(
    r"^Speaker(?:_|\s+)([A-Za-z0-9_]+)\s*\[(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2})\](?::\s*(.*)|\s+(.*))?$"
)

# Versioned Stage 1 regex supporting Unicode en-dash/em-dash and 1-digit hours
HEADER_STAGE1 = re.compile(
    r"^Speaker(?:_|\s+)([A-Za-z0-9_]+)\s*\[(\d{1,2}:\d{2}:\d{2})\s*([\-\u2013\u2014])\s*(\d{1,2}:\d{2}:\d{2})\](?::\s*(.*)|\s+(.*))?$"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ts_to_sec(ts: str) -> float:
    parts = [int(p) for p in ts.strip().split(":")]
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])


def probe_audio_file(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bits_per_sample:format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(res.stdout)
    stream = payload["streams"][0]
    fmt = payload["format"]
    return {
        "codec_name": stream.get("codec_name", ""),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": int(stream.get("channels", 0)),
        "duration_sec": float(fmt.get("duration", 0.0)),
        "size_bytes": int(fmt.get("size", 0)),
    }


def validate_wav_signal(wav_path: Path, expected_dur: float) -> Dict[str, Any]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw_bytes = wf.readframes(nframes)

    dur = nframes / float(sr) if sr > 0 else 0.0
    samples_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
    samples_f32 = samples_i16.astype(np.float32) / 32768.0

    nan_count = int(np.isnan(samples_f32).sum())
    inf_count = int(np.isinf(samples_f32).sum())
    abs_samples = np.abs(samples_f32)
    peak = float(abs_samples.max()) if samples_f32.size > 0 else 0.0
    clip_count = int((np.abs(samples_i16) >= 32767).sum())
    clip_fraction = float(clip_count / samples_f32.size) if samples_f32.size > 0 else 0.0
    rms = float(np.sqrt(np.mean(samples_f32 ** 2))) if samples_f32.size > 0 else 0.0
    rms_dbfs = float(20.0 * math.log10(max(rms, 1e-12)))
    dur_diff = abs(dur - expected_dur)

    valid = (
        sr == 16000
        and channels == 1
        and sampwidth == 2
        and nan_count == 0
        and inf_count == 0
        and dur_diff <= 0.05
    )

    return {
        "wav_path": str(wav_path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(wav_path),
        "sample_rate": sr,
        "channels": channels,
        "sample_width_bytes": sampwidth,
        "num_samples": int(samples_f32.size),
        "duration_sec": round(dur, 4),
        "source_duration_sec": round(expected_dur, 4),
        "duration_diff_sec": round(dur_diff, 5),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "peak_amplitude": round(peak, 6),
        "clipped_sample_count": clip_count,
        "clipped_fraction": round(clip_fraction, 8),
        "rms_dbfs": round(rms_dbfs, 3),
        "validation_passed": valid,
    }


def apply_legacy_bwc_curation(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exact replica of step00_ground_truth.py::apply_legacy_reference_curation (L38-72) for R07 comparison."""
    dropped = {
        (19.0, 20.0, "Speaker_2", "yeah yeah"),
        (84.0, 85.0, "Speaker_2", "Yeah they are hard and square"),
        (105.0, 106.0, "Speaker_1", "Oi"),
        (110.0, 110.0, "Speaker_2", "i'm on the floor"),
        (127.0, 127.0, "Speaker_3", "yeah"),
        (137.0, 138.0, "Speaker_1", "Alright"),
    }
    curated: List[Dict[str, Any]] = []
    for seg in segments:
        key = (
            float(seg["start"]),
            float(seg["end"]),
            str(seg["speaker"]),
            str(seg["text"]),
        )
        if key in dropped:
            continue
        cur = dict(seg)
        if cur["start"] == 134.0 and cur["end"] == 135.0 and cur["text"] == "Ah you scumbag":
            cur["speaker"] = "Speaker_4"
            cur["global_speaker_id"] = f"{cur['rec_id']}::Speaker_4"
        elif cur["start"] == 149.0 and cur["end"] == 149.0 and cur["text"] == "You alright?":
            cur["speaker"] = "Speaker_3"
            cur["global_speaker_id"] = f"{cur['rec_id']}::Speaker_3"
        elif cur["start"] == 150.0 and cur["end"] == 158.0:
            cur["text"] = "Yeah. You did get me, I'll give you that mate. I honestly don't know what you were thinking then"
            cur["words"] = len(cur["text"].split())
        curated.append(cur)
    return curated


def parse_annotation_file(
    rec_id: str,
    stem: str,
    gt_path: Path,
    audio_dur: Optional[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Non-destructively parses a ground_truth/<stem>.txt file, recording every segment
    (including zero-duration, negative-duration, overlapping, and collective-label segments)
    along with explicit flags and reasons.
    """
    lines = gt_path.read_text(encoding="utf-8").splitlines()
    raw_segments: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []

    i = 0
    seg_idx = 0
    while i < len(lines):
        raw_line = lines[i]
        stripped = raw_line.strip()
        if not stripped:
            i += 1
            continue

        m_stage1 = HEADER_STAGE1.match(stripped)
        m_strict = HEADER_STRICT.match(stripped)

        if m_stage1:
            spk_suffix, st_ts, delim, en_ts, inline1, inline2 = m_stage1.groups()
            inline_text = (inline1 or inline2 or "").strip()
            spk_id = f"Speaker_{spk_suffix}"
            st_sec = ts_to_sec(st_ts)
            en_sec = ts_to_sec(en_ts)
            dur = en_sec - st_sec
            header_line = i + 1

            flags: List[str] = []
            if delim != "-":
                flags.append(f"UNICODE_DELIMITER_U+{ord(delim):04X}")
                anomalies.append({
                    "rec_id": rec_id,
                    "stem": stem,
                    "line": header_line,
                    "seg_idx": seg_idx,
                    "speaker": spk_id,
                    "type": "UNICODE_ENDASH_DELIMITER",
                    "action": "IN_MEMORY_NORMALIZED_TO_HYPHEN",
                    "reason": f"Delimiter '{delim}' (U+{ord(delim):04X}) normalized in-memory to '-'",
                })
            if len(st_ts.split(":")[0]) != 2 or len(en_ts.split(":")[0]) != 2:
                flags.append("SINGLE_DIGIT_HOUR_NORMALIZED")
                anomalies.append({
                    "rec_id": rec_id,
                    "stem": stem,
                    "line": header_line,
                    "seg_idx": seg_idx,
                    "speaker": spk_id,
                    "type": "SINGLE_DIGIT_HOUR",
                    "action": "IN_MEMORY_ZERO_PADDED_HOUR",
                    "reason": f"Timestamp [{st_ts} - {en_ts}] zero-padded to HH:MM:SS in-memory",
                })
            if spk_suffix == "all":
                flags.append("COLLECTIVE_SPEAKER_ALL")
                anomalies.append({
                    "rec_id": rec_id,
                    "stem": stem,
                    "line": header_line,
                    "seg_idx": seg_idx,
                    "speaker": spk_id,
                    "type": "COLLECTIVE_SPEAKER_LABEL",
                    "action": "RETAINED_IN_LEDGER_FLAGGED_AMBIGUOUS",
                    "reason": "Label 'Speaker_all' denotes chorus/crowd speech, not a single speaker identity",
                })

            i += 1
            text_lines: List[str] = []
            if inline_text:
                text_lines.append(inline_text)
                if inline2 and not inline1:
                    flags.append("MISSING_COLON_INLINE_HEADER")
            while i < len(lines) and not HEADER_STAGE1.match(lines[i].strip()):
                if lines[i].strip():
                    text_lines.append(lines[i].strip())
                i += 1

            if len(text_lines) > 1:
                flags.append(f"MULTI_PARAGRAPH_MERGED_{len(text_lines)}_LINES")
                anomalies.append({
                    "rec_id": rec_id,
                    "stem": stem,
                    "line": header_line,
                    "seg_idx": seg_idx,
                    "speaker": spk_id,
                    "type": "MULTI_PARAGRAPH_BLOCK",
                    "action": "MERGED_LINES_WITH_FLAG",
                    "reason": f"{len(text_lines)} consecutive text paragraphs under single {dur:.1f}s header",
                })

            full_text = " ".join(text_lines).strip()
            if "*" in full_text or "[ __ ]" in full_text:
                flags.append("CONTAINS_CENSORED_TOKEN")

            if dur < 0:
                flags.append("NEGATIVE_DURATION_MALFORMED")
                anomalies.append({
                    "rec_id": rec_id,
                    "stem": stem,
                    "line": header_line,
                    "seg_idx": seg_idx,
                    "speaker": spk_id,
                    "type": "NEGATIVE_DURATION",
                    "action": "EXCLUDED_FROM_RTTM_KEPT_IN_LEDGER",
                    "reason": f"End ({en_sec}s) < Start ({st_sec}s), duration={dur}s",
                })
            elif dur == 0:
                flags.append("ZERO_DURATION_1S_ROUNDING")
                anomalies.append({
                    "rec_id": rec_id,
                    "stem": stem,
                    "line": header_line,
                    "seg_idx": seg_idx,
                    "speaker": spk_id,
                    "type": "ZERO_DURATION_SEGMENT",
                    "action": "OMITTED_FROM_RTTM_RETAINED_IN_JSON_AND_LEDGER",
                    "reason": f"Start == End ({st_sec}s) due to 1-second timestamp rounding ({len(full_text.split())} words)",
                })

            clipped_end = en_sec
            if audio_dur is not None and en_sec > audio_dur:
                flags.append(f"END_EXCEEDS_AUDIO_BY_{en_sec - audio_dur:.3f}S")
                anomalies.append({
                    "rec_id": rec_id,
                    "stem": stem,
                    "line": header_line,
                    "seg_idx": seg_idx,
                    "speaker": spk_id,
                    "type": "TIMESTAMP_EXCEEDS_AUDIO_DURATION",
                    "action": "RECORDED_UNCLIPPED_AND_CLIPPED_BOUNDS",
                    "reason": f"Annotated end {en_sec}s exceeds physical audio duration {audio_dur:.3f}s",
                })
                clipped_end = max(st_sec, audio_dur)

            if raw_segments:
                prev = raw_segments[-1]
                if st_sec == prev["start"] and en_sec == prev["end"]:
                    flags.append("IDENTICAL_TIMESTAMP_TO_PREV")
                elif st_sec < prev["start"]:
                    flags.append("NON_MONOTONIC_START")
                    anomalies.append({
                        "rec_id": rec_id,
                        "stem": stem,
                        "line": header_line,
                        "seg_idx": seg_idx,
                        "speaker": spk_id,
                        "type": "NON_MONOTONIC_START_TIME",
                        "action": "SORTED_FOR_RTTM_PRESERVED_IN_LEDGER",
                        "reason": f"Start {st_sec}s < previous segment start {prev['start']}s (line {prev['line']})",
                    })

            raw_segments.append({
                "rec_id": rec_id,
                "stem": stem,
                "seg_idx": seg_idx,
                "line": header_line,
                "speaker": spk_id,
                "global_speaker_id": f"{rec_id}::{spk_id}",
                "start": st_sec,
                "end": en_sec,
                "end_clipped_to_audio": round(clipped_end, 4),
                "duration": dur,
                "words": len(full_text.split()),
                "text": full_text,
                "strict_step00_compatible": m_strict is not None and dur >= 0,
                "included_in_rttm": dur > 0,
                "flags": flags,
            })
            seg_idx += 1
        else:
            anomalies.append({
                "rec_id": rec_id,
                "stem": stem,
                "line": i + 1,
                "seg_idx": -1,
                "speaker": "UNKNOWN",
                "type": "UNMATCHED_LINE",
                "action": "SKIPPED_WITH_EXPLICIT_LOG",
                "reason": f"Unmatched line outside header block: {stripped[:80]}",
            })
            i += 1

    # Compute exact overlap intervals for each positive-duration segment
    pos_indices = [idx for idx, s in enumerate(raw_segments) if s["duration"] > 0]
    for a in pos_indices:
        sa = raw_segments[a]
        ov_partners = []
        for b in pos_indices:
            if a == b:
                continue
            sb = raw_segments[b]
            ov = min(sa["end"], sb["end"]) - max(sa["start"], sb["start"])
            if ov > 0:
                ov_partners.append((sb["seg_idx"], sb["speaker"], ov))
        sa["has_gt_overlap"] = len(ov_partners) > 0
        sa["overlap_partner_count"] = len(ov_partners)
        if ov_partners:
            sa["flags"].append(f"OVERLAPS_WITH_{len(ov_partners)}_SEGS")
        # Determine eligibility for clean single-speaker embedding evaluation
        sa["usable_for_clean_speaker_eval"] = (
            sa["duration"] > 0
            and not sa["has_gt_overlap"]
            and sa["speaker"] != "Speaker_all"
            and audio_dur is not None
        )
        if sa["usable_for_clean_speaker_eval"]:
            sa["eval_status_reason"] = "VALID_CLEAN_POSITIVE_SEGMENT"
        elif audio_dur is None:
            sa["eval_status_reason"] = "EXCLUDED_NO_SOURCE_AUDIO"
        elif sa["speaker"] == "Speaker_all":
            sa["eval_status_reason"] = "AMBIGUOUS_COLLECTIVE_SPEAKER_ALL"
        elif sa["has_gt_overlap"]:
            sa["eval_status_reason"] = "AMBIGUOUS_MULTI_SPEAKER_GT_OVERLAP"
        else:
            sa["eval_status_reason"] = "OTHER"

    for s in raw_segments:
        if s["duration"] <= 0:
            s["has_gt_overlap"] = False
            s["overlap_partner_count"] = 0
            s["usable_for_clean_speaker_eval"] = False
            s["eval_status_reason"] = (
                "ZERO_DURATION_1S_ROUNDING" if s["duration"] == 0 else "MALFORMED_NEGATIVE_DURATION"
            )

    return raw_segments, anomalies


def write_rttm_file(rttm_path: Path, rec_uri: str, segments: List[Dict[str, Any]]) -> int:
    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        [s for s in segments if s["duration"] > 0],
        key=lambda x: (float(x["start"]), float(x["end"]), str(x["speaker"])),
    )
    lines = []
    for seg in ordered:
        lines.append(
            f"SPEAKER {rec_uri} 1 {seg['start']:.3f} {seg['duration']:.3f} <NA> <NA> {seg['speaker']} <NA> <NA>"
        )
    rttm_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> None:
    for sub in ["audio", "annotations_json", "annotations_rttm", "manifests", "reports"]:
        (STAGE1_DIR / sub).mkdir(parents=True, exist_ok=True)

    recording_manifest: List[Dict[str, Any]] = []
    audio_validations: List[Dict[str, Any]] = []
    all_segments_ledger: List[Dict[str, Any]] = []
    all_anomalies: List[Dict[str, Any]] = []
    recording_quality_rows: List[Dict[str, Any]] = []
    speaker_quality_rows: List[Dict[str, Any]] = []

    # 1. Process the 8 audio-available recordings
    for rec_id, stem in CANONICAL_RECORDINGS:
        mp3_path = REPO_ROOT / "raw_audio" / f"{stem}.mp3"
        gt_path = REPO_ROOT / "ground_truth" / f"{stem}.txt"
        assert mp3_path.exists(), f"Missing MP3: {mp3_path}"
        assert gt_path.exists(), f"Missing GT: {gt_path}"

        mp3_info = probe_audio_file(mp3_path)
        rec_audio_dir = STAGE1_DIR / "audio" / rec_id
        rec_audio_dir.mkdir(parents=True, exist_ok=True)

        mono_wav = rec_audio_dir / f"{rec_id}_16k_mono.wav"
        norm_wav = rec_audio_dir / f"{rec_id}_normalized.wav"

        if not mono_wav.exists():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(mp3_path),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(mono_wav),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if not norm_wav.exists():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(mono_wav),
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                    str(norm_wav),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        val_mono = validate_wav_signal(mono_wav, mp3_info["duration_sec"])
        val_mono["rec_id"] = rec_id
        val_mono["audio_variant"] = "16k_mono"
        val_norm = validate_wav_signal(norm_wav, mp3_info["duration_sec"])
        val_norm["rec_id"] = rec_id
        val_norm["audio_variant"] = "normalized"
        audio_validations.extend([val_mono, val_norm])

        # Parse annotations non-destructively
        segs, anoms = parse_annotation_file(rec_id, stem, gt_path, val_mono["duration_sec"])
        all_segments_ledger.extend(segs)
        all_anomalies.extend(anoms)

        # Write uniform policy JSON and RTTM (preserves all speakers/segments, no bwc-specific mutation)
        json_uniform = STAGE1_DIR / "annotations_json" / f"{rec_id}.json"
        rttm_uniform = STAGE1_DIR / "annotations_rttm" / f"{rec_id}.rttm"
        json_uniform.write_text(
            json.dumps({"rec_id": rec_id, "recording": stem, "policy": "uniform_v1", "segments": segs}, indent=2) + "\n",
            encoding="utf-8",
        )
        rttm_count = write_rttm_file(rttm_uniform, rec_id, segs)

        # For R07_bwc_cheap_shot, also write the exact Phase 20 legacy curated policy for direct comparison
        if rec_id == "R07_bwc_cheap_shot":
            curated_segs = apply_legacy_bwc_curation(segs)
            json_cur = STAGE1_DIR / "annotations_json" / f"{rec_id}_phase20_curated.json"
            rttm_cur = STAGE1_DIR / "annotations_rttm" / f"{rec_id}_phase20_curated.rttm"
            json_cur.write_text(
                json.dumps({"rec_id": rec_id, "recording": stem, "policy": "phase20_legacy_curated", "segments": curated_segs}, indent=2) + "\n",
                encoding="utf-8",
            )
            write_rttm_file(rttm_cur, rec_id, curated_segs)

        pos_segs = [s for s in segs if s["duration"] > 0]
        clean_segs = [s for s in segs if s["usable_for_clean_speaker_eval"]]
        zero_segs = [s for s in segs if s["duration"] == 0]
        spks_all = sorted(set(s["speaker"] for s in segs))
        spks_pos = sorted(set(s["speaker"] for s in pos_segs))

        recording_manifest.append({
            "rec_id": rec_id,
            "original_stem": stem,
            "source_mp3": str(mp3_path.relative_to(REPO_ROOT)),
            "source_mp3_sha256": sha256_file(mp3_path),
            "source_gt_txt": str(gt_path.relative_to(REPO_ROOT)),
            "source_gt_sha256": sha256_file(gt_path),
            "mono_wav": str(mono_wav.relative_to(REPO_ROOT)),
            "norm_wav": str(norm_wav.relative_to(REPO_ROOT)),
            "duration_sec": val_mono["duration_sec"],
            "json_uniform": str(json_uniform.relative_to(REPO_ROOT)),
            "rttm_uniform": str(rttm_uniform.relative_to(REPO_ROOT)),
            "num_speakers_all_gt": len(spks_all),
            "num_speakers_positive_rttm": len(spks_pos),
            "total_segments_gt": len(segs),
            "positive_segments_rttm": rttm_count,
            "zero_duration_segments": len(zero_segs),
            "clean_non_overlapping_segments": len(clean_segs),
        })

    # 2. Also parse the 4 missing-audio GT files into the ledger so nothing is silently omitted
    for rec_id, stem in MISSING_AUDIO_RECORDINGS:
        gt_path = REPO_ROOT / "ground_truth" / f"{stem}.txt"
        segs, anoms = parse_annotation_file(rec_id, stem, gt_path, None)
        all_segments_ledger.extend(segs)
        all_anomalies.extend(anoms)

    # 3. Build recording-level and speaker-level quality tables across all 12 files
    for rec_id, stem in CANONICAL_RECORDINGS + MISSING_AUDIO_RECORDINGS:
        segs = [s for s in all_segments_ledger if s["rec_id"] == rec_id]
        has_audio = rec_id.startswith("R")
        pos_segs = [s for s in segs if s["duration"] > 0]
        zero_segs = [s for s in segs if s["duration"] == 0]
        neg_segs = [s for s in segs if s["duration"] < 0]
        ov_segs = [s for s in pos_segs if s["has_gt_overlap"]]
        clean_segs = [s for s in segs if s["usable_for_clean_speaker_eval"]]
        spks_all = sorted(set(s["speaker"] for s in segs))
        spks_pos = sorted(set(s["speaker"] for s in pos_segs))
        spks_clean = sorted(set(s["speaker"] for s in clean_segs))

        recording_quality_rows.append({
            "rec_id": rec_id,
            "original_stem": stem,
            "has_audio": has_audio,
            "total_segments": len(segs),
            "positive_dur_segments": len(pos_segs),
            "zero_dur_segments": len(zero_segs),
            "neg_dur_segments": len(neg_segs),
            "overlapping_pos_segments": len(ov_segs),
            "clean_single_speaker_segments": len(clean_segs),
            "speakers_annotated_total": len(spks_all),
            "speakers_with_positive_dur": len(spks_pos),
            "speakers_with_zero_dur_only": len(spks_all) - len(spks_pos),
            "speakers_with_clean_segments": len(spks_clean),
            "sum_pos_duration_sec": round(sum(s["duration"] for s in pos_segs), 2),
            "sum_clean_duration_sec": round(sum(s["duration"] for s in clean_segs), 2),
            "anomalies_logged": sum(1 for a in all_anomalies if a["rec_id"] == rec_id),
        })

        for spk in spks_all:
            s_spk = [s for s in segs if s["speaker"] == spk]
            s_pos = [s for s in s_spk if s["duration"] > 0]
            s_clean = [s for s in s_spk if s["usable_for_clean_speaker_eval"]]
            s_clean_1s = [s for s in s_clean if s["duration"] >= 1.0]
            speaker_quality_rows.append({
                "rec_id": rec_id,
                "global_speaker_id": f"{rec_id}::{spk}",
                "speaker": spk,
                "has_audio": has_audio,
                "is_collective_speaker_all": spk == "Speaker_all",
                "total_segments": len(s_spk),
                "positive_dur_segments": len(s_pos),
                "zero_dur_segments": sum(1 for s in s_spk if s["duration"] == 0),
                "neg_dur_segments": sum(1 for s in s_spk if s["duration"] < 0),
                "overlapping_segments": sum(1 for s in s_pos if s["has_gt_overlap"]),
                "clean_single_speaker_segments": len(s_clean),
                "clean_segments_ge_1s": len(s_clean_1s),
                "total_pos_duration_sec": round(sum(s["duration"] for s in s_pos), 2),
                "total_clean_duration_sec": round(sum(s["duration"] for s in s_clean), 2),
                "total_words": sum(s["words"] for s in s_spk),
                "speaker_status": (
                    "COLLECTIVE_CROWD_LABEL" if spk == "Speaker_all"
                    else ("NO_AUDIO_FILE" if not has_audio
                    else ("ZERO_POSITIVE_DURATION_1S_ROUNDING" if len(s_pos) == 0
                    else ("OVERLAP_ONLY_NO_CLEAN_SEGMENTS" if len(s_clean) == 0
                    else ("SINGLE_CLEAN_SEGMENT_ONLY" if len(s_clean) == 1
                    else "VIABLE_MULTI_SEGMENT_SPEAKER"))))
                ),
            })

    # Write manifests and reports
    (STAGE1_DIR / "manifests" / "recording_manifest.json").write_text(
        json.dumps(recording_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (STAGE1_DIR / "reports" / "audio_validation_report.json").write_text(
        json.dumps(audio_validations, indent=2) + "\n", encoding="utf-8"
    )
    (STAGE1_DIR / "reports" / "stage1_anomalies_and_exclusions.json").write_text(
        json.dumps(all_anomalies, indent=2) + "\n", encoding="utf-8"
    )

    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        flat_rows = []
        for r in rows:
            fr = dict(r)
            if "flags" in fr and isinstance(fr["flags"], list):
                fr["flags"] = "|".join(fr["flags"])
            flat_rows.append(fr)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(flat_rows)

    write_csv(STAGE1_DIR / "manifests" / "recording_manifest.csv", recording_manifest)
    write_csv(STAGE1_DIR / "reports" / "audio_validation_report.csv", audio_validations)
    write_csv(STAGE1_DIR / "reports" / "stage1_recording_quality.csv", recording_quality_rows)
    write_csv(STAGE1_DIR / "reports" / "stage1_speaker_quality.csv", speaker_quality_rows)
    write_csv(STAGE1_DIR / "reports" / "stage1_segment_ledger.csv", all_segments_ledger)

    all_passed = all(v["validation_passed"] for v in audio_validations)
    print(f"Stage 1 audio validations: {len(audio_validations)} files checked, all_passed={all_passed}")
    print(f"Stage 1 segment ledger: {len(all_segments_ledger)} total segments recorded")
    print(f"Stage 1 anomalies/exclusions logged: {len(all_anomalies)}")


if __name__ == "__main__":
    main()
