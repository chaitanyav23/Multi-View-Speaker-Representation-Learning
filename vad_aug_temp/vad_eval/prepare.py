from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

from .common import audio_info, parse_noise_snr, relative_path, safe_file_id, write_csv
from .config import AUDIO_DIR, FRAME_SHIFT_SEC, LABEL_DIR, METADATA_PATH, RAW_DATA, SAMPLE_RATE

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".sph", ".ogg", ".m4a"}
SILENCE_PHONES = {"h#", "pau", "epi", "sil", "sp", "spn", ""}


def parse_annotation(path: Path, sample_rate: int, duration: float) -> list[tuple[float, float]]:
    suffix = path.suffix.lower()
    if suffix in {".phn", ".wrd"}:
        intervals = []
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3 or parts[2].lower() in SILENCE_PHONES:
                    continue
                intervals.append((float(parts[0]) / sample_rate, float(parts[1]) / sample_rate))
        return merge_intervals(intervals)
    if suffix == ".rttm":
        intervals = []
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 5 and parts[0].upper() == "SPEAKER":
                    start, length = float(parts[3]), float(parts[4])
                    intervals.append((start, min(duration, start + length)))
        return merge_intervals(intervals)
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        return merge_intervals(
            [
                (float(row["start"]), float(row["end"]))
                for row in rows
                if row.get("label", "speech").lower() in {"speech", "1", "true"}
            ]
        )
    raise ValueError(f"Unsupported annotation format: {path}")


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def find_annotation(audio_path: Path) -> Path | None:
    candidates = []
    for suffix in (".phn", ".PHN", ".wrd", ".WRD", ".rttm", ".RTTM", ".csv"):
        candidate = audio_path.with_suffix(suffix)
        if candidate.exists():
            candidates.append(candidate)
    return candidates[0] if candidates else None


def normalize_audio(source: Path, target: Path, sample_rate: int, channels: int) -> Path:
    if source.suffix.lower() == ".wav" and sample_rate == SAMPLE_RATE and channels == 1:
        return source.resolve()
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to convert non-16 kHz mono audio")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        check=True,
    )
    return target.resolve()


def prepare_dataset(raw_root: Path = RAW_DATA, limit: int | None = None) -> list[dict]:
    audio_files = sorted(path for path in raw_root.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)
    if limit:
        audio_files = audio_files[:limit]
    if not audio_files:
        raise FileNotFoundError(f"No audio files found under {raw_root}")

    rows = []
    skipped = []
    for source in audio_files:
        annotation = find_annotation(source)
        if annotation is None:
            skipped.append(str(source))
            continue
        file_id = safe_file_id(source, raw_root)
        sample_rate, channels, duration = audio_info(source)
        target_audio = AUDIO_DIR / f"{file_id}.wav"
        normalized = normalize_audio(source, target_audio, sample_rate, channels)
        normalized_rate, _, normalized_duration = audio_info(normalized)
        intervals = parse_annotation(annotation, sample_rate, duration)
        label_path = LABEL_DIR / f"{file_id}.csv"
        label_rows = []
        cursor = 0.0
        for start, end in intervals:
            if start > cursor:
                label_rows.append({"file_id": file_id, "start": cursor, "end": start, "label": "non_speech"})
            label_rows.append({"file_id": file_id, "start": start, "end": end, "label": "speech"})
            cursor = end
        if cursor < normalized_duration:
            label_rows.append(
                {"file_id": file_id, "start": cursor, "end": normalized_duration, "label": "non_speech"}
            )
        write_csv(label_path, label_rows, ["file_id", "start", "end", "label"])
        noise, snr = parse_noise_snr(source)
        rows.append(
            {
                "file_id": file_id,
                "audio_path": relative_path(normalized),
                "label_path": relative_path(label_path),
                "noise_type": noise,
                "snr": snr,
                "duration_sec": f"{normalized_duration:.6f}",
                "split": "test",
                "notes": f"source={relative_path(source)}; annotation={relative_path(annotation)}; "
                f"sample_rate={normalized_rate}; frame_shift={FRAME_SHIFT_SEC}",
            }
        )
    if not rows:
        raise RuntimeError("Audio was found, but no matching .phn, .rttm, or interval .csv labels were found")
    write_csv(
        METADATA_PATH,
        rows,
        ["file_id", "audio_path", "label_path", "noise_type", "snr", "duration_sec", "split", "notes"],
    )
    if skipped:
        (METADATA_PATH.parent / "skipped_unlabelled.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
    return rows
