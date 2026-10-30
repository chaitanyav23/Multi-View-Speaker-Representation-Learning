from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import align_probabilities, atomic_save_npy, audio_info, frames_to_intervals, postprocess, write_csv
from .config import FRAME_SHIFT_SEC, MODEL_NAME
from .inference import load_model

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif", ".sph"}
RESULT_FIELDS = [
    "file_path",
    "noise_type",
    "dataset_split",
    "duration_seconds",
    "detected_speech_segment_count",
    "detected_speech_total_duration_seconds",
    "false_positive_ratio",
    "average_segment_duration",
    "max_segment_duration",
    "vad_threshold",
    "min_speech_duration_ms",
    "min_silence_duration_ms",
    "pre_padding_ms",
    "post_padding_ms",
    "denoising_enabled",
    "denoising_model",
    "run_timestamp",
    "output_path_for_vad_segments",
]

GRID_RESULT_FIELDS = [
    "file_path",
    "noise_type",
    "dataset_split",
    "duration_sec",
    "vad_threshold",
    "min_speech_ms",
    "min_silence_ms",
    "pre_padding_ms",
    "post_padding_ms",
    "false_positive_segments",
    "false_positive_duration_sec",
    "false_positive_ratio",
]


def discover_inputs(input_dir: Path | None, input_list: Path | None) -> list[Path]:
    paths: list[Path] = []
    if input_dir:
        paths.extend(path for path in input_dir.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)
    if input_list:
        with input_list.open(encoding="utf-8-sig") as handle:
            for line in handle:
                value = line.strip()
                if value and not value.startswith("#"):
                    path = Path(value).expanduser()
                    if not path.is_absolute():
                        path = (input_list.parent / path).resolve()
                    paths.append(path)
    unique = sorted({path.resolve() for path in paths})
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input list contains missing files, first: {missing[0]}")
    if not unique:
        raise RuntimeError("No supported audio files were found")
    return unique


def _cache_name(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    return f"{path.stem}_{digest}.npy"


def infer_noise_type(path: Path) -> str:
    value = re.sub(r"_[0-9]+$", "", path.stem)
    aliases = {
        "AirportAnnouncement": "AirportAnnouncements",
        "CafeTeria": "Cafe",
        "Neighbor": "NeighborSpeaking",
        "Washing": "WasherDryer",
    }
    return aliases.get(value, value)


def infer_dataset_split(path: Path) -> str:
    for part in path.parts:
        if part in {"noise_train", "noise_test"}:
            return part.removeprefix("noise_")
    return "unknown"


def generate_probabilities(
    files: list[Path],
    cache_dir: Path,
    model_name: str = MODEL_NAME,
    device: str = "auto",
    overwrite: bool = False,
    split_duration: float = 50.0,
    window_length: float = 0.63,
) -> dict[Path, Path]:
    from nemo.collections.asr.parts.utils import vad_utils

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = {path: cache_dir / _cache_name(path) for path in files}
    pending = [path for path in files if overwrite or not cached[path].exists()]
    if not pending:
        return cached

    model = load_model(model_name, device)
    with tempfile.TemporaryDirectory(prefix="marblenet_noise_", dir=cache_dir) as temp_name:
        temp = Path(temp_name)
        inputs_dir = temp / "inputs"
        inputs_dir.mkdir()
        staged: dict[Path, Path] = {}
        for path in pending:
            staged_path = inputs_dir / f"{_cache_name(path)[:-4]}.wav"
            sample_rate, channels, _ = audio_info(path)
            if path.suffix.lower() == ".wav" and sample_rate == 16000 and channels == 1:
                staged_path.symlink_to(path)
            else:
                import soundfile as sf

                from .synthesis import load_mono_16k

                sf.write(staged_path, load_mono_16k(path), 16000, subtype="PCM_16")
            staged[path] = staged_path
        input_manifest = temp / "input.json"
        with input_manifest.open("w", encoding="utf-8") as handle:
            for path in pending:
                handle.write(
                    json.dumps(
                        {"audio_filepath": str(staged[path]), "offset": 0, "duration": None, "label": "infer"}
                    )
                    + "\n"
                )
        prepared = vad_utils.prepare_manifest(
            {
                "input": str(input_manifest),
                "out_dir": str(temp),
                "split_duration": split_duration,
                "window_length_in_sec": window_length,
                "num_workers": 1,
            }
        )
        prepared_path = Path(prepared)
        prepared_rows = []
        with prepared_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                row["label"] = "background"
                prepared_rows.append(row)
        with prepared_path.open("w", encoding="utf-8") as handle:
            for row in prepared_rows:
                handle.write(json.dumps(row) + "\n")
        model.setup_test_data(
            test_data_config={
                "manifest_filepath": prepared,
                "sample_rate": 16000,
                "labels": ["background", "speech"],
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 0,
                "vad_stream": True,
            }
        )
        frame_dir = temp / "frames"
        frame_dir.mkdir()
        vad_utils.generate_vad_frame_pred(
            model,
            window_length_in_sec=window_length,
            shift_length_in_sec=FRAME_SHIFT_SEC,
            manifest_vad_input=prepared,
            out_dir=str(frame_dir),
        )
        for index, path in enumerate(pending, start=1):
            frame_path = frame_dir / f"{staged[path].stem}.frame"
            if not frame_path.exists():
                raise RuntimeError(f"NeMo did not create predictions for {path}")
            probabilities = np.loadtxt(frame_path, dtype=np.float32, ndmin=1)
            duration = audio_info(path)[2]
            probabilities = align_probabilities(
                probabilities, max(1, int(np.ceil(duration / FRAME_SHIFT_SEC)))
            )
            atomic_save_npy(cached[path], probabilities)
            print(f"Cached probabilities {index}/{len(pending)}: {path.name}")
    return cached


def _segments_from_probabilities(
    probabilities: np.ndarray,
    duration: float,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    pre_padding_ms: int,
    post_padding_ms: int,
) -> list[tuple[float, float]]:
    labels = (probabilities >= threshold).astype(np.uint8)
    labels = postprocess(
        labels,
        max(1, round(min_speech_ms / 1000 / FRAME_SHIFT_SEC)),
        max(1, round(min_silence_ms / 1000 / FRAME_SHIFT_SEC)),
    )
    intervals = [
        (max(0.0, start - pre_padding_ms / 1000), min(duration, end + post_padding_ms / 1000))
        for start, end, value in frames_to_intervals(labels)
        if value == 1
    ]
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def evaluate_threshold(
    files: list[Path],
    probability_paths: dict[Path, Path],
    threshold: float,
    output_csv: Path,
    summary_path: Path,
    segments_dir: Path,
    min_speech_ms: int,
    min_silence_ms: int,
    pre_padding_ms: int,
    post_padding_ms: int,
    denoising_enabled: bool = False,
    denoising_model: str = "",
) -> tuple[list[dict], dict]:
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    segments_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        duration = audio_info(path)[2]
        probabilities = np.load(probability_paths[path])
        segments = _segments_from_probabilities(
            probabilities,
            duration,
            threshold,
            min_speech_ms,
            min_silence_ms,
            pre_padding_ms,
            post_padding_ms,
        )
        durations = [end - start for start, end in segments]
        total = sum(durations)
        segment_path = segments_dir / f"{path.stem}_{_cache_name(path)[-16:-4]}.json"
        segment_path.write_text(
            json.dumps(
                [{"start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3)}
                 for start, end in segments],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "file_path": str(path),
                "noise_type": infer_noise_type(path),
                "dataset_split": infer_dataset_split(path),
                "duration_seconds": duration,
                "detected_speech_segment_count": len(segments),
                "detected_speech_total_duration_seconds": total,
                "false_positive_ratio": total / duration if duration else 0.0,
                "average_segment_duration": statistics.mean(durations) if durations else 0.0,
                "max_segment_duration": max(durations, default=0.0),
                "vad_threshold": threshold,
                "min_speech_duration_ms": min_speech_ms,
                "min_silence_duration_ms": min_silence_ms,
                "pre_padding_ms": pre_padding_ms,
                "post_padding_ms": post_padding_ms,
                "denoising_enabled": denoising_enabled,
                "denoising_model": denoising_model or "none",
                "run_timestamp": timestamp,
                "output_path_for_vad_segments": str(segment_path),
            }
        )
    write_csv(output_csv, rows, RESULT_FIELDS)
    total_duration = sum(float(row["duration_seconds"]) for row in rows)
    false_duration = sum(float(row["detected_speech_total_duration_seconds"]) for row in rows)
    triggered = sum(int(row["detected_speech_segment_count"] > 0) for row in rows)
    summary = {
        "vad_threshold": threshold,
        "files_tested": len(rows),
        "total_noise_duration_seconds": total_duration,
        "total_false_positive_speech_duration_seconds": false_duration,
        "global_false_positive_ratio": false_duration / total_duration if total_duration else 0.0,
        "mean_false_positive_ratio": statistics.mean(float(row["false_positive_ratio"]) for row in rows),
        "median_false_positive_ratio": statistics.median(float(row["false_positive_ratio"]) for row in rows),
        "files_with_false_positives": triggered,
        "percentage_files_with_false_positives": 100 * triggered / len(rows),
        "min_speech_duration_ms": min_speech_ms,
        "min_silence_duration_ms": min_silence_ms,
        "pre_padding_ms": pre_padding_ms,
        "post_padding_ms": post_padding_ms,
        "denoising_enabled": denoising_enabled,
        "denoising_model": denoising_model or "none",
        "top_10_by_false_positive_ratio": sorted(
            rows, key=lambda row: float(row["false_positive_ratio"]), reverse=True
        )[:10],
        "top_10_by_false_positive_duration": sorted(
            rows, key=lambda row: float(row["detected_speech_total_duration_seconds"]), reverse=True
        )[:10],
    }
    lines = [
        "VAD Noise False-Positive Summary",
        f"Files tested: {summary['files_tested']}",
        f"Total noise duration seconds: {total_duration:.3f}",
        f"Total detected false-positive speech seconds: {false_duration:.3f}",
        f"Global false-positive ratio: {summary['global_false_positive_ratio']:.6f}",
        f"Mean file false-positive ratio: {summary['mean_false_positive_ratio']:.6f}",
        f"Median file false-positive ratio: {summary['median_false_positive_ratio']:.6f}",
        f"Files with false positives: {triggered} ({summary['percentage_files_with_false_positives']:.2f}%)",
        f"VAD threshold: {threshold}",
        f"Minimum speech duration: {min_speech_ms} ms",
        f"Minimum silence duration: {min_silence_ms} ms",
        f"Padding: pre={pre_padding_ms} ms, post={post_padding_ms} ms",
        f"Denoising: {denoising_enabled} ({denoising_model or 'none'})",
        "",
        "Top 10 by false-positive ratio:",
    ]
    lines.extend(
        f"{row['false_positive_ratio']:.6f}\t{row['detected_speech_total_duration_seconds']:.3f}s\t{row['file_path']}"
        for row in summary["top_10_by_false_positive_ratio"]
    )
    lines.append("")
    lines.append("Top 10 by false-positive duration:")
    lines.extend(
        f"{row['detected_speech_total_duration_seconds']:.3f}s\t{row['false_positive_ratio']:.6f}\t{row['file_path']}"
        for row in summary["top_10_by_false_positive_duration"]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows, summary


def write_threshold_comparison(path: Path, summaries: list[dict]) -> None:
    fields = [
        "vad_threshold",
        "total_noise_duration_seconds",
        "total_false_positive_speech_duration_seconds",
        "global_false_positive_ratio",
        "files_with_false_positives",
        "percentage_files_with_false_positives",
    ]
    write_csv(path, summaries, fields)


def _aggregate_grid_rows(rows: list[dict], group_fields: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in groups.items():
        total_duration = sum(float(row["duration_sec"]) for row in members)
        fp_duration = sum(float(row["false_positive_duration_sec"]) for row in members)
        ratios = [float(row["false_positive_ratio"]) for row in members]
        output.append(
            {
                **dict(zip(group_fields, key)),
                "total_files": len(members),
                "total_audio_duration_sec": total_duration,
                "total_false_positive_segments": sum(int(row["false_positive_segments"]) for row in members),
                "total_false_positive_duration_sec": fp_duration,
                "global_false_positive_ratio": fp_duration / total_duration if total_duration else 0.0,
                "mean_false_positive_ratio": statistics.mean(ratios),
                "median_false_positive_ratio": statistics.median(ratios),
                "max_false_positive_ratio": max(ratios, default=0.0),
                "files_with_false_positives": sum(float(row["false_positive_duration_sec"]) > 0 for row in members),
            }
        )
    return sorted(output, key=lambda row: tuple(str(row[field]) for field in group_fields))


def evaluate_parameter_grid(
    files: list[Path],
    probability_paths: dict[Path, Path],
    thresholds: list[float],
    min_speech_values: list[int],
    min_silence_values: list[int],
    pre_padding_ms: int,
    post_padding_values: list[int],
    output_dir: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    file_data = []
    for path in files:
        file_data.append((path, audio_info(path)[2], np.load(probability_paths[path])))
    rows = []
    total_configs = len(thresholds) * len(min_speech_values) * len(min_silence_values) * len(
        post_padding_values
    )
    config_index = 0
    for threshold in thresholds:
        for min_speech in min_speech_values:
            for min_silence in min_silence_values:
                for post_padding in post_padding_values:
                    config_index += 1
                    for path, duration, probabilities in file_data:
                        segments = _segments_from_probabilities(
                            probabilities,
                            duration,
                            threshold,
                            min_speech,
                            min_silence,
                            pre_padding_ms,
                            post_padding,
                        )
                        false_duration = sum(end - start for start, end in segments)
                        rows.append(
                            {
                                "file_path": str(path),
                                "noise_type": infer_noise_type(path),
                                "dataset_split": infer_dataset_split(path),
                                "duration_sec": duration,
                                "vad_threshold": threshold,
                                "min_speech_ms": min_speech,
                                "min_silence_ms": min_silence,
                                "pre_padding_ms": pre_padding_ms,
                                "post_padding_ms": post_padding,
                                "false_positive_segments": len(segments),
                                "false_positive_duration_sec": false_duration,
                                "false_positive_ratio": false_duration / duration if duration else 0.0,
                            }
                        )
                    print(f"Evaluated configuration {config_index}/{total_configs}")
    config_fields = [
        "vad_threshold",
        "min_speech_ms",
        "min_silence_ms",
        "pre_padding_ms",
        "post_padding_ms",
    ]
    summary = _aggregate_grid_rows(rows, config_fields)
    by_noise = _aggregate_grid_rows(rows, config_fields + ["noise_type"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "vad_noise_false_positive_report.csv", rows, GRID_RESULT_FIELDS)
    write_csv(output_dir / "vad_noise_false_positive_summary.csv", summary, list(summary[0]))
    write_csv(output_dir / "vad_noise_false_positive_by_noise_type.csv", by_noise, list(by_noise[0]))
    return rows, summary, by_noise


def select_candidate_configurations(summary: list[dict]) -> list[dict]:
    candidates = []
    for threshold in (0.5, 0.6, 0.7):
        rows = [
            row
            for row in summary
            if abs(float(row["vad_threshold"]) - threshold) < 1e-9
            and int(row["min_speech_ms"]) == 250
            and int(row["min_silence_ms"]) == 200
            and int(row["post_padding_ms"]) == 200
        ]
        if rows:
            candidates.append(rows[0])
    return candidates
