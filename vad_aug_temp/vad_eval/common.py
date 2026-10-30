from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError:  # Allows the pre-install setup check to run.
    np = None  # type: ignore[assignment]

from .config import FRAME_SHIFT_SEC, PROJECT_ROOT


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def safe_file_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    value = "__".join(rel.parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def audio_info(path: Path) -> tuple[int, int, float]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return int(info.samplerate), int(info.channels), float(info.duration)
    except Exception:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        stream = json.loads(result.stdout)["streams"][0]
        return (
            int(stream["sample_rate"]),
            int(stream["channels"]),
            float(stream.get("duration") or 0.0),
        )


def parse_noise_snr(path: Path) -> tuple[str, str]:
    text = "/".join(path.parts).lower()
    match = re.search(r"(?<!\d)([-+]?\d+(?:\.\d+)?)\s*d?b(?:\b|_)", text)
    if not match:
        match = re.search(r"(?:snr)[_-]?([-+]?\d+(?:\.\d+)?)", text)
    snr = match.group(1) if match else "unknown"
    known = (
        "cafe",
        "car",
        "home",
        "street",
        "reverb",
        "noise",
        "restaurant",
        "station",
    )
    noise = next((item for item in known if item in text), "unknown")
    return noise, snr


def intervals_to_frames(
    intervals: Iterable[tuple[float, float]], duration: float, shift: float = FRAME_SHIFT_SEC
) -> np.ndarray:
    count = max(1, int(math.ceil(duration / shift)))
    labels = np.zeros(count, dtype=np.uint8)
    for start, end in intervals:
        left = max(0, int(math.floor(start / shift)))
        right = min(count, int(math.ceil(end / shift)))
        if right > left:
            labels[left:right] = 1
    return labels


def frames_to_intervals(labels: np.ndarray, shift: float = FRAME_SHIFT_SEC) -> list[tuple[float, float, int]]:
    if labels.size == 0:
        return []
    changes = np.flatnonzero(np.diff(labels.astype(np.int8)) != 0) + 1
    bounds = np.concatenate(([0], changes, [labels.size]))
    return [(a * shift, b * shift, int(labels[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def load_reference_frames(label_path: Path, duration: float) -> np.ndarray:
    rows = read_csv(label_path)
    intervals = [
        (float(row["start"]), float(row["end"]))
        for row in rows
        if row.get("label", "").strip().lower() in {"speech", "1", "true"}
    ]
    return intervals_to_frames(intervals, duration)


def align_probabilities(probabilities: np.ndarray, frame_count: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if probabilities.size == frame_count:
        return probabilities
    if probabilities.size == 0:
        raise ValueError("Model produced no probability frames")
    old_x = np.linspace(0.0, 1.0, probabilities.size, endpoint=False)
    new_x = np.linspace(0.0, 1.0, frame_count, endpoint=False)
    return np.interp(new_x, old_x, probabilities).astype(np.float32)


def postprocess(labels: np.ndarray, min_on_frames: int, min_off_frames: int) -> np.ndarray:
    result = labels.astype(np.uint8).copy()
    for start, end, value in frames_to_intervals(result, 1.0):
        if value == 1 and end - start < min_on_frames:
            result[int(start) : int(end)] = 0
    segments = frames_to_intervals(result, 1.0)
    for index, (start, end, value) in enumerate(segments):
        if (
            value == 0
            and end - start < min_off_frames
            and index > 0
            and index + 1 < len(segments)
            and segments[index - 1][2] == 1
            and segments[index + 1][2] == 1
        ):
            result[int(start) : int(end)] = 1
    return result


def confusion(reference: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    size = min(reference.size, predicted.size)
    ref = reference[:size].astype(bool)
    pred = predicted[:size].astype(bool)
    return {
        "tp": int(np.sum(ref & pred)),
        "tn": int(np.sum(~ref & ~pred)),
        "fp": int(np.sum(~ref & pred)),
        "fn": int(np.sum(ref & ~pred)),
    }


def metrics_from_counts(counts: dict[str, int], shift: float = FRAME_SHIFT_SEC) -> dict[str, float]:
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))

    def divide(a: float, b: float) -> float:
        return a / b if b else 0.0

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": divide(2 * precision * recall, precision + recall),
        "false_positive_rate": divide(fp, fp + tn),
        "false_negative_rate": divide(fn, fn + tp),
        "false_alarm_seconds": fp * shift,
        "missed_speech_seconds": fn * shift,
        "detection_error_rate": divide(fp + fn, tp + fn),
        "total_duration_seconds": (tp + tn + fp + fn) * shift,
        "speech_duration_seconds": (tp + fn) * shift,
        "non_speech_duration_seconds": (tn + fp) * shift,
    }


def aggregate_metrics(rows: list[dict]) -> dict[str, float]:
    counts = {key: sum(int(row[key]) for row in rows) for key in ("tp", "tn", "fp", "fn")}
    return metrics_from_counts(counts)


def environment_info() -> dict[str, str]:
    usage = shutil.disk_usage("/")
    values = {
        "experiment_name": "MarbleNet VAD evaluation on QUT-NOISE-TIMIT",
        "date_time": datetime.now(timezone.utc).isoformat(),
        "vm_cloud": "GCP",
        "gpu_name": "unavailable",
        "gpu_memory": "unavailable",
        "cpu_count": str(os.cpu_count() or "unknown"),
        "ram_gb": "unknown",
        "disk_total": f"{usage.total / 2**30:.2f} GiB",
        "disk_free_start": f"{usage.free / 2**30:.2f} GiB",
        "python_version": platform.python_version(),
        "dataset_name": "QUT-NOISE-TIMIT",
    }
    try:
        import psutil

        values["ram_gb"] = f"{psutil.virtual_memory().total / 2**30:.2f}"
    except ImportError:
        pass
    for module_name, field in (("torch", "torch_version"), ("nemo", "nemo_version")):
        try:
            module = __import__(module_name)
            values[field] = str(module.__version__)
        except Exception:
            values[field] = "not installed"
    try:
        import torch

        values["cuda_available"] = str(torch.cuda.is_available())
        values["cuda_version"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            values["gpu_name"] = properties.name
            values["gpu_memory"] = f"{properties.total_memory / 2**30:.2f} GiB"
    except Exception:
        values["cuda_available"] = "False"
        values["cuda_version"] = "unavailable"
    return values
