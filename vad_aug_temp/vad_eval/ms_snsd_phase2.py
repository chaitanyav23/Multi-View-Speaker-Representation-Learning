from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import confusion, intervals_to_frames, metrics_from_counts, postprocess, write_csv
from .config import FRAME_SHIFT_SEC
from .noise_false_positive import _segments_from_probabilities, generate_probabilities
from .synthesis import load_mono_16k, mix_signals

SNRS = (20, 10, 5, 0, -5)
SPEECH_BEARING_NOISE = {
    "AirportAnnouncement",
    "AirportAnnouncements",
    "Babble",
    "Cafe",
    "CafeTeria",
    "Neighbor",
    "NeighborSpeaking",
    "Restaurant",
}
CANDIDATES = (
    {"name": "balanced", "threshold": 0.5, "min_speech_ms": 250},
    {"name": "moderate", "threshold": 0.6, "min_speech_ms": 250},
    {"name": "conservative", "threshold": 0.7, "min_speech_ms": 250},
)


def noise_type(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def select_files(root: Path, limit: int, seed: int) -> tuple[list[Path], list[Path]]:
    clean = sorted((root / "clean_test").glob("*.wav"))
    noise = sorted(
        path
        for path in (root / "noise_test").glob("*.wav")
        if noise_type(path) not in SPEECH_BEARING_NOISE
    )
    if not clean or not noise:
        raise RuntimeError("MS-SNSD clean_test or non-speech noise_test files are missing")
    rng = random.Random(seed)
    clean = rng.sample(clean, min(limit, len(clean)))
    return sorted(clean), noise


def energy_reference(signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    frame_length = round(0.025 * sample_rate)
    hop = round(FRAME_SHIFT_SEC * sample_rate)
    frame_count = max(1, math.ceil(signal.size / hop))
    padded = np.pad(signal.astype(np.float64), (frame_length // 2, frame_length))
    rms_values = np.empty(frame_count, dtype=np.float64)
    for index in range(frame_count):
        frame = padded[index * hop : index * hop + frame_length]
        rms_values[index] = np.sqrt(np.mean(frame * frame))
    peak = max(float(np.max(rms_values)), 1e-12)
    db_relative = 20 * np.log10(np.maximum(rms_values, 1e-12) / peak)
    labels = (db_relative >= -40.0).astype(np.uint8)
    return postprocess(labels, min_on_frames=5, min_off_frames=10)


def _stable_offset(clean: Path, noise: Path, seed: int, noise_length: int) -> int:
    digest = hashlib.sha256(f"{clean}|{noise}|{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % noise_length


def build_dataset(
    dataset_root: Path,
    output_root: Path,
    manifest_path: Path,
    limit: int = 200,
    seed: int = 2026,
) -> list[dict]:
    clean_files, noise_files = select_files(dataset_root, limit, seed)
    output_root.mkdir(parents=True, exist_ok=True)
    label_dir = output_root / "reference_labels"
    label_dir.mkdir(exist_ok=True)
    rows = []
    for index, clean_path in enumerate(clean_files):
        clean = load_mono_16k(clean_path)
        labels = energy_reference(clean)
        label_path = label_dir / f"{clean_path.stem}.npy"
        np.save(label_path, labels)
        speech_ratio = float(np.mean(labels))
        rows.append(
            {
                "file_id": f"{clean_path.stem}__clean",
                "audio_path": str(clean_path.resolve()),
                "clean_path": str(clean_path.resolve()),
                "noise_path": "",
                "noise_type": "clean",
                "snr_db": "clean",
                "condition": "clean",
                "duration_sec": clean.size / 16000,
                "reference_path": str(label_path.resolve()),
                "reference_speech_ratio": speech_ratio,
                "snr_scale": "",
                "final_gain": 1.0,
            }
        )
        selected_noise_path = noise_files[index % len(noise_files)]
        noise = load_mono_16k(selected_noise_path)
        offset = _stable_offset(clean_path, selected_noise_path, seed, noise.size)
        for snr in SNRS:
            mixture, values = mix_signals(clean, noise, snr, offset)
            target = output_root / "audio" / f"snr_{snr:+d}" / (
                f"{clean_path.stem}__{noise_type(selected_noise_path)}__snr_{snr:+d}.wav"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                sf.write(target, mixture, 16000, subtype="PCM_16")
            rows.append(
                {
                    "file_id": target.stem,
                    "audio_path": str(target.resolve()),
                    "clean_path": str(clean_path.resolve()),
                    "noise_path": str(selected_noise_path.resolve()),
                    "noise_type": noise_type(selected_noise_path),
                    "snr_db": snr,
                    "condition": "noisy",
                    "duration_sec": clean.size / 16000,
                    "reference_path": str(label_path.resolve()),
                    "reference_speech_ratio": speech_ratio,
                    "snr_scale": values["noise_scale"],
                    "final_gain": values["final_gain"],
                }
            )
        if (index + 1) % 25 == 0 or index + 1 == len(clean_files):
            print(f"Built {index + 1}/{len(clean_files)} clean utterance groups")
    write_csv(manifest_path, rows, list(rows[0]))
    manifest_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "clean_files": len(clean_files),
                "conditions_per_file": 1 + len(SNRS),
                "total_audio_entries": len(rows),
                "snrs": list(SNRS),
                "seed": seed,
                "reference": "25 ms centered RMS, 10 ms shift, -40 dB relative threshold, "
                "50 ms minimum speech, 100 ms gap merge",
                "excluded_speech_bearing_noise_types": sorted(SPEECH_BEARING_NOISE),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_candidates(
    manifest_path: Path,
    output_dir: Path,
    device: str = "auto",
) -> tuple[list[dict], list[dict]]:
    rows = read_manifest(manifest_path)
    files = [Path(row["audio_path"]) for row in rows]
    probabilities = generate_probabilities(files, output_dir / "probabilities", device=device)
    file_results = []
    for item in rows:
        path = Path(item["audio_path"])
        duration = float(item["duration_sec"])
        reference = np.load(item["reference_path"]).astype(np.uint8)
        probs = np.load(probabilities[path.resolve()])
        size = min(reference.size, probs.size)
        reference, probs = reference[:size], probs[:size]
        for candidate in CANDIDATES:
            segments = _segments_from_probabilities(
                probs,
                duration,
                candidate["threshold"],
                candidate["min_speech_ms"],
                200,
                200,
                200,
            )
            predicted = intervals_to_frames(
                [(start, end) for start, end in segments], duration
            )[:size]
            metrics = metrics_from_counts(confusion(reference, predicted))
            file_results.append(
                {
                    **item,
                    "candidate": candidate["name"],
                    "vad_threshold": candidate["threshold"],
                    "min_speech_ms": candidate["min_speech_ms"],
                    "min_silence_ms": 200,
                    "pre_padding_ms": 200,
                    "post_padding_ms": 200,
                    **metrics,
                }
            )
    file_path = output_dir / "phase2_file_metrics.csv"
    write_csv(file_path, file_results, list(file_results[0]))

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in file_results:
        key = (row["candidate"], str(row["snr_db"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (candidate, snr), members in groups.items():
        counts = {
            key: sum(int(row[key]) for row in members)
            for key in ("tp", "tn", "fp", "fn")
        }
        metrics = metrics_from_counts(counts)
        first = members[0]
        summary.append(
            {
                "candidate": candidate,
                "snr_db": snr,
                "vad_threshold": first["vad_threshold"],
                "min_speech_ms": first["min_speech_ms"],
                "num_files": len(members),
                **metrics,
            }
        )
    summary.sort(key=lambda row: (row["candidate"], row["snr_db"]))
    write_csv(output_dir / "phase2_summary_by_snr.csv", summary, list(summary[0]))
    return file_results, summary
