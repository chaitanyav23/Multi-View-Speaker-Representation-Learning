from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from .synthesis import ENVIRONMENTS, SNR_LEVELS, discover_audio, load_mono_16k, noise_segment, rms


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_dataset(
    speech_root: Path,
    output_root: Path,
    manifest_path: Path,
    seed: int = 2026,
    snr_tolerance_db: float = 0.5,
) -> dict:
    rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest is empty")
    rng = random.Random(seed)

    audio_spec_results = []
    for row in rng.sample(rows, min(5, len(rows))):
        path = Path(row["output_path"])
        info = sf.info(path)
        valid = info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
        audio_spec_results.append(
            {
                "path": str(path),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "subtype": info.subtype,
                "valid": valid,
            }
        )

    snr_results = []
    for target in SNR_LEVELS:
        candidates = [row for row in rows if int(row["snr_db"]) == target]
        row = rng.choice(candidates)
        speech = load_mono_16k(Path(row["speech_path"]))
        speech = speech - np.mean(speech)
        noise = load_mono_16k(Path(row["noise_path"]))
        noise = noise - np.mean(noise)
        selected = noise_segment(noise, speech.size, int(row["noise_offset"]))
        speech_component = speech * float(row["final_gain"])
        noise_component = selected * float(row["noise_scale"]) * float(row["final_gain"])
        measured = 20.0 * math.log10(rms(speech_component) / rms(noise_component))
        snr_results.append(
            {
                "target_snr_db": target,
                "measured_snr_db": measured,
                "error_db": measured - target,
                "valid": abs(measured - target) <= snr_tolerance_db,
                "path": row["output_path"],
            }
        )

    clipping_results = []
    for row in rng.sample(rows, min(20, len(rows))):
        signal, _ = sf.read(row["output_path"], dtype="float64")
        peak = float(np.max(np.abs(signal)))
        clipping_results.append({"path": row["output_path"], "peak": peak, "valid": peak < 1.0})

    clean_count = len(discover_audio(speech_root))
    expected = clean_count * len(ENVIRONMENTS) * len(SNR_LEVELS)
    output_files = [path for path in output_root.rglob("*.wav") if path.is_file()]
    completeness = {
        "clean_speech_files": clean_count,
        "environment_count": len(ENVIRONMENTS),
        "snr_count": len(SNR_LEVELS),
        "expected_output_files": expected,
        "actual_output_files": len(output_files),
        "manifest_rows": len(rows),
        "valid": len(output_files) == expected == len(rows),
    }
    report = {
        "audio_specifications": audio_spec_results,
        "snr_levels": snr_results,
        "audio_integrity": clipping_results,
        "directory_completeness": completeness,
    }
    report["valid"] = (
        all(item["valid"] for item in audio_spec_results)
        and all(item["valid"] for item in snr_results)
        and all(item["valid"] for item in clipping_results)
        and completeness["valid"]
    )
    report_path = manifest_path.parent / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
