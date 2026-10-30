from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import (
    aggregate_metrics,
    confusion,
    load_reference_frames,
    metrics_from_counts,
    postprocess,
    project_path,
    read_csv,
    write_csv,
)
from .config import (
    FRAME_SHIFT_SEC,
    METADATA_PATH,
    METRICS_DIR,
    MIN_DURATION_OFF,
    MIN_DURATION_ON,
    MODEL_NAME,
    PROBABILITIES_DIR,
    THRESHOLDS,
)

FILE_FIELDS = [
    "file_id",
    "test_id",
    "audio_path",
    "noise_type",
    "snr",
    "duration_sec",
    "threshold",
    "mode",
    "tp",
    "tn",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "false_negative_rate",
    "false_alarm_seconds",
    "missed_speech_seconds",
    "detection_error_rate",
]


def test_id(threshold_index: int, mode: str) -> str:
    return f"T{threshold_index * 2 + (1 if mode == 'raw' else 2):03d}"


def evaluate(metadata_path: Path = METADATA_PATH, limit: int | None = None) -> dict[str, list[dict]]:
    metadata = read_csv(metadata_path)
    if limit:
        metadata = metadata[:limit]
    file_rows: list[dict] = []
    missing = []
    for item in metadata:
        probability_path = PROBABILITIES_DIR / f"{item['file_id']}.npy"
        if not probability_path.exists():
            missing.append(item["file_id"])
            continue
        probabilities = np.load(probability_path)
        reference = load_reference_frames(project_path(item["label_path"]), float(item["duration_sec"]))
        size = min(probabilities.size, reference.size)
        probabilities, reference = probabilities[:size], reference[:size]
        for threshold_index, threshold in enumerate(THRESHOLDS):
            raw = (probabilities >= threshold).astype(np.uint8)
            processed = postprocess(
                raw,
                round(MIN_DURATION_ON / FRAME_SHIFT_SEC),
                round(MIN_DURATION_OFF / FRAME_SHIFT_SEC),
            )
            for mode, predicted in (("raw", raw), ("postprocessed", processed)):
                tid = test_id(threshold_index, mode)
                metrics = metrics_from_counts(confusion(reference, predicted))
                file_rows.append(
                    {
                        **item,
                        "test_id": tid,
                        "threshold": threshold,
                        "mode": mode,
                        **metrics,
                    }
                )
    if missing:
        print(f"Warning: skipped {len(missing)} files without probabilities")
    if not file_rows:
        raise RuntimeError("No files could be evaluated; run inference first")

    by_test: dict[str, list[dict]] = defaultdict(list)
    by_snr: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_noise: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in file_rows:
        by_test[row["test_id"]].append(row)
        by_snr[(row["test_id"], row["snr"])].append(row)
        by_noise[(row["test_id"], row["noise_type"])].append(row)

    overall = []
    for tid, rows in sorted(by_test.items()):
        overall.append(
            {
                "test_id": tid,
                "model_name": MODEL_NAME,
                "threshold": rows[0]["threshold"],
                "mode": rows[0]["mode"],
                **aggregate_metrics(rows),
            }
        )

    def grouped(groups: dict, group_field: str) -> list[dict]:
        output = []
        for (tid, group_value), rows in sorted(groups.items()):
            output.append(
                {
                    "test_id": tid,
                    group_field: group_value,
                    "threshold": rows[0]["threshold"],
                    "mode": rows[0]["mode"],
                    **aggregate_metrics(rows),
                    "num_files": len(rows),
                }
            )
        return output

    snr_rows = grouped(by_snr, "snr")
    noise_rows = grouped(by_noise, "noise_type")
    write_csv(METRICS_DIR / "file_level_metrics.csv", file_rows, FILE_FIELDS)
    write_csv(METRICS_DIR / "overall_metrics.csv", overall, list(overall[0]))
    write_csv(METRICS_DIR / "metrics_by_snr.csv", snr_rows, list(snr_rows[0]))
    write_csv(METRICS_DIR / "metrics_by_noise_type.csv", noise_rows, list(noise_rows[0]))
    return {"file": file_rows, "overall": overall, "snr": snr_rows, "noise": noise_rows}


def select_best(overall: list[dict]) -> list[dict]:
    selected = []
    for mode in ("raw", "postprocessed"):
        rows = [row for row in overall if row["mode"] == mode]
        eligible = [row for row in rows if float(row["recall"]) >= 0.85]
        constrained = bool(eligible)
        candidates = eligible or rows
        best = min(candidates, key=lambda row: float(row["false_alarm_seconds"])) if eligible else max(
            candidates, key=lambda row: float(row["f1"])
        )
        selected.append(
            {
                **best,
                "selection_rule": "lowest false alarm seconds with recall >= 0.85",
                "constraint_satisfied": constrained,
                "summary": (
                    "Recall constraint satisfied."
                    if constrained
                    else "No threshold satisfied recall >= 0.85; selected by best F1."
                ),
            }
        )
    return selected
