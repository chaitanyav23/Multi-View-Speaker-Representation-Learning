"""
scripts/utils/csv_reporter.py
==============================
Append per-file summary rows to a cumulative CSV report.

The CSV is used for downstream analysis (pandas, spreadsheet, etc.).
Thread-safe via a file lock — safe to call from parallel workers.
"""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from scripts.utils.json_export import VADResult

log = logging.getLogger("bwc_pipeline.csv_reporter")

_LOCK = threading.Lock()

# ── Column schema ────────────────────────────────────────────────────────────
COLUMNS = [
    "file_name",
    "source_video",
    "duration_sec",
    "speech_duration_sec",
    "silence_duration_sec",
    "speech_ratio",
    "num_speech_segments",
    "avg_segment_duration_sec",
    "min_segment_duration_sec",
    "max_segment_duration_sec",
    "std_segment_duration_sec",
    "short_fragment_count",
    "long_silence_count",
    "fragmentation_index",
    "potential_false_positive_segments",
    "potential_missed_speech_regions",
]


def ensure_csv_header(csv_path: Path) -> None:
    """
    Write the header row to *csv_path* if the file does not yet exist
    or is empty.  Thread-safe.
    """
    with _LOCK:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS)
                writer.writeheader()
            log.debug("CSV header written → %s", csv_path)


def append_vad_result_row(result: VADResult, csv_path: Path) -> None:
    """
    Append a single summary row derived from *result* to *csv_path*.
    Creates the file and writes the header if needed.  Thread-safe.
    """
    ensure_csv_header(csv_path)

    analytics = result.analytics

    row: dict[str, Any] = {
        "file_name": result.file_name,
        "source_video": result.source_video or "",
        "duration_sec": round(result.duration_sec, 3),
        "speech_duration_sec": round(result.speech_duration_sec, 3),
        "silence_duration_sec": round(result.silence_duration_sec, 3),
        "speech_ratio": round(result.speech_ratio, 4),
        "num_speech_segments": len(result.segments),
        "avg_segment_duration_sec": (
            round(analytics.avg_segment_duration_sec, 3) if analytics else ""
        ),
        "min_segment_duration_sec": (
            round(analytics.min_segment_duration_sec, 3) if analytics else ""
        ),
        "max_segment_duration_sec": (
            round(analytics.max_segment_duration_sec, 3) if analytics else ""
        ),
        "std_segment_duration_sec": (
            round(analytics.std_segment_duration_sec, 3) if analytics else ""
        ),
        "short_fragment_count": (
            analytics.short_fragment_count if analytics else ""
        ),
        "long_silence_count": (
            analytics.long_silence_count if analytics else ""
        ),
        "fragmentation_index": (
            round(analytics.fragmentation_index, 4) if analytics else ""
        ),
        "potential_false_positive_segments": (
            analytics.potential_false_positive_segments if analytics else ""
        ),
        "potential_missed_speech_regions": (
            analytics.potential_missed_speech_regions if analytics else ""
        ),
    }

    with _LOCK:
        with Path(csv_path).open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writerow(row)

    log.debug("CSV row appended for %s → %s", result.file_name, csv_path)
