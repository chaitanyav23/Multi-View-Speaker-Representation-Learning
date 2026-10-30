"""
scripts/utils/json_export.py
============================
Structured JSON export helpers for VAD results.

Follows the schema defined in the project specification:

  {
    "file_name": "...",
    "duration_sec": ...,
    "speech_duration_sec": ...,
    "speech_ratio": ...,
    "segments": [
      { "start": ..., "end": ..., "duration": ... }
    ]
  }

with additional analytics fields appended for research use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

log = logging.getLogger("bwc_pipeline.json_export")


# ── Data classes (schema) ────────────────────────────────────────────────────

@dataclass
class SpeechSegment:
    start: float              # seconds
    end: float                # seconds
    duration: float           # seconds
    confidence_mean: Optional[float] = None   # mean Silero probability


@dataclass
class VADAnalytics:
    """
    Per-file analytics computed after VAD and segment merging.
    """
    total_duration_sec: float = 0.0
    speech_duration_sec: float = 0.0
    silence_duration_sec: float = 0.0
    speech_ratio: float = 0.0
    num_speech_segments: int = 0
    avg_segment_duration_sec: float = 0.0
    min_segment_duration_sec: float = 0.0
    max_segment_duration_sec: float = 0.0
    std_segment_duration_sec: float = 0.0

    # Fragmentation / QA indicators
    short_fragment_count: int = 0              # segments below threshold
    long_silence_count: int = 0               # silences above threshold
    fragmentation_index: float = 0.0          # segments / minute of speech
    potential_false_positive_segments: int = 0  # very long isolated segments
    potential_missed_speech_regions: int = 0   # long silences in busy audio


@dataclass
class VADResult:
    """
    Top-level output document for one audio file.
    """
    file_name: str
    source_video: Optional[str]
    processed_audio_path: str
    pipeline_version: str

    duration_sec: float
    speech_duration_sec: float
    silence_duration_sec: float
    speech_ratio: float

    segments: List[SpeechSegment] = field(default_factory=list)
    analytics: Optional[VADAnalytics] = None

    # Config snapshot (for reproducibility)
    vad_config: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ── Serialisation helpers ────────────────────────────────────────────────────

def save_vad_result(result: VADResult, output_path: Path) -> None:
    """
    Serialise *result* to a UTF-8 JSON file at *output_path*.
    Creates parent directories as needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(result.to_json())
    log.debug("VAD JSON written → %s", output_path)


def load_vad_result(json_path: Path) -> dict:
    """
    Load a previously saved VAD JSON file.  Returns raw dict.
    """
    with Path(json_path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_segment_list(
    raw_segments: list[dict],
) -> list[SpeechSegment]:
    """
    Convert a list of raw dicts (``{"start": …, "end": …}``) returned
    by Silero into :class:`SpeechSegment` objects.
    """
    segments = []
    for seg in raw_segments:
        start = float(seg["start"])
        end = float(seg["end"])
        duration = round(end - start, 4)
        segments.append(
            SpeechSegment(
                start=round(start, 4),
                end=round(end, 4),
                duration=duration,
                confidence_mean=seg.get("confidence_mean"),
            )
        )
    return segments
