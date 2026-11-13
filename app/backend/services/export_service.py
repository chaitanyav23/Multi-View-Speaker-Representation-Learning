"""Transcript Export Formatter Service (JSON, TXT, SRT, VTT)."""
from __future__ import annotations

import json
from datetime import timedelta

from engine.schemas import TranscriptResult


class ExportService:
    """Formats canonical TranscriptResult into standard industry export formats."""

    @staticmethod
    def to_json(result: TranscriptResult, indent: int = 2) -> str:
        """Export canonical JSON string."""
        return json.dumps(result.model_dump(), indent=indent)

    @staticmethod
    def to_txt(result: TranscriptResult) -> str:
        """Export clean speaker-attributed conversational text."""
        lines = [
            f"# Transcript: {result.filename}",
            f"# Duration: {result.duration:.1f}s | Speakers: {', '.join(result.speakers)}",
            f"# Average Confidence: {result.summary.get('mean_confidence', 0.0):.1%}",
            "",
        ]
        for turn in result.segments:
            lines.append(f"[{turn.start:.1f}s - {turn.end:.1f}s] {turn.speaker}: {turn.text}")
        return "\n".join(lines)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_vtt_time(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    @classmethod
    def to_srt(cls, result: TranscriptResult) -> str:
        """Export SubRip Subtitle (.srt) format."""
        blocks = []
        for idx, turn in enumerate(result.segments, 1):
            start_str = cls._format_srt_time(turn.start)
            end_str = cls._format_srt_time(turn.end)
            text = f"[{turn.speaker}] {turn.text}"
            blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
        return "\n".join(blocks)

    @classmethod
    def to_vtt(cls, result: TranscriptResult) -> str:
        """Export WebVTT Subtitle (.vtt) format."""
        lines = ["WEBVTT", ""]
        for idx, turn in enumerate(result.segments, 1):
            start_str = cls._format_vtt_time(turn.start)
            end_str = cls._format_vtt_time(turn.end)
            lines.append(f"{idx}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(f"<v {turn.speaker}>{turn.text}")
            lines.append("")
        return "\n".join(lines)
