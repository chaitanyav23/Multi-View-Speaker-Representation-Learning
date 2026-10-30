"""
scripts/vad_engine.py
=====================
PART 2 — Silero VAD
===================

Core VAD engine wrapping Silero VAD (https://github.com/snakers4/silero-vad).

Responsibilities
----------------
* Load the Silero VAD model once and cache it on the selected device.
* Process long audio files via chunking (avoids GPU OOM).
* Merge overlapping / nearby speech segments.
* Compute per-file analytics.
* Export JSON results and optional segmented WAV clips.

Design choices
--------------
* Uses torch.hub to load Silero — no custom model files needed.
* Model is loaded once per process, not once per file (expensive on GPU).
* Float32 is enforced — Silero does not support float16.
* All operations on GPU tensors; results moved to CPU for serialisation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from scripts.utils.audio_utils import (
    SILERO_SR,
    extract_segment,
    get_duration_sec,
    iter_chunks,
    load_audio,
    save_audio_clip,
)
from scripts.utils.device import (
    clear_gpu_cache,
    get_device,
    log_gpu_memory,
    move_to_device,
)
from scripts.utils.json_export import (
    SpeechSegment,
    VADAnalytics,
    VADResult,
    build_segment_list,
)

log = logging.getLogger("bwc_pipeline.vad_engine")


# ── Model loader (cached per process) ───────────────────────────────────────

_VAD_MODEL: Optional[torch.nn.Module] = None
_VAD_UTILS: Optional[tuple] = None


def load_silero_model(
    device: torch.device,
    use_onnx: bool = False,
    repo: str = "external/silero-vad",
    model_name: str = "silero_vad",
) -> Tuple[torch.nn.Module, tuple]:
    """
    Load the Silero VAD model from torch.hub (cloned locally using git clone).

    Parameters
    ----------
    device:     Target device (CUDA or CPU).
    use_onnx:   Load the ONNX backend instead of PyTorch (CPU fallback path).
    repo:       torch.hub repository identifier.
    model_name: Model name within the repository.

    Returns
    -------
    (model, utils)
        ``utils`` is the tuple returned by torch.hub: contains helper
        functions ``(get_speech_timestamps, save_audio, read_audio,
        VADIterator, collect_chunks)``.
    """
    global _VAD_MODEL, _VAD_UTILS

    if _VAD_MODEL is not None:
        return _VAD_MODEL, _VAD_UTILS

    log.info(
        "Loading Silero VAD from torch.hub  (repo=%s, onnx=%s, device=%s) …",
        repo, use_onnx, device,
    )
    t0 = time.perf_counter()

    model, utils = torch.hub.load(
        repo_or_dir=str(Path(repo).resolve()),
        model=model_name,
        source="local",
        trust_repo=True,
        force_reload=False,
        onnx=use_onnx,
        verbose=False,
    )

    if not use_onnx:
        model = model.to(device)
        model.eval()

    elapsed = time.perf_counter() - t0
    log.info("Silero VAD loaded in %.2fs.", elapsed)

    _VAD_MODEL = model
    _VAD_UTILS = utils
    return model, utils


# ── Low-level VAD call ───────────────────────────────────────────────────────

def _run_silero_on_chunk(
    model: torch.nn.Module,
    waveform_chunk: torch.Tensor,
    sample_rate: int,
    threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    window_size_samples: int,
    speech_pad_ms: int,
    device: torch.device,
    get_speech_timestamps,
) -> List[Dict]:
    """
    Run Silero VAD on a single waveform chunk.

    Returns a list of dicts: ``[{"start": int, "end": int}, …]``
    where start/end are sample indices (relative to the chunk start).
    """
    # Silero expects a 1-D tensor
    chunk_1d = waveform_chunk.squeeze(0)  # (T,)
    chunk_1d = move_to_device(chunk_1d, device)

    with torch.no_grad():
        timestamps = get_speech_timestamps(
            audio=chunk_1d,
            model=model,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=window_size_samples,
            speech_pad_ms=speech_pad_ms,
            return_seconds=False,   # we work in samples; convert later
        )

    # Reset model state so the next chunk starts fresh
    model.reset_states()
    return timestamps


# ── Segment merging & post-processing ────────────────────────────────────────

def merge_segments(
    segments: List[Dict[str, float]],
    merge_gap_sec: float = 0.3,
    min_duration_sec: float = 0.2,
) -> List[Dict[str, float]]:
    """
    Merge temporally close speech segments and discard very short ones.

    Parameters
    ----------
    segments:        List of ``{"start": float, "end": float}`` dicts
                     (times in seconds).
    merge_gap_sec:   Segments separated by less than this are merged.
    min_duration_sec: Merged segments shorter than this are discarded.

    Returns
    -------
    Sorted list of ``{"start": float, "end": float, "duration": float}`` dicts.
    """
    if not segments:
        return []

    # Sort by start time (should already be sorted, but be defensive)
    segs = sorted(segments, key=lambda s: s["start"])

    merged: List[Dict[str, float]] = []
    current = dict(segs[0])

    for seg in segs[1:]:
        gap = seg["start"] - current["end"]
        if gap <= merge_gap_sec:
            current["end"] = max(current["end"], seg["end"])
        else:
            duration = current["end"] - current["start"]
            if duration >= min_duration_sec:
                current["duration"] = round(duration, 4)
                merged.append(current)
            current = dict(seg)

    # Flush last segment
    duration = current["end"] - current["start"]
    if duration >= min_duration_sec:
        current["duration"] = round(duration, 4)
        merged.append(current)

    return merged


# ── Analytics computation ────────────────────────────────────────────────────

def compute_analytics(
    segments: List[SpeechSegment],
    total_duration_sec: float,
    short_fragment_threshold_sec: float = 0.5,
    long_silence_threshold_sec: float = 5.0,
    max_segment_duration_sec: float = 30.0,
    speech_ratio_busy_threshold: float = 0.4,
) -> VADAnalytics:
    """
    Compute rich per-file analytics from a list of merged speech segments.
    """
    if not segments:
        return VADAnalytics(
            total_duration_sec=total_duration_sec,
            speech_duration_sec=0.0,
            silence_duration_sec=total_duration_sec,
            speech_ratio=0.0,
        )

    durations = np.array([s.duration for s in segments], dtype=np.float64)
    speech_duration = float(durations.sum())
    silence_duration = max(0.0, total_duration_sec - speech_duration)
    speech_ratio = speech_duration / total_duration_sec if total_duration_sec > 0 else 0.0

    # Short fragments
    short_count = int((durations < short_fragment_threshold_sec).sum())

    # Long silences between consecutive segments
    long_silence_count = 0
    for i in range(1, len(segments)):
        gap = segments[i].start - segments[i - 1].end
        if gap > long_silence_threshold_sec:
            long_silence_count += 1

    # Fragmentation index: segments per minute of total audio
    fragmentation_index = (
        len(segments) / (total_duration_sec / 60.0)
        if total_duration_sec > 60
        else len(segments)
    )

    # Potential false positives: very long isolated segments
    fp_segments = int((durations > max_segment_duration_sec).sum())

    # Potential missed speech: long silences in busy audio (high speech ratio)
    missed_speech_regions = 0
    if speech_ratio > speech_ratio_busy_threshold:
        missed_speech_regions = long_silence_count

    return VADAnalytics(
        total_duration_sec=round(total_duration_sec, 4),
        speech_duration_sec=round(speech_duration, 4),
        silence_duration_sec=round(silence_duration, 4),
        speech_ratio=round(speech_ratio, 6),
        num_speech_segments=len(segments),
        avg_segment_duration_sec=round(float(durations.mean()), 4),
        min_segment_duration_sec=round(float(durations.min()), 4),
        max_segment_duration_sec=round(float(durations.max()), 4),
        std_segment_duration_sec=round(float(durations.std()), 4),
        short_fragment_count=short_count,
        long_silence_count=long_silence_count,
        fragmentation_index=round(fragmentation_index, 4),
        potential_false_positive_segments=fp_segments,
        potential_missed_speech_regions=missed_speech_regions,
    )


# ── Full per-file VAD pipeline ───────────────────────────────────────────────

def run_vad_on_file(
    wav_path: Path,
    model: torch.nn.Module,
    utils: tuple,
    device: torch.device,
    source_video: Optional[str] = None,
    # VAD thresholds (all configurable)
    threshold: float = 0.35,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 512,
    speech_pad_ms: int = 300,
    # Chunking
    chunk_size_sec: float = 60.0,
    chunk_overlap_sec: float = 1.0,
    # Post-processing
    merge_gap_sec: float = 1.0,
    min_segment_duration_sec: float = 0.2,
    max_segment_duration_sec: float = 30.0,
    # Analytics
    short_fragment_threshold_sec: float = 0.5,
    long_silence_threshold_sec: float = 5.0,
    # Output
    output_dir_clips: Optional[Path] = None,
    pipeline_version: str = "1.0.0",
    vad_config: Optional[dict] = None,
) -> VADResult:
    """
    Run the complete VAD pipeline on a single WAV file.

    Steps
    -----
    1. Load and validate audio.
    2. Chunk the waveform for GPU-memory-safe processing.
    3. Run Silero VAD on each chunk, adjusting timestamps to file-relative
       positions.
    4. Merge nearby segments.
    5. Compute analytics.
    6. Optionally save per-segment WAV clips.
    7. Return a :class:`VADResult` ready for JSON / CSV export.

    Parameters
    ----------
    wav_path:      Path to the input WAV file (mono 16kHz).
    model:         Loaded Silero model (on correct device).
    utils:         Silero utils tuple from torch.hub.
    device:        Torch device.
    source_video:  Original video filename for provenance tracking.
    ...            (see configs/stage1_config.yaml for all thresholds)
    output_dir_clips: If set, write per-segment WAV clips here.
    pipeline_version: Stored in the JSON for reproducibility.
    vad_config:    Config snapshot to embed in the JSON.
    """
    get_speech_timestamps = utils[0]  # first util from Silero

    wav_path = Path(wav_path)
    log.info("VAD processing: %s", wav_path.name)
    t0 = time.perf_counter()

    # ── 1. Load audio ─────────────────────────────────────────────────────────
    waveform, sr = load_audio(wav_path, target_sr=SILERO_SR, mono=True)
    total_duration = get_duration_sec(waveform, sr)
    log.debug("  Duration: %.2f sec  |  samples: %d", total_duration, waveform.shape[-1])

    log_gpu_memory(device, tag="pre-vad")

    # ── 2. Chunk & run VAD ────────────────────────────────────────────────────
    raw_segments_sec: List[Dict[str, float]] = []

    for chunk, chunk_start_sec, chunk_end_sec in iter_chunks(
        waveform, sr,
        chunk_size_sec=chunk_size_sec,
        overlap_sec=chunk_overlap_sec,
    ):
        chunk_timestamps = _run_silero_on_chunk(
            model=model,
            waveform_chunk=chunk,
            sample_rate=sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=window_size_samples,
            speech_pad_ms=speech_pad_ms,
            device=device,
            get_speech_timestamps=get_speech_timestamps,
        )

        for ts in chunk_timestamps:
            # ts["start"] and ts["end"] are sample indices inside the chunk
            abs_start = chunk_start_sec + ts["start"] / sr
            abs_end = chunk_start_sec + ts["end"] / sr

            # Clamp to the non-overlapping region to avoid counting
            # the overlap twice (except for the very first chunk)
            if raw_segments_sec:
                # Avoid duplicating segments in the overlap zone
                # by only accepting segments that start AFTER the previous
                # chunk's non-overlapping end
                prev_chunk_non_overlap_end = chunk_start_sec + chunk_overlap_sec
                if abs_start < prev_chunk_non_overlap_end:
                    continue

            raw_segments_sec.append({
                "start": round(abs_start, 4),
                "end": round(min(abs_end, total_duration), 4),
            })

        log.debug(
            "  Chunk [%.1f–%.1f s]: %d raw segment(s)",
            chunk_start_sec, chunk_end_sec, len(chunk_timestamps),
        )

    log_gpu_memory(device, tag="post-vad")
    clear_gpu_cache(device)

    # ── 3. Merge segments ─────────────────────────────────────────────────────
    merged = merge_segments(
        raw_segments_sec,
        merge_gap_sec=merge_gap_sec,
        min_duration_sec=min_segment_duration_sec,
    )
    log.debug("  Raw: %d segments → Merged: %d segments", len(raw_segments_sec), len(merged))

    # ── 4. Build SpeechSegment objects ────────────────────────────────────────
    segments: List[SpeechSegment] = build_segment_list(merged)

    # ── 5. Analytics ─────────────────────────────────────────────────────────
    analytics = compute_analytics(
        segments=segments,
        total_duration_sec=total_duration,
        short_fragment_threshold_sec=short_fragment_threshold_sec,
        long_silence_threshold_sec=long_silence_threshold_sec,
        max_segment_duration_sec=max_segment_duration_sec,
    )

    _log_file_analytics(wav_path.name, analytics)

    # ── 6. Save speech clips ──────────────────────────────────────────────────
    if output_dir_clips is not None and segments:
        _save_segment_clips(
            waveform=waveform,
            sample_rate=sr,
            segments=segments,
            clip_dir=Path(output_dir_clips) / wav_path.stem,
            source_stem=wav_path.stem,
        )

    elapsed = time.perf_counter() - t0
    rtf = elapsed / total_duration if total_duration > 0 else 0.0
    log.info(
        "  → %d segments  |  speech %.1f%%  |  RTF %.3f  |  %.2fs elapsed",
        len(segments),
        analytics.speech_ratio * 100,
        rtf,
        elapsed,
    )

    # ── 7. Build result ───────────────────────────────────────────────────────
    return VADResult(
        file_name=wav_path.name,
        source_video=source_video,
        processed_audio_path=str(wav_path.resolve()),
        pipeline_version=pipeline_version,
        duration_sec=round(total_duration, 4),
        speech_duration_sec=analytics.speech_duration_sec,
        silence_duration_sec=analytics.silence_duration_sec,
        speech_ratio=analytics.speech_ratio,
        segments=segments,
        analytics=analytics,
        vad_config=vad_config,
    )


# ── Clip saving helper ────────────────────────────────────────────────────────

def _save_segment_clips(
    waveform: torch.Tensor,
    sample_rate: int,
    segments: List[SpeechSegment],
    clip_dir: Path,
    source_stem: str,
) -> None:
    """Write one WAV file per speech segment to *clip_dir*."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    pad = len(str(len(segments)))  # zero-pad width

    for idx, seg in enumerate(segments):
        clip = extract_segment(waveform, sample_rate, seg.start, seg.end)
        clip_name = f"{source_stem}_seg{str(idx).zfill(pad)}_{seg.start:.3f}_{seg.end:.3f}.wav"
        save_audio_clip(clip, sample_rate, clip_dir / clip_name)

    log.debug("  Saved %d clips to %s", len(segments), clip_dir)


# ── Analytics log helper ──────────────────────────────────────────────────────

def _log_file_analytics(filename: str, a: VADAnalytics) -> None:
    log.info(
        "  [%s]  total=%.2fs  speech=%.2fs  silence=%.2fs  "
        "ratio=%.1f%%  segs=%d  avg_seg=%.2fs  "
        "short=%d  long_silence=%d  frag_idx=%.2f",
        filename,
        a.total_duration_sec,
        a.speech_duration_sec,
        a.silence_duration_sec,
        a.speech_ratio * 100,
        a.num_speech_segments,
        a.avg_segment_duration_sec,
        a.short_fragment_count,
        a.long_silence_count,
        a.fragmentation_index,
    )
    if a.potential_false_positive_segments:
        log.warning(
            "  [%s] ⚠ %d potential false positive segment(s) (>%.0fs)",
            filename, a.potential_false_positive_segments, 30.0,
        )
    if a.potential_missed_speech_regions:
        log.warning(
            "  [%s] ⚠ %d potential missed speech region(s) (long silence in busy audio)",
            filename, a.potential_missed_speech_regions,
        )
