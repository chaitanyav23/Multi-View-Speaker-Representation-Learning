"""
scripts/utils/audio_utils.py
============================
Low-level audio I/O helpers used by both the extraction and VAD stages.

All functions are stateless and side-effect-free unless they write a file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

log = logging.getLogger("bwc_pipeline.audio_utils")

# Target sample rate required by Silero VAD
SILERO_SR = 16_000


# ── Loading ──────────────────────────────────────────────────────────────────

def load_audio(
    path: Path,
    target_sr: int = SILERO_SR,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file, resample to *target_sr*, and optionally convert
    to mono.

    Returns
    -------
    (waveform, sample_rate)
        ``waveform`` shape: ``(1, num_samples)`` (always 2-D, mono).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))   # (C, T)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=target_sr
        )
        waveform = resampler(waveform)
        log.debug(
            "Resampled %s: %d Hz → %d Hz", path.name, sr, target_sr
        )

    return waveform, target_sr


def get_duration_sec(waveform: torch.Tensor, sample_rate: int) -> float:
    """Return duration in seconds from waveform tensor."""
    return waveform.shape[-1] / sample_rate


# ── Saving ───────────────────────────────────────────────────────────────────

def save_audio_clip(
    waveform: torch.Tensor,
    sample_rate: int,
    output_path: Path,
    subtype: str = "PCM_16",
) -> None:
    """
    Save *waveform* to *output_path* as a WAV file.

    Parameters
    ----------
    waveform:     Shape ``(1, T)`` or ``(T,)``.
    sample_rate:  Sample rate in Hz.
    output_path:  Destination ``.wav`` file path.
    subtype:      soundfile subtype — ``"PCM_16"`` or ``"PCM_32"``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np_audio = waveform.squeeze().cpu().numpy()
    if np_audio.ndim == 1:
        np_audio = np_audio  # soundfile writes (T,) as mono
    sf.write(str(output_path), np_audio, sample_rate, subtype=subtype)
    log.debug("Saved audio clip → %s", output_path)


# ── Chunking ─────────────────────────────────────────────────────────────────

def iter_chunks(
    waveform: torch.Tensor,
    sample_rate: int,
    chunk_size_sec: float = 60.0,
    overlap_sec: float = 1.0,
) -> Generator[Tuple[torch.Tensor, float, float], None, None]:
    """
    Yield overlapping chunks of *waveform* for memory-safe processing.

    Yields
    ------
    (chunk, chunk_start_sec, chunk_end_sec)
        ``chunk`` shape: ``(1, chunk_samples)``.
    """
    total_samples = waveform.shape[-1]
    chunk_samples = int(chunk_size_sec * sample_rate)
    overlap_samples = int(overlap_sec * sample_rate)
    step = chunk_samples - overlap_samples

    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, start:end]
        yield chunk, start / sample_rate, end / sample_rate
        if end == total_samples:
            break
        start += step


# ── Segment extraction ───────────────────────────────────────────────────────

def extract_segment(
    waveform: torch.Tensor,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> torch.Tensor:
    """
    Slice *waveform* between *start_sec* and *end_sec*.

    Returns
    -------
    torch.Tensor  shape ``(1, N)``.
    """
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    end_sample = min(end_sample, waveform.shape[-1])
    return waveform[:, start_sample:end_sample]


# ── Validation ───────────────────────────────────────────────────────────────

def validate_wav(path: Path) -> bool:
    """
    Return True if *path* is a readable WAV file at SILERO_SR with ≥1 sample.
    """
    try:
        info = torchaudio.info(str(path))
        if info.num_frames < 1:
            log.warning("Empty audio file: %s", path)
            return False
        if info.sample_rate != SILERO_SR:
            log.warning(
                "Unexpected sample rate %d Hz in %s (expected %d)",
                info.sample_rate, path, SILERO_SR,
            )
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Cannot validate %s: %s", path, exc)
        return False
