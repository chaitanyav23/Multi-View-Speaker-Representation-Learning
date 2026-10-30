"""
scripts/extract_audio.py
========================
PART 1 — Video → Audio Extraction
==================================

Converts raw video files (MP4, MKV, AVI, etc.) to mono 16kHz PCM WAV files
using ffmpeg.  Supports batch processing with a configurable thread pool.

Design goals
------------
* Zero audio-quality loss beyond the intentional mono/16kHz downmix.
* Deterministic output filenames (source stem + .wav).
* Skip already-processed files unless ``--overwrite`` is passed.
* Structured logging with per-file timing.
* Designed for datasets that exceed RAM — ffmpeg streams to disk directly.

Usage
-----
CLI (recommended):
  python -m scripts.extract_audio \
    --input  raw_video/ \
    --output raw_audio/ \
    --workers 4

Import:
  from scripts.extract_audio import batch_extract_audio
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from scripts.utils.config import load_config
from scripts.utils.logger import get_logger

# Module logger (child of pipeline root)
log = get_logger("bwc_pipeline.extract_audio")


# ── Core extraction function ─────────────────────────────────────────────────

def extract_audio(
    video_path: Path,
    output_dir: Path,
    sample_rate: int = 16_000,
    channels: int = 1,
    codec: str = "pcm_s16le",
    ffmpeg_loglevel: str = "error",
    overwrite: bool = False,
) -> Tuple[bool, Path, str]:
    """
    Extract and normalise audio from a single video file.

    Uses a subprocess ffmpeg call — pure Python, no external C bindings
    required at runtime.

    Parameters
    ----------
    video_path:       Source video file.
    output_dir:       Directory to write the WAV file.
    sample_rate:      Target sample rate in Hz (default 16 000).
    channels:         Number of output channels (1 = mono).
    codec:            PCM codec string passed to ``-acodec`` (default
                      ``pcm_s16le`` = 16-bit LE PCM).
    ffmpeg_loglevel:  Verbosity passed to ``-loglevel`` (default ``error``).
    overwrite:        If False, skip if output WAV already exists.

    Returns
    -------
    (success, output_path, message)
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = output_dir / (video_path.stem + ".wav")

    if wav_path.exists() and not overwrite:
        log.info("  SKIP (already exists): %s", wav_path.name)
        return True, wav_path, "skipped"

    cmd = [
        "ffmpeg",
        "-y",                                 # overwrite output silently
        "-i", str(video_path),                # input file
        "-vn",                                # drop video stream
        "-acodec", codec,                     # output codec
        "-ar", str(sample_rate),              # sample rate
        "-ac", str(channels),                 # channel count
        "-loglevel", ffmpeg_loglevel,
        str(wav_path),
    ]

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,    # we inspect returncode manually
        )
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            err = result.stderr.strip()
            log.error(
                "  FAILED: %s  (ffmpeg rc=%d)\n    %s",
                video_path.name, result.returncode, err,
            )
            return False, wav_path, f"ffmpeg error rc={result.returncode}: {err}"

        # Verify output was created and is non-empty
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            log.error("  FAILED: output WAV missing or empty: %s", wav_path)
            return False, wav_path, "output WAV missing or empty"

        size_mb = wav_path.stat().st_size / 1024 ** 2
        log.info(
            "  OK  %-40s  →  %-40s  [%.1f MB, %.2fs]",
            video_path.name, wav_path.name, size_mb, elapsed,
        )
        return True, wav_path, "ok"

    except FileNotFoundError:
        msg = (
            "ffmpeg not found on PATH.  "
            "Install via: sudo apt-get install ffmpeg  OR  conda install ffmpeg"
        )
        log.error(msg)
        return False, wav_path, msg
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error extracting %s: %s", video_path.name, exc)
        return False, wav_path, str(exc)


# ── Batch extraction ─────────────────────────────────────────────────────────

def find_video_files(
    input_dir: Path,
    extensions: List[str],
    recursive: bool = False,
) -> List[Path]:
    """
    Discover all video files in *input_dir* matching *extensions*.

    Parameters
    ----------
    input_dir:   Root directory to search.
    extensions:  List of lowercase dot-prefixed extensions, e.g.
                 ``[".mp4", ".mkv"]``.
    recursive:   If True, search all subdirectories.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        log.warning("Input directory does not exist: %s", input_dir)
        return []

    glob_fn = input_dir.rglob if recursive else input_dir.glob
    files: List[Path] = []
    for ext in extensions:
        files.extend(glob_fn(f"*{ext}"))
        files.extend(glob_fn(f"*{ext.upper()}"))   # handle .MP4 etc.

    # Deduplicate and sort deterministically
    files = sorted(set(files))
    log.info("Found %d video file(s) in %s", len(files), input_dir)
    return files


def batch_extract_audio(
    input_dir: Path,
    output_dir: Path,
    sample_rate: int = 16_000,
    channels: int = 1,
    codec: str = "pcm_s16le",
    ffmpeg_loglevel: str = "error",
    overwrite: bool = False,
    supported_extensions: Optional[List[str]] = None,
    num_workers: int = 2,
    recursive: bool = False,
) -> dict:
    """
    Batch-extract audio from all video files in *input_dir*.

    Uses a :class:`concurrent.futures.ThreadPoolExecutor` to run multiple
    ffmpeg processes in parallel.  ffmpeg itself is CPU/IO bound and does
    not hold the Python GIL, so threads are appropriate here.

    Parameters
    ----------
    input_dir:            Directory containing raw video files.
    output_dir:           Directory where WAV files will be written.
    sample_rate:          Target sample rate (Hz).
    channels:             1 = mono.
    codec:                PCM codec for ffmpeg.
    ffmpeg_loglevel:      ffmpeg verbosity.
    overwrite:            Overwrite existing WAV files.
    supported_extensions: Video file extensions to discover.
    num_workers:          Parallel ffmpeg processes.
    recursive:            Search subdirectories.

    Returns
    -------
    dict with keys: ``total``, ``succeeded``, ``failed``, ``skipped``,
    ``results`` (list of per-file dicts).
    """
    if supported_extensions is None:
        supported_extensions = [
            ".mp4", ".mkv", ".avi", ".mov", ".wmv",
            ".flv", ".webm", ".m4v", ".ts",
        ]

    video_files = find_video_files(input_dir, supported_extensions, recursive)
    if not video_files:
        log.warning("No video files found.  Nothing to do.")
        return {"total": 0, "succeeded": 0, "failed": 0, "skipped": 0, "results": []}

    log.info(
        "Starting batch extraction: %d file(s)  |  workers=%d  |  "
        "target=%dHz mono %s",
        len(video_files), num_workers, sample_rate, codec,
    )

    stats = {"total": len(video_files), "succeeded": 0, "failed": 0, "skipped": 0}
    results = []

    def _worker(vp: Path) -> dict:
        success, wav_path, msg = extract_audio(
            video_path=vp,
            output_dir=output_dir,
            sample_rate=sample_rate,
            channels=channels,
            codec=codec,
            ffmpeg_loglevel=ffmpeg_loglevel,
            overwrite=overwrite,
        )
        return {
            "video_file": str(vp),
            "wav_file": str(wav_path),
            "success": success,
            "message": msg,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_worker, vf): vf for vf in video_files}
        with tqdm(
            total=len(video_files),
            desc="Extracting audio",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                r = fut.result()
                results.append(r)
                if r["success"]:
                    if r["message"] == "skipped":
                        stats["skipped"] += 1
                    else:
                        stats["succeeded"] += 1
                else:
                    stats["failed"] += 1
                pbar.update(1)

    stats["results"] = results
    _log_batch_summary(stats)
    return stats


def _log_batch_summary(stats: dict) -> None:
    log.info(
        "Extraction complete — total: %d  |  ok: %d  |  skipped: %d  |  failed: %d",
        stats["total"], stats["succeeded"], stats["skipped"], stats["failed"],
    )
    if stats["failed"] > 0:
        log.warning(
            "%d file(s) failed extraction — check logs for details.",
            stats["failed"],
        )


# ── CLI entry-point ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.extract_audio",
        description=(
            "Stage 1 Part 1 — Extract mono 16kHz PCM WAV audio from video files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="Path to YAML config file.  Defaults to configs/stage1_config.yaml.",
    )
    p.add_argument(
        "--input", "-i", type=Path,
        help="Input directory containing video files (overrides config).",
    )
    p.add_argument(
        "--output", "-o", type=Path,
        help="Output directory for WAV files (overrides config).",
    )
    p.add_argument(
        "--workers", "-w", type=int,
        help="Number of parallel ffmpeg workers (overrides config).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing WAV files.",
    )
    p.add_argument(
        "--recursive", "-r", action="store_true",
        help="Search input directory recursively.",
    )
    p.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None, help="Console log verbosity.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    # CLI args override config
    log_level = args.log_level or cfg.logging.level
    log_dir = Path(cfg.paths.logs)

    # Re-initialise logger with resolved settings
    global log
    log = get_logger(
        "bwc_pipeline.extract_audio",
        level=log_level,
        log_dir=log_dir,
        log_filename=cfg.logging.log_filename,
        rich_console=cfg.logging.rich_console,
    )

    input_dir = Path(args.input) if args.input else Path(cfg.paths.raw_video)
    output_dir = Path(args.output) if args.output else Path(cfg.paths.raw_audio)
    workers = args.workers if args.workers else cfg.extraction.num_workers

    log.info("=" * 60)
    log.info("BWC Pipeline — Stage 1 Part 1: Audio Extraction")
    log.info("  Input  : %s", input_dir.resolve())
    log.info("  Output : %s", output_dir.resolve())
    log.info("  Workers: %d", workers)
    log.info("=" * 60)

    stats = batch_extract_audio(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_rate=cfg.extraction.target_sample_rate,
        channels=cfg.extraction.channels,
        codec=cfg.extraction.codec,
        ffmpeg_loglevel=cfg.extraction.ffmpeg_loglevel,
        overwrite=args.overwrite or cfg.extraction.overwrite_existing,
        supported_extensions=list(cfg.extraction.supported_video_extensions),
        num_workers=workers,
        recursive=args.recursive,
    )

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
