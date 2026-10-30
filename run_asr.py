"""
scripts/run_asr.py
==================
Batch ASR + alignment (optional diarization) over raw_audio/ using WhisperX.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import whisperx
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from scripts.utils.logger import get_logger

PROJECT_ROOT = Path.cwd()

RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
ALIGNMENTS_DIR = PROJECT_ROOT / "alignments"
DIARIZATION_DIR = PROJECT_ROOT / "diarization"
LOGS_DIR = PROJECT_ROOT / "logs"

EXTERNAL_WHISPERX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".opus",
}


def _load_module(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing WhisperX module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_dirs() -> None:
    for d in (TRANSCRIPTS_DIR, ALIGNMENTS_DIR, DIARIZATION_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _discover_audio(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def _format_plain_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        start = round(seg["start"], 2)
        end = round(seg["end"], 2)
        text = seg["text"]
        lines.append(f"[{start}s - {end}s] {text}")
    return "\n".join(lines)


def _json_default(obj):
    try:
        import numpy as np
    except ImportError:
        np = None

    try:
        import torch
    except ImportError:
        torch = None

    if np is not None:
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()

    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    return str(obj)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        if text:
            f.write(text.strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch WhisperX ASR + alignment for raw_audio/."
    )
    parser.add_argument(
        "--model-size",
        default="large-v3",
        help="Whisper model size to use (e.g., small, medium, large-v3).",
    )
    parser.add_argument(
        "--input",
        default=str(RAW_AUDIO_DIR),
        help="Input audio file or directory (default: raw_audio/).",
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        help="Enable speaker diarization (requires HF_TOKEN in external/.env).",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Force a language code (e.g., en, es) to skip auto-detection.",
    )
    parser.add_argument(
        "--align-model",
        default=None,
        help="Override the WhisperX alignment model name.",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Skip alignment and use raw ASR segments.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional max duration in seconds for validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    args = parser.parse_args()

    _ensure_dirs()

    log = get_logger(
        name="bwc_pipeline.asr",
        level="INFO",
        log_dir=LOGS_DIR,
        log_filename="asr_pipeline.log",
    )

    if not EXTERNAL_WHISPERX_DIR.exists():
        raise FileNotFoundError(
            "WhisperX repo not found at "
            f"{EXTERNAL_WHISPERX_DIR}. Clone it into external/ first."
        )

    whisperx_utils = _load_module(
        "whisperx_utils",
        EXTERNAL_WHISPERX_DIR / "utils.py",
    )
    whisperx_diarize = _load_module(
        "whisperx_diarize",
        EXTERNAL_WHISPERX_DIR / "diarize.py",
    )

    input_path = Path(args.input)
    audio_files = _discover_audio(input_path)

    if not audio_files:
        log.warning("No audio files found in %s", input_path)
        return

    log.info("Discovered %d audio file(s).", len(audio_files))

    asr_model = whisperx_utils.load_models(model_size=args.model_size)
    diarize_model: Optional[object] = None

    failed: List[str] = []

    for audio_path in tqdm(audio_files):
        transcript_path = TRANSCRIPTS_DIR / f"{audio_path.stem}.txt"
        alignment_path = ALIGNMENTS_DIR / f"{audio_path.stem}_aligned.json"
        diarization_path = DIARIZATION_DIR / f"{audio_path.stem}_diarized.json"

        if not args.overwrite:
            already_done = (
                transcript_path.exists()
                and alignment_path.exists()
                and (not args.diarization or diarization_path.exists())
            )
            if already_done:
                log.info("Skipping %s (outputs already exist).", audio_path.name)
                continue

        if args.max_duration is not None:
            whisperx_utils.validate_audio_length(
                str(audio_path),
                max_duration=args.max_duration,
            )

        try:
            log.info("Transcribing: %s", audio_path.name)

            audio = whisperx.load_audio(str(audio_path))
            if args.language:
                result = asr_model.transcribe(audio, language=args.language)
            else:
                result = asr_model.transcribe(audio)

            aligned_result = None
            alignment_error = None
            language_code = args.language or result.get("language")

            if not args.no_align:
                try:
                    if not language_code:
                        raise ValueError(
                            "Language could not be detected. "
                            "Specify --language to continue."
                        )
                    model_a, metadata = whisperx.load_align_model(
                        language_code=language_code,
                        device=whisperx_utils.DEVICE,
                        model_name=args.align_model,
                    )
                    aligned_result = whisperx.align(
                        result["segments"],
                        model_a,
                        metadata,
                        audio,
                        whisperx_utils.DEVICE,
                    )
                except Exception as e:
                    alignment_error = str(e)
                    log.error(
                        "Alignment failed for %s: %s",
                        audio_path.name,
                        alignment_error,
                    )

            if args.diarization and (aligned_result is None):
                log.error(
                    "Diarization requested but alignment was skipped "
                    "or failed for %s. Skipping diarization.",
                    audio_path.name,
                )

            if aligned_result is None:
                alignment_error = alignment_error or (
                    "Alignment skipped via --no-align."
                    if args.no_align
                    else "Alignment failed."
                )
                alignment_payload = {
                    "aligned": False,
                    "language": language_code,
                    "error": alignment_error,
                    "segments": result.get("segments", []),
                }
                transcript_segments = result.get("segments", [])
            else:
                alignment_payload = aligned_result
                transcript_segments = aligned_result["segments"]

            if args.diarization and aligned_result is not None:
                if diarize_model is None:
                    diarize_model = whisperx_utils.load_diarization_model()

                diarize_segments = whisperx_diarize.perform_diarization(
                    audio,
                    diarize_model,
                )
                final_result = whisperx_diarize.assign_speakers(
                    diarize_segments,
                    aligned_result,
                )
                transcript = whisperx_utils.format_transcript(
                    final_result["segments"],
                )
                _write_json(diarization_path, final_result)
                log.info("Saved diarization → %s", diarization_path.name)
            else:
                transcript = _format_plain_transcript(
                    transcript_segments,
                )

            _write_text(transcript_path, transcript)
            _write_json(alignment_path, alignment_payload)

            log.info("Saved transcript → %s", transcript_path.name)
            log.info("Saved alignment → %s", alignment_path.name)

        except Exception as e:
            failed.append(audio_path.name)
            log.error("FAILED: %s", audio_path.name)
            log.error(str(e))

    if failed:
        log.warning("Failed files: %d", len(failed))
        for name in failed:
            log.warning("  - %s", name)


if __name__ == "__main__":
    main()
