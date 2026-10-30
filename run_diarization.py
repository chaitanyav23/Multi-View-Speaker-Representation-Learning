"""
run_diarization.py
==================
Standalone Stage 3: Speaker Diarization.
Processes audio from raw_audio/ and saves diarization segments to diarization/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import whisperx
from pathlib import Path
from typing import List

from tqdm import tqdm

from scripts.utils.logger import get_logger

PROJECT_ROOT = Path.cwd()

RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
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
    for d in (DIARIZATION_DIR, LOGS_DIR):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Speaker Diarization (Stage 3)."
    )
    parser.add_argument(
        "--input",
        default=str(RAW_AUDIO_DIR),
        help="Input audio file or directory (default: raw_audio/).",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Optional minimum number of speakers.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Optional maximum number of speakers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )
    args = parser.parse_args()

    _ensure_dirs()

    log = get_logger(
        name="bwc_pipeline.diarization",
        level="INFO",
        log_dir=LOGS_DIR,
        log_filename="diarization_stage3.log",
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

    # Load diarization model (requires HF_TOKEN)
    diarize_model = whisperx_utils.load_diarization_model()

    failed: List[str] = []

    for audio_path in tqdm(audio_files):
        diarization_path = DIARIZATION_DIR / f"{audio_path.stem}_diarized.json"

        if not args.overwrite and diarization_path.exists():
            log.info("Skipping %s (output already exists).", audio_path.name)
            continue

        try:
            log.info("Diarizing: %s", audio_path.name)

            audio = whisperx.load_audio(str(audio_path))
            
            # Run diarization
            diarize_segments = diarize_model(
                audio, 
                min_speakers=args.min_speakers, 
                max_speakers=args.max_speakers
            )

            # Convert DataFrame to list of dicts if necessary
            # WhisperX DiarizationPipeline returns a pandas DataFrame
            # with columns: start, end, speaker
            if hasattr(diarize_segments, "to_dict"):
                segments_list = diarize_segments.to_dict(orient="records")
            else:
                segments_list = diarize_segments

            payload = {
                "file": audio_path.name,
                "segments": segments_list
            }

            _write_json(diarization_path, payload)
            log.info("Saved diarization → %s", diarization_path.name)

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
