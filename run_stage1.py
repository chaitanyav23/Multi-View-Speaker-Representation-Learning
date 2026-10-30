"""
scripts/run_stage1.py
=====================

Stage-1 Operational Audio Pipeline
==================================

Pipeline:
---------
raw_video
    → audio extraction
    → mono 16kHz wav
    → Silero VAD
    → speech segment extraction
    → JSON export
    → CSV analytics
    → dataset manifest generation

This script orchestrates the COMPLETE Stage-1 pipeline.

Compatible with:
- extract_audio.py
- vad_engine.py
- utils/*
already present in the project.

------------------------------------------------------------
USAGE
------------------------------------------------------------

python -m scripts.run_stage1

OR

python scripts/run_stage1.py

------------------------------------------------------------
EXPECTED PROJECT STRUCTURE
------------------------------------------------------------

project/
│
├── raw_video/
├── raw_audio/
├── vad_outputs/
├── speech_segments/
├── reports/
├── logs/
├── scripts/
│   ├── extract_audio.py
│   ├── vad_engine.py
│   └── utils/
│
└── configs/

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

1. raw_audio/*.wav
2. vad_outputs/*_vad.json
3. speech_segments/<video_name>/*.wav
4. reports/stage1_summary.csv
5. reports/dataset_manifest.csv
6. logs/stage1_pipeline.log
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Internal imports
# ─────────────────────────────────────────────────────────────

from scripts.extract_audio import extract_audio

from scripts.vad_engine import (
    load_silero_model,
    run_vad_on_file,
)

from scripts.utils.csv_reporter import append_vad_result_row

from scripts.utils.device import get_device

from scripts.utils.json_export import save_vad_result

from scripts.utils.logger import get_logger

from scripts.evaluate_vad import (
    evaluate_vad,
    append_vad_eval_row,
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path.cwd()

RAW_VIDEO_DIR = PROJECT_ROOT / "raw_video"
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"

VAD_OUTPUT_DIR = PROJECT_ROOT / "vad_outputs"
SPEECH_SEGMENTS_DIR = PROJECT_ROOT / "speech_segments"

REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

CSV_SUMMARY_PATH = REPORTS_DIR / "stage1_summary.csv"
MANIFEST_PATH = REPORTS_DIR / "dataset_manifest.csv"

GT_MAPPING = {
    "bwc": PROJECT_ROOT / "ground_truth" / "bwc.txt",
    "podcast": PROJECT_ROOT / "ground_truth" / "podcast.txt",
}

# ─────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────

log = get_logger(
    name="bwc_pipeline",
    level="INFO",
    log_dir=LOGS_DIR,
    log_filename="stage1_pipeline.log",
)


# ─────────────────────────────────────────────────────────────
# Supported video extensions
# ─────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
}


# ─────────────────────────────────────────────────────────────
# Directory setup
# ─────────────────────────────────────────────────────────────

def ensure_directories() -> None:
    dirs = [
        RAW_VIDEO_DIR,
        RAW_AUDIO_DIR,
        VAD_OUTPUT_DIR,
        SPEECH_SEGMENTS_DIR,
        REPORTS_DIR,
        LOGS_DIR,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Video discovery
# ─────────────────────────────────────────────────────────────

def discover_videos() -> List[Path]:
    videos = []

    for path in RAW_VIDEO_DIR.iterdir():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    videos = sorted(videos)

    log.info("Discovered %d video(s).", len(videos))

    return videos


# ─────────────────────────────────────────────────────────────
# Metadata heuristics
# ─────────────────────────────────────────────────────────────

def infer_source_type(filename: str) -> str:
    fname = filename.lower()

    if "bwc" in fname:
        return "bwc_police"

    if "podcast" in fname:
        return "podcast"

    return "unknown"


def classify_conversation(fragmentation_index: float) -> str:
    if fragmentation_index >= 15:
        return "fragmented"

    if fragmentation_index >= 8:
        return "mixed"

    return "sustained"


def classify_speech_density(speech_ratio: float) -> str:
    if speech_ratio >= 0.75:
        return "high"

    if speech_ratio >= 0.5:
        return "medium"

    return "low"


# ─────────────────────────────────────────────────────────────
# Dataset manifest update
# ─────────────────────────────────────────────────────────────

def update_manifest(vad_result) -> None:
    row = {
        "video_id": Path(vad_result.file_name).stem,
        "filename": vad_result.file_name,
        "source_type": infer_source_type(vad_result.file_name),

        "duration_sec": vad_result.duration_sec,
        "speech_duration_sec": vad_result.speech_duration_sec,
        "silence_duration_sec": vad_result.silence_duration_sec,

        "speech_ratio": vad_result.speech_ratio,

        "num_segments": vad_result.analytics.num_speech_segments,

        "avg_segment_duration_sec":
            vad_result.analytics.avg_segment_duration_sec,

        "fragmentation_index":
            vad_result.analytics.fragmentation_index,

        "conversation_structure":
            classify_conversation(
                vad_result.analytics.fragmentation_index
            ),

        "speech_density":
            classify_speech_density(
                vad_result.speech_ratio
            ),

        "short_fragment_count":
            vad_result.analytics.short_fragment_count,

        "long_silence_count":
            vad_result.analytics.long_silence_count,

        "potential_missed_speech_regions":
            vad_result.analytics.potential_missed_speech_regions,

        "notes": "",
    }

    if MANIFEST_PATH.exists():
        df = pd.read_csv(MANIFEST_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(MANIFEST_PATH, index=False)


# ─────────────────────────────────────────────────────────────
# Single-file pipeline
# ─────────────────────────────────────────────────────────────

def process_video(
    video_path: Path,
    model,
    utils,
    device,
) -> None:

    log.info("=" * 80)
    log.info("PROCESSING: %s", video_path.name)
    log.info("=" * 80)

    # --------------------------------------------------------
    # Audio extraction
    # --------------------------------------------------------

    wav_output_path = RAW_AUDIO_DIR / f"{video_path.stem}.wav"

    if not wav_output_path.exists():

        log.info("Extracting audio → %s", wav_output_path.name)

        extract_audio(
            video_path=video_path,
            output_dir=RAW_AUDIO_DIR,
            sample_rate=16000,
            channels=1,
        )

    else:
        log.info("Audio already exists. Skipping extraction.")

    # --------------------------------------------------------
    # VAD
    # --------------------------------------------------------

    vad_config = {
        "threshold": 0.35,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "window_size_samples": 512,
        "speech_pad_ms": 300,
        "chunk_size_sec": 60,
        "chunk_overlap_sec": 1,
        "merge_gap_sec": 1.0,
        "min_segment_duration_sec": 0.2,
    }

    vad_result = run_vad_on_file(
        wav_path=wav_output_path,
        model=model,
        utils=utils,
        device=device,

        source_video=video_path.name,

        threshold=vad_config["threshold"],
        min_speech_duration_ms=
            vad_config["min_speech_duration_ms"],

        min_silence_duration_ms=
            vad_config["min_silence_duration_ms"],

        window_size_samples=
            vad_config["window_size_samples"],

        speech_pad_ms=
            vad_config["speech_pad_ms"],

        chunk_size_sec=
            vad_config["chunk_size_sec"],

        chunk_overlap_sec=
            vad_config["chunk_overlap_sec"],

        merge_gap_sec=
            vad_config["merge_gap_sec"],

        min_segment_duration_sec=
            vad_config["min_segment_duration_sec"],

        output_dir_clips=SPEECH_SEGMENTS_DIR,

        pipeline_version="1.0.0",

        vad_config=vad_config,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_output_path = (
        VAD_OUTPUT_DIR /
        f"{video_path.stem}_vad.json"
    )

    save_vad_result(
        vad_result,
        json_output_path,
    )

    log.info("Saved JSON → %s", json_output_path.name)

    # --------------------------------------------------------
    # CSV analytics
    # --------------------------------------------------------

    append_vad_result_row(
        vad_result,
        CSV_SUMMARY_PATH,
    )

    log.info("Updated CSV summary.")

    # --------------------------------------------------------
    # Dataset manifest
    # --------------------------------------------------------

    update_manifest(vad_result)

    log.info("Updated dataset manifest.")


    # --------------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------------

    video_id = video_path.stem

    if video_id in GT_MAPPING:
    
        metrics = evaluate_vad(
            gt_file=GT_MAPPING[video_id],
            vad_json=json_output_path,
        )

        append_vad_eval_row(
            video_id=video_path.stem,
            metrics=metrics,
            csv_path=REPORTS_DIR / "vad_eval.csv",
        )

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:

    ensure_directories()

    videos = discover_videos()

    if not videos:
        log.warning("No videos found in raw_video/")
        return

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device(
        prefer_cuda=True,
    )

    # --------------------------------------------------------
    # Load Silero once
    # --------------------------------------------------------

    model, utils = load_silero_model(
        device=device,
        repo="external/silero-vad",
    )

    # --------------------------------------------------------
    # Process all videos
    # --------------------------------------------------------

    failed = []

    for video_path in tqdm(videos):

        try:
            process_video(
                video_path=video_path,
                model=model,
                utils=utils,
                device=device,
            )

        except Exception as e:

            failed.append(video_path.name)

            log.error(
                "FAILED: %s",
                video_path.name,
            )

            log.error(str(e))
            log.error(traceback.format_exc())

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    log.info("=" * 80)
    log.info("STAGE-1 PIPELINE COMPLETE")
    log.info("=" * 80)

    log.info("Processed: %d", len(videos))
    log.info("Failed: %d", len(failed))

    if failed:
        log.warning("Failed files:")
        for f in failed:
            log.warning("  - %s", f)

    log.info("Outputs:")
    log.info("  raw_audio/")
    log.info("  vad_outputs/")
    log.info("  speech_segments/")
    log.info("  reports/")


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
