"""
custom_pipeline_stage3_align.py
================================
Implements Section 2.4: Forced Phoneme Alignment.
Uses Wav2Vec2.0 to obtain word-level timestamps from the grounded text.
"""

import json
import argparse
import torch
import whisperx
from pathlib import Path
from tqdm import tqdm
import importlib.util

PROJECT_ROOT = Path.cwd()
EXTERNAL_WHISPERX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"
RAW_TEXT_DIR = PROJECT_ROOT / "transcripts_custom"
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
OUTPUT_DIR = PROJECT_ROOT / "asr_outputs_wx_custom"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

whisperx_utils = _load_module("whisperx_utils", EXTERNAL_WHISPERX_DIR / "utils.py")

def run_alignment(audio_path, raw_text_path, overwrite=False):
    out_json = OUTPUT_DIR / f"{audio_path.stem}_aligned.json"
    if not overwrite and out_json.exists():
        print(f"Skipping {audio_path.stem} (already aligned).")
        return

    print(f"\n>>> Aligning {audio_path.name}...")

    # 1. Load data
    audio = whisperx.load_audio(str(audio_path))
    with open(raw_text_path, "r") as f:
        data = json.load(f)
    
    language = data.get("language", "en")
    segments = data.get("segments", [])

    # 2. Section 2.4: Load Alignment Model
    # Paper Default: wav2vec2.0 BASE 960H
    model_a, metadata = whisperx.load_align_model(
        language_code=language, 
        device=whisperx_utils.DEVICE
    )

    # 3. Perform Alignment
    # "Applying Dynamic Time Warping (DTW) on the resulting logits matrix..."
    result = whisperx.align(
        segments, 
        model_a, 
        metadata, 
        audio, 
        whisperx_utils.DEVICE, 
        return_char_alignments=False
    )

    # 4. Save Final Output
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {out_json.name}")

def main():
    parser = argparse.ArgumentParser(description="Custom Phoneme Alignment.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing.")
    args = parser.parse_args()

    raw_files = sorted(list(RAW_TEXT_DIR.glob("*_raw_text.json")))
    for raw_path in raw_files:
        stem = raw_path.stem.replace("_raw_text", "")
        audio_path = RAW_AUDIO_DIR / f"{stem}.wav"
        
        if audio_path.exists():
            run_alignment(audio_path, raw_path, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
