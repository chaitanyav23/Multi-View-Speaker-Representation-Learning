"""
custom_pipeline_stage3_align_conditioned.py
===========================================
Alignment script for the conditioned transcription outputs.
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
RAW_TEXT_DIR = PROJECT_ROOT / "transcripts_custom_conditioned"
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
OUTPUT_DIR = PROJECT_ROOT / "asr_outputs_wx_custom_conditioned"
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

    print(f"\n>>> Aligning (Conditioned) {audio_path.name}...")

    audio = whisperx.load_audio(str(audio_path))
    with open(raw_text_path, "r") as f:
        data = json.load(f)
    
    language = data.get("language", "en")
    segments = data.get("segments", [])

    model_a, metadata = whisperx.load_align_model(
        language_code=language, 
        device=whisperx_utils.DEVICE
    )

    result = whisperx.align(
        segments, 
        model_a, 
        metadata, 
        audio, 
        whisperx_utils.DEVICE, 
        return_char_alignments=False
    )

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved: {out_json.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_files = sorted(list(RAW_TEXT_DIR.glob("*_raw_text.json")))
    for raw_path in raw_files:
        stem = raw_path.stem.replace("_raw_text", "")
        audio_path = RAW_AUDIO_DIR / f"{stem}.wav"
        
        if audio_path.exists():
            run_alignment(audio_path, raw_path, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
