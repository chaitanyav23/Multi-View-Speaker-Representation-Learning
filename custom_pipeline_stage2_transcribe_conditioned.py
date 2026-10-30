"""
custom_pipeline_stage2_transcribe_conditioned.py
================================================
Modified custom transcription pipeline with context conditioning enabled.
"""

import json
import argparse
import torch
import whisperx
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scripts.utils.audio_utils import load_audio
import importlib.util

PROJECT_ROOT = Path.cwd()
EXTERNAL_WHISPERX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"
GROUNDING_DIR = PROJECT_ROOT / "vad_custom_grounding"
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
OUTPUT_DIR = PROJECT_ROOT / "transcripts_custom_conditioned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

whisperx_utils = _load_module("whisperx_utils", EXTERNAL_WHISPERX_DIR / "utils.py")

def transcribe_manual_slices(audio_path, grounding_path, model_pipeline, language="en", overwrite=False):
    out_json = OUTPUT_DIR / f"{audio_path.stem}_raw_text.json"
    if not overwrite and out_json.exists():
        print(f"Skipping {audio_path.stem} (output exists).")
        return

    # 1. Load data
    audio = whisperx.load_audio(str(audio_path))
    with open(grounding_path, "r") as f:
        grounding = json.load(f)
    
    segments = grounding["segments"]
    print(f"\n>>> Transcribing (Conditioned) {audio_path.name} ({len(segments)} segments)")

    results = []
    sample_rate = 16000 
    
    for i, seg in enumerate(tqdm(segments, desc="Segments")):
        start_samp = int(seg["start"] * sample_rate)
        end_samp = int(seg["end"] * sample_rate)
        audio_slice = audio[start_samp:end_samp]
        
        # Hybrid Approach: We slice but we allow Whisper to use context 
        # from the PREVIOUS SLICE if available.
        
        gen_result, _ = model_pipeline.model.transcribe(
            audio_slice,
            language=language,
            task="transcribe",
            word_timestamps=False,
            condition_on_previous_text=True # Re-enabled for noisy BWC context
        )
        
        full_text = ""
        for s in gen_result:
            full_text += s.text
            
        results.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": full_text.strip()
        })

    final_output = {
        "segments": results,
        "language": language
    }
    
    with open(out_json, "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Saved: {out_json.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-size", default="large-v3")
    args = parser.parse_args()

    print(f"Loading Whisper {args.model_size}...")
    model = whisperx.load_model(
        args.model_size, 
        whisperx_utils.DEVICE, 
        compute_type="float16"
    )
    
    grounding_files = sorted(list(GROUNDING_DIR.glob("*.json")))
    for grounding_path in grounding_files:
        stem = grounding_path.stem.replace("_grounding", "")
        audio_path = RAW_AUDIO_DIR / f"{stem}.wav"
        
        if audio_path.exists():
            transcribe_manual_slices(audio_path, grounding_path, model, language=args.language, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
