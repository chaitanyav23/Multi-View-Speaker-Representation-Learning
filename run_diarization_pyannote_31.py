"""
run_diarization_pyannote_31.py
==============================
Standalone Speaker Diarization using Pyannote 3.1 directly.
Outputs speaker labels and time segments to diarization_outputs_pyannote_3.1/.
"""

import os
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from pyannote.audio import Pipeline
from dotenv import dotenv_values

# Setup paths
PROJECT_ROOT = Path.cwd()
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
OUTPUT_DIR = PROJECT_ROOT / "diarization_outputs_pyannote_3.1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load HF Token from .env
ENV_PATH = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform" / ".env"
if not ENV_PATH.exists():
    ENV_PATH = PROJECT_ROOT / ".env"

ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_diarization(audio_path, pipeline, num_speakers=None, min_speakers=None, max_speakers=None):
    """
    Run Pyannote 3.1 diarization on a single audio file.
    """
    print(f">>> Diarizing {audio_path.name}...")
    
    # Run the pipeline
    output = pipeline(str(audio_path), num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
    
    # In pyannote-audio >= 3.3, the pipeline returns a DiarizeOutput object.
    # We must access the .speaker_diarization attribute (which is an Annotation object).
    diarization = output.speaker_diarization
    
    # Process results into a list of segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker
        })
    
    return segments

def main():
    parser = argparse.ArgumentParser(description="Diarization with Pyannote 3.1")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Error: HF_TOKEN not found in .env file.")
        return

    print(f"Loading Pyannote 3.1 pipeline on {DEVICE}...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        ).to(DEVICE)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    audio_files = sorted(list(RAW_AUDIO_DIR.glob("*.wav")))
    
    for audio_path in audio_files:
        output_json = OUTPUT_DIR / f"{audio_path.stem}_diarization.json"
        
        if not args.overwrite and output_json.exists():
            print(f"Skipping {audio_path.name} (exists).")
            continue
            
        try:
            segments = run_diarization(
                audio_path, 
                pipeline, 
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers
            )
            
            output_data = {
                "file": audio_path.name,
                "segments": segments
            }
            
            with open(output_json, "w") as f:
                json.dump(output_data, f, indent=2)
                
            print(f"Saved: {output_json.name}")
            
        except Exception as e:
            print(f"Failed to process {audio_path.name}: {e}")

if __name__ == "__main__":
    main()
