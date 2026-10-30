"""
run_grounded_diarization_test.py
================================
Experimental script to run diarization on 'grounded' audio.
Replaces non-speech segments with silence before processing.
"""

import json
import torch
import numpy as np
import whisperx
from pathlib import Path
from tqdm import tqdm
from pyannote.audio import Pipeline
from dotenv import dotenv_values
import argparse

# Setup paths
PROJECT_ROOT = Path.cwd()
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
GROUNDING_DIR = PROJECT_ROOT / "vad_custom_grounding"
OUTPUT_DIR = PROJECT_ROOT / "diarization_grounded_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Auth
ENV_PATH = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform" / ".env"
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_grounded_audio(audio_path, grounding_path):
    """Loads audio and mutes everything not in grounding."""
    audio = whisperx.load_audio(str(audio_path))
    with open(grounding_path, "r") as f:
        grounding = json.load(f)
    
    # Create a mask (1.0 for speech, 0.0 for silence)
    mask = np.zeros_like(audio)
    sr = 16000
    for seg in grounding["segments"]:
        start_idx = int(seg["start"] * sr)
        end_idx = int(seg["end"] * sr)
        mask[start_idx:end_idx] = 1.0
        
    return audio * mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["whisperx", "pyannote31"], required=True)
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Error: HF_TOKEN not found.")
        return

    # Load Model
    if args.model == "pyannote31":
        print("Loading Pyannote 3.1...")
        model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=HF_TOKEN).to(DEVICE)
    else:
        print("Loading WhisperX Diarizer...")
        from whisperx.diarize import DiarizationPipeline
        model = DiarizationPipeline(token=HF_TOKEN, device=str(DEVICE))

    grounding_files = sorted(list(GROUNDING_DIR.glob("*.json")))
    
    for g_path in grounding_files:
        stem = g_path.stem.replace("_grounding", "")
        audio_path = RAW_AUDIO_DIR / f"{stem}.wav"
        
        if not audio_path.exists(): continue
        
        print(f"\n>>> Processing {stem} with {args.model} (Grounded Input)")
        grounded_audio = load_grounded_audio(audio_path, g_path)
        
        # Run Diarization
        if args.model == "pyannote31":
            # Pyannote expects a dict with 'waveform' and 'sample_rate'
            audio_tensor = torch.from_numpy(grounded_audio[None, :]).to(DEVICE)
            res = model({"waveform": audio_tensor, "sample_rate": 16000})
            diarization = res.speaker_diarization
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": float(round(turn.start, 3)), 
                    "end": float(round(turn.end, 3)), 
                    "speaker": str(speaker)
                })
        else:
            # WhisperX expects a numpy array
            df = model(grounded_audio, min_speakers=2, max_speakers=2)
            segments = []
            for _, row in df.iterrows():
                segments.append({
                    "start": float(round(row["start"], 3)),
                    "end": float(round(row["end"], 3)),
                    "speaker": str(row["speaker"])
                })

        # Save
        out_json = OUTPUT_DIR / f"{stem}_{args.model}_grounded.json"
        with open(out_json, "w") as f:
            json.dump({"file": audio_path.name, "segments": segments}, f, indent=2)
        print(f"Saved: {out_json.name}")

if __name__ == "__main__":
    main()
