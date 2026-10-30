import os
import json
import torch
import yaml
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub
from dotenv import dotenv_values

ENV_PATH = Path("external/WhisperX-Audio-Intelligence-Platform/.env")
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set in .env")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path("experiments/phase10_dual_branch_fusion")
AUDIO_DIR = BASE_DIR / "audio_variants"
NORM_WAV = AUDIO_DIR / "bwc_normalized.wav"

GROUNDING_DIR = BASE_DIR / "grounding_master"
GROUNDING_OUT = GROUNDING_DIR / "bwc_vad.json"

def load_pyannote_vad_pipeline(hf_token):
    model_id = "pyannote/voice-activity-detection"
    config_path = download_from_hf_hub(model_id, AssetFileName.Pipeline, token=hf_token)
    if config_path is None:
        raise ValueError(f"Could not download pipeline config for {model_id}.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    segmentation = config.get("pipeline", {}).get("params", {}).get("segmentation")
    if isinstance(segmentation, str) and "@" in segmentation:
        checkpoint, revision = segmentation.split("@", 1)
        config["pipeline"]["params"]["segmentation"] = {
            "checkpoint": checkpoint,
            "revision": revision,
        }
    return Pipeline.from_pretrained(config, token=hf_token)

def run_vad_pyannote31(audio_path, merge_gap=0.15, min_duration=0.1):
    print(f"Running Pyannote 3.1 VAD on {audio_path.name}...")
    pipeline = load_pyannote_vad_pipeline(HF_TOKEN).to(torch.device(DEVICE))
    vad = pipeline(str(audio_path))
    
    segments = []
    for s in vad.get_timeline().support():
        segments.append({"start": round(s.start, 3), "end": round(s.end, 3)})
    
    # Cleaning
    cleaned = []
    if segments:
        curr = segments[0]
        for next_seg in segments[1:]:
            if next_seg["start"] - curr["end"] < merge_gap:
                curr["end"] = next_seg["end"]
            else:
                if (curr["end"] - curr["start"]) >= min_duration:
                    cleaned.append(curr)
                curr = next_seg
        if (curr["end"] - curr["start"]) >= min_duration:
            cleaned.append(curr)

    return cleaned

def main():
    GROUNDING_DIR.mkdir(parents=True, exist_ok=True)
    
    if GROUNDING_OUT.exists():
        print(f"Grounding already exists at {GROUNDING_OUT}, skipping...")
        return
        
    segments = run_vad_pyannote31(NORM_WAV)
    
    out_data = {
        "file": NORM_WAV.name,
        "segments": segments
    }
    
    with open(GROUNDING_OUT, "w") as f:
        json.dump(out_data, f, indent=2)
        
    print(f"Generated {len(segments)} VAD segments.")
    print(f"Saved to {GROUNDING_OUT}")

if __name__ == "__main__":
    main()
