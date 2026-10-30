import os
import json
import torch
import yaml
import argparse
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub
from dotenv import dotenv_values

# Setup paths
PROJECT_ROOT = Path.cwd()

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

def run_vad_pyannote31(pipeline, audio_path, merge_gap=0.15, min_duration=0.1):
    print(f"Running Pyannote 3.1 VAD on {audio_path.name}...")
    vad = pipeline(str(audio_path))
    
    segments = []
    for s in vad.get_timeline().support():
        segments.append({"start": round(s.start, 3), "end": round(s.end, 3)})
    
    raw_count = len(segments)
    
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

    merged_count = len(cleaned)
    total_speech_duration = sum(s["end"] - s["start"] for s in cleaned)
    
    return cleaned, raw_count, merged_count, total_speech_duration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(Path(__file__).parent))
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    audio_path = Path(args.audio_path)
    grounding_path = Path(args.output_json)

    env_path = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform" / ".env"
    if not env_path.exists():
        env_path = PROJECT_ROOT / ".env"

    env_values = dotenv_values(env_path)
    hf_token = env_values.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set in .env")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    grounding_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading Pyannote VAD Pipeline...")
    pipeline = load_pyannote_vad_pipeline(hf_token).to(torch.device(device))
    
    segments, raw_count, merged_count, total_speech_duration = run_vad_pyannote31(pipeline, audio_path)
    
    out_data = {
        "file": audio_path.name,
        "segments": segments,
        "raw_count": raw_count,
        "merged_count": merged_count,
        "total_speech_duration": round(total_speech_duration, 3)
    }
    
    with open(grounding_path, "w") as f:
        json.dump(out_data, f, indent=2)
    
    print(f"Saved grounding to {grounding_path}")
    print(f"Number of raw segments: {raw_count}")
    print(f"Number of merged segments: {merged_count}")
    print(f"Total speech duration: {total_speech_duration:.3f}s")

if __name__ == "__main__":
    main()

