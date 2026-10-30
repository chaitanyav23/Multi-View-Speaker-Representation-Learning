import json
import torch
import numpy as np
import argparse
from pathlib import Path
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from dotenv import dotenv_values
import sys

# Add project root to path
sys.path.append(str(Path.cwd()))
from scripts.utils.device import get_device

PROJECT_ROOT = Path.cwd()
EXTERNAL_WX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"
ENV_VALUES = dotenv_values(EXTERNAL_WX_DIR / ".env")
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

def merge_segments(segments, max_dur_sec):
    if not segments: return []
    merged = []
    curr_start = segments[0][0]
    curr_end = segments[0][1]
    for i in range(1, len(segments)):
        next_start, next_end = segments[i]
        if (next_end - curr_start) <= max_dur_sec:
            curr_end = next_end
        else:
            merged.append({"start": round(curr_start, 3), "end": round(curr_end, 3)})
            curr_start, curr_end = next_start, next_end
    merged.append({"start": round(curr_start, 3), "end": round(curr_end, 3)})
    return merged

def process_audio(audio_path, pipeline, output_dir: Path):
    print(f"Processing: {audio_path.name}")
    vad_res = pipeline(str(audio_path))
    
    raw_segs = []
    for segment in vad_res.itersegments():
        raw_segs.append((segment.start, segment.end))
        
    # Apply merge logic (30s target for context)
    final_segments = merge_segments(raw_segs, 28.0)
    
    out_path = output_dir / f"{audio_path.stem}_grounding.json"
    with open(out_path, "w") as f:
        json.dump({"file": audio_path.name, "segments": final_segments, "vad_type": "pyannote_v3"}, f, indent=2)
    print(f"Created {len(final_segments)} grounding segments for {audio_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Pyannote VAD")
    parser.add_argument("--audio-dir", default="raw_audio")
    parser.add_argument("--output-dir", default="experiments/phase1_vad/grounding_pyannote")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    # Use segmentation 3.0 model wrapped in a VAD pipeline
    model = Model.from_pretrained("pyannote/segmentation-3.0", token=HF_TOKEN)
    pipeline = VoiceActivityDetection(segmentation=model)
    # Explicitly instantiate with parameters to avoid NotImplementedError/RuntimeError
    pipeline.instantiate({"min_duration_on": 0.1, "min_duration_off": 0.1})
    pipeline.to(torch.device(device))
    
    for audio_path in audio_dir.glob("*.wav"):
        process_audio(audio_path, pipeline, output_dir)

if __name__ == "__main__":
    main()
