import json
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path.cwd()))

from scripts.utils.audio_utils import load_audio, SILERO_SR, get_duration_sec
from scripts.utils.device import get_device

PROJECT_ROOT = Path.cwd()

def apply_min_cut(scores, start_idx, end_idx, max_len):
    duration = end_idx - start_idx
    if duration <= max_len:
        return [(start_idx, end_idx)]
    search_start, search_end = start_idx + max_len // 2, min(start_idx + max_len, end_idx)
    if search_start >= search_end: return [(start_idx, end_idx)]
    cut_idx = search_start + np.argmin(scores[search_start:search_end])
    return apply_min_cut(scores, start_idx, cut_idx, max_len) + apply_min_cut(scores, cut_idx, end_idx, max_len)

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

def process_audio(audio_path, model, device, output_dir: Path):
    waveform, sr = load_audio(audio_path, target_sr=SILERO_SR, mono=True)
    window_size = 512
    timestep = window_size / sr
    waveform_1d = waveform.squeeze(0).to(device)
    scores = []
    with torch.no_grad():
        for i in range(0, waveform_1d.shape[0], window_size):
            chunk = waveform_1d[i:i+window_size]
            if chunk.shape[0] < window_size: chunk = torch.nn.functional.pad(chunk, (0, window_size-chunk.shape[0]))
            scores.append(model(chunk, sr).item())
    scores_arr = np.array(scores)

    active = False
    start_f = 0
    raw_indices = []
    for i, s in enumerate(scores_arr):
        if not active and s > 0.3:
            start_f, active = i, True
        elif active and s < 0.25:
            raw_indices.append((start_f, i))
            active = False
    if active: raw_indices.append((start_f, len(scores_arr)))

    max_f = int(30.0 / timestep)
    cut_indices = []
    for s, e in raw_indices:
        cut_indices.extend(apply_min_cut(scores_arr, s, e, max_f))
    
    raw_segs_sec = [(s * timestep, e * timestep) for s, e in cut_indices]
    final_segments = merge_segments(raw_segs_sec, 28.0)

    out_path = output_dir / f"{audio_path.stem}_grounding.json"
    with open(out_path, "w") as f:
        json.dump({"file": audio_path.name, "segments": final_segments, "vad_type": "silero"}, f, indent=2)
    print(f"Created {len(final_segments)} grounding segments for {audio_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Silero VAD")
    parser.add_argument("--audio-dir", default="raw_audio")
    parser.add_argument("--output-dir", default="experiments/phase1_vad/grounding_silero")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    repo = PROJECT_ROOT / "external" / "silero-vad"
    model, _ = torch.hub.load(repo_or_dir=str(repo), model="silero_vad", source="local")
    model = model.to(device)
    
    for audio_path in audio_dir.glob("*.wav"):
        process_audio(audio_path, model, device, output_dir)

if __name__ == "__main__":
    main()
