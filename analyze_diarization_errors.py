import os
import json
import re
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# Regex for Ground Truth: Speaker X [HH:MM:SS - HH:MM:SS]
GT_PATTERN = re.compile(
    r"(Speaker\s+\d+)\s*\[(\d+):(\d+):(\d+)\s*-\s*(\d+):(\d+):(\d+)\]"
)

def hms_to_seconds(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)

def parse_ground_truth(gt_file: Path) -> List[Dict]:
    segments = []
    if not gt_file.exists(): return []
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            match = GT_PATTERN.search(line)
            if not match: continue
            speaker, h1, m1, s1, h2, m2, s2 = match.groups()
            start = hms_to_seconds(h1, m1, s1)
            end = hms_to_seconds(h2, m2, s2)
            if end <= start: continue
            segments.append({"start": float(start), "end": float(end), "speaker": speaker})
    return segments

def load_prediction(pred_json: Path) -> List[Dict]:
    with open(pred_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = []
    # Support both list format and object with 'segments' key
    segs = data.get("segments", data) if isinstance(data, dict) else data
    for seg in segs:
        segments.append({
            "start": float(seg["start"]), 
            "end": float(seg["end"]), 
            "speaker": seg.get("speaker", "UNKNOWN")
        })
    return segments

def identify_error_regions(gt_segments, pred_segments, step=0.01, collar=0.25):
    if not gt_segments and not pred_segments:
        return []
        
    max_t = max([s["end"] for s in gt_segments + pred_segments])
    n_frames = int(max_t / step) + 1
    
    gt_speakers = sorted(list(set(s["speaker"] for s in gt_segments)))
    pred_speakers = sorted(list(set(s["speaker"] for s in pred_segments)))
    
    gt_map = {name: i for i, name in enumerate(gt_speakers)}
    pred_map = {name: i for i, name in enumerate(pred_speakers)}
    
    gt_frames = np.full(n_frames, -1, dtype=int)
    pred_frames = np.full(n_frames, -1, dtype=int)
    
    for s in gt_segments:
        gt_frames[int(s["start"]/step):int(s["end"]/step)] = gt_map[s["speaker"]]
    for s in pred_segments:
        pred_frames[int(s["start"]/step):int(s["end"]/step)] = pred_map[s["speaker"]]

    ignore_frames = np.zeros(n_frames, dtype=bool)
    collar_frames = int(collar / step)
    for s in gt_segments:
        s_idx = int(s["start"] / step)
        e_idx = int(s["end"] / step)
        ignore_frames[max(0, s_idx-collar_frames):s_idx+collar_frames] = True
        ignore_frames[max(0, e_idx-collar_frames):e_idx+collar_frames] = True

    valid_mask = ~ignore_frames
    
    # Speaker Mapping (Greedy)
    mapping = {}
    if gt_speakers and pred_speakers:
        cost_matrix = np.zeros((len(gt_speakers), len(pred_speakers)))
        for i in range(n_frames):
            if valid_mask[i] and gt_frames[i] != -1 and pred_frames[i] != -1:
                cost_matrix[gt_frames[i], pred_frames[i]] += 1
        
        temp_matrix = cost_matrix.copy()
        for _ in range(min(len(gt_speakers), len(pred_speakers))):
            gt_idx, pred_idx = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
            if temp_matrix[gt_idx, pred_idx] == 0: break
            mapping[gt_idx] = pred_idx
            temp_matrix[gt_idx, :] = -1
            temp_matrix[:, pred_idx] = -1

    error_types = np.full(n_frames, None, dtype=object)
    for i in range(n_frames):
        if not valid_mask[i]: continue
        gt_spk = gt_frames[i]
        pred_spk = pred_frames[i]
        
        if gt_spk != -1:
            if pred_spk == -1:
                error_types[i] = "MISS"
            elif pred_spk != mapping.get(gt_spk, -1):
                error_types[i] = "CONFUSION"
        elif pred_spk != -1:
            error_types[i] = "FALSE_ALARM"

    regions = []
    if n_frames == 0: return []
    
    current_type = error_types[0]
    current_start = 0
    
    for i in range(1, n_frames):
        if error_types[i] != current_type:
            if current_type is not None:
                regions.append({
                    "start": round(current_start * step, 3),
                    "end": round(i * step, 3),
                    "error_type": current_type
                })
            current_type = error_types[i]
            current_start = i
            
    if current_type is not None:
        regions.append({
            "start": round(current_start * step, 3),
            "end": round(n_frames * step, 3),
            "error_type": current_type
        })
        
    return regions

def extract_audio_context(source_wav: Path, output_wav: Path, start: float, end: float, context: float = 2.0):
    duration = end - start
    # Calculate window with context
    win_start = max(0, start - context)
    win_end = end + context
    win_duration = win_end - win_start
    
    cmd = [
        "ffmpeg", "-y", "-ss", str(win_start), "-t", str(win_duration),
        "-i", str(source_wav), "-acodec", "pcm_s16le", str(output_wav)
    ]
    subprocess.run(cmd, capture_output=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="final_bwc_outputs")
    parser.add_argument("--audio-dir", default="raw_audio")
    parser.add_argument("--gt-dir", default="ground_truth")
    parser.add_argument("--output-dir", default="error_analysis_hardened")
    args = parser.parse_args()
    
    pred_dir = Path(args.pred_dir)
    audio_dir = Path(args.audio_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_errors = []
    
    # Target *_hardened.json
    for pred_path in list(pred_dir.glob("*_hardened.json")) + list(pred_dir.glob("*_final.json")):
        stem = pred_path.name.replace("_hardened.json", "").replace("_final.json", "")
        gt_path = gt_dir / f"{stem}.txt"
        source_wav = audio_dir / f"{stem}.wav"
        
        if not gt_path.exists():
            print(f"Skipping {stem}: Ground truth not found at {gt_path}")
            continue
        if not source_wav.exists():
            print(f"Skipping {stem}: Source audio not found at {source_wav}")
            continue
            
        print(f"Analyzing errors for: {stem}")
        gt_segments = parse_ground_truth(gt_path)
        pred_segments = load_prediction(pred_path)
        
        error_regions = identify_error_regions(gt_segments, pred_segments)
        
        file_output_dir = output_dir / stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, reg in enumerate(error_regions):
            reg["file"] = stem
            reg["error_id"] = i
            all_errors.append(reg)
            
            # Extract audio
            error_audio_path = file_output_dir / f"error_{i:03d}_{reg['error_type']}_{reg['start']:.2f}_{reg['end']:.2f}.wav"
            extract_audio_context(source_wav, error_audio_path, reg["start"], reg["end"])
            
    if all_errors:
        df = pd.DataFrame(all_errors)
        df.to_csv(output_dir / "diarization_errors.csv", index=False)
        print(f"Saved {len(all_errors)} error regions to {output_dir / 'diarization_errors.csv'}")
    else:
        print("No errors found.")

if __name__ == "__main__":
    main()
