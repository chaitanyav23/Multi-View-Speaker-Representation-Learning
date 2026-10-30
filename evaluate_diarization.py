"""
evaluate_diarization.py
=======================
Evaluate speaker diarization using ground-truth timestamps and predicted
diarization outputs. Computes DER (Diarization Error Rate).
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

PROJECT_ROOT = Path.cwd()
GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"
DIARIZATION_DIR = PROJECT_ROOT / "diarization"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Regex for Ground Truth: Speaker X [HH:MM:SS - HH:MM:SS]
GT_PATTERN = re.compile(
    r"(Speaker\s+\d+)\s*\[(\d+):(\d+):(\d+)\s*-\s*(\d+):(\d+):(\d+)\]"
)

def hms_to_seconds(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)

def parse_ground_truth(gt_file: Path) -> List[Dict]:
    segments = []
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            match = GT_PATTERN.search(line)
            if not match:
                continue
            
            speaker, h1, m1, s1, h2, m2, s2 = match.groups()
            start = hms_to_seconds(h1, m1, s1)
            end = hms_to_seconds(h2, m2, s2)
            
            if end <= start:
                continue
            
            segments.append({
                "start": float(start),
                "end": float(end),
                "speaker": speaker
            })
    return segments

def load_prediction(pred_json: Path) -> List[Dict]:
    with open(pred_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = []
    for seg in data.get("segments", []):
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker": seg["speaker"]
        })
    return segments

def get_total_duration(gt_segments, pred_segments):
    max_gt = max([s["end"] for s in gt_segments]) if gt_segments else 0
    max_pred = max([s["end"] for s in pred_segments]) if pred_segments else 0
    return max(max_gt, max_pred)

def compute_der(gt_segments, pred_segments, step=0.01):
    """
    Simple frame-based DER calculation.
    step=0.01 -> 10ms resolution
    """
    duration = get_total_duration(gt_segments, pred_segments)
    n_frames = int(duration / step) + 1
    
    # Maps for speakers
    gt_speakers = sorted(list(set(s["speaker"] for s in gt_segments)))
    pred_speakers = sorted(list(set(s["speaker"] for s in pred_segments)))
    
    gt_map = {name: i for i, name in enumerate(gt_speakers)}
    pred_map = {name: i for i, name in enumerate(pred_speakers)}
    
    # Initialize frame labels (-1 means no speaker)
    gt_frames = np.full(n_frames, -1, dtype=int)
    pred_frames = np.full(n_frames, -1, dtype=int)
    
    for s in gt_segments:
        start_idx = int(s["start"] / step)
        end_idx = int(s["end"] / step)
        gt_frames[start_idx:end_idx] = gt_map[s["speaker"]]
        
    for s in pred_segments:
        start_idx = int(s["start"] / step)
        end_idx = int(s["end"] / step)
        pred_frames[start_idx:end_idx] = pred_map[s["speaker"]]
        
    # Calculate basic components
    miss = 0
    false_alarm = 0
    speech_total = 0
    
    for i in range(n_frames):
        gt_spk = gt_frames[i]
        pred_spk = pred_frames[i]
        
        if gt_spk != -1:
            speech_total += 1
            if pred_spk == -1:
                miss += 1
        elif pred_spk != -1:
            false_alarm += 1
            
    # Calculate Speaker Confusion (requires optimal mapping)
    # We use a greedy approach for speaker mapping to minimize confusion
    confusion = 0
    if gt_speakers and pred_speakers:
        # Build cost matrix (number of frames where gt_i and pred_j overlap)
        cost_matrix = np.zeros((len(gt_speakers), len(pred_speakers)))
        for i in range(n_frames):
            if gt_frames[i] != -1 and pred_frames[i] != -1:
                cost_matrix[gt_frames[i], pred_frames[i]] += 1
        
        # Optimal mapping (Hungarian algorithm replacement - Greedy for simplicity)
        mapping = {}
        temp_matrix = cost_matrix.copy()
        for _ in range(min(len(gt_speakers), len(pred_speakers))):
            gt_idx, pred_idx = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
            if temp_matrix[gt_idx, pred_idx] == 0:
                break
            mapping[gt_idx] = pred_idx
            temp_matrix[gt_idx, :] = -1
            temp_matrix[:, pred_idx] = -1
            
        # Count confusion
        for i in range(n_frames):
            if gt_frames[i] != -1 and pred_frames[i] != -1:
                expected_pred = mapping.get(gt_frames[i], -1)
                if pred_frames[i] != expected_pred:
                    confusion += 1
                    
    if speech_total == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    der = (miss + false_alarm + confusion) / speech_total
    return der, miss/speech_total, false_alarm/speech_total, confusion/speech_total

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    pred_files = list(DIARIZATION_DIR.glob("*_diarized.json"))
    for pred_path in pred_files:
        stem = pred_path.name.replace("_diarized.json", "")
        gt_path = GROUND_TRUTH_DIR / f"{stem}.txt"
        
        if not gt_path.exists():
            print(f"Skipping {pred_path.name}: Ground truth not found at {gt_path}")
            continue
            
        print(f"Evaluating {stem}...")
        
        gt_segments = parse_ground_truth(gt_path)
        pred_segments = load_prediction(pred_path)
        
        der, miss, fa, conf = compute_der(gt_segments, pred_segments)
        
        results.append({
            "file": stem,
            "der": round(der, 4),
            "miss": round(miss, 4),
            "false_alarm": round(fa, 4),
            "confusion": round(conf, 4)
        })
        
    if not results:
        print("No results to display.")
        return
        
    df = pd.DataFrame(results)
    print("\nDIARIZATION EVALUATION RESULTS")
    print("==============================")
    print(df.to_string(index=False))
    
    # Save to CSV
    output_path = REPORTS_DIR / "diarization_eval.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
