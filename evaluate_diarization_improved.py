"""
evaluate_diarization_improved.py
================================
Enhanced DER evaluation with a 0.25s collar and improved speaker matching.
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from typing import List, Dict, Tuple

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
    for seg in data.get("segments", []):
        spk = seg.get("speaker", "UNK")
        segments.append({"start": float(seg["start"]), "end": float(seg["end"]), "speaker": spk})
    return segments

def compute_der_improved(gt_segments, pred_segments, step=0.01, collar=0.25):
    """
    DER calculation with collar.
    Frames within 'collar' distance of a GT boundary are ignored for Miss/FA/Conf.
    """
    max_t = max([s["end"] for s in gt_segments + pred_segments])
    n_frames = int(max_t / step) + 1
    
    gt_speakers = sorted(list(set(s["speaker"] for s in gt_segments)))
    pred_speakers = sorted(list(set(s["speaker"] for s in pred_segments)))
    
    gt_map = {name: i for i, name in enumerate(gt_speakers)}
    pred_map = {name: i for i, name in enumerate(pred_speakers)}
    
    gt_frames = np.full(n_frames, -1, dtype=int)
    pred_frames = np.full(n_frames, -1, dtype=int)
    
    # 1. Fill ground truth frames
    for s in gt_segments:
        gt_frames[int(s["start"]/step):int(s["end"]/step)] = gt_map[s["speaker"]]
        
    # 2. Fill prediction frames
    for s in pred_segments:
        pred_frames[int(s["start"]/step):int(s["end"]/step)] = pred_map[s["speaker"]]

    # 3. Mark collar zones (ignore zones)
    ignore_frames = np.zeros(n_frames, dtype=bool)
    collar_frames = int(collar / step)
    for s in gt_segments:
        s_idx = int(s["start"] / step)
        e_idx = int(s["end"] / step)
        # Apply collar around start and end
        ignore_frames[max(0, s_idx-collar_frames):s_idx+collar_frames] = True
        ignore_frames[max(0, e_idx-collar_frames):e_idx+collar_frames] = True

    # 4. Filter frames (only count non-ignored frames for DER components)
    valid_mask = ~ignore_frames
    
    # 5. Calculate components
    miss = 0
    false_alarm = 0
    speech_total = 0
    
    for i in range(n_frames):
        if not valid_mask[i]: continue
        
        gt_spk = gt_frames[i]
        pred_spk = pred_frames[i]
        
        if gt_spk != -1:
            speech_total += 1
            if pred_spk == -1:
                miss += 1
        elif pred_spk != -1:
            false_alarm += 1
            
    # 6. Speaker Confusion (Greedy Mapping on valid frames)
    confusion = 0
    if gt_speakers and pred_speakers:
        cost_matrix = np.zeros((len(gt_speakers), len(pred_speakers)))
        for i in range(n_frames):
            if valid_mask[i] and gt_frames[i] != -1 and pred_frames[i] != -1:
                cost_matrix[gt_frames[i], pred_frames[i]] += 1
        
        mapping = {}
        temp_matrix = cost_matrix.copy()
        for _ in range(min(len(gt_speakers), len(pred_speakers))):
            gt_idx, pred_idx = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
            if temp_matrix[gt_idx, pred_idx] == 0: break
            mapping[gt_idx] = pred_idx
            temp_matrix[gt_idx, :] = -1
            temp_matrix[:, pred_idx] = -1
            
        for i in range(n_frames):
            if valid_mask[i] and gt_frames[i] != -1 and pred_frames[i] != -1:
                if pred_frames[i] != mapping.get(gt_frames[i], -1):
                    confusion += 1
                    
    if speech_total == 0: return 0, 0, 0, 0
    der = (miss + false_alarm + confusion) / speech_total
    return der, miss/speech_total, false_alarm/speech_total, confusion/speech_total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    
    gt_dir = Path("ground_truth")
    pred_dir = Path(args.pred_dir)
    results = []
    
    for pred_path in pred_dir.glob("*.json"):
        # Handle both naming conventions
        stem = pred_path.name.replace("_diarized.json", "").replace("_diarization.json", "").replace("_hardened.json", "").replace("_final.json", "")
        gt_path = gt_dir / f"{stem}.txt"
        
        if gt_path.exists():
            print(f"Evaluating Diarization: {stem}")
            gt_segments = parse_ground_truth(gt_path)
            pred_segments = load_prediction(pred_path)
            der, miss, fa, conf = compute_der_improved(gt_segments, pred_segments)
            results.append({"file": stem, "der": round(der, 4), "miss": round(miss, 4), "fa": round(fa, 4), "conf": round(conf, 4)})
            
    df = pd.DataFrame(results)
    output_path = Path("reports") / f"diarization_eval_improved_{args.output_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
