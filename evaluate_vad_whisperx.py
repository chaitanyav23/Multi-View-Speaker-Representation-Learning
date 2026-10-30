"""
evaluate_vad_whisperx.py
========================
Evaluates and compares the standard VAD against the new WhisperX-style VAD.
Includes metrics from Wang et al. and the Marston paper.
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.evaluate_vad import (
    parse_ground_truth, 
    load_vad_segments, 
    segments_to_frame_labels, 
    compute_segment_metrics
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path.cwd()
GT_DIR = PROJECT_ROOT / "ground_truth"
VAD_STD_DIR = PROJECT_ROOT / "vad_outputs"
VAD_WX_DIR = PROJECT_ROOT / "vad_outputs_whisperx"
REPORTS_DIR = PROJECT_ROOT / "reports" / "whisperx_vad"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_advanced_metrics(y_true, y_pred, gt_segments, pred_segments, duration_sec):
    # Frame-level basic
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Speech Ratio
    gt_speech_ratio = sum(s[1] - s[0] for s in gt_segments) / duration_sec
    pred_speech_ratio = sum(s[1] - s[0] for s in pred_segments) / duration_sec
    speech_ratio_error = abs(gt_speech_ratio - pred_speech_ratio)
    
    # Fragmentation (Segments per minute)
    gt_frag = len(gt_segments) / (duration_sec / 60) if duration_sec > 0 else 0
    pred_frag = len(pred_segments) / (duration_sec / 60) if duration_sec > 0 else 0
    frag_error = abs(gt_frag - pred_frag)

    # Segment metrics (Recall, Precision, IoU, Boundary Error)
    seg_metrics = compute_segment_metrics(gt_segments, pred_segments)
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "miss_rate": miss_rate,
        "false_alarm_rate": false_alarm_rate,
        "gt_speech_ratio": gt_speech_ratio,
        "pred_speech_ratio": pred_speech_ratio,
        "speech_ratio_error": speech_ratio_error,
        "gt_segments": len(gt_segments),
        "pred_segments": len(pred_segments),
        "gt_frag_index": gt_frag,
        "pred_frag_index": pred_frag,
        "frag_error": frag_error,
        **seg_metrics
    }

def main():
    results = []
    
    # Discover matching ground truth and VAD files
    gt_files = list(GT_DIR.glob("*.txt"))
    
    for gt_path in gt_files:
        stem = gt_path.stem
        
        # Paths for both VAD versions
        std_vad_path = VAD_STD_DIR / f"{stem}_vad.json"
        wx_vad_path = VAD_WX_DIR / f"{stem}_vad.json"
        
        if not std_vad_path.exists() and not wx_vad_path.exists():
            continue
            
        gt_segments = parse_ground_truth(gt_path)
        
        # Evaluate Standard VAD
        if std_vad_path.exists():
            pred_std, duration = load_vad_segments(std_vad_path)
            y_true = segments_to_frame_labels(gt_segments, duration)
            y_pred = segments_to_frame_labels(pred_std, duration)
            metrics = calculate_advanced_metrics(y_true, y_pred, gt_segments, pred_std, duration)
            metrics["pipeline"] = "standard"
            metrics["file"] = stem
            results.append(metrics)
            
        # Evaluate WhisperX VAD
        if wx_vad_path.exists():
            pred_wx, duration = load_vad_segments(wx_vad_path)
            y_true = segments_to_frame_labels(gt_segments, duration)
            y_pred = segments_to_frame_labels(pred_wx, duration)
            metrics = calculate_advanced_metrics(y_true, y_pred, gt_segments, pred_wx, duration)
            metrics["pipeline"] = "whisperx_mincut"
            metrics["file"] = stem
            results.append(metrics)

    if not results:
        print("No results found. Make sure you ran both VAD pipelines.")
        return

    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    cols = ["file", "pipeline", "f1", "miss_rate", "false_alarm_rate", "mean_iou", "boundary_error_sec", "speech_ratio_error", "frag_error"]
    display_df = df[cols].sort_values(["file", "pipeline"])
    
    print("\nVAD PIPELINE COMPARISON")
    print("=======================")
    print(display_df.to_string(index=False))
    
    # Save to CSV
    output_path = REPORTS_DIR / "vad_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDetailed report saved to {output_path}")

if __name__ == "__main__":
    main()
