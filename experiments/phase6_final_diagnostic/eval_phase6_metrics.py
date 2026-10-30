"""
experiments/phase6_final_diagnostic/eval_phase6_metrics.py
=========================================================
Multi-tier evaluation (VAD, ASR, Diar, Attribution) against Human GT.
Generates comprehensive reports.
"""

import json
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd
import argparse
from typing import List, Dict, Tuple
from evaluate_asr_improved import normalize_text_improved, levenshtein_distance
from evaluate_diarization_improved import parse_ground_truth, compute_der_improved

# Paths
PROJECT_ROOT = Path.cwd()
PHASE_DIR = PROJECT_ROOT / "experiments" / "phase6_final_diagnostic"
GT_DIR = PROJECT_ROOT / "ground_truth"
REPORTS_DIR = PHASE_DIR / "reports"

def evaluate_vad(gt_segments, vad_segments, step=0.01):
    """VAD Prec/Rec/F1 at frame level."""
    if not gt_segments: return 0, 0, 0
    max_t = max([s["end"] for s in gt_segments + vad_segments])
    n_frames = int(max_t / step) + 1
    
    gt_frames = np.zeros(n_frames, dtype=bool)
    vad_frames = np.zeros(n_frames, dtype=bool)
    
    for s in gt_segments:
        gt_frames[int(s["start"]/step):int(s["end"]/step)] = True
    for s in vad_segments:
        vad_frames[int(s["start"]/step):int(s["end"]/step)] = True
        
    tp = np.sum(gt_frames & vad_frames)
    fp = np.sum(~gt_frames & vad_frames)
    fn = np.sum(gt_frames & ~vad_frames)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return round(precision, 4), round(recall, 4), round(f1, 4)

def compute_cp_wer(gt_segments, pred_segments):
    """
    Concatenated Permutation WER.
    1. Concatenate all segments for each GT speaker.
    2. Concatenate all segments for each Pred speaker.
    3. Find the best permutation mapping GT to Pred to minimize WER.
    """
    gt_spks = sorted(list(set(s["speaker"] for s in gt_segments)))
    pred_spks = sorted(list(set(s["speaker"] for s in pred_segments)))
    
    gt_texts = {spk: "" for spk in gt_spks}
    # Note: This is a simplified cp-WER. Real cp-WER needs careful alignment.
    # For this script, we'll focus on the primary WER metrics first.
    return 0.0 # Placeholder for complex implementation

def evaluate_path(enhancer_label):
    print(f"Evaluating: {enhancer_label}")
    output_dir = PHASE_DIR / "outputs" / enhancer_label
    grounding_dir = PHASE_DIR / "grounding"
    
    vad_results = []
    asr_results = []
    diar_results = []
    attr_results = []

    for pred_path in output_dir.glob("*_final.json"):
        stem = pred_path.name.replace("_final.json", "")
        gt_path = GT_DIR / f"{stem}.txt"
        vad_path = grounding_dir / f"{stem}_{enhancer_label}_vad.json"
        
        if not gt_path.exists(): continue
        
        # 1. Load Data
        gt_segments = parse_ground_truth(gt_path)
        with open(pred_path, "r") as f:
            pred_data = json.load(f)
        pred_segments = pred_data["segments"]
        with open(vad_path, "r") as f:
            vad_segments = json.load(f)["segments"]

        # 2. VAD Metrics
        prec, rec, f1 = evaluate_vad(gt_segments, vad_segments)
        vad_results.append({"enhancer": enhancer_label, "file": stem, "precision": prec, "recall": rec, "f1": f1})

        # 3. ASR Metrics
        # GT text reconstruction
        with open(gt_path, "r") as f: gt_text = f.read()
        hyp_text = " ".join([s["text"] for s in pred_segments])
        ref_words = normalize_text_improved(gt_text).split()
        hyp_words = normalize_text_improved(hyp_text).split()
        
        if ref_words:
            errs, s, i, d = levenshtein_distance(ref_words, hyp_words)
            wer = errs / len(ref_words)
            asr_results.append({
                "enhancer": enhancer_label, "file": stem, "wer": round(wer, 4), 
                "ref_count": len(ref_words), "subs": s, "dels": d, "ins": i
            })

        # 4. Diarization Metrics
        der, miss, fa, conf = compute_der_improved(gt_segments, pred_segments)
        diar_results.append({
            "enhancer": enhancer_label, "file": stem, "der": round(der, 4), 
            "miss": round(miss, 4), "fa": round(fa, 4), "conf": round(conf, 4)
        })

    return vad_results, asr_results, diar_results

def main():
    enhancers = ["baseline", "deepfilternet", "speechbrain"]
    all_vad = []
    all_asr = []
    all_diar = []

    for enh in enhancers:
        v, a, d = evaluate_path(enh)
        all_vad.extend(v)
        all_asr.extend(a)
        all_diar.extend(d)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_vad).to_csv(REPORTS_DIR / "vad_metrics.csv", index=False)
    pd.DataFrame(all_asr).to_csv(REPORTS_DIR / "asr_metrics.csv", index=False)
    pd.DataFrame(all_diar).to_csv(REPORTS_DIR / "diarization_metrics.csv", index=False)
    
    # Create a summary report
    summary = []
    for enh in enhancers:
        enh_asr = [r["wer"] for r in all_asr if r["enhancer"] == enh]
        enh_der = [r["der"] for r in all_diar if r["enhancer"] == enh]
        if enh_asr:
            summary.append({
                "enhancer": enh,
                "avg_wer": round(np.mean(enh_asr), 4),
                "avg_der": round(np.mean(enh_der), 4)
            })
    pd.DataFrame(summary).to_csv(REPORTS_DIR / "final_comparison.csv", index=False)
    print(f"Reports saved to {REPORTS_DIR}")

if __name__ == "__main__":
    main()
