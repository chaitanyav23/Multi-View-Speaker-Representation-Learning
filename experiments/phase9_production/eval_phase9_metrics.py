"""
experiments/phase9_production/eval_phase9_metrics.py
====================================================
Phase 9 Evaluation Script.
Computes comprehensive VAD, ASR, Diarization, and Conversational metrics.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from evaluate_asr_improved import normalize_text_improved, levenshtein_distance
from evaluate_diarization_improved import parse_ground_truth, compute_der_improved
from evaluate_conversational_timeline import (
    parse_ground_truth_with_text,
    compute_sawer_and_wder,
    compute_cpwer,
    normalize_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_DIR = PROJECT_ROOT / "experiments" / "phase9_production"
GT_DIR = PROJECT_ROOT / "ground_truth"

GROUNDING_DIR = PHASE_DIR / "grounding_master"
OUTPUT_DIR = PHASE_DIR / "outputs_final"
ASR_DIR = PHASE_DIR / "outputs_asr"
REPORTS_DIR = PHASE_DIR / "reports"

def evaluate_vad(gt_segments, vad_segments, step=0.01):
    if not gt_segments:
        return {}
    
    max_t = max([s["end"] for s in gt_segments + vad_segments])
    n_frames = int(max_t / step) + 1

    gt_frames = np.zeros(n_frames, dtype=bool)
    vad_frames = np.zeros(n_frames, dtype=bool)

    for s in gt_segments:
        gt_frames[int(s["start"] / step):int(s["end"] / step)] = True
    for s in vad_segments:
        vad_frames[int(s["start"] / step):int(s["end"] / step)] = True

    tp = np.sum(gt_frames & vad_frames)
    fp = np.sum(~gt_frames & vad_frames)
    fn = np.sum(gt_frames & ~vad_frames)
    tn = np.sum(~gt_frames & ~vad_frames)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "miss_rate": round(miss_rate, 4),
        "false_alarm_rate": round(false_alarm_rate, 4)
    }

def evaluate_asr(gt_path: Path, asr_path: Path) -> dict:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_text = f.read()
    with open(asr_path, "r", encoding="utf-8") as f:
        asr_data = json.load(f)
    
    hyp_text = " ".join([s.get("text", "") for s in asr_data.get("segments", [])])

    ref_words = normalize_text_improved(gt_text).split()
    hyp_words = normalize_text_improved(hyp_text).split()
    
    if not ref_words:
        return {"wer": 0.0, "subs": 0, "dels": 0, "ins": 0, "ref_count": 0}

    errs, s, i, d = levenshtein_distance(ref_words, hyp_words)
    return {
        "wer": round(errs / len(ref_words), 4),
        "subs": s,
        "dels": d,
        "ins": i,
        "ref_count": len(ref_words),
    }

def evaluate_path():
    all_results = []
    
    # We only process bwc.wav
    stem = "bwc"
    gt_path = GT_DIR / f"{stem}.txt"
    pred_path = OUTPUT_DIR / f"{stem}_final.json"
    vad_path = GROUNDING_DIR / f"{stem}_grounding.json"
    asr_path = ASR_DIR / f"{stem}_asr.json"

    if not gt_path.exists() or not pred_path.exists():
        print(f"Skipping {stem}, missing GT or prediction JSON.")
        return []

    # 1. Load data
    gt_segments = parse_ground_truth(gt_path)
    ref_segs = parse_ground_truth_with_text(gt_path)
    
    with open(pred_path, "r") as f:
        pred_data = json.load(f)
    pred_segments = pred_data.get("segments", [])

    hyp_segs = []
    for s in pred_segments:
        hyp_segs.append({
            "start": s.get("start", 0.0),
            "end": s.get("end", 0.0),
            "speaker": s.get("speaker", "UNKNOWN"),
            "text": normalize_text(s.get("text", "")),
        })

    # 2. VAD Metrics
    vad_metrics = {}
    if vad_path.exists():
        with open(vad_path, "r") as f:
            vad_segs = json.load(f)["segments"]
        vad_metrics = evaluate_vad(gt_segments, vad_segs)

    # 3. ASR Metrics
    asr_metrics = {}
    if asr_path.exists():
        asr_metrics = evaluate_asr(gt_path, asr_path)

    # 4. Diarization Metrics
    der, miss, fa, conf = compute_der_improved(gt_segments, pred_segments)
    diar_metrics = {
        "der": round(der, 4),
        "miss": round(miss, 4),
        "fa": round(fa, 4),
        "conf": round(conf, 4)
    }

    # 5. Conversational Metrics
    base_wer, sa_wer, wder, subs, dels, ins = compute_sawer_and_wder(ref_segs, hyp_segs)
    cp_wer = compute_cpwer(ref_segs, hyp_segs)
    
    conv_metrics = {
        "wder": round(wder, 4),
        "sa_wer": round(sa_wer, 4),
        "cp_wer": round(cp_wer, 4)
    }

    combined = {
        "file": stem,
        **vad_metrics,
        **{f"asr_{k}": v for k, v in asr_metrics.items()},
        **diar_metrics,
        **conv_metrics
    }
    
    return [combined]

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate_path()

    if results:
        df = pd.DataFrame(results)
        df.to_csv(REPORTS_DIR / "phase9_comprehensive_metrics.csv", index=False)
        print("\n--- PHASE 9 BWC.WAV METRICS ---")
        for k, v in results[0].items():
            print(f"{k}: {v}")
        print(f"\nReports saved to {REPORTS_DIR}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
