"""
experiments/phase8_sentence_grouped/eval_phase8_metrics.py
=========================================================
Phase 8 evaluation: VAD, ASR, Diarization, cp-WER, SA-WER.
Against Human Ground Truth.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

import numpy as np
import pandas as pd

from evaluate_asr_improved import normalize_text_improved, levenshtein_distance
from evaluate_diarization_improved import parse_ground_truth, compute_der_improved
from evaluate_conversational_timeline import (
    parse_ground_truth_with_text,
    compute_sawer_and_wder,
    compute_cpwer,
    normalize_text,
)

PROJECT_ROOT = Path.cwd()
PHASE_DIR = PROJECT_ROOT / "experiments" / "phase8_sentence_grouped"
GT_DIR = PROJECT_ROOT / "ground_truth"

GROUNDING_DIR = PHASE_DIR / "grounding"
OUTPUT_DIR = PHASE_DIR / "outputs"
ASR_DIR = OUTPUT_DIR / "asr"
REPORTS_DIR = PHASE_DIR / "reports"

def evaluate_vad(gt_segments, vad_segments, step=0.01):
    if not gt_segments: return 0, 0, 0
    max_t = max([s["end"] for s in gt_segments + vad_segments])
    n_frames = int(max_t / step) + 1
    gt_frames = np.zeros(n_frames, dtype=bool)
    vad_frames = np.zeros(n_frames, dtype=bool)
    for s in gt_segments: gt_frames[int(s["start"]/step):int(s["end"]/step)] = True
    for s in vad_segments: vad_frames[int(s["start"]/step):int(s["end"]/step)] = True
    tp = np.sum(gt_frames & vad_frames)
    fp = np.sum(~gt_frames & vad_frames)
    fn = np.sum(gt_frames & ~vad_frames)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return round(precision, 4), round(recall, 4), round(f1, 4)

def evaluate_asr(gt_path: Path, asr_path: Path) -> dict:
    with open(gt_path, "r", encoding="utf-8") as f: gt_text = f.read()
    with open(asr_path, "r", encoding="utf-8") as f: asr_data = json.load(f)
    hyp_text = " ".join([s["text"] for s in asr_data.get("segments", [])])
    ref_words = normalize_text_improved(gt_text).split()
    hyp_words = normalize_text_improved(hyp_text).split()
    if not ref_words: return {"wer": 0.0, "subs": 0, "dels": 0, "ins": 0, "ref_count": 0}
    errs, s, i, d = levenshtein_distance(ref_words, hyp_words)
    return {"wer": round(errs / len(ref_words), 4), "subs": s, "dels": d, "ins": i, "ref_count": len(ref_words)}

def evaluate_path():
    vad_results, asr_results, diar_results, cp_sawer_results = [], [], [], []
    for pred_path in OUTPUT_DIR.glob("*_final.json"):
        stem = pred_path.name.replace("_final.json", "")
        gt_path = GT_DIR / f"{stem}.txt"
        vad_path = GROUNDING_DIR / f"{stem}_soft_vad.json"
        asr_path = ASR_DIR / f"{stem}_asr.json"
        if not gt_path.exists(): continue
        gt_segments = parse_ground_truth(gt_path)
        with open(pred_path, "r") as f: pred_data = json.load(f)
        pred_segments = pred_data.get("segments", [])
        if vad_path.exists():
            with open(vad_path, "r") as f: vad_segments = json.load(f)["segments"]
            prec, rec, f1 = evaluate_vad(gt_segments, vad_segments)
            vad_results.append({"file": stem, "precision": prec, "recall": rec, "f1": f1})
        if asr_path.exists():
            metrics = evaluate_asr(gt_path, asr_path)
            metrics["file"] = stem
            asr_results.append(metrics)
        der, miss, fa, conf = compute_der_improved(gt_segments, pred_segments)
        diar_results.append({"file": stem, "der": round(der, 4), "miss": round(miss, 4), "fa": round(fa, 4), "conf": round(conf, 4)})
        ref_segs = parse_ground_truth_with_text(gt_path)
        hyp_segs = [{"start": s["start"], "end": s["end"], "speaker": s.get("speaker", "UNKNOWN"), "text": normalize_text(s.get("text", ""))} for s in pred_segments]
        base_wer, sa_wer, wder, subs, dels, ins = compute_sawer_and_wder(ref_segs, hyp_segs)
        cp_wer = compute_cpwer(ref_segs, hyp_segs)
        cp_sawer_results.append({"file": stem, "base_wer": round(base_wer, 4), "sa_wer": round(sa_wer, 4), "wder": round(wder, 4), "cp_wer": round(cp_wer, 4), "subs": subs, "dels": dels, "ins": ins})
    return vad_results, asr_results, diar_results, cp_sawer_results

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    v, a, d, c = evaluate_path()
    if v: pd.DataFrame(v).to_csv(REPORTS_DIR / "vad_metrics.csv", index=False)
    if a: pd.DataFrame(a).to_csv(REPORTS_DIR / "asr_metrics.csv", index=False)
    if d: pd.DataFrame(d).to_csv(REPORTS_DIR / "diarization_metrics.csv", index=False)
    if c: pd.DataFrame(c).to_csv(REPORTS_DIR / "cp_sawer_metrics.csv", index=False)
    summary = {}
    if a: summary["avg_wer"] = round(np.mean([r["wer"] for r in a]), 4)
    if d: summary["avg_der"] = round(np.mean([r["der"] for r in d]), 4)
    if c:
        summary["avg_cpwer"] = round(np.mean([r["cp_wer"] for r in c]), 4)
        summary["avg_sawer"] = round(np.mean([r["sa_wer"] for r in c]), 4)
    if summary: pd.DataFrame([summary]).to_csv(REPORTS_DIR / "final_summary.csv", index=False)
    print(f"Reports saved to {REPORTS_DIR}")

if __name__ == "__main__":
    main()
