
import json
import re
import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from jiwer import wer, process_words
from scipy.optimize import linear_sum_assignment
from itertools import permutations

# Add project root to path
sys.path.append(str(Path.cwd()))

from evaluate_asr_improved import normalize_text_improved, levenshtein_distance
from evaluate_diarization_improved import parse_ground_truth, compute_der_improved

# --- Constants & Paths ---
PROJECT_ROOT = Path.cwd()
PHASE_DIR = PROJECT_ROOT / "experiments" / "phase6_final_diagnostic"
V2_DIR = PHASE_DIR / "v2"
GT_DIR = PROJECT_ROOT / "ground_truth"
REPORTS_DIR = V2_DIR / "reports"

# Regex for Ground Truth segments with text
GT_TEXT_PATTERN = re.compile(
    r"(Speaker\s+\d+)\s*\[(\d+):(\d+):(\d+)\s*-\s*(\d+):(\d+):(\d+)\]:\s*(.*)"
)

def hms_to_seconds(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)

def parse_ground_truth_with_text(gt_file: Path) -> List[Dict]:
    segments = []
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            match = GT_TEXT_PATTERN.search(line)
            if not match: continue
            spk, h1, m1, s1, h2, m2, s2, text = match.groups()
            segments.append({
                "start": float(hms_to_seconds(h1, m1, s1)),
                "end": float(hms_to_seconds(h2, m2, s2)),
                "speaker": spk.strip(),
                "text": normalize_text_improved(text)
            })
    return segments

def evaluate_vad(gt_segments, vad_segments, step=0.01):
    """VAD metrics at frame level."""
    if not gt_segments: return {}
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
    tn = np.sum(~gt_frames & ~vad_frames)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / n_frames
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "miss_rate": round(miss_rate, 4),
        "fa_rate": round(fa_rate, 4)
    }

def compute_sawer_and_wder(ref_segments, hyp_segments):
    """
    SA-WER and WDER.
    """
    ref_spks = sorted(list(set(s.get("speaker", "UNKNOWN") for s in ref_segments)))
    hyp_spks = sorted(list(set(s.get("speaker", "UNKNOWN") for s in hyp_segments)))
    
    mapping = {}
    if ref_spks and hyp_spks:
        overlap_matrix = np.zeros((len(ref_spks), len(hyp_spks)))
        ref_map = {name: i for i, name in enumerate(ref_spks)}
        hyp_map = {name: i for i, name in enumerate(hyp_spks)}
        
        for r in ref_segments:
            for h in hyp_segments:
                overlap = min(r["end"], h["end"]) - max(r["start"], h["start"])
                if overlap > 0:
                    overlap_matrix[ref_map[r.get("speaker", "UNKNOWN")], hyp_map[h.get("speaker", "UNKNOWN")]] += overlap
        
        row_ind, col_idx = linear_sum_assignment(overlap_matrix, maximize=True)
        for r_i, h_i in zip(row_ind, col_idx):
            mapping[ref_spks[r_i]] = hyp_spks[h_i]
            
    rev_mapping = {v: k for k, v in mapping.items()}

    all_ref_text = " ".join([s["text"] for s in ref_segments])
    all_hyp_text = " ".join([s["text"] for s in hyp_segments])
    
    ref_word_spks = [s.get('speaker', 'UNKNOWN') for s in ref_segments for w in s["text"].split()]
    hyp_word_spks = [rev_mapping.get(s.get('speaker', 'UNKNOWN'), 'UNKNOWN') for s in hyp_segments for w in s["text"].split()]

    out = process_words(all_ref_text, all_hyp_text)
    base_wer = out.wer
    subs = out.substitutions
    dels = out.deletions
    ins = out.insertions
    
    if not ref_word_spks:
        return base_wer, sa_wer, 0.0, subs, dels, ins

    diarization_errors = 0
    for chunk in out.alignments[0]:
        if chunk.type == 'equal':
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                r_idx = chunk.ref_start_idx + i
                h_idx = chunk.hyp_start_idx + i
                if r_idx < len(ref_word_spks) and h_idx < len(hyp_word_spks):
                    if ref_word_spks[r_idx] != hyp_word_spks[h_idx]:
                        diarization_errors += 1

    wder_val = diarization_errors / len(ref_word_spks) if ref_word_spks else 0
    sa_wer_val = base_wer + wder_val

    return round(base_wer, 4), round(sa_wer_val, 4), round(wder_val, 4), subs, dels, ins

def compute_cpwer(ref_segments, hyp_segments):
    """Concatenated Minimum-Permutation WER."""
    ref_spks = sorted(list(set(s.get("speaker", "UNKNOWN") for s in ref_segments)))
    hyp_spks = sorted(list(set(s.get("speaker", "UNKNOWN") for s in hyp_segments)))
    
    ref_transcripts = {spk: " ".join([s["text"] for s in ref_segments if s.get("speaker", "UNKNOWN") == spk]) for spk in ref_spks}
    hyp_transcripts = {spk: " ".join([s["text"] for s in hyp_segments if s.get("speaker", "UNKNOWN") == spk]) for spk in hyp_spks}
    
    min_wer = float('inf')
    if len(hyp_spks) > 5: return 0.5
        
    for p in permutations(hyp_spks, min(len(ref_spks), len(hyp_spks))):
        current_ref = []
        current_hyp = []
        for i, ref_spk in enumerate(ref_spks[:len(p)]):
            current_ref.append(ref_transcripts[ref_spk])
            current_hyp.append(hyp_transcripts[p[i]])
        
        error = wer(" ".join(current_ref), " ".join(current_hyp))
        if error < min_wer:
            min_wer = error
            
    return round(min_wer, 4) if min_wer != float('inf') else 1.0

def evaluate_approach(enhancer_label):
    print(f"Evaluating: {enhancer_label}")
    output_dir = V2_DIR / "outputs" / enhancer_label
    grounding_dir = V2_DIR / "grounding"
    
    results = []

    for pred_path in output_dir.glob("bwc_final.json"):
        stem = "bwc"
        gt_path = GT_DIR / f"{stem}.txt"
        vad_path = grounding_dir / f"{stem}_{enhancer_label}_vad.json"
        
        if not gt_path.exists(): continue
        
        # Load
        gt_segments = parse_ground_truth_with_text(gt_path)
        with open(pred_path, "r") as f:
            pred_data = json.load(f)
        pred_segments = pred_data["segments"]
        for s in pred_segments:
            s["text"] = normalize_text_improved(s.get("text", ""))
            if "speaker" not in s:
                s["speaker"] = "UNKNOWN"
        
        with open(vad_path, "r") as f:
            vad_segments = json.load(f)["segments"]

        # VAD
        vad_metrics = evaluate_vad(gt_segments, vad_segments)
        
        # ASR & Conversational
        base_wer, sa_wer, wder, subs, dels, ins = compute_sawer_and_wder(gt_segments, pred_segments)
        cp_wer = compute_cpwer(gt_segments, pred_segments)
        
        # Diarization
        der, miss, fa, conf = compute_der_improved(gt_segments, pred_segments)
        
        res = {
            "enhancer": enhancer_label,
            "file": stem,
            "VAD_Acc": vad_metrics["accuracy"],
            "VAD_Prec": vad_metrics["precision"],
            "VAD_Rec": vad_metrics["recall"],
            "VAD_F1": vad_metrics["f1"],
            "VAD_Miss": vad_metrics["miss_rate"],
            "VAD_FA": vad_metrics["fa_rate"],
            "WER": base_wer,
            "Subs": subs,
            "Dels": dels,
            "Ins": ins,
            "DER": round(der, 4),
            "Diar_Miss": round(miss, 4),
            "Diar_FA": round(fa, 4),
            "Diar_Conf": round(conf, 4),
            "SA-WER": sa_wer,
            "WDER": wder,
            "cpWER": cp_wer
        }
        results.append(res)
    return results

def main():
    enhancers = ["baseline", "deepfilternet", "speechbrain"]
    all_results = []

    for enh in enhancers:
        all_results.extend(evaluate_approach(enh))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(REPORTS_DIR / "phase6_v2_metrics.csv", index=False)
    
    # Save a summary markdown
    with open(REPORTS_DIR / "phase6_v2_summary.md", "w") as f:
        f.write("# Phase 6 V2 Evaluation Summary\n\n")
        f.write(df.to_markdown(index=False))
        
    print(f"Evaluation complete. Reports saved to {REPORTS_DIR}")

if __name__ == "__main__":
    main()
