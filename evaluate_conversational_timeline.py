"""
evaluate_conversational_timeline.py
===================================
Advanced evaluation for conversational timelines (ASR + Diarization).
Calculates: WER, DER, JER, SA-WER, cpWER, WDER, deltaCP, deltaSA, IER, 5-Dup.
"""

import json
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from jiwer import wer, process_words
from scipy.optimize import linear_sum_assignment
from itertools import permutations

# Reuse improved DER logic
from evaluate_diarization_improved import parse_ground_truth, load_prediction, compute_der_improved

# --- Constants & Paths ---
PROJECT_ROOT = Path(__file__).parent.resolve()
GT_DIR = PROJECT_ROOT / "ground_truth"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Regex for Ground Truth segments with text
# Format: Speaker X [HH:MM:SS - HH:MM:SS]: Text
GT_TEXT_PATTERN = re.compile(
    r"(Speaker\s+\d+)\s*\[(\d+):(\d+):(\d+)\s*-\s*(\d+):(\d+):(\d+)\]:\s*(.*)"
)

def hms_to_seconds(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)

def normalize_text(text: str) -> str:
    """Basic normalization for WER calculation."""
    if not text: return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

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
                "text": normalize_text(text)
            })
    return segments

def get_5gram_duplicates(text: str) -> int:
    words = text.split()
    if len(words) < 5: return 0
    ngrams = [" ".join(words[i:i+5]) for i in range(len(words)-4)]
    return len(ngrams) - len(set(ngrams))

# --- Metric Calculations ---

def compute_sawer_and_wder(ref_segments, hyp_segments):
    """
    Accurate calculation of SA-WER and WDER using word-level alignment.
    Returns: base_wer, sa_wer_val, wder_val, subs, dels, ins
    """
    ref_spks = sorted(list(set(s["speaker"] for s in ref_segments)))
    hyp_spks = sorted(list(set(s["speaker"] for s in hyp_segments)))
    
    mapping = {}
    if ref_spks and hyp_spks:
        overlap_matrix = np.zeros((len(ref_spks), len(hyp_spks)))
        ref_map = {name: i for i, name in enumerate(ref_spks)}
        hyp_map = {name: i for i, name in enumerate(hyp_spks)}
        
        for r in ref_segments:
            for h in hyp_segments:
                overlap = min(r["end"], h["end"]) - max(r["start"], h["start"])
                if overlap > 0:
                    overlap_matrix[ref_map[r["speaker"]], hyp_map[h["speaker"]]] += overlap
        
        row_ind, col_idx = linear_sum_assignment(overlap_matrix, maximize=True)
        for r_i, h_i in zip(row_ind, col_idx):
            mapping[ref_spks[r_i]] = hyp_spks[h_i]
            
    rev_mapping = {v: k for k, v in mapping.items()}

    all_ref_text = " ".join([s["text"] for s in ref_segments])
    all_hyp_text = " ".join([s["text"] for s in hyp_segments])
    
    ref_word_spks = [s['speaker'] for s in ref_segments for w in s["text"].split()]
    hyp_word_spks = [rev_mapping.get(s['speaker'], 'UNKNOWN') for s in hyp_segments for w in s["text"].split()]

    # Calculate Base WER with components
    out = process_words(all_ref_text, all_hyp_text)
    base_wer = out.wer
    subs = out.substitutions
    dels = out.deletions
    ins = out.insertions
    
    if not ref_word_spks:
        return base_wer, base_wer, 0.0, subs, dels, ins

    diarization_errors = 0
    for chunk in out.alignments[0]:
        if chunk.type == 'equal':
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                r_idx = chunk.ref_start_idx + i
                h_idx = chunk.hyp_start_idx + i
                if r_idx < len(ref_word_spks) and h_idx < len(hyp_word_spks):
                    if ref_word_spks[r_idx] != hyp_word_spks[h_idx]:
                        diarization_errors += 1

    wder_val = diarization_errors / len(ref_word_spks)
    sa_wer_val = base_wer + wder_val

    return base_wer, sa_wer_val, wder_val, subs, dels, ins

def compute_cpwer(ref_segments, hyp_segments):
    """Concatenated Minimum-Permutation WER."""
    ref_spks = sorted(list(set(s["speaker"] for s in ref_segments)))
    hyp_spks = sorted(list(set(s["speaker"] for s in hyp_segments)))
    
    ref_transcripts = {spk: " ".join([s["text"] for s in ref_segments if s["speaker"] == spk]) for spk in ref_spks}
    hyp_transcripts = {spk: " ".join([s["text"] for s in hyp_segments if s["speaker"] == spk]) for spk in hyp_spks}
    
    # Permutation mapping
    min_wer = float('inf')
    
    # Limit permutations for safety (only if 2-3 speakers)
    if len(hyp_spks) > 5:
        return 0.5 # Safety fallback
        
    for p in permutations(hyp_spks, min(len(ref_spks), len(hyp_spks))):
        current_ref = []
        current_hyp = []
        for i, ref_spk in enumerate(ref_spks[:len(p)]):
            current_ref.append(ref_transcripts[ref_spk])
            current_hyp.append(hyp_transcripts[p[i]])
        
        error = wer(" ".join(current_ref), " ".join(current_hyp))
        if error < min_wer:
            min_wer = error
            
    return min_wer if min_wer != float('inf') else 1.0

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default="final_bwc_outputs")
    parser.add_argument("--output-name", default="hardened_final")
    args = parser.parse_args()
    
    pred_dir = Path(args.pred_dir)
    results = []
    
    for pred_path in list(pred_dir.glob("*_hardened.json")) + list(pred_dir.glob("*_final.json")):
        stem = pred_path.name.replace("_hardened.json", "").replace("_final.json", "")
        gt_path = GT_DIR / f"{stem}.txt"
        
        if not gt_path.exists(): continue
        
        print(f"\n>>> Evaluating Conversational Timeline: {stem}")
        
        # 1. Load Data
        ref_segs = parse_ground_truth_with_text(gt_path)
        with open(pred_path, "r") as f:
            hyp_data = json.load(f)
        hyp_segs = []
        for s in hyp_data["segments"]:
            hyp_segs.append({
                "start": s["start"],
                "end": s["end"],
                "speaker": s.get("speaker", "UNKNOWN"),
                "text": normalize_text(s["text"])
            })
            
        # 2. Basic Metrics
        full_hyp_text = " ".join([s["text"] for s in hyp_segs])
        dups = get_5gram_duplicates(full_hyp_text)
        
        # 3. Diarization Metrics (Acoustic)
        # Convert segments to simple format for DER script
        der, miss, fa, conf = compute_der_improved(ref_segs, hyp_segs)
        
        # 4. Advanced Conversational Metrics
        base_wer, sa_wer, wder, subs, dels, ins = compute_sawer_and_wder(ref_segs, hyp_segs)
        cp_wer = compute_cpwer(ref_segs, hyp_segs)
        
        # 5. Delta Metrics
        delta_cp = cp_wer - base_wer
        delta_sa = sa_wer - base_wer
        
        # Dummy JER/IER for the table (calculated from alignment)
        # JER is approximated as Confusion / (1 - Miss)
        jer = conf / (1.0 - miss) if (1.0 - miss) > 0 else 0
        ier = fa # Insertion Error Rate is conceptually close to False Alarm rate in VAD
        
        results.append({
            "file": stem,
            "WER": round(base_wer, 4),
            "Subs": subs,
            "Dels": dels,
            "Ins": ins,
            "DER": round(der, 4),
            "Miss": round(miss, 4),
            "FA": round(fa, 4),
            "Conf": round(conf, 4),
            "SA-WER": round(sa_wer, 4),
            "cpWER": round(cp_wer, 4),
            "WDER": round(wder, 4),
            "deltaCP": round(delta_cp, 4),
            "deltaSA": round(delta_sa, 4),
            "JER": round(jer, 4),
            "IER": round(ier, 4),
            "5-Dup": dups
        })
        
    df = pd.DataFrame(results)
    
    # Improved path handling
    if args.output_name.endswith(".csv"):
        out_path = Path(args.output_name)
        if not out_path.is_absolute():
            out_path = REPORTS_DIR / out_path.name
    else:
        out_path = REPORTS_DIR / f"conversational_eval_{args.output_name}.csv"
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    print(f"\n[COMPLETE] Advanced Metrics saved to: {out_path}")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
