"""
evaluate_asr_improved.py
========================
Enhanced WER evaluation with advanced text normalization.
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple

# Common English contractions for normalization
CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am"
}

def normalize_text_improved(text: str) -> str:
    # 1. Remove timestamps [HH:MM:SS] or [00.00s]
    text = re.sub(r"\[[^\]]+\]", " ", text)
    # 2. Remove Speaker labels
    text = re.sub(r"Speaker\s+\d+[:]*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"SPEAKER_\d+[:]*", " ", text, flags=re.IGNORECASE)
    
    text = text.lower()
    
    # 3. Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
        
    # 4. Remove all punctuation except apostrophes in words
    text = re.sub(r"[^\w\s']", " ", text)
    
    # 5. Clean up whitespace
    return " ".join(text.split())

def levenshtein_distance(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    n = len(ref)
    m = len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1): dp[i, 0] = i
    for j in range(m + 1): dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = min(dp[i-1, j-1], dp[i, j-1], dp[i-1, j]) + 1
    
    # Backtrack to find S, I, D
    i, j = n, m
    subs, ins, dels = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i, j] == dp[i-1, j-1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i, j] == dp[i, j-1] + 1:
            ins += 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i-1, j] + 1:
            dels += 1
            i -= 1
            
    return dp[n, m], subs, ins, dels

import numpy as np

def evaluate_file(gt_path: Path, hyp_path: Path, is_json=False) -> dict:
    with open(gt_path, "r", encoding="utf-8") as f:
        ref_text = f.read()
    
    if is_json:
        with open(hyp_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        hyp_text = " ".join([s["text"] for s in data.get("segments", [])])
    else:
        with open(hyp_path, "r", encoding="utf-8") as f:
            hyp_text = f.read()
            
    ref_words = normalize_text_improved(ref_text).split()
    hyp_words = normalize_text_improved(hyp_text).split()
    
    if not ref_words:
        return {"wer": 0.0, "s": 0, "i": 0, "d": 0, "count": 0}
        
    errs, s, i, d = levenshtein_distance(ref_words, hyp_words)
    wer = errs / len(ref_words)
    
    return {
        "wer": round(wer, 4),
        "substitutions": s,
        "insertions": i,
        "deletions": d,
        "ref_count": len(ref_words),
        "hyp_count": len(hyp_words)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp-dir", required=True)
    parser.add_argument("--is-json", action="store_true")
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    
    gt_dir = Path("ground_truth")
    hyp_dir = Path(args.hyp_dir)
    results = []
    
    ext = "*.json" if args.is_json else "*.txt"
    for hyp_path in hyp_dir.glob(ext):
        stem = hyp_path.stem.replace("_asr", "").replace("_aligned", "").replace("_final", "")
        gt_path = gt_dir / f"{stem}.txt"
        
        if gt_path.exists():
            print(f"Evaluating ASR: {stem}")
            metrics = evaluate_file(gt_path, hyp_path, is_json=args.is_json)
            metrics["file"] = stem
            results.append(metrics)
            
    df = pd.DataFrame(results)
    output_path = Path("reports") / f"asr_eval_improved_{args.output_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

import pandas as pd
if __name__ == "__main__":
    main()
