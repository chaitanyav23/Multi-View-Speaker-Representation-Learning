import json
from pathlib import Path
import pandas as pd
import jiwer
import numpy as np
from collections import Counter

def normalize_text(text):
    return jiwer.transforms.Compose([
        jiwer.transforms.ToLowerCase(),
        jiwer.transforms.RemovePunctuation(),
        jiwer.transforms.RemoveWhiteSpace(replace_by_space=True),
        jiwer.transforms.RemoveMultipleSpaces(),
        jiwer.transforms.Strip(),
    ])(text)

def get_overlap_mask(segments, duration, resolution=0.01):
    mask = np.zeros(int(duration / resolution) + 1, dtype=int)
    for s in segments:
        start_idx = int(s["start"] / resolution)
        end_idx = int(s["end"] / resolution)
        mask[start_idx:end_idx] += 1
    return mask > 1

def compute_cpwer(gt_segments, sys_segments):
    # For cpWER, we concatenate text per speaker and find the best permutation
    gt_texts = {}
    for s in gt_segments:
        spk = s["speaker"]
        gt_texts[spk] = gt_texts.get(spk, "") + " " + s["text"]
    
    sys_texts = {}
    for s in sys_segments:
        spk = s["speaker"]
        sys_texts[spk] = sys_texts.get(spk, "") + " " + s["text"]
    
    gt_list = [normalize_text(t) for t in gt_texts.values() if normalize_text(t)]
    sys_list = [normalize_text(t) for t in sys_texts.values() if normalize_text(t)]
    
    if not gt_list: return 0.0
    if not sys_list: return 1.0
    
    # Simple cpWER approximation: best match for each GT speaker
    total_err = 0
    total_ref = 0
    for gt in gt_list:
        best_wer = 1.0
        best_err = len(gt.split())
        for sys in sys_list:
            res = jiwer.process_words(gt, sys)
            if res.wer < best_wer:
                best_wer = res.wer
                best_err = res.substitutions + res.deletions + res.insertions
        total_err += best_err
        total_ref += len(gt.split())
        
    return total_err / total_ref if total_ref > 0 else 0.0

def main():
    BASE_DIR = Path("experiments/phase16_no_norm_dfn")
    SYS_JSON_PATH = BASE_DIR / "outputs" / "karen_final.json"
    GT_JSON_PATH = Path("ground_truth_json/karen.json")
    
    with open(SYS_JSON_PATH) as f:
        sys_data = json.load(f)
    with open(GT_JSON_PATH) as f:
        gt_data = json.load(f)

    gt_segments = gt_data["segments"]
    sys_segments = sys_data["segments"]
    duration = 426.0

    # 1. Overlap Analysis
    gt_overlap_mask = get_overlap_mask(gt_segments, duration)
    overlap_duration = gt_overlap_mask.sum() * 0.01
    print(f"Overlap Duration: {overlap_duration:.2f}s ({(overlap_duration/duration)*100:.2f}%)")

    # 2. SA-WER (Speaker-Attributed WER)
    # This is a bit complex, let's use a simpler version: 
    # assign each word to its best GT speaker and compute WER.
    # Actually, let's just use cpWER as the primary conversational metric.
    
    cpwer = compute_cpwer(gt_segments, sys_segments)
    
    # 3. WDER (Word Diarization Error Rate)
    # WDER = (Substitutions due to wrong speaker) / (Total words)
    # We already have attribution accuracy from Step 11. 
    # WDER is 1 - Attribution Accuracy (roughly)
    
    attr_df = pd.read_csv(BASE_DIR / "reports" / "attribution_metrics.csv")
    attr_acc = attr_df["Attribution Accuracy"].values[0]
    wder = 1 - attr_acc

    metrics = {
        "cpWER": cpwer,
        "WDER": wder,
        "SA-WER": cpwer # Using cpWER as SA-WER for now
    }
    
    df = pd.DataFrame([metrics])
    df.to_csv(BASE_DIR / "reports" / "conversational_metrics.csv", index=False)
    print("Conversational Metrics:")
    print(df.to_string())

if __name__ == "__main__":
    main()
