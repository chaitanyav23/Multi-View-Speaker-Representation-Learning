import json
from pathlib import Path
import pandas as pd
import jiwer
import numpy as np

def normalize_text(text):
    return jiwer.transforms.Compose([
        jiwer.transforms.ToLowerCase(),
        jiwer.transforms.RemovePunctuation(),
        jiwer.transforms.RemoveWhiteSpace(replace_by_space=True),
        jiwer.transforms.RemoveMultipleSpaces(),
        jiwer.transforms.Strip(),
    ])(text)

def compute_cpwer(gt_segments, sys_segments):
    gt_texts = {}
    for s in gt_segments:
        spk = s["speaker"]
        gt_texts[spk] = gt_texts.get(spk, "") + " " + s.get("text", s.get("word", ""))
    
    sys_texts = {}
    for s in sys_segments:
        spk = s["speaker"]
        sys_texts[spk] = sys_texts.get(spk, "") + " " + s.get("text", s.get("word", ""))
    
    gt_list = [normalize_text(t) for t in gt_texts.values() if normalize_text(t)]
    sys_list = [normalize_text(t) for t in sys_texts.values() if normalize_text(t)]
    
    if not gt_list: return 0.0
    if not sys_list: return 1.0
    
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

def compute_wder(gt_segments, sys_segments):
    # WDER based on Attribution Accuracy
    sys_words = []
    for s in sys_segments:
        if "word" in s:
            sys_words.append(s)
        else:
            # Approximate words by splitting
            text = s["text"]
            start = s["start"]
            end = s["end"]
            words = text.split()
            if not words: continue
            dur = (end - start) / len(words)
            for i, w in enumerate(words):
                sys_words.append({
                    "word": w,
                    "start": start + i * dur,
                    "end": start + (i + 1) * dur,
                    "speaker": s["speaker"]
                })
    
    attribution_pairs = []
    for word in sys_words:
        if "start" not in word or word["start"] is None:
            continue
        mid = (word["start"] + word["end"]) / 2
        gt_spk = "UNK"
        for gt in gt_segments:
            if gt["start"] <= mid <= gt["end"]:
                gt_spk = gt["speaker"]
                break
        
        attribution_pairs.append({
            "sys_spk": word["speaker"],
            "gt_spk": gt_spk
        })

    sys_spks = sorted(list(set(p["sys_spk"] for p in attribution_pairs)))
    gt_spks = sorted(list(set(p["gt_spk"] for p in attribution_pairs)))
    if not gt_spks or not sys_spks: return 1.0
    
    matrix = np.zeros((len(gt_spks), len(sys_spks)))
    gt_map = {s: i for i, s in enumerate(gt_spks)}
    sys_map = {s: i for i, s in enumerate(sys_spks)}
    
    for p in attribution_pairs:
        matrix[gt_map[p["gt_spk"]], sys_map[p["sys_spk"]]] += 1
        
    mapping = {}
    temp_matrix = matrix.copy()
    for _ in range(min(len(gt_spks), len(sys_spks))):
        gt_idx, sys_idx = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
        if temp_matrix[gt_idx, sys_idx] == 0:
            break
        mapping[sys_spks[sys_idx]] = gt_spks[gt_idx]
        temp_matrix[gt_idx, :] = -1
        temp_matrix[:, sys_idx] = -1

    correct = 0
    total = 0
    for p in attribution_pairs:
        if p["gt_spk"] == "UNK": continue
        total += 1
        if mapping.get(p["sys_spk"]) == p["gt_spk"]:
            correct += 1
            
    accuracy = correct / total if total > 0 else 0
    return 1 - accuracy

def main():
    BASE_DIR = Path("experiments/phase18_tich_karen")
    GT_PATH = BASE_DIR / "grounding" / "rich_karen_gt.json"
    
    with open(GT_PATH) as f:
        gt_segments = json.load(f)["segments"]

    pipelines = [
        ("1. Standard Normalized", BASE_DIR / "outputs" / "rich_karen_transcript.json"),
        ("2. No Normalization", BASE_DIR / "no_norm" / "outputs" / "rich_karen_transcript.json"),
        ("3. Oracle Grounded", BASE_DIR / "grounded" / "outputs" / "rich_karen_transcript.json"),
        ("4. ASR-Grounded", BASE_DIR / "asr_grounded" / "outputs" / "asr_grounded_transcript.json"),
        ("5. DFN Enhanced", BASE_DIR / "dfn_no_norm" / "outputs" / "rich_karen_transcript.json"),
        ("6. Dual-Path", BASE_DIR / "dual_path" / "outputs" / "dual_path_transcript.json")
    ]

    results = []
    for name, path in pipelines:
        if not path.exists():
            print(f"Missing {path}")
            continue
            
        with open(path) as f:
            sys_segments = json.load(f)
            
        cpwer = compute_cpwer(gt_segments, sys_segments)
        wder = compute_wder(gt_segments, sys_segments)
        
        results.append({
            "Pipeline Variation": name,
            "cpWER": f"{cpwer:.4f}",
            "WDER": f"{wder:.4f}",
            "SA-WER": f"{cpwer:.4f}" # Approximation based on cpWER
        })

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
