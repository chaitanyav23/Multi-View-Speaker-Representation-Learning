import json
from pathlib import Path
import pandas as pd
from collections import Counter
import numpy as np

def main():
    BASE_DIR = Path("experiments/phase16_no_norm_dfn")
    SYS_JSON_PATH = BASE_DIR / "outputs" / "karen_final.json"
    GT_JSON_PATH = Path("ground_truth_json/karen.json")
    
    with open(SYS_JSON_PATH) as f:
        sys_data = json.load(f)
    with open(GT_JSON_PATH) as f:
        gt_data = json.load(f)

    # 1. Prepare GT segments for lookup
    gt_segments = gt_data["segments"]
    
    # 2. Assign GT speaker to each SYS word
    sys_words = sys_data["words"]
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

    # 3. Find optimal mapping (Greedy)
    # Build confusion matrix
    sys_spks = sorted(list(set(p["sys_spk"] for p in attribution_pairs)))
    gt_spks = sorted(list(set(p["gt_spk"] for p in attribution_pairs)))
    
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

    # 4. Compute Accuracy
    correct = 0
    total = 0
    for p in attribution_pairs:
        if p["gt_spk"] == "UNK": continue
        total += 1
        if mapping.get(p["sys_spk"]) == p["gt_spk"]:
            correct += 1
            
    accuracy = correct / total if total > 0 else 0
    
    metrics = {
        "Attribution Accuracy": accuracy,
        "Total Words Evaluated": total,
        "Correct Attributions": correct
    }
    
    df = pd.DataFrame([metrics])
    df.to_csv(BASE_DIR / "reports" / "attribution_metrics.csv", index=False)
    print("Attribution Metrics:")
    print(df.to_string())

    # Diagnostics
    # Confusion Matrix
    conf_df = pd.DataFrame(matrix, index=gt_spks, columns=sys_spks)
    conf_df.to_csv(BASE_DIR / "diagnostics" / "attribution" / "attribution_confusion.csv")

if __name__ == "__main__":
    main()
