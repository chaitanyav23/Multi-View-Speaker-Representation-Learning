import json
import numpy as np
import pandas as pd
from pathlib import Path

def get_activity_mask(segments, duration, resolution=0.01):
    mask = np.zeros(int(duration / resolution) + 1, dtype=bool)
    for s in segments:
        start_idx = int(s["start"] / resolution)
        end_idx = int(s["end"] / resolution)
        mask[start_idx:end_idx] = True
    return mask

def main():
    BASE_DIR = Path("experiments/phase17_bwc_no_norm")
    SYS_VAD_PATH = BASE_DIR / "grounding" / "bwc_grounding.json"
    GT_JSON_PATH = Path("ground_truth_json/bwc.json")
    
    with open(SYS_VAD_PATH) as f:
        sys_data = json.load(f)
    with open(GT_JSON_PATH) as f:
        gt_data = json.load(f)

    duration = 426.0
    res = 0.01
    
    sys_mask = get_activity_mask(sys_data["segments"], duration, res)
    gt_mask = get_activity_mask(gt_data["segments"], duration, res)

    tp = np.logical_and(sys_mask, gt_mask).sum()
    tn = np.logical_and(np.logical_not(sys_mask), np.logical_not(gt_mask)).sum()
    fp = np.logical_and(sys_mask, np.logical_not(gt_mask)).sum()
    fn = np.logical_and(np.logical_not(sys_mask), gt_mask).sum()

    accuracy = (tp + tn) / len(sys_mask)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    fa_rate = fp / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Miss Rate": miss_rate,
        "False Alarm Rate": fa_rate
    }

    df = pd.DataFrame([metrics])
    df.to_csv(BASE_DIR / "reports" / "vad_metrics.csv", index=False)
    print("VAD Metrics:")
    print(df.to_string())

    # Diagnostics
    miss_regions = []
    fa_regions = []
    
    # Find contiguous regions of FN and FP
    diff_miss = np.diff(np.logical_and(np.logical_not(sys_mask), gt_mask).astype(int))
    miss_starts = np.where(diff_miss == 1)[0]
    miss_ends = np.where(diff_miss == -1)[0]
    
    for s, e in zip(miss_starts, miss_ends):
        miss_regions.append({"start": s * res, "end": e * res})
        
    diff_fa = np.diff(np.logical_and(sys_mask, np.logical_not(gt_mask)).astype(int))
    fa_starts = np.where(diff_fa == 1)[0]
    fa_ends = np.where(diff_fa == -1)[0]
    
    for s, e in zip(fa_starts, fa_ends):
        fa_regions.append({"start": s * res, "end": e * res})

    with open(BASE_DIR / "diagnostics" / "vad" / "miss_regions.json", "w") as f:
        json.dump(miss_regions, f, indent=2)
    with open(BASE_DIR / "diagnostics" / "vad" / "fa_regions.json", "w") as f:
        json.dump(fa_regions, f, indent=2)

if __name__ == "__main__":
    main()
