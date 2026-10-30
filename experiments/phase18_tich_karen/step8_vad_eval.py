import json
from pathlib import Path

def get_speech_mask(segments, duration, resolution=0.01):
    mask = [False] * int(duration / resolution + 1)
    for s in segments:
        start_idx = int(s["start"] / resolution)
        end_idx = int(s["end"] / resolution)
        for i in range(start_idx, min(end_idx, len(mask))):
            mask[i] = True
    return mask

def main():
    BASE_DIR = Path("experiments/phase18_tich_karen")
    GT_PATH = BASE_DIR / "grounding" / "rich_karen_gt.json"
    PRED_PATH = BASE_DIR / "grounding" / "rich_karen_grounding.json"
    
    with open(GT_PATH) as f: gt = json.load(f)
    with open(PRED_PATH) as f: pred = json.load(f)
    
    max_time = max(max([s["end"] for s in gt["segments"]]), max([s["end"] for s in pred["segments"]]))
    res = 0.01
    
    gt_mask = get_speech_mask(gt["segments"], max_time, res)
    pred_mask = get_speech_mask(pred["segments"], max_time, res)
    
    tp = fp = tn = fn = 0
    for g, p in zip(gt_mask, pred_mask):
        if g and p: tp += 1
        elif not g and p: fp += 1
        elif not g and not p: tn += 1
        elif g and not p: fn += 1
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_speech_gt": sum(gt_mask) * res,
        "total_speech_pred": sum(pred_mask) * res
    }
    
    print(f"VAD Metrics: {metrics}")
    with open(BASE_DIR / "reports" / "vad_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
