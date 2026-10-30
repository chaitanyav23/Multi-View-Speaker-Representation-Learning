import json
import jiwer
import pandas as pd
from pathlib import Path
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

def get_speech_mask(segments, duration, resolution=0.01):
    mask = [False] * int(duration / resolution + 1)
    for s in segments:
        start_idx = int(s["start"] / resolution)
        end_idx = int(s["end"] / resolution)
        for i in range(start_idx, min(end_idx, len(mask))):
            mask[i] = True
    return mask

def normalize_text(text):
    return jiwer.ToLowerCase()(jiwer.RemovePunctuation()(jiwer.RemoveMultipleSpaces()(jiwer.Strip()(text))))

def load_rttm(path):
    annotation = Annotation()
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                annotation[Segment(start, start + duration)] = speaker
    return annotation

def main():
    BASE_DIR = Path("experiments/phase18_tich_karen")
    DP_DIR = BASE_DIR / "dual_path"
    
    GT_JSON_PATH = DP_DIR / "grounding" / "rich_karen_gt.json"
    GT_RTTM_PATH = DP_DIR / "grounding" / "rich_karen_gt.rttm"
    
    PRED_GROUNDING = DP_DIR / "grounding" / "rich_karen_grounding.json"
    PRED_TRANSCRIPT = DP_DIR / "outputs" / "dual_path_transcript.json"
    PRED_RTTM = DP_DIR / "diarization" / "dual_path_msdd.rttm"
    
    # 1. VAD Eval
    with open(GT_JSON_PATH) as f: gt = json.load(f)
    with open(PRED_GROUNDING) as f: pred = json.load(f)
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
    
    vad_metrics = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    
    # 2. ASR Eval
    with open(PRED_TRANSCRIPT) as f: pred_data = json.load(f)
    gt_text = " ".join([s["text"] for s in gt["segments"]])
    pred_text = " ".join([s["text"] for s in pred_data])
    gt_norm = normalize_text(gt_text)
    pred_norm = normalize_text(pred_text)
    wer = jiwer.wer(gt_norm, pred_norm)
    asr_metrics = {"wer": round(wer, 4), "gt_words": len(gt_norm.split()), "pred_words": len(pred_norm.split())}
    
    # 3. Diarization Eval
    reference = load_rttm(GT_RTTM_PATH)
    hypothesis = load_rttm(PRED_RTTM)
    metric = DiarizationErrorRate()
    der_val = metric(reference, hypothesis, detailed=True)
    total_time = der_val['total']
    diar_metrics = {
        "DER": round(der_val['diarization error rate'], 4),
        "Miss": round(der_val['missed detection'] / total_time if total_time > 0 else 0, 4),
        "False Alarm": round(der_val['false alarm'] / total_time if total_time > 0 else 0, 4),
        "Confusion": round(der_val['confusion'] / total_time if total_time > 0 else 0, 4)
    }
    
    (DP_DIR / "reports").mkdir(exist_ok=True)
    with open(DP_DIR / "reports" / "eval_summary.json", "w") as f:
        json.dump({"VAD": vad_metrics, "ASR": asr_metrics, "DIARIZATION": diar_metrics}, f, indent=2)
        
    print("\n--- Dual-Path Evaluation ---")
    print(f"VAD: {vad_metrics}")
    print(f"ASR: {asr_metrics}")
    print(f"DIAR: {diar_metrics}")

if __name__ == "__main__":
    main()
