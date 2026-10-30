import json
import jiwer
from pathlib import Path

def normalize_text(text):
    return jiwer.ToLowerCase()(jiwer.RemovePunctuation()(jiwer.RemoveMultipleSpaces()(jiwer.Strip()(text))))

def main():
    BASE_DIR = Path("experiments/phase18_tich_karen/no_norm")
    GT_PATH = BASE_DIR / "grounding" / "rich_karen_gt.json"
    TRANSCRIPT_PATH = BASE_DIR / "outputs" / "rich_karen_transcript.json"
    
    with open(GT_PATH) as f: gt_data = json.load(f)
    with open(TRANSCRIPT_PATH) as f: pred_data = json.load(f)
    
    gt_text = " ".join([s["text"] for s in gt_data["segments"]])
    pred_text = " ".join([s["text"] for s in pred_data])
    
    gt_norm = normalize_text(gt_text)
    pred_norm = normalize_text(pred_text)
    
    wer = jiwer.wer(gt_norm, pred_norm)
    mer = jiwer.mer(gt_norm, pred_norm)
    wil = jiwer.wil(gt_norm, pred_norm)
    
    metrics = {
        "wer": round(wer, 4),
        "mer": round(mer, 4),
        "wil": round(wil, 4),
        "gt_word_count": len(gt_norm.split()),
        "pred_word_count": len(pred_norm.split())
    }
    
    print(f"ASR Metrics: {metrics}")
    with open(BASE_DIR / "reports" / "asr_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
