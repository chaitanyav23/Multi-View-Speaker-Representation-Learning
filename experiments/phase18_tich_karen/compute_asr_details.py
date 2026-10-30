import json
import jiwer
import pandas as pd
from pathlib import Path

def normalize_text(text):
    return jiwer.transforms.Compose([
        jiwer.transforms.ToLowerCase(),
        jiwer.transforms.RemovePunctuation(),
        jiwer.transforms.RemoveWhiteSpace(replace_by_space=True),
        jiwer.transforms.RemoveMultipleSpaces(),
        jiwer.transforms.Strip(),
    ])(text)

def main():
    BASE_DIR = Path("experiments/phase18_tich_karen")
    GT_PATH = BASE_DIR / "grounding" / "rich_karen_gt.json"
    
    with open(GT_PATH) as f:
        gt_data = json.load(f)
    gt_text = " ".join([s["text"] for s in gt_data["segments"]])
    gt_norm = normalize_text(gt_text)

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
            continue
            
        with open(path) as f:
            pred_data = json.load(f)
            
        pred_text = " ".join([s["text"] for s in pred_data])
        pred_norm = normalize_text(pred_text)
        
        # Calculate full jiwer alignment
        out = jiwer.process_words(gt_norm, pred_norm)
        
        results.append({
            "Pipeline Variation": name,
            "WER": f"{out.wer:.4f}",
            "Subs": out.substitutions,
            "Dels": out.deletions,
            "Ins": out.insertions,
            "Total GT Words": len(gt_norm.split())
        })

    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
