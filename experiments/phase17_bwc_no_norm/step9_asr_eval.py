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
    BASE_DIR = Path("experiments/phase17_bwc_no_norm")
    SYS_PATH = BASE_DIR / "outputs" / "bwc_final.json"
    GT_PATH = Path("ground_truth_json/bwc.json")
    
    with open(SYS_PATH) as f:
        sys_data = json.load(f)
    with open(GT_PATH) as f:
        gt_data = json.load(f)

    sys_text = " ".join([s["text"] for s in sys_data["segments"]])
    gt_text = " ".join([s["text"] for s in gt_data["segments"]])

    sys_norm = normalize_text(sys_text)
    gt_norm = normalize_text(gt_text)

    wer_res = jiwer.process_words(gt_norm, sys_norm)
    
    metrics = {
        "WER": wer_res.wer,
        "Ref Words": len(gt_norm.split()),
        "Substitutions": wer_res.substitutions,
        "Deletions": wer_res.deletions,
        "Insertions": wer_res.insertions
    }

    df = pd.DataFrame([metrics])
    df.to_csv(BASE_DIR / "reports" / "asr_metrics.csv", index=False)
    print("ASR Metrics:")
    print(df.to_string())

    # Diagnostics
    # Save a comparison file
    with open(BASE_DIR / "diagnostics" / "asr" / "asr_comparison.txt", "w") as f:
        f.write("GROUND TRUTH:\n")
        f.write(gt_norm + "\n\n")
        f.write("SYSTEM:\n")
        f.write(sys_norm + "\n")

if __name__ == "__main__":
    main()
