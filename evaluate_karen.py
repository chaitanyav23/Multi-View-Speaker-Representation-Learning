import json
import re
import pandas as pd
from pathlib import Path
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from jiwer import process_words

KAREN_DIR = Path("experiments/karen_11A_audit")
OUTPUT_DIR = KAREN_DIR / "outputs"
GT_PATH = Path("ground_truth/karen.txt")
VAD_FILE = KAREN_DIR / "grounding" / "karen_grounding.json"
DURATION_LIMIT = 426.0

def hms_to_seconds(hms):
    h, m, s = map(float, hms.split(':'))
    return h * 3600 + m * 60 + s

def parse_gt_vad(gt_path):
    segments = []
    with open(gt_path, 'r') as f:
        for line in f:
            match = re.search(r'\[(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2})\]', line)
            if match:
                segments.append((hms_to_seconds(match.group(1)), hms_to_seconds(match.group(2))))
    return segments

def segments_to_frame_labels(segments, duration, frame_rate=100):
    frames = np.zeros(int(duration * frame_rate) + 1, dtype=int)
    for s, e in segments:
        if s >= duration: continue
        frames[int(s * frame_rate):min(int(e * frame_rate), len(frames))] = 1
    return frames

# 1. Evaluate VAD (Truncated)
print(">>> Evaluating VAD (Truncated to 426s)...")
gt_segments = parse_gt_vad(GT_PATH)
with open(VAD_FILE) as f:
    pred_data = json.load(f)
    pred_segments = [(s["start"], s["end"]) for s in pred_data["segments"]]

duration = DURATION_LIMIT
y_true = segments_to_frame_labels(gt_segments, duration)
y_pred = segments_to_frame_labels(pred_segments, duration)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
vad_metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred, zero_division=0),
    "f1": f1_score(y_true, y_pred, zero_division=0),
    "miss_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
    "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0
}

# 2. Run Conversational Evaluation
print(">>> Evaluating Conversational Timeline...")
subprocess.run([
    "python3", "evaluate_conversational_timeline.py",
    "--pred-dir", str(OUTPUT_DIR),
    "--output-name", "karen_audit"
], capture_output=True)

# 3. Compile Results
conv_report_path = "reports/conversational_eval_karen_audit.csv"
if not Path(conv_report_path).exists():
    potential = list(Path("reports").glob("conversational_eval_karen_audit*.csv"))
    if potential: conv_report_path = potential[0]

conv_res = pd.read_csv(conv_report_path).iloc[0]

metrics = {
    "VAD Accuracy": vad_metrics["accuracy"],
    "VAD Precision": vad_metrics["precision"],
    "VAD Recall": vad_metrics["recall"],
    "VAD F1": vad_metrics["f1"],
    "VAD Miss Rate": vad_metrics["miss_rate"],
    "VAD FA Rate": vad_metrics["false_alarm_rate"],
    "WER": conv_res["WER"],
    "Subs": conv_res["Subs"],
    "Dels": conv_res["Dels"],
    "Ins": conv_res["Ins"],
    "DER": conv_res["DER"],
    "DER Miss Rate": conv_res["Miss"],
    "DER FA Rate": conv_res["FA"],
    "DER Conf Rate": conv_res["Conf"],
    "SA-WER": conv_res["SA-WER"],
    "WDER": conv_res["WDER"],
    "cp-WER": conv_res["cpWER"]
}

df = pd.DataFrame([metrics])
print("\n" + "="*40)
print("FINAL KAREN 11A AUDIT RESULTS")
print("="*40)
print(df.T)
df.to_csv(KAREN_DIR / "final_metrics.csv")

# 4. ASR Error Report (Word-level)
print(">>> Generating ASR Error Detail...")
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

def parse_gt_text(gt_path):
    text = []
    with open(gt_path, 'r') as f:
        for line in f:
            if ":" in line and "[" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    text.append(normalize_text(parts[1]))
    return " ".join(text)

with open(OUTPUT_DIR / "karen_final.json") as f:
    pred_data = json.load(f)
    pred_text = " ".join([normalize_text(s["text"]) for s in pred_data["segments"]])

ref_text = parse_gt_text(GT_PATH)
out = process_words(ref_text, pred_text)

# Extract specific errors
with open(KAREN_DIR / "asr_error_detail.txt", "w") as f:
    f.write(f"WER: {out.wer:.4f}\n")
    f.write(f"Substitutions: {out.substitutions}\n")
    f.write(f"Deletions: {out.deletions}\n")
    f.write(f"Insertions: {out.insertions}\n\n")
    
    f.write("ALIGNMENT (Ref top, Hyp bottom):\n")
    # jiwer might return words as lists in some versions, ensure they are strings
    for i in range(0, len(out.references), 10):
        ref_chunk = out.references[i:i+10]
        hyp_chunk = out.hypotheses[i:i+10]
        f.write("REF: " + " ".join([f"{str(w)[:14]:15}" for w in ref_chunk]) + "\n")
        f.write("HYP: " + " ".join([f"{str(w)[:14]:15}" for w in hyp_chunk]) + "\n\n")

print(f">>> ASR Error Detail saved to {KAREN_DIR / 'asr_error_detail.txt'}")
