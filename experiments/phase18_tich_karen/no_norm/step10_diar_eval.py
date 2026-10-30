import pandas as pd
from pathlib import Path
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import json

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
    BASE_DIR = Path("experiments/phase18_tich_karen/no_norm")
    SYS_RTTM_PATH = BASE_DIR / "diarization" / "rich_karen_msdd.rttm"
    GT_RTTM_PATH = BASE_DIR / "grounding" / "rich_karen_gt.rttm"
    
    if not SYS_RTTM_PATH.exists() or not GT_RTTM_PATH.exists():
        print("Missing RTTM files for evaluation.")
        return

    reference = load_rttm(GT_RTTM_PATH)
    hypothesis = load_rttm(SYS_RTTM_PATH)

    metric = DiarizationErrorRate()
    der_val = metric(reference, hypothesis, detailed=True)

    total_time = der_val['total']
    metrics = {
        "DER": round(der_val['diarization error rate'], 4),
        "Miss": round(der_val['missed detection'] / total_time if total_time > 0 else 0, 4),
        "False Alarm": round(der_val['false alarm'] / total_time if total_time > 0 else 0, 4),
        "Confusion": round(der_val['confusion'] / total_time if total_time > 0 else 0, 4)
    }

    df = pd.DataFrame([metrics])
    df.to_csv(BASE_DIR / "reports" / "diarization_metrics.csv", index=False)
    
    # Also save as JSON for consistency
    with open(BASE_DIR / "reports" / "diarization_eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Diarization Metrics:")
    print(df.to_string())

if __name__ == "__main__":
    main()
