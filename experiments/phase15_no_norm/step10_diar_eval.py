import pandas as pd
from pathlib import Path
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

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
    BASE_DIR = Path("experiments/phase15_no_norm")
    SYS_RTTM_PATH = BASE_DIR / "diarization" / "karen_msdd.rttm"
    GT_RTTM_PATH = Path("ground_truth_rttm/karen.rttm")
    
    reference = load_rttm(GT_RTTM_PATH)
    hypothesis = load_rttm(SYS_RTTM_PATH)

    metric = DiarizationErrorRate()
    der_val = metric(reference, hypothesis, detailed=True)

    total_time = der_val['total']
    metrics = {
        "DER": der_val['diarization error rate'],
        "Miss": der_val['missed detection'] / total_time if total_time > 0 else 0,
        "False Alarm": der_val['false alarm'] / total_time if total_time > 0 else 0,
        "Confusion": der_val['confusion'] / total_time if total_time > 0 else 0
    }

    df = pd.DataFrame([metrics])
    df.to_csv(BASE_DIR / "reports" / "diarization_metrics.csv", index=False)
    print("Diarization Metrics:")
    print(df.to_string())

if __name__ == "__main__":
    main()
