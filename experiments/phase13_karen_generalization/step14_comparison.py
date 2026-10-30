import pandas as pd
from pathlib import Path

def main():
    BASE_DIR = Path("experiments/phase13_karen_generalization")
    
    # Karen Metrics
    asr_df = pd.read_csv(BASE_DIR / "reports" / "asr_metrics.csv")
    diar_df = pd.read_csv(BASE_DIR / "reports" / "diarization_metrics.csv")
    conv_df = pd.read_csv(BASE_DIR / "reports" / "conversational_metrics.csv")
    
    karen_metrics = {
        "WER": asr_df["WER"].values[0],
        "DER": diar_df["DER"].values[0],
        "Miss": diar_df["Miss"].values[0],
        "FA": diar_df["False Alarm"].values[0],
        "Conf": diar_df["Confusion"].values[0],
        "SA-WER": conv_df["SA-WER"].values[0],
        "WDER": conv_df["WDER"].values[0],
        "cpWER": conv_df["cpWER"].values[0]
    }
    
    # BWC Phase 11A Metrics (Manual entry from cat output)
    bwc_metrics = {
        "WER": 0.3002,
        "DER": 0.4679,
        "Miss": 0.2878,
        "FA": 0.0766,
        "Conf": 0.1035,
        "SA-WER": 0.4380,
        "WDER": 0.1136,
        "cpWER": 0.5032
    }
    
    comparison = []
    for metric in bwc_metrics.keys():
        bwc_val = bwc_metrics[metric]
        karen_val = karen_metrics[metric]
        delta = karen_val - bwc_val
        comparison.append({
            "Metric": metric,
            "BWC Phase 11A": bwc_val,
            "Karen": karen_val,
            "Delta": delta
        })
        
    df = pd.DataFrame(comparison)
    df.to_csv(BASE_DIR / "comparisons" / "karen_vs_bwc.csv", index=False)
    print("Karen vs BWC Comparison:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
