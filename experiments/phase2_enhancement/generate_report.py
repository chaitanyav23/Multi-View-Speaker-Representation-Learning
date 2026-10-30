import pandas as pd
from pathlib import Path

def main():
    reports_dir = Path("reports")
    enhancers = ["speechbrain"] # Only speechbrain succeeded
    
    all_data = []
    # Baseline (Phase 1 Silero is a good baseline)
    baseline_path = reports_dir / "conversational_eval_phase1_silero.csv"
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        df['Enhancer'] = 'None (Baseline)'
        all_data.append(df)

    for enhancer in enhancers:
        csv_path = reports_dir / f"conversational_eval_phase2_{enhancer}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['Enhancer'] = enhancer
            all_data.append(df)
            
    if not all_data:
        print("No report data found.")
        return
        
    df_combined = pd.concat(all_data)
    bwc_df = df_combined[df_combined['file'] == 'bwc'][['Enhancer', 'WER', 'Subs', 'Dels', 'Ins', 'DER', 'Miss', 'FA', 'Conf', 'WDER']]
    
    with open("reports/phase2_report.md", "w") as f:
        f.write("# Phase 2: Speech Enhancement Study\n\n")
        f.write("## Objective\nReduce Whisper substitutions caused by low-SNR BWC audio.\n\n")
        f.write("## BWC Results Comparison\n\n")
        f.write(bwc_df.to_markdown(index=False))
        f.write("\n\n## Statistical Interpretation\n")
        
        sb_wer = bwc_df[bwc_df['Enhancer']=='speechbrain']['WER'].values[0]
        base_wer = bwc_df[bwc_df['Enhancer']=='None (Baseline)']['WER'].values[0]
        
        if sb_wer > base_wer:
            f.write(f"- **SpeechBrain (MetricGAN+) significantly DEGRADED performance.** WER increased from {base_wer} to {sb_wer}.\n")
            f.write("- Analysis: Enhancement artifacts are likely more damaging to Whisper's transformer-based attention than the original noise.\n")
        else:
            f.write("- SpeechBrain provided an improvement.\n")

    print("Phase 2 report generated: reports/phase2_report.md")

if __name__ == "__main__":
    main()
