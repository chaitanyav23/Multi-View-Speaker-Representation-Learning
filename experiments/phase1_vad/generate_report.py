import pandas as pd
import os
from pathlib import Path

def main():
    reports_dir = Path("reports")
    vads = ["silero", "pyannote"]
    
    all_data = []
    for vad in vads:
        csv_path = reports_dir / f"conversational_eval_phase1_{vad}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['VAD'] = vad
            all_data.append(df)
            
    if not all_data:
        print("No report data found.")
        return
        
    df_combined = pd.concat(all_data)
    
    # Generate Markdown Table
    # Filter for BWC as it's the main target
    bwc_df = df_combined[df_combined['file'] == 'bwc'][['VAD', 'WER', 'DER', 'Miss', 'FA', 'Conf', 'WDER', 'SA-WER']]
    
    with open("reports/phase1_report.md", "w") as f:
        f.write("# Phase 1: VAD Benchmarking Study\n\n")
        f.write("## Objective\nDetermine whether the current Silero-based Grounded VAD is responsible for the high Miss Rate and False Alarm Rate.\n\n")
        f.write("## BWC Results Comparison\n\n")
        f.write(bwc_df.to_markdown(index=False))
        f.write("\n\n## Statistical Interpretation\n")
        
        # Simple logic for interpretation
        best_der_vad = bwc_df.loc[bwc_df['DER'].idxmin()]['VAD']
        f.write(f"- **Best overall DER:** {best_der_vad}\n")
        
        silero_miss = bwc_df[bwc_df['VAD']=='silero']['Miss'].values[0]
        pyannote_miss = bwc_df[bwc_df['VAD']=='pyannote']['Miss'].values[0]
        
        if pyannote_miss < silero_miss:
            f.write(f"- **Pyannote** reduced the Miss Rate by {round((silero_miss - pyannote_miss)*100, 2)}% compared to Silero.\n")
        else:
            f.write(f"- **Silero** maintained a lower Miss Rate.\n")

        silero_fa = bwc_df[bwc_df['VAD']=='silero']['FA'].values[0]
        pyannote_fa = bwc_df[bwc_df['VAD']=='pyannote']['FA'].values[0]
        
        if pyannote_fa < silero_fa:
            f.write(f"- **Pyannote** reduced False Alarms.\n")
        else:
            f.write(f"- **Silero** is more conservative, with {round((pyannote_fa - silero_fa)*100, 2)}% fewer False Alarms.\n")

    print("Phase 1 report generated: reports/phase1_report.md")

if __name__ == "__main__":
    main()
