"""
compare_pipelines_v3.py
=======================
Automates the improved evaluation and generates a side-by-side comparison.
"""

import os
import subprocess
import pandas as pd
from pathlib import Path

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1. Run Improved ASR Evaluations
    run_cmd("python3 evaluate_asr_improved.py --hyp-dir transcripts/ --output-name standard")
    run_cmd("python3 evaluate_asr_improved.py --hyp-dir asr_outputs_wx/ --is-json --output-name whisperx_parallel")
    
    # 2. Run Improved Diarization Evaluations
    run_cmd("python3 evaluate_diarization_improved.py --pred-dir diarization/ --output-name standard")
    run_cmd("python3 evaluate_diarization_improved.py --pred-dir diarization_outputs_wx/ --output-name whisperx_parallel")
    
    # 3. Load and Compare Results
    asr_std = pd.read_csv("reports/asr_eval_improved_standard.csv")
    asr_std['pipeline'] = 'standard'
    asr_wx = pd.read_csv("reports/asr_eval_improved_whisperx_parallel.csv")
    asr_wx['pipeline'] = 'whisperx_parallel'
    
    diar_std = pd.read_csv("reports/diarization_eval_improved_standard.csv")
    diar_std['pipeline'] = 'standard'
    diar_wx = pd.read_csv("reports/diarization_eval_improved_whisperx_parallel.csv")
    diar_wx['pipeline'] = 'whisperx_parallel'
    
    # Combined report
    asr_all = pd.concat([asr_std, asr_wx])
    diar_all = pd.concat([diar_std, diar_wx])
    
    print("\n" + "="*50)
    print("ASR (WER) COMPARISON")
    print("="*50)
    asr_pivot = asr_all.pivot(index="file", columns="pipeline", values="wer")
    if 'standard' in asr_pivot.columns and 'whisperx_parallel' in asr_pivot.columns:
        asr_pivot['improvement (%)'] = ((asr_pivot['standard'] - asr_pivot['whisperx_parallel']) / asr_pivot['standard'] * 100).round(2)
    print(asr_pivot)
    
    print("\n" + "="*50)
    print("DIARIZATION (DER) COMPARISON (0.25s Collar)")
    print("="*50)
    diar_pivot = diar_all.pivot(index="file", columns="pipeline", values="der")
    if 'standard' in diar_pivot.columns and 'whisperx_parallel' in diar_pivot.columns:
        diar_pivot['improvement (%)'] = ((diar_pivot['standard'] - diar_pivot['whisperx_parallel']) / diar_pivot['standard'] * 100).round(2)
    print(diar_pivot)
    
    # Save final report
    summary_path = Path("reports") / "final_pipeline_comparison.csv"
    asr_pivot.to_csv(Path("reports") / "asr_comparison_summary.csv")
    diar_pivot.to_csv(Path("reports") / "diarization_comparison_summary.csv")
    print(f"\nSummaries saved to reports/")

if __name__ == "__main__":
    main()
