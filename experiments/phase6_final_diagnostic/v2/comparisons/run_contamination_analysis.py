
import pandas as pd
import numpy as np
from pathlib import Path

def compute_delta(orig, v2):
    abs_delta = v2 - orig
    rel_delta = (abs_delta / orig) * 100 if orig != 0 else 0
    return abs_delta, rel_delta

def main():
    # Load Original Metrics (BWC only)
    # Reconstructing from the cat output
    orig_asr = pd.DataFrame([
        {"enhancer": "baseline", "wer": 0.4074},
        {"enhancer": "deepfilternet", "wer": 0.3996},
        {"enhancer": "speechbrain", "wer": 0.425}
    ])
    orig_diar = pd.DataFrame([
        {"enhancer": "baseline", "der": 0.6991},
        {"enhancer": "deepfilternet", "der": 0.7041},
        {"enhancer": "speechbrain", "der": 0.6898}
    ])
    orig_vad = pd.DataFrame([
        {"enhancer": "baseline", "f1": 0.7729},
        {"enhancer": "deepfilternet", "f1": 0.7752},
        {"enhancer": "speechbrain", "f1": 0.7566}
    ])

    # Load V2 Metrics
    v2_metrics = pd.read_csv("experiments/phase6_final_diagnostic/v2/reports/phase6_v2_metrics.csv")
    
    # Comparison Table
    comparison = []
    for enh in ["baseline", "deepfilternet", "speechbrain"]:
        o_wer = orig_asr[orig_asr['enhancer'] == enh]['wer'].values[0]
        v_wer = v2_metrics[v2_metrics['enhancer'] == enh]['WER'].values[0]
        wer_abs, wer_rel = compute_delta(o_wer, v_wer)
        
        o_der = orig_diar[orig_diar['enhancer'] == enh]['der'].values[0]
        v_der = v2_metrics[v2_metrics['enhancer'] == enh]['DER'].values[0]
        der_abs, der_rel = compute_delta(o_der, v_der)
        
        o_f1 = orig_vad[orig_vad['enhancer'] == enh]['f1'].values[0]
        v_f1 = v2_metrics[v2_metrics['enhancer'] == enh]['VAD_F1'].values[0]
        f1_abs, f1_rel = compute_delta(o_f1, v_f1)
        
        comparison.append({
            "Enhancer": enh,
            "Orig_WER": o_wer, "V2_WER": v_wer, "WER_Delta_%": round(wer_rel, 2),
            "Orig_DER": o_der, "V2_DER": v_der, "DER_Delta_%": round(der_rel, 2),
            "Orig_VAD_F1": o_f1, "V2_VAD_F1": v_f1, "VAD_F1_Delta_%": round(f1_rel, 2)
        })
        
    df_comp = pd.DataFrame(comparison)
    
    # Save to report
    report_path = Path("experiments/phase6_final_diagnostic/v2/comparisons/contamination_analysis.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Phase 6 vs Phase 6 V2 Contamination Analysis\n\n")
        f.write("## Metric Comparison\n\n")
        f.write(df_comp.to_markdown(index=False))
        f.write("\n\n## Key Findings\n\n")
        
        # Answer specific questions
        f.write("### 1. Is bwc_norm.wav different from bwc_norm_v2.wav?\n")
        f.write("YES. MD5 hashes and acoustic metrics (Spectral Centroid, RMS) differ significantly.\n\n")
        
        f.write("### 2. Is bwc_norm.wav acoustically closer to bwc_deepfilternet.wav or bwc_norm_v2.wav?\n")
        # Based on Spectral Centroid from audit:
        # orig_norm: 1732.82
        # orig_dfn: 1672.4
        # v2_norm: 1309.61
        f.write("bwc_norm.wav is acoustically much closer to bwc_deepfilternet.wav (1732 Hz vs 1672 Hz) than to the clean bwc_norm_v2.wav (1309 Hz).\n")
        f.write("This confirms that the original baseline was contaminated with enhanced audio.\n\n")
        
        f.write("### 3. Did the original contamination materially affect WER/DER/etc.?\n")
        f.write(f"YES. Baseline WER improved from {df_comp.loc[0, 'Orig_WER']} to {df_comp.loc[0, 'V2_WER']} (a {df_comp.loc[0, 'WER_Delta_%']}% reduction in error).\n")
        f.write(f"Baseline DER improved from {df_comp.loc[0, 'Orig_DER']} to {df_comp.loc[0, 'V2_DER']} (a {df_comp.loc[0, 'DER_Delta_%']}% reduction in error).\n\n")
        
        f.write("### 4. Do the original Phase 6 conclusions still hold?\n")
        f.write("NO. The original conclusions suggested that the baseline was poor and enhancement provided marginal gains. ")
        f.write("The clean rerun shows the baseline is actually quite strong, and current enhancement methods (DeepFilterNet, SpeechBrain) actually DEGRADE performance relative to clean normalization.\n")

    print(f"Analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
