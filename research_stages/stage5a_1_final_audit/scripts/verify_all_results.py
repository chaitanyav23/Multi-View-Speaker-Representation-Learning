#!/usr/bin/env python3
"""
Independent Verification and Reconciliation Script:
Recomputes and verifies all Stage 5A.1 results directly from raw artifacts.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Paths
STAGE5A_1_DIR = REPO_ROOT / "research_stages" / "stage5a_1"

def load_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# 1. Verify Verification Per-Recording & Macro Metrics
verif_per_rec = load_csv(STAGE5A_1_DIR / "verification" / "stage5a_1_verification_per_recording.csv")
verif_summary = load_csv(STAGE5A_1_DIR / "verification" / "stage5a_1_verification_summary.csv")

methods = sorted(set(r["method"] for r in verif_per_rec))

recomputed_verif_summary = []
for m in methods:
    rows_m = [r for r in verif_per_rec if r["method"] == m]
    
    # Check eligible recordings (test_pos_pairs > 0 and test_neg_pairs > 0)
    eligible_rows = [r for r in rows_m if int(r["test_pos_pairs"]) > 0 and int(r["test_neg_pairs"]) > 0]
    rec_ids_eligible = [r["held_out_rec_id"] for r in eligible_rows]
    
    aucs = [float(r["held_out_roc_auc"]) for r in eligible_rows]
    macro_auc_5rec = float(np.mean(aucs)) if aucs else None
    
    hters = [float(r["hter_at_frozen_thr"]) for r in eligible_rows]
    macro_hter_5rec = float(np.mean(hters)) if hters else None
    
    recomputed_verif_summary.append({
        "method": m,
        "eligible_recordings_count": len(eligible_rows),
        "eligible_recordings": rec_ids_eligible,
        "recomputed_macro_5rec_roc_auc": round(macro_auc_5rec, 4),
        "recomputed_macro_5rec_frozen_hter": round(macro_hter_5rec, 4),
    })

print("=== RECOMPUTED VERIFICATION MACRO SUMMARY (5 ELIGIBLE RECORDINGS) ===")
for r in recomputed_verif_summary:
    print(f"Method: {r['method']}")
    print(f"  Eligible: {r['eligible_recordings_count']} recordings: {r['eligible_recordings']}")
    print(f"  Macro ROC-AUC (5 rec): {r['recomputed_macro_5rec_roc_auc']}")
    print(f"  Macro Frozen HTER (5 rec): {r['recomputed_macro_5rec_frozen_hter']}")

# 2. Verify Downstream Diarization Metrics
diar_per_rec = load_csv(STAGE5A_1_DIR / "downstream_metrics" / "stage5a_1_per_recording_diarization.csv")
diar_summary = load_csv(STAGE5A_1_DIR / "downstream_metrics" / "stage5a_1_summary_diarization.csv")

recomputed_diar_summary = []
for m in methods:
    rows_m = [r for r in diar_per_rec if r["method"] == m]
    tot_d = sum(float(r["duration_sec"]) for r in rows_m)
    
    macro_der_0 = float(np.mean([float(r["der_collar0_ovl"]) for r in rows_m]))
    dur_der_0 = float(sum(float(r["der_collar0_ovl"]) * float(r["duration_sec"]) for r in rows_m) / tot_d)
    macro_der_025 = float(np.mean([float(r["der_collar025_noovl"]) for r in rows_m]))
    dur_der_025 = float(sum(float(r["der_collar025_noovl"]) * float(r["duration_sec"]) for r in rows_m) / tot_d)
    macro_jer = float(np.mean([float(r["jer_collar0"]) for r in rows_m]))
    macro_fa = float(np.mean([float(r["false_alarm"]) for r in rows_m]))
    macro_miss = float(np.mean([float(r["miss"]) for r in rows_m]))
    macro_conf = float(np.mean([float(r["confusion"]) for r in rows_m]))
    spk_mae = float(np.mean([float(r["speaker_count_error_abs"]) for r in rows_m]))
    spk_bias = float(np.mean([float(r["speaker_count_signed_bias"]) for r in rows_m]))
    collapse_cnt = sum(1 for r in rows_m if r["single_speaker_collapse"] == "True")
    macro_cpwer = float(np.mean([float(r["cpwer"]) for r in rows_m]))
    macro_sawer = float(np.mean([float(r["sawer"]) for r in rows_m]))
    micro_sacc = float(np.mean([float(r["micro_sacc"]) for r in rows_m]))
    macro_sacc = float(np.mean([float(r["macro_sacc"]) for r in rows_m]))
    
    recomputed_diar_summary.append({
        "method": m,
        "macro_der_collar0_ovl": round(macro_der_0, 4),
        "dur_weighted_der_collar0_ovl": round(dur_der_0, 4),
        "macro_der_collar025_noovl": round(macro_der_025, 4),
        "dur_weighted_der_collar025_noovl": round(dur_der_025, 4),
        "macro_jer": round(macro_jer, 4),
        "macro_false_alarm": round(macro_fa, 4),
        "macro_miss": round(macro_miss, 4),
        "macro_confusion": round(macro_conf, 4),
        "spk_count_mae": round(spk_mae, 4),
        "spk_count_signed_bias": round(spk_bias, 4),
        "single_spk_collapse_count": collapse_cnt,
        "macro_cpwer": round(macro_cpwer, 4),
        "macro_sawer": round(macro_sawer, 4),
        "macro_micro_sacc": round(micro_sacc, 4),
        "macro_macro_sacc": round(macro_sacc, 4),
    })

print("\n=== RECOMPUTED DIARIZATION SUMMARY ===")
for r in recomputed_diar_summary:
    print(f"{r['method']}: DER={r['macro_der_collar0_ovl']}, cpWER={r['macro_cpwer']}, spkMAE={r['spk_count_mae']}, macroSACC={r['macro_macro_sacc']}, Conf={r['macro_confusion']}")

# 3. Check exact matches against reported summaries
print("\n=== CHECKING SUMMARY MATCHES ===")
for rep in diar_summary:
    m = rep["method"]
    rec = next(r for r in recomputed_diar_summary if r["method"] == m)
    match_der = (float(rep["macro_der_collar0_ovl"]) == rec["macro_der_collar0_ovl"])
    match_cpwer = (float(rep["macro_cpwer"]) == rec["macro_cpwer"])
    print(f"{m}: DER match={match_der}, cpWER match={match_cpwer}")

# 4. Check C vs J exact equality
row_c = [r for r in diar_per_rec if r["method"] == "condition_C_norm_equal_weight_stage5a"]
row_j = [r for r in diar_per_rec if r["method"] == "condition_J_norm_equal_weight_fusion"]
c_j_diffs = []
for rc, rj in zip(row_c, row_j):
    for k in rc.keys():
        if rc[k] != rj[k] and k not in ["condition_id", "method"]:
            c_j_diffs.append((rc["rec_id"], k, rc[k], rj[k]))
print(f"\nCondition C vs Condition J per-recording discrepancies count: {len(c_j_diffs)}")

# 5. Check H vs A and I vs B exact equality
row_a = [r for r in diar_per_rec if r["method"] == "condition_A_raw_titanet_dfn"]
row_h = [r for r in diar_per_rec if r["method"] == "condition_H_norm_titanet_dfn_alone"]
h_a_diffs = []
for ra, rh in zip(row_a, row_h):
    for k in ra.keys():
        if ra[k] != rh[k] and k not in ["condition_id", "method"]:
            h_a_diffs.append((ra["rec_id"], k, ra[k], rh[k]))
print(f"Condition H vs Condition A per-recording discrepancies count: {len(h_a_diffs)}")

row_b = [r for r in diar_per_rec if r["method"] == "condition_B_raw_resnet293_mono"]
row_i = [r for r in diar_per_rec if r["method"] == "condition_I_norm_resnet293_mono_alone"]
i_b_diffs = []
for rb, ri in zip(row_b, row_i):
    for k in rb.keys():
        if rb[k] != ri[k] and k not in ["condition_id", "method"]:
            i_b_diffs.append((rb["rec_id"], k, rb[k], ri[k]))
print(f"Condition I vs Condition B per-recording discrepancies count: {len(i_b_diffs)}")

# 6. Verify Error Complementarity
comp_csv = load_csv(STAGE5A_1_DIR / "verification" / "stage5a_1_error_complementarity.csv")
print("\n=== ERROR COMPLEMENTARITY VERIFICATION ===")
for r in comp_csv:
    total = int(r["total_pairs"])
    m1_only = int(r["method_1_only_errors"])
    m2_only = int(r["method_2_only_errors"])
    shared = int(r["shared_errors"])
    both_correct = int(r["both_correct"])
    unique_err = int(r["total_unique_errors"])
    comp_frac = float(r["complementary_error_fraction"])
    jaccard = float(r["error_jaccard_similarity"])
    
    assert m1_only + m2_only + shared == unique_err, "Error sum mismatch!"
    assert unique_err + both_correct == total, "Total pairs mismatch!"
    calc_comp_frac = (m1_only + m2_only) / unique_err
    calc_jaccard = shared / unique_err
    print(f"Comparison: {r['comparison']}")
    print(f"  Total pairs: {total}, Both correct: {both_correct}, Unique errors: {unique_err}")
    print(f"  M1 only: {m1_only}, M2 only: {m2_only}, Shared: {shared}")
    print(f"  Calculated Comp Frac: {calc_comp_frac:.4f} (Reported: {comp_frac})")
    print(f"  Calculated Jaccard: {calc_jaccard:.4f} (Reported: {jaccard})")

# Save audit results
audit_results = {
    "verification_macro_recomputed": recomputed_verif_summary,
    "diarization_summary_recomputed": recomputed_diar_summary,
    "c_vs_j_differences_count": len(c_j_diffs),
    "h_vs_a_differences_count": len(h_a_diffs),
    "i_vs_b_differences_count": len(i_b_diffs),
}
out_p = REPO_ROOT / "research_stages" / "stage5a_1_final_audit" / "reconciliation" / "recomputed_audit_metrics.json"
out_p.write_text(json.dumps(audit_results, indent=2))
print(f"\nSaved recomputed audit metrics to {out_p}")
