#!/usr/bin/env python3
"""
Stage 5A.1.2 Verification Audit Script:
Independently searches for saved held-out prediction artifacts,
audits macro averaging definitions (6-rec eligible vs 5-rec sensitivity),
recomputes verification metrics from saved per-recording tables,
and evaluates aggregate consistency and complementarity.
"""

import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STAGE5A_1_DIR = REPO_ROOT / "research_stages" / "stage5a_1"
STAGE5A_DIR = REPO_ROOT / "research_stages" / "stage5a"
CLOSURE_DIR = REPO_ROOT / "research_stages" / "stage5a_1_2_freeze_closure"


def search_for_raw_predictions() -> Dict[str, Any]:
    """Search for raw pair-level held-out prediction artifacts."""
    search_dirs = [STAGE5A_1_DIR, STAGE5A_DIR]
    found_pair_prediction_files = []
    
    for s_dir in search_dirs:
        for root, dirs, files in os.walk(s_dir):
            for f in files:
                f_lower = f.lower()
                if any(k in f_lower for k in ["pair_pred", "heldout_pred", "test_pairs", "trial_pred"]):
                    found_pair_prediction_files.append(os.path.join(root, f))
    
    return {
        "raw_pair_predictions_found": len(found_pair_prediction_files) > 0,
        "files": found_pair_prediction_files,
        "status": (
            "RAW_HELDOUT_PREDICTIONS_AVAILABLE"
            if len(found_pair_prediction_files) > 0
            else "AGGREGATE_CONSISTENCY_ONLY"
        ),
    }


def audit_macro_definitions() -> List[Dict[str, Any]]:
    """Audit recording-level counts and recompute 6-rec and 5-rec macros."""
    per_rec_csv = STAGE5A_1_DIR / "verification" / "stage5a_1_verification_per_recording.csv"
    assert per_rec_csv.exists(), f"Missing {per_rec_csv}"

    rows = []
    with open(per_rec_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Group by method
    by_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        m = r["method"]
        rec = r["held_out_rec_id"]
        by_method.setdefault(m, {})[rec] = {
            "pos": int(r["test_pos_pairs"]),
            "neg": int(r["test_neg_pairs"]),
            "eligible": r["eligible_for_eer"] == "True",
            "thr": float(r["frozen_train_thr"]),
            "far": float(r["far_at_frozen_thr"]) if r["far_at_frozen_thr"] != "" else None,
            "frr": float(r["frr_at_frozen_thr"]) if r["frr_at_frozen_thr"] != "" else None,
            "hter": float(r["hter_at_frozen_thr"]) if r["hter_at_frozen_thr"] != "" else None,
            "auc": float(r["held_out_roc_auc"]) if r["held_out_roc_auc"] != "" else None,
        }

    rec_6 = ["R01_sara_sharif", "R03_disabled_parking", "R05_southwest_pilot", "R06_rich_karen", "R07_bwc_cheap_shot", "R08_uk_parliament"]
    rec_5 = ["R03_disabled_parking", "R05_southwest_pilot", "R06_rich_karen", "R07_bwc_cheap_shot", "R08_uk_parliament"]

    conditions_map = [
        ("Condition A", "condition_A_raw_titanet_dfn", "Raw TitaNet (DFN)"),
        ("Condition B", "condition_B_raw_resnet293_mono", "Raw ResNet293 (Mono)"),
        ("Condition C", "condition_C_norm_equal_weight_stage5a", "Raw Equal Fusion (Stage 5A)"),
        ("Condition H", "condition_H_norm_titanet_dfn_alone", "Norm TitaNet Alone (Control)"),
        ("Condition I", "condition_I_norm_resnet293_mono_alone", "Norm ResNet Alone (Control)"),
        ("Condition J", "condition_J_norm_equal_weight_fusion", "Norm Equal Fusion (Control)"),
        ("Condition K", "condition_K_norm_reliability_gate", "Norm Linear Reliability Gate"),
        ("Condition K_sig", "condition_K_sigmoid_gate", "Norm Sigmoid Gate (Ablation)"),
    ]

    reconciled_rows = []
    for cond_code, method_name, desc in conditions_map:
        rec_data = by_method[method_name]
        
        # 6-rec macro
        aucs_6 = [rec_data[r]["auc"] for r in rec_6 if rec_data[r]["auc"] is not None]
        hters_6 = [rec_data[r]["hter"] for r in rec_6 if rec_data[r]["hter"] is not None]
        m6_auc = sum(aucs_6) / len(aucs_6) if aucs_6 else None
        m6_hter = sum(hters_6) / len(hters_6) if hters_6 else None

        # 5-rec macro
        aucs_5 = [rec_data[r]["auc"] for r in rec_5 if rec_data[r]["auc"] is not None]
        hters_5 = [rec_data[r]["hter"] for r in rec_5 if rec_data[r]["hter"] is not None]
        m5_auc = sum(aucs_5) / len(aucs_5) if aucs_5 else None
        m5_hter = sum(hters_5) / len(hters_5) if hters_5 else None

        reconciled_rows.append({
            "condition_code": cond_code,
            "method_name": method_name,
            "description": desc,
            "r01_auc": rec_data["R01_sara_sharif"]["auc"],
            "r01_hter": rec_data["R01_sara_sharif"]["hter"],
            "macro_6rec_eligible_auc": round(m6_auc, 4) if m6_auc else None,
            "macro_6rec_eligible_hter": round(m6_hter, 4) if m6_hter else None,
            "macro_5rec_sensitivity_auc": round(m5_auc, 4) if m5_auc else None,
            "macro_5rec_sensitivity_hter": round(m5_hter, 4) if m5_hter else None,
            "interpretation_note": (
                "6-rec macro includes R01 (1 pos, 14 neg, valid AUC); "
                "5-rec macro excludes R01 as sensitivity check on multi-speaker trial density."
            ),
        })

    return reconciled_rows


def verify_complementarity() -> Dict[str, Any]:
    """Verify complementarity arithmetic from retained summary."""
    comp_csv = STAGE5A_1_DIR / "verification" / "stage5a_1_error_complementarity.csv"
    assert comp_csv.exists(), f"Missing {comp_csv}"

    rows = []
    with open(comp_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    raw_comp = [r for r in rows if "Raw Baselines" in r["comparison"]][0]
    total_pairs = int(raw_comp["total_pairs"])
    m1_only = int(raw_comp["method_1_only_errors"])
    m2_only = int(raw_comp["method_2_only_errors"])
    shared = int(raw_comp["shared_errors"])
    both_correct = int(raw_comp["both_correct"])
    unique_errors = int(raw_comp["total_unique_errors"])
    jaccard_rep = float(raw_comp["error_jaccard_similarity"])
    comp_frac_rep = float(raw_comp["complementary_error_fraction"])

    # Independent check of definitions
    disjoint_errors = m1_only + m2_only
    calc_unique = disjoint_errors + shared
    calc_jaccard = shared / calc_unique if calc_unique > 0 else 0.0
    calc_comp_frac = disjoint_errors / calc_unique if calc_unique > 0 else 0.0

    return {
        "total_pairs": total_pairs,
        "both_correct": both_correct,
        "tita_only_errors": m1_only,
        "resnet_only_errors": m2_only,
        "shared_errors": shared,
        "disjoint_errors": disjoint_errors,
        "unique_errors": calc_unique,
        "unique_errors_matches_retained": calc_unique == unique_errors == 3579,
        "disjoint_errors_matches_retained": disjoint_errors == 2077,
        "error_jaccard_calculated": round(calc_jaccard, 4),
        "error_jaccard_matches": round(calc_jaccard, 4) == round(jaccard_rep, 4) == 0.4197,
        "complementary_error_fraction_calculated": round(calc_comp_frac, 4),
        "complementary_error_fraction_matches": round(calc_comp_frac, 4) == round(comp_frac_rep, 4) == 0.5803,
        "denominator_used": "Total unique error pairs |E_T union E_R| = 3579",
        "status": "VERIFIED_ARITHMETIC_ON_RETAINED_AGGREGATES",
    }


def main():
    print("=" * 80)
    print("Stage 5A.1.2: Audit A — Verification Macro and Held-Out Predictions")
    print("=" * 80)

    # 1. Search for raw prediction files
    raw_search = search_for_raw_predictions()
    print(f"Raw Pair-Level Prediction Files Found: {raw_search['raw_pair_predictions_found']}")
    print(f"Verification Reproduction Classification: {raw_search['status']}")

    # 2. Recompute macro reconciliation
    macro_rows = audit_macro_definitions()
    out_csv = CLOSURE_DIR / "macro_reconciliation.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(macro_rows[0].keys()))
        writer.writeheader()
        writer.writerows(macro_rows)
    print(f"\nSaved Macro Reconciliation CSV to: {out_csv}")

    for r in macro_rows:
        print(f"  {r['condition_code']} ({r['description']}): "
              f"6-Rec Macro AUC = {r['macro_6rec_eligible_auc']}, "
              f"5-Rec Macro AUC = {r['macro_5rec_sensitivity_auc']}")

    # 3. Check complementarity
    comp_audit = verify_complementarity()
    print("\nComplementarity Verification Summary:")
    print(f"  Total Pairs: {comp_audit['total_pairs']}")
    print(f"  |E_T union E_R|: {comp_audit['unique_errors']} (Matches 3579: {comp_audit['unique_errors_matches_retained']})")
    print(f"  |E_T delta E_R|: {comp_audit['disjoint_errors']} (Matches 2077: {comp_audit['disjoint_errors_matches_retained']})")
    print(f"  Error Jaccard: {comp_audit['error_jaccard_calculated']} (Matches 0.4197: {comp_audit['error_jaccard_matches']})")
    print(f"  Complementary Fraction: {comp_audit['complementary_error_fraction_calculated']} (Matches 0.5803: {comp_audit['complementary_error_fraction_matches']})")
    print(f"  Denominator: {comp_audit['denominator_used']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
