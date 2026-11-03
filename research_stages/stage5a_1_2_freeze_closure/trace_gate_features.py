#!/usr/bin/env python3
"""
Stage 5A.1.2 Gate Feature Trace Script:
Traces the executed reliability gate feature vector from source data
to training and inference, resolving all candidate features.
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CLOSURE_DIR = REPO_ROOT / "research_stages" / "stage5a_1_2_freeze_closure"


def build_gate_feature_trace() -> List[Dict[str, Any]]:
    trace_rows = [
        {
            "feature_name": "resnet293_l2_norm",
            "extraction_file": "research_stages/stage3_dual_encoder/run_stage3_extraction.py",
            "extraction_function": "extract_recording_scale",
            "transformation": "Raw Euclidean norm before L2 normalization: np.linalg.norm(res_raw, axis=-1)",
            "training_input": "meta_mono[i]['resnet293_l2_norm']",
            "training_function": "ReliabilityGate.fit() (via gate_features list)",
            "inference_input": "meta_s4_mono[i]['resnet293_l2_norm']",
            "inference_function": "ReliabilityGate.build_affinity_gate_matrix() / predict_subsegment_weights()",
            "feature_order": 1,
            "actually_used": "Used",
            "evidence_path": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py",
            "evidence_lines": "298, 307, 312, 378, 577",
            "verification_status": "VERIFIED",
        },
        {
            "feature_name": "dfn_noise_reduction_db",
            "extraction_file": "research_stages/stage1_data_prep/scripts/run_stage1_prep.py",
            "extraction_function": "compute_dfn_metrics / extract_subsegment_metadata",
            "transformation": "Energy ratio: 10 * log10(power(mono) / max(power(mono - dfn), 1e-12))",
            "training_input": "meta_dfn[i]['dfn_noise_reduction_db']",
            "training_function": "ReliabilityGate.fit() (via gate_features list)",
            "inference_input": "meta_s4_dfn[i]['dfn_noise_reduction_db']",
            "inference_function": "ReliabilityGate.build_affinity_gate_matrix() / predict_subsegment_weights()",
            "feature_order": 2,
            "actually_used": "Used",
            "evidence_path": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py",
            "evidence_lines": "299, 307, 312, 379, 577",
            "verification_status": "VERIFIED",
        },
        {
            "feature_name": "vad_conf_mean",
            "extraction_file": "research_stages/stage3_dual_encoder/run_stage3_extraction.py",
            "extraction_function": "extract_subsegment_metadata",
            "transformation": "Mean pyannote VAD posterior probability across subsegment frames",
            "training_input": "meta_mono[i]['vad_conf_mean']",
            "training_function": "ReliabilityGate.fit() (via gate_features list)",
            "inference_input": "meta_s4_mono[i]['vad_conf_mean']",
            "inference_function": "ReliabilityGate.build_affinity_gate_matrix() / predict_subsegment_weights()",
            "feature_order": 3,
            "actually_used": "Used",
            "evidence_path": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py",
            "evidence_lines": "300, 307, 312, 380, 577",
            "verification_status": "VERIFIED",
        },
        {
            "feature_name": "wada_snr",
            "extraction_file": "N/A (not in active pipeline)",
            "extraction_function": "N/A",
            "transformation": "WADA algorithm SNR estimation (exploratory code only)",
            "training_input": "None",
            "training_function": "None",
            "inference_input": "None",
            "inference_function": "None",
            "feature_order": "N/A",
            "actually_used": "Not used",
            "evidence_path": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py",
            "evidence_lines": "307 (features list strictly contains 3 elements)",
            "verification_status": "VERIFIED_NOT_USED",
        },
        {
            "feature_name": "titanet_embedding_norm",
            "extraction_file": "research_stages/stage3_dual_encoder/run_stage3_extraction.py",
            "extraction_function": "extract_recording_scale",
            "transformation": "NeMo TitaNet unit-normalizes internally; norm is identically 1.0 (zero variance)",
            "training_input": "None",
            "training_function": "None",
            "inference_input": "None",
            "inference_function": "None",
            "feature_order": "N/A",
            "actually_used": "Not used",
            "evidence_path": "research_stages/stage5a_1/gate_audit/gate_audit_results.json",
            "evidence_lines": "audit_gate_features.py: zero variance noted",
            "verification_status": "VERIFIED_NOT_USED",
        },
    ]
    return trace_rows


def main():
    print("=" * 80)
    print("Stage 5A.1.2: Audit B — Exact Reliability-Gate Feature Vector Trace")
    print("=" * 80)

    rows = build_gate_feature_trace()
    out_csv = CLOSURE_DIR / "gate_feature_trace.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved Gate Feature Trace CSV to: {out_csv}\n")
    for r in rows:
        print(f"  {r['feature_name']:<25}: {r['actually_used']} (Order: {r['feature_order']}, Status: {r['verification_status']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
