#!/usr/bin/env python3
"""
Generates STAGE5A_1_CLAIM_AUDIT.csv:
Audits every major quantitative claim across Stage 5A and Stage 5A.1 reports,
linking to source code, source output, reported value, verified value, and status.
"""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FINAL_AUDIT_DIR = REPO_ROOT / "research_stages" / "stage5a_1_final_audit"

claims = [
    {
        "claim_id": "CLM-001",
        "report_location": "STAGE5A_1_NORMALIZATION_AUDIT.md:Sec 6",
        "claim": "Single-encoder diarization is invariant to offdiag_zscore_affinity normalization (H vs A and I vs B).",
        "source_script": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py:407-448",
        "source_output": "research_stages/stage5a_1/downstream_metrics/stage5a_1_summary_diarization.csv",
        "reported_value": "Delta DER = 0.0000, Delta cpWER = 0.0000 across all 8 recordings",
        "verified_value": "Delta DER = 0.0000, Delta cpWER = 0.0000 across all 8 recordings (0 discrepancies across all 8x8 metrics)",
        "status": "VERIFIED",
        "explanation": "NeMo NME-SC selects top-p neighbors via row-wise argsort rank. Because offdiag_zscore_affinity is strictly monotonic on off-diagonal similarities, row-wise argsort is mathematically identical."
    },
    {
        "claim_id": "CLM-002",
        "report_location": "STAGE5A_1_GATE_IMPLEMENTATION_AUDIT.md:Sec 2.1",
        "claim": "Stage 5A executable gate implements linear mapping with clipping, not sigmoid.",
        "source_script": "research_stages/stage5a/scripts/fusion_models.py:96-103",
        "source_output": "research_stages/stage5a_1/gate_audit/gate_audit_results.json",
        "reported_value": "g_res = clip(0.5 + 2.0 * pred_delta, 0.15, 0.85)",
        "verified_value": "g_res = np.clip(0.5 + self.gain_factor * pred_delta, self.clip_min, self.clip_max) with gain=2.0, clip=[0.15, 0.85]",
        "status": "VERIFIED",
        "explanation": "Code audit confirms exact linear clipped mapping. The prior report text had written sigma(2.0*delta), which was an analytical discrepancy in documentation, now resolved."
    },
    {
        "claim_id": "CLM-003",
        "report_location": "STAGE5A_1_GATE_IMPLEMENTATION_AUDIT.md:Sec 3",
        "claim": "resnet293_l2_norm is measured prior to unit normalization.",
        "source_script": "research_stages/stage3_dual_encoder/run_stage3_extraction.py:558-569",
        "source_output": "research_stages/stage3_dual_encoder/cache/16k_mono/R06_rich_karen/subsegments_metadata__scale4.json",
        "reported_value": "Raw embedding Euclidean norm before L2 normalization",
        "verified_value": "np.linalg.norm(res_raw, axis=-1) on raw linear projection head output",
        "status": "VERIFIED",
        "explanation": "Direct code inspection confirms resnet293_l2_norm stores res_norms before unit-normalization, capturing embedding confidence."
    },
    {
        "claim_id": "CLM-004",
        "report_location": "STAGE5A_1_GATE_IMPLEMENTATION_AUDIT.md:Sec 3",
        "claim": "vad_conf_mean has near-zero variance in clean speech LOTO subsegments.",
        "source_script": "research_stages/stage5a_1/scripts/audit_gate_features.py:108-120",
        "source_output": "research_stages/stage5a_1/gate_audit/gate_audit_results.json",
        "reported_value": "Variance = 0.0102, median = 0.9909, 75% of segments > 0.962",
        "verified_value": "Mean = 0.9542, Std = 0.1012, Median = 0.9909, P25 = 0.9621, Var = 0.010239",
        "status": "VERIFIED",
        "explanation": "Clean speech LOTO turns filter out non-speech and ambiguous frames, leaving VAD confidence saturated near 1.0."
    },
    {
        "claim_id": "CLM-005",
        "report_location": "STAGE5A_1_VERIFICATION_RESULTS.md:Sec 1",
        "claim": "Total valid within-recording turn pairs is 9,489 from 250 clean turns.",
        "source_script": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py:114-123",
        "source_output": "research_stages/stage5a_1/verification/stage5a_1_error_complementarity.csv",
        "reported_value": "250 clean turns, 9,489 within-recording pairs",
        "verified_value": "250 clean turns, 9,489 within-recording pairs (reconciled against Stage 5A's 254 turns / 9,817 pairs)",
        "status": "VERIFIED_WITH_QUALIFICATION",
        "explanation": "Stage 5A.1 applied stricter subsegment containment (+-50ms) and excluded overlap subsegments, dropping 4 turns (R03 seg 41, R06 seg 66, R06 seg 85, R07 seg 44)."
    },
    {
        "claim_id": "CLM-006",
        "report_location": "STAGE5A_1_VERIFICATION_RESULTS.md:Sec 3",
        "claim": "Macro ROC-AUC increases from 0.8238/0.8230 to 0.8450 (Equal Fusion) and 0.8458 (Reliability Gate).",
        "source_script": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py:645-665",
        "source_output": "research_stages/stage5a_1/verification/stage5a_1_verification_summary.csv",
        "reported_value": "TitaNet=0.8238, ResNet=0.8230, Equal=0.8450, Gate=0.8458",
        "verified_value": "6-rec macro: Tita=0.8238, Res=0.8230, Eq=0.8450, Gate=0.8458. 5-rec macro (exc R01): Tita=0.8172, Res=0.8304, Eq=0.8426, Gate=0.8436.",
        "status": "VERIFIED_WITH_QUALIFICATION",
        "explanation": "The reported 0.8238/0.8450 values are the 6-recording average (R01, R03, R05, R06, R07, R08). R01 was included in the CSV (AUC=0.8571) but omitted from the report text table."
    },
    {
        "claim_id": "CLM-007",
        "report_location": "STAGE5A_1_VERIFICATION_RESULTS.md:Sec 5",
        "claim": "Error complementarity between TitaNet and ResNet on within-recording pairs is 58.03%.",
        "source_script": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py:670-710",
        "source_output": "research_stages/stage5a_1/verification/stage5a_1_error_complementarity.csv",
        "reported_value": "Total errors=3,579; Tita-only=1,035; Res-only=1,042; Shared=1,502; Comp Frac=58.03%; Jaccard=0.4197",
        "verified_value": "Total errors=3,579; Tita-only=1,035; Res-only=1,042; Shared=1,502; Comp Frac=58.0330%; Jaccard=0.41967",
        "status": "VERIFIED",
        "explanation": "Independently recomputed and verified exactly from held-out frozen-threshold decisions across all 9,489 pairs."
    },
    {
        "claim_id": "CLM-008",
        "report_location": "STAGE5A_1_DOWNSTREAM_RESULTS.md:Sec 2",
        "claim": "Macro DER collar 0.0s ovl: TitaNet=46.22%, ResNet=45.98%, Equal Fusion=46.17%, Gate=45.92%.",
        "source_script": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py:475-500",
        "source_output": "research_stages/stage5a_1/downstream_metrics/stage5a_1_summary_diarization.csv",
        "reported_value": "A=46.22%, B=45.98%, J=46.17%, K=45.92%",
        "verified_value": "A=0.4622, B=0.4598, J=0.4617, K=0.4592, H=0.4622, I=0.4598 (exact match)",
        "status": "VERIFIED",
        "explanation": "Recomputed directly from per-recording RTTMs evaluated via pyannote.metrics."
    },
    {
        "claim_id": "CLM-009",
        "report_location": "STAGE5A_1_DOWNSTREAM_RESULTS.md:Sec 2",
        "claim": "DER error decomposition: False Alarm ~13.33%, Miss ~24.41%, Speaker Confusion ~8.18% - 8.48%.",
        "source_script": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py:480-490",
        "source_output": "research_stages/stage5a_1/downstream_metrics/stage5a_1_summary_diarization.csv",
        "reported_value": "FA=13.33%, Miss=24.41%, Conf: A=8.48%, B=8.24%, J=8.43%, K=8.18%",
        "verified_value": "FA=0.1333, Miss=0.2441, Conf: A=0.0848, B=0.0824, J=0.0843, K=0.0818 (exact match)",
        "status": "VERIFIED",
        "explanation": "Confirms that representation fusion affects only the 8.2% confusion component, leaving the 37.7% FA+Miss ceiling unchanged."
    },
    {
        "claim_id": "CLM-010",
        "report_location": "STAGE5A_1_DOWNSTREAM_RESULTS.md:Sec 4.1",
        "claim": "Equal-weight fusion achieves statistically significant reduction in cpWER over TitaNet baseline.",
        "source_script": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py:720-770",
        "source_output": "research_stages/stage5a_1/bootstrap/stage5a_1_paired_bootstrap_differences.csv",
        "reported_value": "Delta cpWER = -1.02%, 95% CI = [-2.07%, -0.10%], Wins: 5 / 1 / 2",
        "verified_value": "Point diff = -0.0102, 95% CI = [-0.0207, -0.0010], CI width = 0.0197, includes_zero = False, Wins: 5 / 1 / 2",
        "status": "VERIFIED_WITH_QUALIFICATION",
        "explanation": "Bootstrap interval strictly excludes zero. However, cluster count is small (N=8 recordings), and multiple comparison correction was not performed; significance must be interpreted as exploratory."
    },
    {
        "claim_id": "CLM-011",
        "report_location": "STAGE5A_1_DOWNSTREAM_RESULTS.md:Sec 3",
        "claim": "Single-speaker collapse occurs on R07 across all evaluated conditions.",
        "source_script": "research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py:440-442",
        "source_output": "research_stages/stage5a_1/downstream_metrics/stage5a_1_per_recording_diarization.csv",
        "reported_value": "Predicted speakers = 1, Ref speakers = 3 across all conditions (A, B, C, H, I, J, K, K_sig)",
        "verified_value": "Predicted speakers = 1 on R07 across all 8 conditions; DER = 45.16% on all",
        "status": "VERIFIED",
        "explanation": "Eigenvalues are flattened by severe overlap and background saturation, causing spectral gap to select k=1."
    },
    {
        "claim_id": "CLM-012",
        "report_location": "STAGE5A_1_EXECUTION_REPORT.md:Sec 2.2",
        "claim": "Execution environment runs NeMo 2.7.3.",
        "source_script": "research_stages/.venv_phase20/lib/python3.10/site-packages/nemo/package_info.py",
        "source_output": "terminal inspection: nemo.__version__",
        "reported_value": "NeMo 2.7.3 (correcting Stage 5A report text which wrote 2.1.0)",
        "verified_value": "nemo.__version__ == '2.7.3'",
        "status": "VERIFIED",
        "explanation": "Verified directly in the python environment. NeMo 2.7.3 matches the frozen Phase20 environment."
    }
]

out_p = FINAL_AUDIT_DIR / "reports" / "STAGE5A_1_CLAIM_AUDIT.csv"
fieldnames = ["claim_id", "report_location", "claim", "source_script", "source_output", "reported_value", "verified_value", "status", "explanation"]

with open(out_p, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(claims)

print(f"Generated {out_p} with {len(claims)} audited claims.")
