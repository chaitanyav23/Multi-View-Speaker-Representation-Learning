# Stage 5B.2 V4.1: Audit Execution Log

**Audit Phase**: Stage 5B.2 V4.1  
**Date**: November 4, 2026  
**Auditor**: Independent Forensic Research Engineer  
**Status**: COMPLETE

---

## 1. Chronological Timeline of Actions

1. **Workspace Setup & Boundary Verification**:
   - Inspected git status: working tree clean at commit `e64286b`.
   - Created isolated directory: `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/` with subdirectories `scripts/`, `tests/`, `configs/`, `metrics/`, `predictions/`, `plots/`.
   - Confirmed strict workspace isolation.

2. **Audit A: Forensic Metric Investigation**:
   - Compared SHA-256 hashes of all four CSV files across `audit_corrections_v3`, `audit_corrections_v4`, and `audit_corrections_v4_1`.
   - Proved that `STAGE5B_2_CORRECTED_RESULTS.csv`, `STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv`, `STAGE5B_2_CORRECTED_OVERLAP_METRICS.csv`, and `STAGE5B_2_CORRECTED_WORD_METRICS.csv` were 100% bit-for-bit identical across all versions.
   - Traced step 2626 in conversation transcript to uncover the origin of the 24.31% number: confirmed as a clerical markdown hallucination where fabricated recording names and baseline numbers were written into the report text while the real deltas were copied from the CSV.
   - Independently computed DER from raw hypothesis RTTMs using Pyannote DiarizationErrorRate. Exact match found: `O02 = 44.90%`, `O42-R2 = 45.42%`.

3. **Audit B: M2 Prototype-Purity Interval Union**:
   - Inspected `load_dual_encoder_prototypes_pure` in V4.0. Identified vulnerability where naive summation `spk_coverage[spk] += inter` could inflate speaker coverage for duplicate or overlapping intervals.
   - Implemented exact per-speaker 1D interval union in `run_stage5b_2_corrected_v4_1.py`.
   - Handled floating-point tolerance at boundaries (`- 1e-6`).
   - Developed production test suite with 19 tests in `tests/test_stage5b_2_v4_1_production.py`. Executed via pytest: 19/19 passed in 0.85s.
   - Performed dataset scan across all 1,840 subsegments in the 8 benchmark recordings: confirmed 0 coverage differences, 0 decision flips, and identical 1,426 admitted prototypes.

4. **Minimal Benchmark Rerun**:
   - Executed `run_stage5b_2_corrected_v4_1.py --configs minimal --seed 42`.
   - Processed 6 configurations across 8 recordings (48 runs) in 62.37 seconds.
   - Successfully generated native CSVs and plots in `audit_corrections_v4_1/`.

5. **Statistical Significance Testing**:
   - Recomputed paired differences on authoritative data: 1 win (`R04`: -8.09%), 3 losses (`R03`: +2.64%, `R05`: +2.82%, `R06`: +6.81%), 4 ties (`R01`, `R02`, `R07`, `R08`: 0.00%).
   - Ran Wilcoxon signed-rank test on non-zero pairs: $W = 4.0, p = 0.8750$.
   - Computed 10,000-sample bootstrap 95% CI of mean difference: $[-2.35\%, +3.04\%]$.

6. **Cryptographic Integrity Verification**:
   - Verified 298/298 Stage 1–5A.1.2 frozen artifacts against `frozen_artifact_hashes_after.csv` (100% match).
   - Verified 125 V3 artifacts and 129 V4 artifacts intact.

7. **Documentation & Manifest Generation**:
   - Generated all 13 required audit and verification reports in `audit_corrections_v4_1/`.
   - Generated `STAGE5B_2_V4_1_MANIFEST.json` containing cryptographic SHA-256 hashes of all deliverables.
