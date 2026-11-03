# Stage 5B.1: Correction Execution Log

## 1. Execution Environment
- **Operating System**: Linux 6.6.137+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.137-1+rpt1 (2026-02-10) aarch64
- **Host / User**: `priyanka@raspberrypi` / `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`
- **Execution Workspace**: `research_stages/stage5b/stage5b_1_vad_optimization/audit_corrections/`
- **Python Environment**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python` (Python 3.10 with Pyannote 3.3.2, jiwer, torch 2.6.0)
- **Execution Date**: 2026-11-03

## 2. Chronological Record of Corrective Actions
1. **Protected Workspace Setup**:
   Created `audit_corrections/` with subdirectories `scripts/`, `metrics/`, `predictions/`, `configs/`.
2. **Pre-Audit Protected Artifact Integrity Verification**:
   Executed SHA-256 validation against `frozen_artifact_hashes_after.csv`. Verified all 298 frozen artifacts matched bitwise (0 mismatches, 0 missing).
3. **Pre-Audit Stage 5B.1 File Hashing**:
   Recorded SHA-256 hashes of all 434 existing Stage 5B.1 files into `audit_corrections/stage5b_1_pre_audit_hashes.json`.
4. **Issue B Audit (Consensus Alignment)**:
   Inspected timestamp arrays in `_dfn_posteriors.npz` and `_16k_mono_posteriors.npz` across all 8 recordings. Verified bitwise equality (`max_deviation = 0.0s`, `stride = 0.016875s`, `start = 0.030969s`). Confirmed that truncation did not drop any frames. Produced `STAGE5B_1_CONSENSUS_ALIGNMENT_AUDIT.md`.
5. **Issue C Audit & Implementation (Union Denominator Normalization)**:
   Identified that `total_gt_speech` summed individual RTTM segments ($T_{\text{sum}}$), double-counting overlapping speech in R02 (+75.3%) and R04 (+52.7%). Implemented exact union speech computation ($T_{\text{union}}$). Produced `STAGE5B_1_VAD_UNION_DENOMINATOR_AUDIT.md`.
6. **Issue D Audit & Instrumentation (Projected DER Nearest-Neighbor Quantification)**:
   Audited `project_vad_to_diarization()`. Measured nearest-neighbor label propagation duration and percentage across all 208 runs. Re-labeled results as "VAD-mask-projected DER under frozen baseline speaker assignments". Produced `STAGE5B_1_PROJECTED_DER_METHOD_LIMITATIONS.md`.
7. **Issue E Audit & Schema Fix (cpWER)**:
   Diagnosed that `p20_common.write_json(tmp_attr_path, segs_attr)` wrote a JSON list rather than a dict `{"segments": segs_attr}`, causing `AttributeError` in `fusion_metrics.py`. Fixed schema, removed unsafe fallback, and recomputed authentic cpWER across all 208 runs. Produced `STAGE5B_1_CPWER_AUDIT.md`.
8. **Issue F Audit (Posterior Timestamp Conventions)**:
   Verified `frame_times` represent window centers `middle`. Confirmed $>99.6\%$ frame agreement and $>99.4\%$ IoU with Pyannote oracle timeline.
9. **Execution of Corrected Sweep**:
   Executed `audit_corrections/scripts/run_corrected_evaluations.py` across 26 configurations and 8 recordings (208 evaluations).
   - Generated:
     - `STAGE5B_1_CORRECTED_VAD_METRICS_PER_RECORDING.csv`
     - `STAGE5B_1_CORRECTED_DOWNSTREAM_METRICS_PER_RECORDING.csv`
     - `STAGE5B_1_CORRECTED_SUMMARY_COMPARISON.csv`
     - `STAGE5B_1_METRIC_RECONCILIATION_DELTA.csv`
10. **Deliverables & Synthesis**:
    Emitted master audit report `STAGE5B_1_CORRECTION_AUDIT.md` and addendum `STAGE5B_1_FINAL_REPORT_ADDENDUM.md`.
