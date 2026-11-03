# Stage 5B.2: Execution Log

## 1. Environment & Hardware
- **Operating System**: Linux 6.6.137+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.137-1+rpt1 (2026-02-10) aarch64
- **Host / User**: `priyanka@raspberrypi` / `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`
- **Execution Workspace**: `research_stages/stage5b/stage5b_2_overlap_optimization/`
- **Python Environment**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python` (Python 3.10 with Pyannote 3.3.2, jiwer, torch 2.6.0)
- **Execution Date**: 2026-11-03

## 2. Chronological Record of Experimental Actions
1. **Workspace Setup**:
   Created `research_stages/stage5b/stage5b_2_overlap_optimization/` with subdirectories `scripts/`, `configs/`, `metrics/`, `plots/`, `predictions/`, `overlap_cache/`.
2. **Pre-Execution Protected Artifact Integrity Verification**:
   Executed SHA-256 validation against `frozen_artifact_hashes_after.csv`. Verified all 298 frozen artifacts matched bitwise (0 mismatches, 0 missing).
3. **Stage 5B.1 Closure**:
   Formally closed Stage 5B.1 with `STAGE5B_1_CLOSURE_RECORD.md`.
4. **PyanNet Multi-Speaker Segmentation Inspection**:
   Verified that the local checkpoint `pytorch_model.bin` in WhisperX contains the full 3-speaker multi-label PyanNet segmentation architecture (`classes=['speaker#1', 'speaker#2', 'speaker#3']`).
5. **Continuous Overlap Posterior Extraction & Caching**:
   Constructed continuous overlap aggregator via `pre_aggregation_hook=overlap_hook` (top-2 second-highest probability).
   Extracted and cached continuous overlap curves for all 8 recordings across both `mono` and `dfn` streams in `overlap_cache/` (16 `.npz` files, ~1.5 MB total).
6. **Execution of Overlap Optimization Suite (`run_stage5b_2_experiments.py`)**:
   Evaluated 17 configurations across all 8 recordings (136 total experimental evaluations).
   - Produced 136 multi-speaker hypothesis RTTM files in `predictions/{config_id}/`.
   - Computed Diarization metrics (DER collar=0.0s, DER collar=0.25s, Miss, FA, Confusion, JER).
   - Computed Overlap-specific detection metrics (Overlap Precision, Recall, F1, Ground Truth Overlap Duration, Predicted Overlap Duration).
   - Computed Conversational Word Attribution metrics (cpWER and saWER) using verified dictionary schema `{"segments": segs_attr}`.
   - Saved `STAGE5B_2_METRICS_PER_RECORDING.csv` (136 rows) and `STAGE5B_2_SUMMARY_RESULTS.csv` (17 rows).
7. **Analytical Plot Generation**:
   - `plots/overlap_threshold_sensitivity_r02_r04.png`
   - `plots/macro_der_overlap_attribution_comparison.png`
8. **Synthesis & Documentation**:
   - `STAGE5B_2_EXPERIMENT_PROTOCOL.md`
   - `STAGE5B_2_OVERLAP_AUDIT.md`
   - `STAGE5B_2_METHOD_COMPARISON.md`
   - `STAGE5B_2_FAILURE_ANALYSIS.md`
   - `STAGE5B_2_FINAL_REPORT.md`
   - `STAGE5B_2_EXECUTION_LOG.md`
   - `STAGE5B_2_MANIFEST.json`
