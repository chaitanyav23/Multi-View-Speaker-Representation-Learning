# Stage 5B.1: Execution Log

## 1. Environment & Hardware
- **Operating System**: Linux 6.6.137+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.137-1+rpt1 (2026-02-10) aarch64
- **Host / User**: `priyanka@raspberrypi` / `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`
- **Python Version**: Python 3.13.7 (Anaconda environment)
- **PyTorch Version**: 2.6.0
- **Pyannote Audio Version**: 3.3.2
- **Execution Date**: 2026-11-03

## 2. Experimental Execution Chronology
1. **Directory Setup**:
   Created `research_stages/stage5b/stage5b_1_vad_optimization/` with subdirectories `configs/`, `scripts/`, `predictions/`, `metrics/`, and `plots/`.
2. **Pre-Execution Integrity Verification**:
   Executed SHA-256 validation against `frozen_artifact_hashes_after.csv`. Verified all 298 frozen artifacts matched bitwise (0 mismatches, 0 missing).
3. **Audit of Baseline VAD**:
   Executed `scripts/audit_baseline_vad.py` to reconcile Stage 5B.0 reconnaissance numbers against ground truth across all 8 recordings.
   - Identified that Stage 5B.0 reported numbers were from legacy single-file evaluation on R07.
4. **Configuration Generation**:
   Emitted `configs/stage5b_1_config.json` containing 26 experimental configurations covering Experiments 0 through 6.
5. **Experimental Grid Sweep**:
   Executed `scripts/run_stage5b_1_experiments.py`:
   - Evaluated 26 configurations across 8 recordings (208 total evaluations).
   - Produced 208 hypothesis RTTM files in `predictions/{config_id}/`.
   - Computed frame-level VAD metrics (Precision, Recall, F1, Miss, FA relative to reference speech, FA relative to silence).
   - Computed downstream Diarization metrics (DER collar=0.0s, DER collar=0.25s, Miss, False Alarm, Speaker Confusion, JER).
   - Output files:
     - `STAGE5B_1_VAD_METRICS_PER_RECORDING.csv`
     - `STAGE5B_1_DOWNSTREAM_METRICS_PER_RECORDING.csv`
     - `metrics/stage5b_1_summary_comparison.csv`
     - `STAGE5B_1_PARAMETER_SENSITIVITY.csv`
   - Total runtime: ~22.4 seconds (fully vectorized with NumPy binary search).
6. **Analytical Plot Generation**:
   Generated 4 high-resolution figures:
   - `plots/vad_precision_recall_tradeoff.png`
   - `plots/macro_der_vs_vad_config.png`
   - `plots/der_error_decomposition.png`
   - `plots/per_recording_false_alarm_reduction.png`
7. **Documentation & Synthesis**:
   Created:
   - `STAGE5B_1_EXPERIMENT_PROTOCOL.md`
   - `STAGE5B_1_BASELINE_REPRODUCTION.md`
   - `STAGE5B_1_SHORT_TURN_ANALYSIS.md`
   - `STAGE5B_1_RECORDING_DIAGNOSTICS.md`
   - `STAGE5B_1_FINAL_REPORT.md`
   - `STAGE5B_1_EXECUTION_LOG.md`
   - `STAGE5B_1_MANIFEST.json`
