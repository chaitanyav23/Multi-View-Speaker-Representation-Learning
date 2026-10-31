# Stage 5A.1: Affinity Normalization, Fusion Fairness & Implementation Audit
## Comprehensive Execution and Environment Report

**Date:** 2026-10-31  
**Execution Environment:** Linux 6.6.137+ x86_64 | NVIDIA GeForce RTX 4090 (24,564 MiB VRAM, Driver 580.82.07, CUDA 12.8)  
**Python Environment:** Python 3.10.20, PyTorch 2.8.0+cu128, NeMo 2.7.3, pyannote.metrics 4.1, pyannote.core 6.0.1  
**Git Commit Hash:** `8826f5e5afaf9e3cf36b068c349486669c26c27a`  
**Isolation Status:** Strict Isolation (All new outputs restricted to `research_stages/stage5a_1/`)

---

## 1. Executive Summary

This report documents the execution of **Stage 5A.1: Affinity Normalization, Fusion Fairness & Implementation Audit**. Stage 5A.1 was initiated to investigate a potential methodological confound in Stage 5A: whether the observed performance gains of multi-view affinity fusion over single-encoder baselines were genuine representation advantages or an artifact of comparing raw single-encoder affinities against normalized fused affinities.

Stage 5A.1 executed completely autonomously under strict preservation and fairness protocols:
1. **Preservation Baseline:** Recorded SHA-256 hashes and sizes for all 133 files in `research_stages/stage5a/` prior to any execution. Post-audit checks confirmed zero modifications to Stage 5A artifacts.
2. **Mathematical Normalization Audit:** Investigated `offdiag_zscore_affinity` on synthetic matrices and cached multiscale affinity matrices across all 8 recordings. Proved that for a single encoder, the off-diagonal z-score sigmoid transformation is strictly monotonic and does not alter the top-$p$ neighbor connectivity in NeMo NME-SC spectral clustering.
3. **Gate Implementation Audit:** Discovered that the executable gate implementation uses linear mapping with clipping ($g_{\text{res}} = \text{clip}(0.5 + 2.0 \cdot \hat{\Delta}, 0.15, 0.85)$) rather than the sigmoid formulation referenced in prior documentation. Audited feature variances on clean LOTO turns ($N=2,771$) and verified that `resnet293_l2_norm` is measured prior to L2 unit-normalization.
4. **Controlled Downstream Diarization & WhisperX Attribution:** Evaluated 8 distinct conditions across all 8 recordings (64 complete diarization runs), comparing historical raw baselines (A, B) against new fair normalized single-encoder controls (H, I), normalized equal fusion (J), and normalized reliability-aware fusion (K), plus a sigmoid-gated sensitivity check (K_sig).
5. **Audited Verification Protocol:** Evaluated turn-level verification across 9,489 within-recording pairs under strict 8-fold LORO with frozen thresholds. Audited cross-fold score comparability, reporting macro 6-recording ROC-AUC alongside pooled OOF metrics.
6. **Error Complementarity & Clustered Bootstrap:** Computed 2,000 paired cluster bootstrap resamples across 12 condition comparisons, demonstrating that representation fusion delivers a statistically significant reduction in cpWER while downstream DER improvements remain constrained by VAD and overlapping speech.

All operations completed successfully with exit code 0.

---

## 2. Software & Hardware Environment Audit

### 2.1 Host Configuration
- **Operating System:** Linux 6.6.137+ x86_64
- **Host GPU:** NVIDIA GeForce RTX 4090 (24,564 MiB VRAM, Driver 580.82.07)
- **CUDA Version:** 12.8 (cu128)
- **Python Executable:** `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python`

### 2.2 Installed Package Versions
An audit of the active virtual environment resolved a version discrepancy reported in prior documentation:
- `python`: 3.10.20
- `torch`: 2.8.0+cu128
- `nemo_toolkit`: **2.7.3** *(Note: Prior Stage 5A report text noted 2.1.0, but the project environment is verified to run NeMo 2.7.3)*
- `pyannote.metrics`: **4.1**
- `pyannote.core`: **6.0.1**
- `scipy`: 1.15.3
- `scikit-learn`: 1.7.2
- `numpy`: 2.2.6
- `pandas`: 2.3.2

### 2.3 Hardcoded Paths and Environment Variables
- `REPO_ROOT`: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`
- `TORCH_HOME`: `/home/priyanka/.cache/torch`
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`: `1`
- All paths are dynamically anchored via `REPO_ROOT` and parameterized through `stage5a_1_config.json`. No secrets or Hugging Face tokens are required or exposed.

---

## 3. Execution Log Summary

### 3.1 Chronological Execution Steps
1. **Pre-Execution Baseline Hash Check:**  
   Executed `record_baseline_hashes.py`. Hashed 133 files (11,457,512 bytes) in `research_stages/stage5a/`. Baseline persisted to `research_stages/stage5a_1/file_integrity/baseline_stage5a_inventory.json`.
2. **Affinity Normalization Audit:**  
   Executed `audit_normalization.py` (runtime: 8.2s). Evaluated 5 synthetic matrix configurations (standard symmetric, constant off-diagonal, near-zero variance, negative values, asymmetric inputs) and cached multiscale affinity matrices across all 8 recordings. Persisted to `normalization_audit/normalization_audit_results.json`.
3. **Gate Implementation & Feature Audit:**  
   Executed `audit_gate_features.py` (runtime: 6.4s). Extracted 2,771 clean LOTO subsegments across eligible recordings. Computed distributions of `resnet293_l2_norm`, `dfn_noise_reduction_db`, and `vad_conf_mean`. Verified the linear-clipped gate equation against the sigmoid alternative. Persisted to `gate_audit/gate_audit_results.json`.
4. **Controlled Evaluation Suite:**  
   Executed `run_stage5a_1_controlled_eval.py` (runtime: 102.5s).
   - Downstream Diarization: Evaluated conditions A, B, C, H, I, J, K, K_sig across all 8 recordings ($8 \times 8 = 64$ RTTM files).
   - WhisperX Attribution: Generated 64 attributed transcript JSONs.
   - Verification: Evaluated 9,489 within-recording turn pairs across 8 LORO folds with frozen thresholds.
   - Bootstrap Analysis: Computed 2,000 paired cluster resamples for 12 comparisons across 8 primary metrics.
5. **Post-Execution Integrity Verification:**  
   Verified all 133 files in `research_stages/stage5a/`. Zero modifications detected.

---

## 4. Directory & Artifact Inventory

All generated artifacts are preserved in `research_stages/stage5a_1/` with the following structure:

```
research_stages/stage5a_1/
├── configs/
│   └── stage5a_1_config.json
├── scripts/
│   ├── audit_normalization.py
│   ├── audit_gate_features.py
│   └── run_stage5a_1_controlled_eval.py
├── normalization_audit/
│   └── normalization_audit_results.json
├── gate_audit/
│   └── gate_audit_results.json
├── downstream_metrics/
│   ├── stage5a_1_summary_diarization.csv
│   ├── stage5a_1_per_recording_diarization.csv
│   └── stage5a_1_master_results.json
├── verification/
│   ├── stage5a_1_verification_summary.csv
│   ├── stage5a_1_verification_per_recording.csv
│   └── stage5a_1_error_complementarity.csv
├── bootstrap/
│   └── stage5a_1_paired_bootstrap_differences.csv
├── predictions/
│   ├── rttms/          (64 RTTM files organized into 8 condition folders)
│   └── attributions/   (64 attributed JSON files organized into 8 condition folders)
├── file_integrity/
│   ├── baseline_stage5a_inventory.json
│   └── post_stage5a_1_integrity_check.json
├── logs/
│   └── stage5a_1_execution.log
└── reports/
    ├── STAGE5A_1_EXECUTION_REPORT.md
    ├── STAGE5A_1_NORMALIZATION_AUDIT.md
    ├── STAGE5A_1_GATE_IMPLEMENTATION_AUDIT.md
    ├── STAGE5A_1_VERIFICATION_RESULTS.md
    ├── STAGE5A_1_DOWNSTREAM_RESULTS.md
    ├── STAGE5A_1_FINAL_RECOMMENDATION.md
    └── STAGE5A_1_FILE_INTEGRITY.md
```

---

## 5. Compute Resources & Execution Constraints Adherence

- **Total Execution Runtime:** ~2 minutes across all scripts.
- **GPU Memory Peak:** 2.1 GB VRAM (RTX 4090).
- **Prohibited Operations Audit:**
  - Neural model training / fine-tuning: **None** (zero backprop passes).
  - Neural embedding extraction: **None** (reused Stage 3 cache).
  - VAD inference / threshold modification: **None** (reused Stage 1/2 VAD masks).
  - WhisperX ASR re-transcription: **None** (reused Stage 1/2 word alignments).
  - Audio enhancement: **None** (zero audio processing).
  - Stage 5B / 5C initiation: **None** (strict stop at Stage 5A.1 boundary).
