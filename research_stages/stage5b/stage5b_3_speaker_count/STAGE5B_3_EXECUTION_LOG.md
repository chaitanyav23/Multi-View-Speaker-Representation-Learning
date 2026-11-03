# Stage 5B.3 — Execution Log & Environment Audit

**Execution Date**: 2026-11-03  
**Stage**: Stage 5B.3 (Speaker-Count Estimation & Clustering Calibration)  
**Status**: COMPLETE, VERIFIED & FROZEN  
**Working Directory**: `research_stages/stage5b/stage5b_3_speaker_count/`  
**Execution Host**: `priyanka-Precision-7920-Tower` (Linux 6.8.0-49-generic, x86_64)  
**GPU Acceleration**: NVIDIA GeForce RTX 4090 (24,564 MiB VRAM, CUDA Driver 550.120, CUDA 12.4)  
**Python Runtime**: 3.10.12 (`research_stages/.venv_phase20/bin/python`)  
**PyTorch Version**: 2.6.0+cu124  

---

## 1. Pre-Execution Integrity Audit

Prior to beginning Stage 5B.3, the working tree and all 298 previously frozen artifacts were audited against `frozen_artifact_hashes_after.csv`:

```
======================================================================
PRE-EXECUTION AUDIT (2026-11-03T20:04:14Z):
Git Commit HEAD: c09dd6751f2b77a81672fb7cb0fb1c4b41a7ecb5
Frozen Commit: ddf089de986f9e7cf026aa507324f3fccaf0cd8f
Checked Artifacts: 298
Missing Artifacts: 0
Hash Mismatches: 0
Status: ALL 298 FROZEN ARTIFACTS VERIFIED BITWISE IDENTICAL!
======================================================================
```

---

## 2. Execution Chronology & Milestone Trace

### Step 1: Workspace Initialization
- Created dedicated isolated directories:
  - `research_stages/stage5b/stage5b_3_speaker_count/configs/`
  - `research_stages/stage5b/stage5b_3_speaker_count/scripts/`
  - `research_stages/stage5b/stage5b_3_speaker_count/predictions/`
  - `research_stages/stage5b/stage5b_3_speaker_count/metrics/`

### Step 2: R07 Oracle-Count Reproduction Diagnostic
- Executed diagnostic script on `R07_bwc_cheap_shot` using cached multiscale embeddings and Stage 5A.1 Condition J fused affinity matrix ($434 \times 434$).
- Discovered and resolved f-string backslash syntax incompatibility in Python 3.10.
- Successfully reproduced cluster sizes `[359, 75]` ($k=2$) and `[215, 157, 62]` ($k=3$).
- Traced ground truth durations and subsegments: Speaker 1 (107s, 87%), Speaker 2 (15s, 12.2%), Speaker 3 (1s, 0.8%).
- Established that forced $k=2$ produces DER = 47.27% (Confusion = 11.65%), and forced $k=3$ produces DER = 71.72% (Confusion = 36.10%), debunking the preliminary Stage 5B.0 note of 21.80% DER.
- Computed intra- vs inter-speaker affinities, proving Speaker 2 intra-speaker similarity (0.3422) is separated from Speaker 1 by only 0.0130.

### Step 3: Full 8-Recording Multi-Method Controlled Evaluation
- Script: `research_stages/stage5b/stage5b_3_speaker_count/scripts/run_stage5b_3_experiments.py`.
- Evaluated all 8 benchmark recordings across 6 methods:
  - M1: `nmesc_auto` (Automatic NME-SC baseline)
  - M2: `oracle_k` (Oracle count diagnostic upper bound)
  - M3: `floor_k2` (Conversational lower bound)
  - M4: `rel_eigengap` (Relative eigengap selection)
  - M5: `silhouette` (Silhouette-guided spectral selection)
  - M6: `ahc_oracle` (Agglomerative hierarchical clustering with oracle count)
- Generated:
  - [STAGE5B_3_ALL_RECORDINGS_RESULTS.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_3_speaker_count/STAGE5B_3_ALL_RECORDINGS_RESULTS.csv)
  - [stage5b_3_summary_comparison.csv](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_3_speaker_count/metrics/stage5b_3_summary_comparison.csv)
  - Full set of RTTM predictions for all methods and recordings under `predictions/`.

### Step 4: Parameter Sensitivity & Stability Analysis
- Script: `research_stages/stage5b/stage5b_3_speaker_count/scripts/run_stage5b_3_sensitivity.py`.
- Swept candidate counts $k \in [1, 10]$ for R07 and R06.
- Swept graph neighborhood parameters $p \in [2, 100]$ measuring density, node degrees, and connectivity.
- Evaluated $k$-means random trials $\{1, 5, 10, 20\}$ across multiple random seeds (`42`, `100`, `2024`).
- Generated:
  - [stage5b_3_sensitivity_results.json](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_3_speaker_count/metrics/stage5b_3_sensitivity_results.json)

---

## 3. Computational Resource Consumption

All experiments executed using pre-extracted representation caches, requiring zero neural network inference:
- **Total Execution Time**: Approximately 2.5 minutes (well below the 15-minute budget).
- **Peak GPU VRAM**: 1.84 GB (used exclusively for PyTorch tensor operations and eigensolvers).
- **Peak CPU RAM**: ~1.2 GB.
- **Disk Footprint**: ~4.2 MB total (JSON/CSV metrics and hypothesis RTTMs).

---

## 4. Post-Execution Integrity Verification

At the conclusion of Stage 5B.3, all frozen artifacts were re-audited against `frozen_artifact_hashes_after.csv` to ensure zero regression or tampering:

```
Checked: 298 frozen artifacts
Mismatches: 0
Missing: 0
Result: 100% BITWISE IDENTICAL
```
