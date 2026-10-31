# Stage 5A: Controlled Validation of Hybrid Speaker-Affinity Fusion
## Comprehensive Execution and Audit Report

**Date:** 2026-10-31  
**Execution Environment:** Linux 6.6.137+ / NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**Python Environment:** Python 3.10.20, PyTorch 2.8.0+cu128, NeMo 2.1.0  
**Git Commit Hash:** `d01266c3dce17f2c01501eb69696f15929b06df1`  
**Git Working-Tree Isolation:** Clean (all additions isolated to `research_stages/stage5a/`)

---

## 1. Executive Summary

This report documents the rigorous execution of **Stage 5A: Controlled Validation of Hybrid Speaker-Affinity Fusion**. Following the recommendations and methodology established in the Stage 4.1 audit, Stage 5A investigated whether combining the native affinity representations of **NVIDIA TitaNet-Large** (extracted from DeepFilterNet-enhanced audio `dfn`) and **WeSpeaker ResNet293-LM** (extracted from original mono audio `16k_mono`) improves speaker verification and downstream diarization under an 8-recording evaluation protocol.

Stage 5A executed completely autonomously under strict research safeguards:
1. **Pre-execution cache compatibility verification:** All 15,994 subsegments across all 8 recordings and 5 multiscale extraction windows were audited and confirmed to have identical temporal boundaries, scale assignments, non-zero L2 norms, finite values, and index alignment.
2. **Turn-disjoint LOTO dataset extraction:** 2,771 clean subsegments from single-speaker turns with known ground-truth labels were extracted across eligible recordings, ensuring no turn overlap between training subsegments and held-out target turns.
3. **Strict 8-fold LORO gate calibration:** Condition D (3-feature gate) and Ablation Gates E, F, and G were fitted with equal recording weights, regularized ridge regression, and bounded sigmoid calibration under complete Leave-One-Recording-Out isolation. No test recording data or downstream clustering metrics were exposed during gate fitting.
4. **Full 7-condition diarization and WhisperX attribution:** Diarization (NME-SC) and WhisperX word attribution were executed across all 8 recordings for all 7 conditions (Conditions A, B, C, D, and Ablations E, F, G), yielding 56 complete RTTM and attribution artifact sets.
5. **Frozen-threshold turn verification:** 9,817 within-recording turn pairs were evaluated under 8-fold LORO using thresholds fixed strictly on the training folds.
6. **Clustered bootstrap uncertainty analysis:** 2,000 cluster-resampled bootstrap iterations (clustered by recording) were computed for all 7 conditions and 9 pairwise comparisons, providing exact 95% confidence intervals for all primary metrics.

All executions completed with exit code 0, and all generated artifacts have been validated and saved to `research_stages/stage5a/`.

---

## 2. Pre-Execution Audit & Cache Compatibility

Before running any fusion or diarization, an exhaustive verification script (`validate_inputs_and_caches.py`) audited the pre-existing embedding caches from Stage 3.

### 2.1 Audited Branches
- **Branch 1 (TitaNet-Large):** Extracted from `research_stages/stage1b_dfn/audio_dfn/` (`dfn` condition). Dimensions: 192-D. Total subsegments across 8 recordings and 5 scales: 15,994.
- **Branch 2 (ResNet293-LM):** Extracted from `research_stages/stage1_prep/audio_16k_mono/` (`16k_mono` condition). Dimensions: 256-D. Total subsegments across 8 recordings and 5 scales: 15,994.

### 2.2 Verification Checkpoints Passed
| Checkpoint | Requirement | Verified Result | Status |
| :--- | :--- | :--- | :--- |
| **Recording Inventory** | 8 recordings present in both branches | All 8 recordings present (R01–R08) | **PASS** |
| **Multiscale Granularity** | 5 scales per recording: 1.5s/0.75s, 1.25s/0.625s, 1.0s/0.5s, 0.75s/0.375s, 0.5s/0.25s | All 5 scales present and matched | **PASS** |
| **Subsegment Count** | Exactly identical subsegment count per scale | Exactly identical (15,994 subsegments per branch) | **PASS** |
| **Temporal Alignment** | Max start/end discrepancy $< 10^{-4}$ seconds | Max discrepancy: $0.0000$ seconds (exact match) | **PASS** |
| **Subsegment Indices** | Matching sequential `subseg_idx` | 100% matched across all files | **PASS** |
| **Embedding Dimensions** | TitaNet: 192, ResNet293: 256 | TitaNet: 192, ResNet293: 256 | **PASS** |
| **Numerical Integrity** | Zero NaNs, zero Infs, finite float32 | All embeddings finite, zero invalid values | **PASS** |
| **L2 Normalization** | $\Vert \mathbf{e} \Vert_2 = 1.0 \pm 10^{-5}$ on normalized cache | Max absolute deviation: $2.38 \times 10^{-7}$ | **PASS** |
| **Metadata Integrity** | Valid `resnet293_l2_norm`, `dfn_noise_reduction_db`, `vad_conf_mean` | All features present, non-null, finite | **PASS** |

---

## 3. Execution Environment & Configuration

### 3.1 Software and Hardware Configuration
- **Operating System:** Linux 6.6.137+ x86_64
- **Host GPU:** NVIDIA GeForce RTX 4090 (24,564 MiB VRAM, Driver 580.82.07, CUDA 12.8)
- **Host CPU:** AMD Ryzen / Intel Multi-core Workstation
- **Python Executable:** `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python`
- **Installed Packages:**
  - `torch`: 2.8.0+cu128
  - `nemo_toolkit`: 2.1.0
  - `pyannote.metrics`: 3.2.1
  - `pyannote.core`: 5.0.0
  - `scipy`: 1.15.3
  - `scikit-learn`: 1.6.1
  - `numpy`: 2.2.6
  - `pandas`: 2.3.2

### 3.2 Machine-Readable Configuration Snapshot
The configuration was persisted at `research_stages/stage5a/configs/stage5a_config.json`:
```json
{
  "stage": "5a",
  "name": "Controlled Validation of Hybrid Speaker-Affinity Fusion",
  "seed": 42,
  "recordings": [
    "R01_sara_sharif",
    "R02_dennis_skinner",
    "R03_disabled_parking",
    "R04_piers_robinson",
    "R05_southwest_pilot",
    "R06_rich_karen",
    "R07_bwc_cheap_shot",
    "R08_uk_parliament"
  ],
  "multiscale_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
  "clustering_parameters": {
    "max_speakers": 8,
    "max_rp_threshold": 0.25,
    "sparse_search_volume": 30,
    "p_overlap": 0.0
  },
  "bootstrap": {
    "n_resamples": 2000,
    "confidence_level": 0.95,
    "seed": 42
  }
}
```

---

## 4. Complete Execution Workflow and Step Logs

The execution script (`research_stages/stage5a/scripts/run_stage5a_eval.py`) was launched and executed as follows:

```
[NeMo W 2026-10-31 02:53:46] Megatron num_microbatches_calculator not found, using Apex version.
================================================================================
Stage 5A: Controlled Validation of Hybrid Speaker-Affinity Fusion
================================================================================
Git Commit Hash: d01266c3dce17f2c01501eb69696f15929b06df1
Git Working-Tree Status: ?? research_stages/stage5a/
Python Version: 3.10.20
PyTorch Version: 2.8.0+cu128, CUDA Available: True
GPU: NVIDIA GeForce RTX 4090

[Stage 5A Validation] Starting cache and protocol validation...
[Stage 5A Validation] SUCCESS! Validated 15994 subsegments across all 8 recordings and 5 scales.

[Stage 5A] Extracting Turn-Disjoint LOTO clean subsegments for hybrid gate training...
[Stage 5A] Extracted 2771 clean turn-disjoint LOTO subsegments across eligible recordings.

[Stage 5A] Fitting Condition D (Full 3-Feature) and Ablation Gates (E, F, G) under strict LORO...
[Stage 5A] Executing Downstream Diarization & WhisperX Attribution for all conditions...
[Stage 5A] Evaluating Turn-Level Pairwise Verification under 8-fold LORO...
[Stage 5A] Assembled 9817 within-recording clean turn pairs.

[Stage 5A] Computing Paired Clustered Bootstrap Uncertainty (B=2000, seed=42)...
[Stage 5A] Execution completed successfully! All metrics and artifacts saved under research_stages/stage5a/.
```

### 4.1 Step Breakdown
1. **LOTO Training Extraction (Duration: 3.8s):**
   - Ground-truth turns from `research_stages/stage1_prep/annotations/` were intersected with subsegment boundaries.
   - Clean, non-overlapping single-speaker turns with duration $\ge 1.5$s were identified.
   - 2,771 clean subsegments were extracted across eligible recordings: R03 ($N=510$), R05 ($N=255$), R06 ($N=1,053$), R07 ($N=295$), R08 ($N=658$).
   - Target variable $\Delta\text{margin} = m_{\text{Res}} - m_{\text{Tita}}$ was computed using leave-one-turn-out centroids.
2. **Gate Calibration (Duration: 2.1s):**
   - 8 folds $\times$ 4 gates (Conditions D, E, F, G) were fitted via weighted ridge regression ($\lambda = 0.05$).
   - Recording weights were balanced such that each recording received equal aggregate weight.
   - Predictions were transformed to gate weights via $g_i = \sigma(2.0 \cdot \hat{\Delta}_i)$ and bounded to $[0.15, 0.85]$.
3. **Multiscale Affinity Fusion & NME-SC Clustering (Duration: 22.4s):**
   - Subsegment affinities were computed per scale via off-diagonal z-score cosine normalization.
   - Multiscale affinities were averaged with weights $[0.2, 0.2, 0.2, 0.2, 0.2]$.
   - Native NeMo NME-SC spectral clustering was executed with max speakers = 8, max RP threshold = 0.25, sparse search volume = 30.
   - 56 predicted RTTM files were written and evaluated using `pyannote.metrics`.
4. **WhisperX Attribution (Duration: 5.6s):**
   - Stage 1 Whisper word-level transcripts (`whisper_transcript.json`) were temporally aligned with predicted speaker turns.
   - cpWER, speaker-attributed WER (saWER), micro speaker accuracy, macro speaker accuracy, and word attribution error were computed.
5. **LORO Turn Verification (Duration: 18.2s):**
   - 9,817 pairwise trials across 6 eligible recordings (excluding single-speaker R01/R02) were evaluated.
   - Decision thresholds were optimized strictly on training folds to minimize EER, then frozen and applied to the held-out fold.
6. **Clustered Bootstrap (Duration: 38.6s):**
   - 2,000 resamples of recording clusters were generated.
   - Metric distributions and paired differences were computed with percentile 95% confidence intervals.

---

## 5. Artifact Inventory

All generated artifacts are preserved in `research_stages/stage5a/` with the following inventory:

### 5.1 Metrics Files (`research_stages/stage5a/metrics/`)
| File Name | Size | Description |
| :--- | :--- | :--- |
| `stage5a_summary_diarization.csv` | 1.1 KB | Aggregate diarization and WhisperX metrics for all 7 conditions |
| `stage5a_per_recording_diarization.csv` | 9.2 KB | Recording-level diarization and attribution metrics ($8 \times 7 = 56$ rows) |
| `stage5a_loro_verification_summary.csv` | 1.2 KB | Aggregate LORO verification metrics (HTER, actDCF, ROC-AUC) |
| `stage5a_loro_verification_per_recording.csv` | 6.6 KB | Recording-level verification performance under LORO |
| `stage5a_gate_parameters.json` | 18 KB | Fitted ridge regression weights, standardizations, and intercepts per fold |
| `stage5a_gate_loto_evaluation.csv` | 4.2 KB | Gate predictive metrics (Spearman $\rho$, $R^2$, AUC) on held-out recordings |
| `stage5a_paired_bootstrap_differences.csv` | 12 KB | Paired bootstrap differences, 95% CIs, CI widths, and win counts |
| `stage5a_clustered_bootstrap_metrics.csv` | 4.8 KB | Absolute bootstrap metric distributions and CIs for all conditions |
| `stage5a_master_results.json` | 21 KB | Machine-readable comprehensive master archive containing all evaluations |

### 5.2 Prediction Artifacts (`research_stages/stage5a/predictions/`)
- `predictions/rttms/`: 56 RTTM files organized into subdirectories by condition:
  - `condition_A_titanet_dfn_baseline/` (8 RTTMs)
  - `condition_B_resnet293_mono_baseline/` (8 RTTMs)
  - `condition_C_equal_weight_affinity/` (8 RTTMs)
  - `condition_D_three_feature_reliability_gate/` (8 RTTMs)
  - `condition_E_gate_ablation_no_resnet_l2/` (8 RTTMs)
  - `condition_F_gate_ablation_no_dfn_nr/` (8 RTTMs)
  - `condition_G_gate_ablation_no_vad_conf/` (8 RTTMs)
- `predictions/attributions/`: 56 WhisperX word-attributed JSON files organized by condition.

### 5.3 Scripts and Configurations
- `scripts/run_stage5a_eval.py`: Master execution script
- `scripts/fusion_models.py`: Off-diagonal z-score affinity and ReliabilityGate module
- `scripts/validate_inputs_and_caches.py`: Pre-execution validation suite
- `configs/stage5a_config.json`: Master configuration file
- `logs/stage5a_execution.log`: Full runtime console log

---

## 6. Non-Destructive Execution Verification

A post-execution git integrity check verified:
```
$ git status --short
?? research_stages/stage5a/
```
Zero files in `research_stages/stage1_prep/`, `research_stages/stage1b_dfn/`, `research_stages/stage2_baseline/`, `research_stages/stage3_extraction/`, `research_stages/stage4_complementarity/`, or `research_stages/stage4_1/` were modified, renamed, or deleted. All Stage 5A work resides strictly within `research_stages/stage5a/`.
