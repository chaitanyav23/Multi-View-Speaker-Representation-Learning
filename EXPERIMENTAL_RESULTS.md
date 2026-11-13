# Experimental Results: Multi-View Speaker Representation Learning & ConvoFuse

This document presents the complete, provenance-verified experimental results of the **Multi-View Speaker Representation Learning** research program and the final **ConvoFuse** CLI deliverable across the 8-recording Body-Worn Camera (BWC) and multi-talker conversational benchmark (`R01`–`R08`, `1,893.38s` total duration, `4,940` aligned ASR words).

> [!IMPORTANT]
> **Provenance & Evaluation Separation Policy**:
> - **Part A (Multi-View Speaker Representation & Diarization Comparison)** reports the frozen, audited metrics from Stages 2, 3, 4.1, 5A.1.2, and 5B.1–5B.4 across `R01`–`R08`.
> - **Part B (ConvoFuse Generative Speaker Refinement Comparison)** reports the post-inference evaluation of the final CLI run (`convofuse_outputs/final_release/CORPUS_SUMMARY.json`) using `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (`Q4_K_M`) with Locality-Gated Transcript-Preserving Speaker Transfer (`LG-TPST`). Ground truth is strictly isolated from all inference modules and used solely inside `convofuse/evaluator.py` after transcript generation completes.
> - **Part C (Historical Phase20 Standalone Appendix)** reports the historical single-recording (`bwc.wav`) metrics from [`phase20_standalone/results_to_be_published.md`](phase20_standalone/results_to_be_published.md) with explicit dataset and protocol caveats.

---

## Part A — Multi-View Speaker Representation & Fusion Comparison (`R01`–`R08`)

**Authoritative Source Artifacts**:
- [`research_stages/stage2_baseline/reports/STAGE2_BASELINE_REPORT.md`](research_stages/stage2_baseline/reports/STAGE2_BASELINE_REPORT.md)
- [`research_stages/stage3_dual_encoder/reports/STAGE3_EXTRACTION_REPORT.md`](research_stages/stage3_dual_encoder/reports/STAGE3_EXTRACTION_REPORT.md)
- [`research_stages/stage4_1/STAGE4_1_FINAL_AUDIT_REPORT.md`](research_stages/stage4_1/STAGE4_1_FINAL_AUDIT_REPORT.md)
- [`research_stages/stage5a_1_final_audit/STAGE5A_1_CORRECTED_RESULTS.md`](research_stages/stage5a_1_final_audit/STAGE5A_1_CORRECTED_RESULTS.md)
- [`research_stages/stage5a_1_2_freeze_closure/STAGE5A_1_2_FINAL_FREEZE_RECORD.md`](research_stages/stage5a_1_2_freeze_closure/STAGE5A_1_2_FINAL_FREEZE_RECORD.md)
- [`research_stages/stage5b/STAGE5B_CONSOLIDATED_RESULTS_REPORT.md`](research_stages/stage5b/STAGE5B_CONSOLIDATED_RESULTS_REPORT.md)
- [`research_stages/stage5b/stage5b_4_fresh_diarization/r01_pilot/R01_POST_RUN_AUDIT_ADDENDUM.md`](research_stages/stage5b/stage5b_4_fresh_diarization/r01_pilot/R01_POST_RUN_AUDIT_ADDENDUM.md)

### A.1 Benchmark Corpus & Multiscale Extraction Summary

Across `R01`–`R08` (`1,893.38s` / `31.56 min`), Stage 3 extracted `15,994` multiscale subsegments across 5 temporal scales (`1.5s`, `1.2s`, `1.0s`, `0.8s`, `0.5s` with 50% overlap) for both `nvidia/speakerverification_en_titanet_large` (`192-D`) and `WeSpeaker ResNet293-LM` (`256-D`), with `0` NaN/Inf embeddings and `100%` L2-norm conformance.

| Recording ID | Domain / Acoustic Profile | Duration (s) | Gold Speakers | Overlap Ratio (%) | Multiscale Subsegments (5 Scales) | Clean Verification Turns ($\ge 1.0\text{s}$) | Within-Recording Verification Pairs (Pos / Neg) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R01_sara_sharif`** | Domestic BWC arrest (reverberant indoor) | 100.96 | 3 | 1.98% | 697 | 16 | 120 (48 / 72) |
| **`R02_dennis_skinner`** | Parliamentary BWC / crowd heckling | 176.26 | 5 | 9.57% | 995 | 26 | 325 (112 / 213) |
| **`R03_disabled_parking`** | Roadside BWC dispute (traffic/wind noise) | 206.22 | 3 | 9.10% | 1,781 | 44 | 946 (467 / 479) |
| **`R04_piers_robinson`** | Multi-person interview / cross-talk | 204.54 | 4 | 9.65% | 1,797 | 26 | 325 (137 / 188) |
| **`R05_southwest_pilot`** | Cockpit / cabin PA communication | 162.45 | 3 | 5.21% | 1,105 | — (1 clean spk) | — |
| **`R06_rich_karen`** | High-overlap street confrontation | 652.18 | 9 | 12.24% | 5,914 | 122 | 7,381 (1,622 / 5,759) |
| **`R07_bwc_cheap_shot`** | Nighttime street BWC (`bwc.wav`, siren/yelling) | 175.43 | 5 | 4.82% | 1,599 | 2 | 1 (0 / 1, single-class) |
| **`R08_uk_parliament`** | Formal parliamentary debate | 215.34 | 2 | 1.12% | 2,106 | 28 | 378 (366 / 12) |
| **Total / Macro** | **8 BWC & Multi-Talker Recordings** | **1,893.38** | **2–9 (mean 4.25)** | **6.71%** | **15,994** | **250 (6-Rec)** | **9,489 (2,766 / 6,723)** |

---

### A.2 Pairwise Verification Error Complementarity (`TitaNet-DFN` vs. `ResNet293-Mono`)

Evaluated across all `9,489` clean within-recording turn pairs (`250` non-overlapping turns $\ge 1.0\text{s}$) under Leave-One-Recording-Out (LORO) threshold calibration (`STAGE5A_1_2_FINAL_FREEZE_RECORD.md`):

| Pairwise Outcome Category | Pair Count (`N = 9,489`) | Percentage of All Pairs | Percentage of Unique Error Pairs (`N = 3,579`) |
| :--- | :---: | :---: | :---: |
| **Both Encoders Correct** | 5,910 | 62.28% | — |
| **Both Encoders Wrong (Shared Errors)** | 1,502 | 15.83% | 41.97% (`Error Jaccard = 0.4197`) |
| **TitaNet-Large (DFN) Wrong Only** | 1,035 | 10.91% | 28.92% |
| **WeSpeaker ResNet293-LM (Mono) Wrong Only** | 1,042 | 10.98% | 29.11% |
| **Total Disjoint (Complementary) Error Pairs** | **2,077** | **21.89%** | **58.03%** |

- **Empirical Conclusion**: Nearly three-fifths (**`58.03%`**) of all verification errors are committed by only one of the two encoders, confirming strong architectural and acoustic-view complementarity between depthwise-separable 1D convolutions on DFN-enhanced speech (`TitaNet-Large`) and 2D ResNet convolutions on unenhanced mono speech (`WeSpeaker ResNet293-LM`).

---

### A.3 Asymmetric Preprocessing Sensitivity (`Stage 4.1` Audit)

Paired 10,000-resample cluster bootstrap evaluation (`STAGE4_1_FINAL_AUDIT_REPORT.md`) comparing `DeepFilterNet3` (`dfn.wav`) against raw `16 kHz` mono (`16k_mono.wav`) for each backbone under identical VAD and NME-SC clustering:

| Backbone & Preprocessing Condition | 8-Rec Macro DER (`c=0, ovl`) (%) $\downarrow$ | 8-Rec Macro DER (`c=0.25, no-ovl`) (%) $\downarrow$ | 8-Rec Macro JER (%) $\downarrow$ | Spk Count MAE $\downarrow$ | 8-Rec Macro cpWER (%) $\downarrow$ | 8-Rec Macro SA-WER (%) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TitaNet-Large` (192-D) on `dfn.wav` (Condition H)** | 46.22% | 39.72% | 62.14% | **1.500** | 38.07% | 46.58% |
| **`TitaNet-Large` (192-D) on `16k_mono.wav`** | 46.52% | 39.88% | 64.96% | 1.625 (collapses `R01`) | 38.67% | 48.29% |
| **`WeSpeaker ResNet293-LM` (256-D) on `dfn.wav`** | 47.60% | 40.82% | 64.61% | 1.625 | 39.96% | 48.32% |
| **`WeSpeaker ResNet293-LM` (256-D) on `16k_mono.wav` (Condition I)** | **45.98%** | **38.94%** | **62.07%** | 1.625 | **36.44%** | **46.58%** |

- **Statistical Significance (`ResNet293-LM` `16k_mono` vs. `dfn`)**: `WeSpeaker ResNet293-LM` on raw `16k_mono.wav` reduces Macro DER (`c=0, ovl`) by **`-1.63 pp`** (`95% CI [-2.66%, -0.68%]`, winning **7 of 8 recordings**) and reduces Macro `cpWER` by **`-3.52 pp`** (`36.44%` vs `39.96%`). Conversely, `TitaNet-Large` requires `dfn.wav` on `R01_sara_sharif` to prevent single-speaker eigengap collapse (`DER = 38.06%` on `dfn` vs `44.96%` on `16k_mono`).

---

### A.4 Turn-Level Speaker Verification Benchmark (`Stage 5A.1.2` Freeze)

Leave-One-Recording-Out (LORO) verification across the 6-recording valid subset (`R01`, `R02`, `R03`, `R04`, `R06`, `R08`; `250` turns, `9,488` pairs) and the 5-recording strict subset (excluding highly imbalanced `R08`; `222` turns, `9,110` pairs):

| Condition | Description | 6-Rec Macro ROC-AUC $\uparrow$ | 6-Rec Macro HTER (%) $\downarrow$ | 5-Rec Strict Macro ROC-AUC $\uparrow$ | 5-Rec Strict Macro HTER (%) $\downarrow$ | Pooled OOF ROC-AUC (6-Rec / 5-Rec) $\uparrow$ | Pooled OOF HTER (6-Rec / 5-Rec) (%) $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond A** | `TitaNet-Large` (192-D on `dfn`) Alone | 0.8238 | 33.80% | 0.8172 | 34.13% | 0.7681 / 0.7710 | 29.31% / 29.27% |
| **Cond B** | `WeSpeaker ResNet293-LM` (256-D on `16k_mono`) Alone | 0.8230 | 32.40% | 0.8304 | 33.16% | 0.7800 / 0.7836 | 28.90% / 29.50% |
| **Cond C** | Unnormalized Equal Fusion (`0.5 S_T + 0.5 S_W`) | 0.8453 | 32.89% | 0.8430 | 33.01% | 0.7950 / 0.7995 | 27.62% / 28.02% |
| **Cond D** | Unnormalized Reliability Gate (`w_t in [0.35, 0.65]`) | **0.8458** | 32.16% | 0.8435 | 32.92% | 0.7976 / 0.8020 | 27.71% / 28.08% |
| **Cond J** | **Z-Score + Sigmoid Normalized Equal Fusion (`0.5 / 0.5`)** | 0.8450 | 32.69% | 0.8426 | 32.79% | 0.7955 / 0.8001 | 27.82% / 28.14% |
| **Cond K** | **Z-Score + Sigmoid Normalized LORO Reliability Gate** | **0.8458** | **31.90%** | **0.8436** | **32.56%** | **0.7980 / 0.8025** | **27.62% / 27.88%** |

- **Why Off-Diagonal Z-Score Normalization is Mandatory for Clustering**: Although unnormalized fusion (`Cond C/D`) improves pairwise ROC-AUC, raw cosine affinities have incompatible off-diagonal distributions (`TitaNet` $\mu=0.0887, \sigma=0.1610$ vs `ResNet293` $\mu=0.2341, \sigma=0.1492$). Without off-diagonal Z-score normalization, NME-SC collapses `R06_rich_karen` from 9 speakers to 2 (`DER = 62.23%`). Off-diagonal Z-score normalization (`Cond J` and `Cond K`) preserves the full ROC-AUC gain (`+0.0212` to `+0.0228` over single encoders) while stabilizing spectral eigengaps (`R06` recovers 8 speakers under `Cond J`, `DER = 47.41%`).

---

### A.5 Downstream Diarization & Attribution Macro Comparison (`R01`–`R08`, `Stage 5A.1.2` Freeze)

All Stage 3–5A.1 conditions share identical legacy MarbleNet VAD intervals and identical WhisperX `large-v3` ASR words (`Macro ASR WER = 23.22%`), isolating the exact downstream effect of the speaker representation and affinity fusion strategy:

| System / Condition | Macro DER (`c=0, ovl`) (%) $\downarrow$ | Dur-Wtd DER (`c=0, ovl`) (%) $\downarrow$ | Miss / FA / Conf (`c=0, ovl`) (%) $\downarrow$ | Macro DER (`c=0.25, no-ovl`) (%) $\downarrow$ | Macro JER (%) $\downarrow$ | Spk Count MAE (Bias) $\downarrow$ | Macro cpWER (%) $\downarrow$ | Macro SA-WER (%) $\downarrow$ | Macro-SACC / Micro-SACC (%) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 2 Baseline** (Legacy NeMo MSDD `uniform_v1`) | 47.48% | 51.85% | 23.30 / 16.07 / 8.11 | 42.52% | **61.88%** | 1.500 (-1.500) | 39.34% | 47.97% | 49.27% / 83.33% |
| **Cond A / H** (`TitaNet-Large` 192-D on `dfn` alone) | 46.22% | **51.44%** | 24.41 / 13.33 / 8.48 | 39.72% | 62.14% | 1.500 (-1.500) | 38.07% | 46.58% | 49.42% / 84.27% |
| **Cond B / I** (`WeSpeaker ResNet293-LM` 256-D on `16k_mono` alone) | 45.98% | 52.22% | 24.41 / 13.33 / 8.24 | **38.94%** | 62.07% | 1.625 (-1.625) | **36.44%** | 46.58% | 48.94% / **84.29%** |
| **Cond J** (**Normalized Equal Fusion `0.5 / 0.5`**) | 46.17% | 52.18% | 24.41 / 13.33 / 8.43 | 39.65% | 62.20% | **1.250 (-1.250)** | 37.05% | **46.45%** | **50.25%** / 84.01% |
| **Cond K** (**Normalized LORO Reliability Gate**) | **45.92%** | 51.49% | 24.41 / 13.33 / **8.18** | 39.39% | 62.63% | 1.625 (-1.625) | 37.74% | 46.65% | 48.99% / 84.10% |

#### Paired 10,000-Resample Cluster Bootstrap 95% Confidence Intervals (`STAGE5A_1_CORRECTED_RESULTS.md`)

| Comparison Pair | Metric | Point $\Delta$ | 95% Bootstrap CI | Statistical Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Cond J (Equal Fusion) vs. Cond H (`TitaNet-DFN`)** | Macro `cpWER` | **-1.02 pp** | **[-2.07%, -0.10%]** | **Statistically significant improvement ($p < 0.05$, CI excludes 0)** |
| **Cond J (Equal Fusion) vs. Cond H (`TitaNet-DFN`)** | Macro `DER (c=0, ovl)` | -0.05 pp | [-0.36%, +0.29%] | Equivalent acoustic DER with lower `Spk Count MAE` (`1.250` vs `1.500`) |
| **Cond K (Reliability Gate) vs. Cond H (`TitaNet-DFN`)** | Macro `DER (c=0, ovl)` | **-0.30 pp** | [-0.90%, +0.16%] | Lowest corpus-wide Macro DER (`45.92%`) and speaker confusion (`8.18%`) |
| **Cond K (Reliability Gate) vs. Cond I (`ResNet293-Mono`)** | Macro `DER (c=0, ovl)` | **-0.06 pp** | [-1.72%, +1.47%] | Matches/beats `ResNet293-Mono` DER while improving `R06` by `-3.65 pp` |

---

### A.6 Per-Recording Downstream Breakdown (`R01`–`R08`, `Stage 5A.1.2` Freeze)

| Recording ID (Ref Spk) | Metric | Stage 2 Baseline (Legacy MSDD) | Cond A / H (`TitaNet-DFN`) | Cond B / I (`ResNet293-Mono`) | Cond J (Normalized Equal Fusion) | Cond K (Normalized LORO Gate) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`R01_sara_sharif`** (3 Spk) | Pred Spk / DER (`c=0`) / DER (`c=0.25`) / JER / cpWER | 2 / 40.38% / 36.26% / 62.74% / 55.64% | 3 / **38.06%** / 33.76% / 52.89% / 51.13% | 2 / 39.24% / **33.49%** / 61.55% / **42.11%** | 3 / **38.06%** / 33.76% / **52.82%** / 45.86% | 3 / **38.06%** / 33.76% / **52.82%** / 45.86% |
| **`R02_dennis_skinner`** (5 Spk) | Pred Spk / DER (`c=0`) / DER (`c=0.25`) / JER / cpWER | 3 / 59.10% / 56.11% / 66.20% / 44.30% | 3 / 57.15% / 51.85% / **63.70%** / 43.32% | 3 / **56.83%** / **51.40%** / 64.08% / 43.32% | 3 / 57.09% / 51.77% / 64.02% / **43.00%** | 3 / 57.09% / 51.77% / 64.02% / **43.00%** |
| **`R03_disabled_parking`** (3 Spk) | Pred Spk / DER (`c=0`) / DER (`c=0.25`) / JER / cpWER | 3 / 28.04% / 22.92% / 23.09% / 21.37% | 3 / 25.00% / 17.89% / 20.17% / 19.66% | 3 / **24.64%** / **17.44%** / **19.57%** / **19.19%** | 3 / 24.91% / 17.78% / 20.04% / 19.50% | 3 / 24.91% / 17.78% / 20.04% / 19.50% |
| **`R04_piers_robinson`** (4 Spk) | Pred Spk / DER (`c=0`) / DER (`c=0.25`) / JER / cpWER | 4 / 37.81% / 32.89% / **53.71%** / 36.28% | 3 / **36.22%** / **30.07%** / 60.53% / **35.41%** | 3 / 38.67% / 33.14% / 62.66% / 39.04% | 3 / 37.13% / 31.22% / 61.34% / 36.57% | 3 / 36.88% / 30.90% / 61.07% / 36.14% |
| **`R05_southwest_pilot`** (3 Spk) | Pred Spk / DER (`c=0`) / DER (`c=0.25`) / JER / cpWER | 3 / 44.69% / 36.71% / **43.02%** / 31.99% | 3 / 42.90% / 33.12% / 43.39% / 31.45% | 3 / **42.68%** / **32.83%** / 43.13% / **31.18%** | 3 / **42.68%** / **32.83%** / 43.13% / **31.18%** | 3 / **42.68%** / **32.83%** / 43.13% / **31.18%** |
| **`R06_rich_karen`** (9 Spk) | Pred Spk / DER (`c=0`) / DER (`c=0.25`) / JER / cpWER | 6 / 47.41% / 41.32% / 81.14% / 48.05% | 6 / 47.18% / 40.32% / 81.91% / 49.91% | 6 / 48.62% / 42.10% / 71.15% / **43.46%** | **8** / 47.41% / 40.61% / **70.21%** / 49.74% | 5 / **44.97%** / **37.55%** / 83.24% / 55.67% |
| **`R07_bwc_cheap_shot`** (5 Spk) | Pred Spk / DER (`c=0`) / DER (`c=0.25`) / JER / cpWER | 1 / 85.40% / 84.33% / 90.47% / 66.74% | 1 / 85.94% / 84.88% / 90.75% / 63.39% | 1 / **79.77%** / **77.45%** / **89.49%** / 62.97% | 1 / 84.74% / 83.44% / 90.51% / **60.25%** | 1 / 85.42% / 84.26% / 90.64% / 60.25% |
| **`R08_uk_parliament`** (2 Spk) | Pred Spk / DER (`c=0`) / DER (`c=0.25`) / JER / cpWER | 2 / 37.01% / 29.62% / 74.67% / **10.32%** | 2 / **37.33%** / 25.87% / **83.78%** / **10.32%** | 2 / 37.41% / **23.67%** / 84.93% / **10.32%** | 2 / 37.33% / 25.79% / 85.53% / **10.32%** | 2 / 37.33% / 25.79% / 86.08% / **10.32%** |

---

### A.7 Stage 5B Upstream Bottleneck Decomposition & `R01` Fresh Diarization Pilot (`E20` vs. `E21`)

Stage 5B decomposed the remaining diarization error into three upstream components (`STAGE5B_CONSOLIDATED_RESULTS_REPORT.md`) and executed a fresh end-to-end diarization pilot on `R01_sara_sharif` (`R01_POST_RUN_AUDIT_ADDENDUM.md`):

1. **Stage 5B.1 (Consensus VAD `C62`)**: 2-of-3 frame voting across Silero VAD (`16k_mono` + `dfn`) and Pyannote 3.1 segmentation reduces corpus VAD Missed Speech from `24.41%` to `13.84%` (`-10.57 pp`) and False Alarm from `13.33%` to `8.12%` (`-5.21 pp`).
2. **Stage 5B.2 (Overlap Posteriors `V4.2.1`)**: Tuned Pyannote 3.1 overlap posteriors reduce oracle-upper-bound DER by `-0.98 pp`, whereas post-hoc RTTM second-speaker attribution on noisy clustering slightly increases speaker confusion (`+0.26 pp` macro DER).
3. **Stage 5B.3 (Speaker-Count Calibration)**: Sweeping `rp_threshold` (`0.15` vs `0.11`) reduces speaker-count MAE from `1.250` to `0.875` (`bias = -0.375`), confirming that `R07_bwc_cheap_shot` single-speaker collapse is caused by extreme background siren/yelling noise rather than fusion scale mismatch.
4. **Stage 5B.4 (`R01_sara_sharif` Fresh Diarization Pilot)**:

| `R01_sara_sharif` Pipeline Condition | VAD Source | Embedding & Fusion Backend | Overlap Mode | DER (`c=0.0s, ovl`) (%) $\downarrow$ | Miss / FA / Conf (`c=0.0s`) (%) $\downarrow$ | DER (`c=0.25s, no-ovl`) (%) $\downarrow$ | JER (%) $\downarrow$ | TS-WDER / Lex-WDER (%) $\downarrow$ | cpWER (%) $\downarrow$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 5A.1 Condition J** | Legacy MarbleNet (`32` segs) | `0.5 TitaNet-DFN + 0.5 ResNet-Mono` | `M0` (1-Spk) | 38.06% | 26.25 / 9.29 / 2.52 | 33.76% | **52.82%** | — | 45.86% |
| **Stage 5B.4 `E20` (ConvoFuse R01 Default)** | **Stage 5B.1 Consensus `C62` (`19` segs)** | `0.5 TitaNet-DFN + 0.5 ResNet-Mono` | `M0` (1-Spk) | **32.91%** | **18.13 / 1.35** / 13.42 | **31.77%** | 62.18% | **20.08% / 19.38%** | 52.30% |
| **Stage 5B.4 `E21` (Overlap Pilot)** | **Stage 5B.1 Consensus `C62` (`19` segs)** | `0.5 TitaNet-DFN + 0.5 ResNet-Mono` | `M4-DFN` (2-Spk) | **32.58%** | **17.81 / 1.35** / 13.42 | **31.77%** | 61.99% | **20.08% / 19.38%** | 52.30% |

- **Why `E20` (`M0`) is the Canonical Word-Attribution Input**: While `E21` (`M4-DFN`) reduces acoustic `DER (c=0.0s)` by `-0.33 pp` (`32.91% -> 32.58%`) on `R01` by adding two short secondary-speaker RTTM intervals (`0.25s`), `TS-WDER` (`20.08%`), `Lex-WDER` (`19.38%`), and `cpWER` (`52.30%`) are **100% identical** between `E20` and `E21` because single-stream WhisperX ASR emits one lexical sequence per timestamp interval. ConvoFuse therefore uses `M0` (`E20` on `R01`, `Condition J` on `R02`–`R08`) for word attribution while surfacing explicit overlap metadata (`is_overlapping_speech`, `frame_overlap_prob_max`) on every word and turn.

---

## Part B — ConvoFuse Generative Speaker Refinement Comparison (`R01`–`R08`)

**Authoritative Source Artifacts**:
- [`convofuse_outputs/final_release/CORPUS_SUMMARY.json`](convofuse_outputs/final_release/CORPUS_SUMMARY.json)
- [`convofuse_outputs/final_release/CORPUS_INDEX.json`](convofuse_outputs/final_release/CORPUS_INDEX.json)
- [`convofuse_outputs/final_release/RELEASE_MANIFEST.json`](convofuse_outputs/final_release/RELEASE_MANIFEST.json)
- Per-recording `<rec_id>__evaluation_report.json` in [`convofuse_outputs/final_release/`](convofuse_outputs/final_release/)

### B.1 Corpus-Wide Macro Summary (`DiarizationLM-Gemma-4-E4B-v1` + `LG-TPST`)

Executed via `PYTHONPATH=. ./research_stages/.venv_phase20/bin/python -m convofuse.cli --all-recordings --condition auto --output-dir convofuse_outputs/final_release` (`total_corpus_sec = 92.18s`):

| Metric | Acoustic Baseline (`E20` / `Cond J`) | ConvoFuse Refined (`LG-TPST`) | Net $\Delta$ | Recording-Level Outcome Summary (`8` Recordings) |
| :--- | :---: | :---: | :---: | :--- |
| **Macro `cpWER` (%)** $\downarrow$ | 37.23% | **36.98%** | **-0.25 pp** | **5 Improved** (`R02`, `R04`, `R05`, `R06`, `R08`), **2 Unchanged** (`R01`, `R07`), **1 Degraded** (`R03`) |
| **Macro Lexical WDER (`Lex-WDER`) (%)** $\downarrow$ | 13.29% | **13.22%** | **-0.07 pp** | **3 Improved** (`R02`, `R04`, `R05`), **3 Unchanged** (`R01`, `R07`, `R08`), **2 Degraded** (`R03`, `R06`) |
| **Macro Timestamp WDER (`TS-WDER`) (%)** $\downarrow$ | 16.11% | 16.36% | +0.25 pp | **0 Improved**, **6 Unchanged** (`R01`, `R02`, `R04`, `R05`, `R07`, `R08`), **2 Degraded** (`R03`, `R06`) |
| **Total Word Decisions (`4,940` words)** | `4,940` baseline | `85` accepted / `48` rejected / `69` abstained / `4,738` unchanged | — | `19` beneficial / `40` harmful / `26` neutral (under strict timestamp-collar WDER) |
| **Integrity & Invariance Verification** | 100% | **100%** | **0 Violations** | `all_word_sequences_preserved = true`, `all_timestamps_preserved = true`, `all_unk_invariance_verified = true`, `all_frozen_inputs_sha256_verified = true` |

---

### B.2 Per-Recording Generative Refinement Results (`R01`–`R08`)

| Recording ID | Diarization Input Condition | Words (`UNK`) | Turns (Base $\rightarrow$ Ref) | Overlap Turns (Words) | Spk Inv | TS-WDER Base $\rightarrow$ Ref ($\Delta$ pp) | Lex-WDER Base $\rightarrow$ Ref ($\Delta$ pp) | cpWER Base $\rightarrow$ Ref ($\Delta$ pp) | Acc / Rej / Abs / Unch | Beneficial / Harmful / Neutral (Prec) | Runtime (LLM / Total s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R01_sara_sharif`** | `Stage5B.4_E20` | 266 (2) | 11 $\rightarrow$ 13 | 0 (0) | 3 | 20.08% $\rightarrow$ 20.08% (`0.00`) | 19.38% $\rightarrow$ 19.38% (`0.00`) | 52.30% $\rightarrow$ 52.30% (`0.00`) | 1 / 2 / 0 / 263 | 0 / 0 / 1 (`0.000`) | 3.50 / 3.80 |
| **`R02_dennis_skinner`** | `Stage5A.1_Cond_J` | 307 (4) | 27 $\rightarrow$ 26 | 6 (7) | 3 | 37.25% $\rightarrow$ 37.25% (`0.00`) | 18.79% $\rightarrow$ **18.46%** (`-0.33`) | 43.00% $\rightarrow$ **42.67%** (`-0.33`) | 1 / 8 / 1 / 297 | 0 / 1 / 0 (`0.000`) | 4.64 / 5.02 |
| **`R03_disabled_parking`** | `Stage5A.1_Cond_J` | 641 (4) | 59 $\rightarrow$ 59 | 17 (18) | 3 | 8.55% $\rightarrow$ 9.25% (`+0.70`) | 4.83% $\rightarrow$ 5.48% (`+0.65`) | 19.60% $\rightarrow$ 20.52% (`+0.92`) | 11 / 6 / 3 / 621 | 2 / 6 / 3 (`0.182`) | 9.80 / 11.37 |
| **`R04_piers_robinson`** | `Stage5A.1_Cond_J` | 689 (0) | 59 $\rightarrow$ 61 | 36 (44) | 3 | 26.59% $\rightarrow$ 26.59% (`0.00`) | 11.69% $\rightarrow$ **10.88%** (`-0.81`) | 38.57% $\rightarrow$ **37.75%** (`-0.82`) | 22 / 17 / 20 / 630 | 10 / 10 / 2 (`0.476`) | 9.70 / 11.82 |
| **`R05_southwest_pilot`** | `Stage5A.1_Cond_J` | 372 (10) | 46 $\rightarrow$ 47 | 6 (6) | 3 | 2.51% $\rightarrow$ 2.51% (`0.00`) | 8.68% $\rightarrow$ **8.12%** (`-0.56`) | 31.25% $\rightarrow$ **30.21%** (`-1.04`) | 9 / 0 / 10 / 353 | 1 / 1 / 7 (`0.250`) | 6.50 / 7.09 |
| **`R06_rich_karen`** | `Stage5A.1_Cond_J` | 1,719 (35) | 183 $\rightarrow$ 180 | 64 (79) | 8 | 21.19% $\rightarrow$ 22.48% (`+1.29`) | 25.33% $\rightarrow$ 25.82% (`+0.49`) | 49.81% $\rightarrow$ **49.31%** (`-0.50`) | 40 / 10 / 35 / 1,634 | 6 / 22 / 12 (`0.207`) | 26.72 / 38.35 |
| **`R07_bwc_cheap_shot`** | `Stage5A.1_Cond_J` | 478 (1) | 16 $\rightarrow$ 16 | 0 (0) | 1 | 12.53% $\rightarrow$ 12.53% (`0.00`) | 15.93% $\rightarrow$ 15.93% (`0.00`) | 52.99% $\rightarrow$ 52.99% (`0.00`) | 0 / 5 / 0 / 473 | 0 / 0 / 0 (`N/A`) | 6.57 / 7.68 |
| **`R08_uk_parliament`** | `Stage5A.1_Cond_J` | 468 (0) | 13 $\rightarrow$ 14 | 0 (0) | 2 | 0.22% $\rightarrow$ 0.22% (`0.00`) | 1.72% $\rightarrow$ 1.72% (`0.00`) | 10.32% $\rightarrow$ **10.11%** (`-0.21`) | 1 / 0 / 0 / 467 | 0 / 0 / 1 (`N/A`) | 6.21 / 7.05 |
| **Total / Macro** | **8 Recordings** | **4,940 (56)** | **414 $\rightarrow$ 416** | **129 (154)** | — | **16.11% $\rightarrow$ 16.36%** (`+0.25`) | **13.29% $\rightarrow$ 13.22%** (`-0.07`) | **37.23% $\rightarrow$ 36.98%** (`-0.25`) | **85 / 48 / 69 / 4,738** | **19 / 40 / 26** | **73.64 / 92.18** |

---

### B.3 Safeguard Decision Breakdown & Ablation Against Unconstrained Refinement

Every word-level proposal from `DiarizationLM-Gemma-4-E4B-v1` is audited by the `LG-TPST` safeguard filter in [`convofuse/generative_refiner.py`](convofuse/generative_refiner.py):

| Decision Reason Code | `R01` | `R02` | `R03` | `R04` | `R05` | `R06` | `R07` | `R08` | Corpus Total (`4,940` words) | Safeguard Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `UNCHANGED_BASELINE_CONFIRMED` | 263 | 297 | 621 | 630 | 353 | 1,634 | 473 | 467 | **4,738** | Model confirmed baseline acoustic attribution |
| `ACCEPTED_VALIDATED_TRANSFER` | 1 | 1 | 11 | 22 | 9 | 40 | 0 | 1 | **85** | Validated boundary speaker transfer accepted |
| `REJECTED_UNKNOWN_SPEAKER_MAPPING` | 2 | 3 | 1 | 0 | 0 | 0 | 5 | 0 | **11** | Blocked reassignment of `UNK` or out-of-inventory `<speaker:k>` |
| `REJECTED_UNSUPPORTED_OVERLAP_CORRECTION` | 0 | 5 | 5 | 17 | 0 | 10 | 0 | 0 | **37** | Blocked rewrite inside multi-speaker overlap (`ovl >= 0.50`) |
| `ABSTAINED_PRESERVE_UNK` | 0 | 1 | 3 | 0 | 10 | 35 | 0 | 0 | **49** | Preserved baseline `UNK` words without modification |
| `ABSTAINED_LOCALITY_GATE` | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | **20** | Blocked interior multi-word run rewrite ($\ge 3$ words / $\ge 1.0\text{s}$) |

#### Key Scientific Findings from Generative Refinement

1. **Remediation of Stage 6.1 `R01` `UNK` Regression**: In the initial Stage 6.1 prototype (`diarizationlm_adapter.py`), `UNK` was mapped to `<speaker:1>`, allowing `DiarizationLM-Gemma` to reassign two pre-VAD `UNK` words (`"Yes sir."` at `2.052s–2.377s` on `R01_sara_sharif`) to `SPEAKER_00` and degrading `R01` `cpWER` by `+0.70 pp` (`52.30% -> 53.00%`). ConvoFuse isolates `UNK` as `<speaker:0>` and rejects `UNK` reassignments (`REJECTED_UNKNOWN_SPEAKER_MAPPING = 2` on `R01`), completely eliminating the `+0.70 pp` degradation (`cpWER = 52.30%`, $\Delta = 0.00\text{ pp}$).
2. **Effect of Locality Gating (`LG-TPST` vs. Unconstrained on `R04_piers_robinson`)**: On `R04_piers_robinson`, unconstrained `DiarizationLM-Gemma` (`--unconstrained`) proposes rewriting 20 words inside acoustically stable interior speaker segments (`ABSTAINED_LOCALITY_GATE = 20`). Blocking those 20 interior rewrites via `LG-TPST` improves `R04` `Lex-WDER` by **`-0.81 pp`** (`11.69% -> 10.88%`) and `cpWER` by **`-0.82 pp`** (`38.57% -> 37.75%`).
3. **Why `cpWER` and `Lex-WDER` Improve While `TS-WDER` Slightly Increases on `R06`**: On high-overlap recordings (`R06_rich_karen`, 8 discovered speakers, 64 overlapping turns), `DiarizationLM-Gemma` shifts turn boundaries at rapid conversational backchannels, reducing concatenated speaker-attributed word error rate (`cpWER`: `49.81% -> 49.31%`, `-0.50 pp`). However, because reference RTTM intervals on `R06` are coarse acoustic segments rather than word-level turn boundaries, moving a word across a segment boundary can disagree with the coarse RTTM interval at that timestamp (`TS-WDER`: `+1.29 pp` on `R06`).
4. **Why `R03_disabled_parking` Degrades (`+0.92 pp` `cpWER`)**: On `R03_disabled_parking`, the acoustic baseline is already strong (`Lex-WDER = 4.83%`, `cpWER = 19.60%`), and two speakers engage in rapid, short-phrase roadside arguments where lexical context alone is ambiguous (`2` beneficial vs `6` harmful boundary shifts). This demonstrates that text-only LLM post-processing without acoustic confidence weighting can over-smooth rapid back-and-forth exchanges.
5. **Reproducibility Note on Warm vs. Cold Ollama Cache (`R01_sara_sharif`)**: When `diarizationlm-gemma:latest` (`Q4_K_M`, `temperature=0.0`, `seed=42`) is invoked from a cold GPU model load (`runtime ~ 6.1s`), floating-point prompt-evaluation batching leaves word 29 (`"Alright."` at `9.409s–9.850s` on `R01`) unchanged (`0` accepted corrections, `11` turns). Once the model is warm in Ollama GPU memory (`runtime ~ 3.5s`), it deterministically transfers word 29 to `SPEAKER_01` (`1` accepted correction, `13` turns, as packaged in `convofuse_outputs/final_release/`). Both states yield identical `TS-WDER = 20.08%`, `Lex-WDER = 19.38%`, and `cpWER = 52.30%`.

---

## Part C — Historical Phase20 Reference Appendix (`bwc.wav`)

> [!WARNING]
> **Historical Reference Caveat**:
> The tables in this appendix are reproduced strictly from [`phase20_standalone/results_to_be_published.md`](phase20_standalone/results_to_be_published.md) for historical completeness.
> - **Dataset Scope**: Evaluated on a **single recording** (`raw_audio/bwc.wav`, corresponding to `R07_bwc_cheap_shot` in the 8-recording benchmark).
> - **Protocol Differences**: Used an earlier probability-guided Pyannote 3.1 VAD mask (`Path A` / `Path B`), legacy NeMo MSDD configured with `oracle_vad = True` on that mask, and a 2-speaker reference mapping (`speaker_0 -> Speaker_1`, `speaker_1 -> Speaker_2`), rather than the 5-speaker full-recording gold annotation audited in Stages 1–5B (`R07_bwc_cheap_shot`).
> - **Non-Equivalence**: These single-recording numbers must **not** be compared directly against the 8-recording (`R01`–`R08`) macro averages in Parts A and B.

### C.1 Historical Phase20 VAD Metrics (`bwc.wav`, `phase20_standalone`)

| Branch | Precision $\uparrow$ | Recall $\uparrow$ | F1 Score $\uparrow$ | False Alarm (`FA`) $\downarrow$ | Missed Detection (`Miss`) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Path A (Diarization Branch — DFN)** | 0.8448 | 0.7721 | 0.8068 | 0.2991 | 0.2279 |
| **Path B (Transcription Branch — Loudnorm)** | 0.8417 | 0.7670 | 0.8026 | 0.3042 | 0.2330 |

### C.2 Historical Phase20 ASR Metrics (`bwc.wav`, `phase20_standalone`)

| Metric | Value |
| :--- | :---: |
| **Word Error Rate (`WER`)** | `0.3244` (`32.44%`) |
| **Substitutions (`S`)** | `86` |
| **Insertions (`I`)** | `26` |
| **Deletions (`D`)** | `45` |
| **Reference Words (`N_ref`)** | `484` |
| **Hypothesis Words (`N_hyp`)** | `465` |
| **Aligned Words** | `465` |
| **Whisper Segments** | `41` |

### C.3 Historical Phase20 Diarization Metrics (`bwc.wav`, `phase20_standalone`)

| Metric | Value |
| :--- | :---: |
| **Diarization Error Rate (`DER`)** | `0.4471` (`44.71%`) |
| **Jaccard Error Rate (`JER`)** | `0.8424` (`84.24%`) |
| **Missed Speech (`Miss`)** | `0.2300` (`23.00%`) |
| **False Alarm (`FA`)** | `0.1399` (`13.99%`) |
| **Speaker Confusion (`Confusion`)** | `0.0771` (`7.71%`) |

### C.4 Historical Phase20 Fusion & Speaker Attribution Metrics (`bwc.wav`, `phase20_standalone`)

| Metric | Value |
| :--- | :---: |
| **Concatenated Min-Permutation WER (`cpWER`)** | `0.5331` (`53.31%`) |
| **Speaker-Attributed WER (`SA-WER`)** | `0.5331` (`53.31%`) |
| **Speaker Attribution Accuracy** | `0.9027` (`90.27%`, `362 / 401` words evaluated) |
| **Word Attribution Error** | `0.0973` (`9.73%`) |
| **Timeline Accuracy** | `0.7053` (`70.53%`) |
| **Evaluated Speaker Mapping** | `{"speaker_0": "Speaker_1"}` |
