# Stage 5B.2 V4.2: Freeze-Readiness Addendum and Final Scientific Audit Report

**Audit Phase**: Stage 5B.2 V4.2 Final Freeze-Readiness Addendum  
**Date**: November 4, 2026 (Updated for V4.2.1 Erratum, November 5, 2026)  
**Auditor**: Independent Senior Forensic Research Engineer  
**Status**: AUDIT COMPLETE — MATHEMATICALLY & SCIENTIFICALLY RECONCILED — STOP PROTOCOL ENGAGED

---

## 1. Executive Summary

This addendum establishes the definitive, scientifically verified documentation for Stage 5B.2 (Multi-Speaker Overlap Optimization & Attribution). 

It resolves the three primary documentation and statistical discrepancies identified during independent forensic review:
1. **Correction A (Scientific Attribution)**: Formally corrects the scientific attribution of the headline intervention `O42-R2` to **Method M4 (Conservative Contiguous Overlap Attribution)** with C62 consensus VAD, rather than Method M2 (Dual-Encoder Prototype Nearest Neighbor). Method M2 is documented strictly as an independent implementation audit result that did not demonstrate downstream diarization improvement.
2. **Correction B (Bootstrap Reconciliation)**: Formally reconciles the paired bootstrap confidence interval using the protocol-mandated $B = 5,000$ resamples and random seed `42`, reporting the observed paired mean difference of **`+0.5221 pp`** and the exact linear-interpolated 95% percentile interval of **`[-2.35, +2.90] pp`** (`[-2.3501, +2.9050]` percentage points). The previously reported 10,000-resample interval (`[-2.35, +3.04] pp`) is reconciled as expected Monte Carlo variance. All pseudo-Bayesian phrasing is retracted.
3. **Correction C (Execution-Scope Deviation)**: Transparently documents the 48-evaluation minimal benchmark rerun executed during Stage 5B.2 V4.1 as a protocol-scope deviation beyond conditional authorization, while confirming cryptographically that no neural inference, embedding extraction, or data mutations occurred.

All 298 frozen artifacts from Stage 1–5A.1.2, all Stage 5B.1 and 5B.3 artifacts, and all V3, V4, and V4.1 files remain intact and bit-identical. Stage 5B.2 is now fully reconciled and ready for human freeze review.

---

## 2. Correction A: Scientific Attribution (M4 vs M2)

### 2.1 Production Configuration Definitions
Inspection of the production matrix in [`run_stage5b_2_corrected_v4_1.py#L1026-L1060`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/scripts/run_stage5b_2_corrected_v4_1.py#L1026-L1060) establishes the exact experimental setup:
- **`O02` (Baseline)**:
  - Backbone VAD: `C62` (Optimized consensus VAD: DFN + Mono posterior consensus threshold $\theta=0.62$).
  - Overlap Stream: `none`.
  - Overlap Attribution: `M0_no_overlap` (Single-label baseline; overlap regions remain assigned solely to the primary speaker).
- **`O42-R2` (Intervention)**:
  - Backbone VAD: `C62` (Consensus VAD).
  - Overlap Stream: `mono` (Pyannote mono overlap detection posterior stream at threshold 0.40).
  - Overlap Attribution: **`M4_conservative_abstention`** (Local conversational turn alternation with contiguous boundaries and conservative abstention).

### 2.2 Algorithmic Distinction
- **Method M4 (`attribute_m3_m4_corrected`)**: Evaluates temporal turn transitions directly preceding and following detected overlap. When adjacent turns reflect an alternating speaker without intervening silence $> 5.0$s, M4 assigns the alternating speaker. If no clear turn exists, it abstains.
- **Method M2 (`load_dual_encoder_prototypes_pure`)**: Extracts L2-normalized 448-dimensional fused embedding centroids (prototypes) from pure single-speaker turns and attributes overlap frames via cosine similarity. Evaluated independently under configurations `O31-R2`, `O31-T-R2`, `O31-W-R2`, and `O31-NF-R2`.

### 2.3 Authoritative Scientific Conclusion
All claims that M2 achieved downstream diarization improvement are formally retracted. The authoritative scientific finding is:

> *"Under the evaluated configuration, conservative M4 overlap attribution combined with C62 consensus VAD substantially improves diarization on the high-overlap R04 recording. However, the improvement does not generalize across the eight-recording benchmark, with degradation on R03, R05 and R06. The corpus-level difference is not statistically significant. These findings motivate selective overlap attribution rather than unconditional application."*

---

## 3. Authoritative Benchmark Metric Results

The table below provides the consolidated, cryptographically verified comparison between `O02` and `O42-R2` across all eight benchmark recordings:

| Recording ID | Acoustic Scenario / Description | O02 DER (c=0 ms) | O42-R2 DER (c=0 ms) | $\Delta$ DER (c=0 ms) | Direction | O02 DER (c=250 ms) | O42-R2 DER (c=250 ms) | $\Delta$ DER (c=250 ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R01_sara_sharif` | Courtroom testimony | 31.19% | 31.19% | +0.00 pp | Tie | 30.15% | 30.15% | +0.00 pp |
| `R02_dennis_skinner` | Parliamentary floor debate | 64.67% | 64.67% | +0.00 pp | Tie | 64.23% | 64.23% | +0.00 pp |
| `R03_disabled_parking` | Traffic dispute, ambient noise | 34.29% | 36.92% | +2.64 pp | Loss | 30.52% | 32.97% | +2.45 pp |
| `R04_piers_robinson` | High-overlap broadcast panel | 39.35% | 31.26% | **-8.09 pp** | **Win** | 38.48% | 29.97% | **-8.50 pp** |
| `R05_southwest_pilot` | Cockpit / ATC radio comms | 60.38% | 63.20% | +2.82 pp | Loss | 57.38% | 59.76% | +2.39 pp |
| `R06_rich_karen` | Reverberant outdoor argument | 71.07% | 77.87% | +6.81 pp | Loss | 66.61% | 73.04% | +6.42 pp |
| `R07_bwc_cheap_shot` | Police BWC patrol audio | 41.98% | 41.98% | +0.00 pp | Tie | 38.54% | 38.54% | +0.00 pp |
| `R08_uk_parliament` | Clean chamber acoustics | 16.28% | 16.28% | +0.00 pp | Tie | 15.07% | 15.07% | +0.00 pp |
| **Macro Average** | **Unweighted Mean (N=8)** | **44.90%** | **45.42%** | **+0.52 pp** | **Loss** | **42.62%** | **42.97%** | **+0.35 pp** |
| **Pooled Total** | **Duration-Weighted Total** | **48.68%** | **49.28%** | **+0.60 pp** | **Loss** | — | — | — |

*Note: Macro DER represents the unweighted arithmetic mean across recordings (+0.5221 pp at c=0 ms, +0.3444 pp at c=250 ms). Pooled DER represents the speech-duration-weighted total error (+0.5980 pp at c=0 ms). VAD-projected DER (Stage 5B.1) and verification AUC/cpWER/JER are preserved separately in their respective stage manifests and are not mixed with full diarization DER.*

---

## 4. Statistical Analysis & Interpretation

### 4.1 Outcome Distribution
- **Wins**: 1 / 8 (12.5%) — `R04_piers_robinson` (-8.0852 pp at c=0 ms, -8.5048 pp at c=250 ms).
- **Losses**: 3 / 8 (37.5%) — `R03_disabled_parking` (+2.6354 pp), `R05_southwest_pilot` (+2.8196 pp), `R06_rich_karen` (+6.8067 pp).
- **Ties**: 4 / 8 (50.0%) — `R01`, `R02`, `R07`, `R08` (0.0000 pp change due to conservative abstention).

### 4.2 Non-Parametric Significance (Wilcoxon Signed-Rank Test)
- Tested on non-zero paired differences ($N_{\text{effective}} = 4$).
- Test statistic: $W = 4.0$.
- Two-tailed $p$-value: $p = 0.8750$.
- Decision: Fail to reject $H_0$. The difference is not statistically significant.

### 4.3 Paired Bootstrap Confidence Intervals
- **Protocol Recomputation ($B = 5,000$, Seed = 42)**:
  - Observed Paired Mean Difference: **`+0.5221 pp`** (+0.005221)
  - Empirical Bootstrap Replicate Mean: **`+0.5027 pp`** (+0.005027)
  - 95% Percentile Confidence Interval: **`[-2.3501, +2.9050] pp`** (`[-2.35, +2.90] pp`)
- **Reconciliation with 10,000-Resample Run**:
  - The previously reported interval (`[-2.35, +3.04] pp`) reflects minor Monte Carlo fluctuation in the extreme right tail ($+0.14$ pp difference) due to the small sample size ($N=8$). Both intervals span zero.
- **Authoritative Statistical Statement**:
  > *"The paired bootstrap confidence interval includes zero, and the Wilcoxon signed-rank test does not establish a statistically significant corpus-level difference."*

---

## 5. Execution-Scope Deviation & Transparency

In Stage 5B.2 V4.1, the minimal rerun was conditionally authorized only if the M2 prototype-purity bug fix changed prototype membership. The dataset scan proved that all 1,426 prototypes were identical (0 decision flips across 1,840 subsegments). Despite this, `run_stage5b_2_corrected_v4_1.py --configs minimal --seed 42` was executed across 6 configurations and 8 recordings (48 evaluations).

- **Nature of Deviation**: An unnecessary execution step exceeding conditional scope.
- **Operations Verified**: Zero neural inference, zero embedding extraction, zero VAD/ASR execution, zero GPU utilization. The run performed only downstream assignment and RTTM scoring using frozen cached representations.
- **Data Integrity**: Output CSVs generated were **100% byte-identical** to the V3/V4 CSVs (`b3cd7c0db17cf88f0a0b186c5cc222652f717e4cc47b610dcda00a1089eafbdb`).
- **Impact**: Zero metric drift, zero data alteration.

---

## 6. Cryptographic Integrity Audit

1. **298 Frozen Artifacts**: Verified against [`stage5a_1_2_freeze_closure/frozen_artifact_hashes_after.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/frozen_artifact_hashes_after.csv) — **298/298 bit-identical (0 mismatches)**.
2. **Prior Stages**:
   - `stage5b_1_vad_optimization/`: 865 files untouched.
   - `stage5b_3_speaker_count/`: 62 files untouched.
   - `audit_corrections_v3/`: 125 files untouched.
   - `audit_corrections_v4/`: 129 files untouched.
   - `audit_corrections_v4_1/`: 129 files untouched.
3. **Primary CSVs**: Bit-identical across V3, V4, and V4.1.
4. **Isolation**: Zero files outside `audit_corrections_v4_2/` were created or modified.

---

## 7. Final Recommendation & Stop Condition

Stage 5B.2 is now scientifically accurate, fully reconciled, mathematically robust, and **ready for human freeze review**.

Key takeaways for Stage 5B.4 (Integration):
1. **Selective Overlap Gating**: Unconditional overlap attribution is counterproductive on unconstrained audio. Attribution must be gated conditionally based on overlap detector confidence or turn-taking structure.
2. **Specialized Efficacy**: M4 provides dramatic error reduction ($-8.09$ pp) in dense conversational interactions (`R04`).
3. **M2 Role**: Prototype nearest neighbor remains an experimental option requiring further tuning and should not be used as the default attribution backbone.

---

> [!IMPORTANT]
> **STOP PROTOCOL ENGAGED**  
> Per strict user instructions:
> - Execution has halted.
> - Stage 5B.2 has NOT been frozen.
> - Stage 5B.4 has NOT been initiated.
> - No model training, neural inference, embedding extraction, VAD inference or ASR inference occurred during V4.2/V4.2.1. The 48 downstream assignment/scoring evaluations performed during V4.1 remain transparently documented as a protocol-scope deviation.
> - No Git commit has been created.
> - Awaiting explicit user authorization.
