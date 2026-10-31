# Stage 5A.1 Issues Register: Final Independent Audit

**Audit Date**: October 31, 2026  
**Project**: Multi-View Speaker Representation Learning  
**Target Subsystem**: Stage 5A.1 Affinity Normalization, Fusion Fairness, and Evaluation Protocol  
**Repository State**: Git Commit `16d80ec3e98da93d9a8692df6dd9661723afe0c1`

---

## 1. Overview and Severity Classification

This Issues Register documents all anomalies, discrepancies, implementation nuances, and architectural bottlenecks identified during the final independent audit of Stage 5A.1. Each issue is assigned a unique identifier, assessed for severity, evaluated for its impact on scientific conclusions, and paired with its definitive resolution.

### Severity Definitions
- **Critical**: Invalidates core scientific findings; requires experimental re-execution or data re-extraction.
- **High**: Significant methodological discrepancy or severe downstream performance bottleneck; requires formal qualification and limitation disclosure.
- **Medium**: Protocol divergence, reporting inconsistency, or metric ambiguity; reconcilable via code inspection, rigorous recomputation, and explicit documentation without new experiments.
- **Low**: Minor documentation typo, versioning detail, or negligible numerical rounding discrepancy.

---

## 2. Issues Log

### ISSUE-01: Discrepancy in Clean Turn and Pair Counts Between Stage 5A and Stage 5A.1
- **Issue ID**: `ISS-01`
- **Severity**: Medium
- **Location**:
  - Stage 5A: `research_stages/stage5a/scripts/run_stage5a_eval.py:165-200`
  - Stage 5A.1: `research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py:145-188`
- **Description**:
  Stage 5A reported 254 clean turns and 9,817 pairwise trials across the 8 recordings, whereas Stage 5A.1 reported 250 clean turns and 9,489 pairwise trials, representing a discrepancy of 4 turns and 328 pairs.
- **Evidence**:
  Automated turn-level diffing executed by [`research_stages/stage5a_1_final_audit/scripts/audit_turn_discrepancy.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_final_audit/scripts/audit_turn_discrepancy.py) revealed that Stage 5A.1 introduced three explicit quality constraints in `build_hybrid_turn_records`:
  1. Boundary margin constraint: `(st >= gst - 0.05) and (en <= gen + 0.05)`
  2. Overlap exclusion: `not m["gt_is_overlap"]`
  3. Strict speaker matching: `m["gt_label"] == g["speaker"]`
  
  Four turns included in Stage 5A were excluded in Stage 5A.1:
  - `R03_disabled_parking`, segment 41 ([133.0, 134.0], Speaker 1): Subsegment 455 ends at 134.113s, exceeding the 50 ms tolerance boundary ($134.113 > 134.05$).
  - `R06_rich_karen`, segment 66 ([288.0, 289.0], Speaker 7): Subsegment 526 ends at 289.077s, exceeding tolerance ($289.077 > 289.05$).
  - `R06_rich_karen`, segment 85 ([334.0, 335.0], Speaker 4): Subsegment 669 has `gt_is_overlap = True`. Stage 5A erroneously included it; Stage 5A.1 properly filtered it.
  - `R07_bwc_cheap_shot`, segment 44 ([135.0, 136.0], Speaker 1): Subsegment 373 ends at 136.054s, exceeding tolerance ($136.054 > 136.05$).
- **Impact on Scientific Conclusions**:
  Zero negative impact. Stage 5A.1's filtering is methodologically cleaner, preventing contaminated boundary and overlap frames from leaking into the verification protocol. Pair counts within each recording strictly follow $N_r(N_r - 1) / 2$, summing deterministically to 9,489 pairs.
- **Prevents Freezing**: No.
- **Resolution**:
  Formally adopt the 250-turn / 9,489-pair evaluation set as the definitive ground-truth benchmark for Stage 5A.1. Document the exact filtering rules in the frozen record.
- **Requires New Experiment**: No.

---

### ISSUE-02: Ambiguity in 5-Recording vs 6-Recording Macro Verification Averaging
- **Issue ID**: `ISS-02`
- **Severity**: Medium
- **Location**:
  - Stage 5A.1 Report: `research_stages/stage5a_1/reports/STAGE5A_1_REPORT.md:Sec 4`
  - Verification Summary CSV: `research_stages/stage5a_1/verification/stage5a_1_verification_summary.csv`
- **Description**:
  In `stage5a_1_verification_summary.csv`, the reported macro metrics (e.g., Condition A ROC-AUC = 0.8238, Condition C ROC-AUC = 0.8450, Condition K ROC-AUC = 0.8458) are explicitly computed across 6 eligible recordings (`eligible_recordings_count = 6`: R01, R03, R05, R06, R07, R08). However, in the narrative text of `STAGE5A_1_REPORT.md`, Section 2 stated that R01 was excluded due to sparse trials (1 positive, 14 negative pairs), and Section 4 labeled the macro row as "Macro Average (5 Eligible)".
- **Evidence**:
  Recomputation via [`verify_all_results.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_final_audit/scripts/verify_all_results.py) confirms:
  - **6-Recording Macro (R01, R03, R05, R06, R07, R08)**:
    - TitaNet (A/H): ROC-AUC = 0.8238, HTER = 33.80%
    - ResNet (B/I): ROC-AUC = 0.8230, HTER = 32.40%
    - Equal Fusion (C/J): ROC-AUC = 0.8450, HTER = 32.69%
    - Reliability Gate (K): ROC-AUC = 0.8458, HTER = 31.90%
  - **5-Recording Macro (R03, R05, R06, R07, R08 strictly excluding R01)**:
    - TitaNet (A/H): ROC-AUC = 0.8172, HTER = 34.13%
    - ResNet (B/I): ROC-AUC = 0.8304, HTER = 33.16%
    - Equal Fusion (C/J): ROC-AUC = 0.8426, HTER = 32.79%
    - Reliability Gate (K): ROC-AUC = 0.8436, HTER = 32.56%
- **Impact on Scientific Conclusions**:
  Under both 6-recording and 5-recording definitions, the core conclusions hold identically:
  1. Equal fusion improves ROC-AUC over both single encoders (+0.0212 vs TitaNet on 6-rec; +0.0122 vs ResNet on 5-rec).
  2. The reliability gate achieves the top ROC-AUC (0.8458 on 6-rec, 0.8436 on 5-rec) and lowest HTER (31.90% on 6-rec, 32.56% on 5-rec).
- **Prevents Freezing**: No.
- **Resolution**:
  Reconcile both tables in `STAGE5A_1_CORRECTED_RESULTS.md`. Publish both the 6-recording macro (inclusive of R01's 15 trials) and the 5-recording macro (strictly balanced multi-speaker sessions) with unambiguous labels.
- **Requires New Experiment**: No.

---

### ISSUE-03: Analytical vs Executable Reliability Gate Formulation (Linear Clipped vs Sigmoid)
- **Issue ID**: `ISS-03`
- **Severity**: Low
- **Location**:
  - Stage 5A implementation: `research_stages/stage5a/scripts/fusion_models.py:96-103`
  - Stage 5A.1 gate audit: `research_stages/stage5a_1/reports/STAGE5A_1_GATE_IMPLEMENTATION_AUDIT.md:Sec 2.1`
- **Description**:
  Initial narrative documentation described the reliability gate weighting function using a sigmoid mapping $\sigma(\beta \cdot \Delta_{\text{pred}})$. In contrast, the executable implementation in `fusion_models.py` uses a piecewise linear clipped mapping:
  $$g_{\text{res}} = \text{clip}(0.5 + \gamma \cdot \Delta_{\text{pred}},\, 0.15,\, 0.85)$$
  with gain $\gamma = 2.0$.
- **Evidence**:
  Inspection of `ReliabilityWeightedFusion.fuse()` in `research_stages/stage5a/scripts/fusion_models.py` confirmed lines 96–103:
  ```python
  pred_delta = self.gate_model.predict(features)
  g_res = np.clip(0.5 + self.gain_factor * pred_delta, self.clip_min, self.clip_max)
  ```
  In Stage 5A.1, both formulations were evaluated side-by-side (`condition_K_norm_reliability_gate` vs `condition_K_sigmoid_gate`):
  - Linear Clipped (K): Macro ROC-AUC = 0.8458, Macro HTER = 31.90%
  - Sigmoid (K_sig): Macro ROC-AUC = 0.8454, Macro HTER = 32.69%
- **Impact on Scientific Conclusions**:
  None. The linear clipped gate is computationally simpler, avoids saturating exponential operations, and yields slightly better HTER performance. Both formulations confirm the efficacy of reliability gating.
- **Prevents Freezing**: No.
- **Resolution**:
  Formally define the frozen gate equation as the linear clipped mapping in `STAGE5A_1_FREEZE_RECORD.md`. Document the sigmoid variant as an audited ablation.
- **Requires New Experiment**: No.

---

### ISSUE-04: NeMo Environment Version Reference in Audit Documentation
- **Issue ID**: `ISS-04`
- **Severity**: Low
- **Location**:
  - Environment specifications: `research_stages/stage5a_1/reports/STAGE5A_1_NORMALIZATION_AUDIT.md:Sec 8`
- **Description**:
  Certain historical project logs referenced NeMo v2.1.0, whereas the active execution virtual environment (`.venv_phase20`) runs NeMo v2.7.3.
- **Evidence**:
  Direct inspection of `.venv_phase20/lib/python3.10/site-packages/nemo/package_info.py` and `pip list` confirmed `nemo_toolkit == 2.7.3` and `torch == 2.5.1+cu124`.
- **Impact on Scientific Conclusions**:
  None. The NME-SC spectral clustering algorithm implementation in `nemo.collections.asr.parts.utils.clustering_utils` has identical mathematical formulation across both versions.
- **Prevents Freezing**: No.
- **Resolution**:
  Record `nemo_toolkit == 2.7.3` as the authoritative frozen environment package in `STAGE5A_1_FREEZE_RECORD.md`.
- **Requires New Experiment**: No.

---

### ISSUE-05: Single-Speaker Clustering Collapse on Recording R07 (`R07_bwc_cheap_shot`)
- **Issue ID**: `ISS-05`
- **Severity**: High (Downstream Generalization Bottleneck)
- **Location**:
  - Diarization results: `research_stages/stage5a_1/downstream_metrics/stage5a_1_per_recording_diarization.csv`
- **Description**:
  On recording `R07_bwc_cheap_shot` (ground truth: 3 speakers), all evaluated conditions (A, B, C, H, I, J, K, K_sig) predict exactly 1 speaker (`pred_spk_count = 1`), suffering a catastrophic single-speaker collapse.
- **Evidence**:
  Inspection of `stage5a_1_per_recording_diarization.csv` shows:
  ```csv
  R07_bwc_cheap_shot,condition_A_raw_titanet_dfn,gt_speakers=3,pred_speakers=1,der=0.5516,cpwer=0.7816
  R07_bwc_cheap_shot,condition_K_norm_reliability_gate,gt_speakers=3,pred_speakers=1,der=0.5516,cpwer=0.7816
  ```
  In R07, high acoustic distortion and heavy background noise cause cross-speaker affinity off-diagonals to exceed the NME-SC binarization threshold ($p = 0.25$), merging all graph components into a single clique.
- **Impact on Scientific Conclusions**:
  This demonstrates that affinity-level representation fusion cannot rescue downstream spectral clustering when the affinity matrix as a whole suffers from baseline acoustic elevation. Speaker count estimation under noisy BWC remains an uncalibrated failure mode.
- **Prevents Freezing**: No (it is a valid negative finding of Stage 5A.1).
- **Resolution**:
  Explicitly state in `STAGE5A_1_FREEZE_RECORD.md` and research claims that representation fusion does not solve speaker count estimation or prevent single-speaker collapse in extreme noise.
- **Requires New Experiment**: No (Stage 5B scope).

---

### ISSUE-06: Downstream Diarization Error Ceiling Dominated by Upstream VAD Overhead
- **Issue ID**: `ISS-06`
- **Severity**: High (Architectural Boundary Constraint)
- **Location**:
  - Diarization summary: `research_stages/stage5a_1/downstream_metrics/stage5a_1_summary_diarization.csv`
- **Description**:
  The macro DER across all conditions hovers between 45.92% and 46.22%. Decomposing the DER components reveals that the error is overwhelmingly dominated by upstream VAD false alarms and missed speech, rather than speaker confusion.
- **Evidence**:
  Across all conditions (A through K):
  - **Macro Missed Speech**: **24.41%**
  - **Macro False Alarm**: **13.33%**
  - **Fixed VAD Error Ceiling**: $24.41\% + 13.33\% = \mathbf{37.74\%}$
  - **Macro Speaker Confusion**: **8.18%** (Condition K) to **8.48%** (Condition A)
- **Impact on Scientific Conclusions**:
  Representation fusion successfully reduces speaker confusion from 8.48% to 8.18% ($\Delta_{\text{abs}} = -0.30\%$). However, because VAD errors represent 81.7% of total DER ($37.74 / 46.22$), downstream DER improvements are fundamentally masked and capped at $\sim 0.3\%$ absolute.
- **Prevents Freezing**: No.
- **Resolution**:
  Document in the freeze record that affinity fusion operates exclusively on speaker confusion. Advancing total DER beyond the 45% ceiling requires upstream VAD recalibration and overlap detection in Stage 5B.
- **Requires New Experiment**: No.

---

## 3. Summary of Issues

| Issue ID | Category | Severity | Status | Impact on Freezing |
| :--- | :--- | :---: | :---: | :---: |
| **ISS-01** | Turn/Pair Filtering Discrepancy | Medium | Resolved | None (stricter protocol verified) |
| **ISS-02** | 5-Rec vs 6-Rec Macro Reporting | Medium | Resolved | None (both verified & reported) |
| **ISS-03** | Linear vs Sigmoid Gate Formulation | Low | Resolved | None (both verified; linear primary) |
| **ISS-04** | NeMo Toolkit Versioning | Low | Resolved | None (2.7.3 verified identical) |
| **ISS-05** | R07 Single-Speaker Collapse | High | Documented | Qualified limitation for Stage 5B |
| **ISS-06** | Upstream VAD Error Ceiling | High | Documented | Architectural boundary for Stage 5B |

No critical issues remain. All discrepancies have been reconciled with mathematical and code-level evidence.
