# Stage 5B.1: Correction Addendum to Final Experimental Report

This addendum amends [`STAGE5B_1_FINAL_REPORT.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_1_vad_optimization/STAGE5B_1_FINAL_REPORT.md) following the independent audit and corrective validation conducted in `audit_corrections/`.

---

## 1. Metric Designation Corrections

1. **Downstream Diarization Error Rate (DER)**:
   All downstream DER values reported in Stage 5B.1 must be read as:
   **"VAD-mask-projected DER under frozen baseline speaker assignments"**.
   - These results evaluate the effect of updating the speech activity mask while inheriting speaker cluster identities via nearest-neighbor temporal propagation from the frozen Condition J (M1) baseline.
   - On average across the corpus, **$10.58\%$** of the hypothesis speech in C50 (and $20.07\%$ in R01) was assigned speaker identities via nearest-neighbor propagation.
   - Full end-to-end diarization verification with fresh embedding extraction and re-clustering is deferred to **Stage 5B.4**.

2. **Binary VAD Reference Normalization**:
   - The reported VAD Miss Rate and False Alarm Rate have been recomputed using the exact union of reference speech intervals ($T_{\text{union}}$), eliminating the double-counting of simultaneous speakers present in the legacy segment-sum calculation ($T_{\text{sum}}$).
   - The true baseline C00 Macro VAD Miss Rate is **$16.39\%$** (corrected from $15.22\%$).
   - The true raw mono C50 Macro VAD Miss Rate is **$9.15\%$** (corrected from $8.75\%$).
   - In R02 (`dennis_skinner`, $34.6\%$ overlap), the true binary VAD Miss Rate is **$40.04\%$** (corrected from $22.84\%$).

3. **Conversational Attribution (cpWER) Availability**:
   - The missing cpWER values in the original CSV have been resolved and computed using the exact dictionary payload schema `{"segments": segs_attr}` without fallback.
   - Baseline C00 Macro cpWER is established at **$37.12\%$**.
   - The legacy $0.50$s minimum duration filter (`C13`) degrades cpWER to **$37.58\%$** ($+0.46\%$ error), confirming that aggressive turn filtering damages conversational attribution.
   - For C50 (raw mono), cpWER is **$37.59\%$** (essentially flat relative to baseline $37.12\%$), showing that frame-level VAD fringe expansion does not automatically translate into improved word attribution.

---

## 2. Updated Summary Comparison Table

| Config ID | Description | Stream | VAD Prec | VAD Rec | VAD F1 | VAD Miss ($T_{\text{union}}$) | VAD FA ($T_{\text{union}}$) | Projected DER (c=0) | Projected DER (c=0.25) | Macro cpWER | Macro NN Prop % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **C00** | **Baseline (M1)** | DFN | 87.45% | 83.61% | 84.94% | **16.39%** | **13.68%** | **46.17%** | **39.65%** | **37.12%** | **0.00%** |
| **C10** | Short-Turn: 0.05s | DFN | 87.45% | 83.64% | 84.96% | 16.36% | 13.68% | 46.16% | 39.64% | 37.14% | 0.03% |
| **C13** | Legacy Filter: 0.50s | DFN | 87.90% | 82.85% | 84.74% | 17.15% | 12.92% | 45.93% | 39.40% | **37.58%** | 0.00% |
| **C20** | High Sensitivity (T=0.30) | DFN | 86.64% | 86.24% | 85.91% | 13.76% | 15.26% | 46.04% | 39.36% | 37.23% | 4.23% |
| **C25** | High Specificity (T=0.80) | DFN | 89.03% | 78.22% | 82.66% | 21.78% | 10.90% | 47.08% | 40.75% | 37.24% | 0.00% |
| **C31** | Hysteresis: 0.60 / 0.35 | DFN | 87.27% | 84.34% | 85.22% | 15.66% | 14.07% | 46.14% | 39.55% | 37.21% | 1.96% |
| **C41** | Gap Filling: 0.30s | DFN | 87.27% | 84.12% | 85.12% | 15.88% | 14.01% | 46.13% | 39.57% | 37.12% | 0.89% |
| **C50** | **Stream: Raw 16k Mono** | **Mono** | 86.76% | **90.85%** | **88.34%** | **9.15%** | 15.94% | **43.84%** | **36.99%** | 37.59% | **10.58%** |
| **C52** | Linear Consensus | Cons | 87.34% | 86.38% | 86.35% | 13.62% | 14.37% | 45.13% | 38.40% | 37.26% | 4.26% |
| **C61** | Consensus + Hysteresis | Cons | 86.82% | 87.89% | 86.89% | 12.11% | 15.27% | 44.80% | 38.20% | 37.12% | 6.71% |
| **C62** | **Consensus+Hyst+Gap** | **Cons** | 86.81% | 88.17% | 87.03% | **11.83%** | 15.33% | **44.65%** | **38.03%** | **37.12%** | **7.02%** |

---

## 3. Reconciled Status of Stage 5B.1 Deliverables

The definitive, corrected outputs are recorded in:
- Corrected VAD Metrics: [`audit_corrections/metrics/STAGE5B_1_CORRECTED_VAD_METRICS_PER_RECORDING.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_1_vad_optimization/audit_corrections/metrics/STAGE5B_1_CORRECTED_VAD_METRICS_PER_RECORDING.csv)
- Corrected Downstream Metrics: [`audit_corrections/metrics/STAGE5B_1_CORRECTED_DOWNSTREAM_METRICS_PER_RECORDING.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_1_vad_optimization/audit_corrections/metrics/STAGE5B_1_CORRECTED_DOWNSTREAM_METRICS_PER_RECORDING.csv)
- Corrected Summary Metrics: [`audit_corrections/metrics/STAGE5B_1_CORRECTED_SUMMARY_COMPARISON.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_1_vad_optimization/audit_corrections/metrics/STAGE5B_1_CORRECTED_SUMMARY_COMPARISON.csv)
- Reconciliation Delta: [`audit_corrections/metrics/STAGE5B_1_METRIC_RECONCILIATION_DELTA.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_1_vad_optimization/audit_corrections/metrics/STAGE5B_1_METRIC_RECONCILIATION_DELTA.csv)
