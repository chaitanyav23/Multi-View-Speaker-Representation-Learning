# Stage 5B.1: VAD Reference-Duration Normalization Audit

## 1. Executive Summary & Defect Identification

In Voice Activity Detection (VAD), the task is strictly binary: at each time instant $t$, determine whether *any* speaker is active:
$$y(t) = \bigvee_{k=1}^K y_k(t) \in \{0, 1\}$$

When evaluating binary VAD metrics (such as Miss Rate and Speech-Normalized False Alarm Rate), the ground-truth reference speech duration MUST be the measure of the union of speech intervals:
$$T_{\text{union}} = \int_{\mathcal{T}} \mathbb{I}\left(\sum_{k=1}^K y_k(t) \ge 1\right) dt$$

### The Implementation Flaw in the Original Stage 5B.1 Script:
In `scripts/run_stage5b_1_experiments.py` (lines 112–124), the reference speech duration was computed as:
```python
total_gt_speech = sum(s["end"] - s["start"] for s in gt_rttm_segs)
```
This summed the durations of individual speaker segments in the reference RTTM ($T_{\text{sum}} = \sum_k \int y_k(t) dt$).
In recordings with simultaneous overlapping speech, this caused simultaneous speech frames to be **double-counted** in the denominator.
Consequently:
- In `R02_dennis_skinner` ($34.6\%$ overlap), $T_{\text{sum}} = 284.0$s while $T_{\text{union}} = 162.0$s (a **$+75.3\%$ inflation** in the denominator).
- In `R04_piers_robinson` ($51.1\%$ overlap), $T_{\text{sum}} = 284.0$s while $T_{\text{union}} = 186.0$s (a **$+52.7\%$ inflation** in the denominator).

Because `fn_sec` and `fp_sec` (which were derived from frame-level boolean masks) were divided by $T_{\text{sum}}$, the reported `miss_ref` and `fa_ref` were artificially depressed in recordings with overlapping speech.

---

## 2. Granular Reference Speech Durations: Union vs. Sum

| Recording ID | Total Audio (s) | $T_{\text{sum}}$ (Sum Segs) | $T_{\text{union}}$ (Union) | Overlap Inflation ($T_{\text{sum}} - T_{\text{union}}$) | % Inflation |
|---|---|---|---|---|---|
| **R01_sara_sharif** | 108.576 | 93.000s | 93.000s | 0.000s | +0.00% |
| **R02_dennis_skinner** | 176.320 | 284.000s | **162.000s** | **+122.000s** | **+75.31%** |
| **R03_disabled_parking** | 206.272 | 178.000s | 176.000s | +2.000s | +1.14% |
| **R04_piers_robinson** | 204.608 | 284.000s | **186.000s** | **+98.000s** | **+52.69%** |
| **R05_southwest_pilot** | 162.496 | 97.000s | 97.000s | 0.000s | +0.00% |
| **R06_rich_karen** | 652.224 | 371.000s | 369.000s | +2.000s | +0.54% |
| **R07_bwc_cheap_shot** | 175.488 | 123.000s | 122.000s | +1.000s | +0.82% |
| **R08_uk_parliament** | 215.392 | 195.000s | 195.000s | 0.000s | +0.00% |
| **CORPUS TOTAL** | **1995.076s** | **1625.000s** | **1400.000s** | **+225.000s** | **+16.07%** |

---

## 3. Impact on VAD Metrics (Baseline C00 and Mono C50)

### 3.1 Per-Recording Miss Rate Comparison: Corrected Union vs Legacy Sum

| Recording ID | C00 Legacy Miss ($FN/T_{\text{sum}}$) | C00 Corrected Miss ($FN/T_{\text{union}}$) | $\Delta_{\text{C00}}$ | C50 Legacy Miss ($FN/T_{\text{sum}}$) | C50 Corrected Miss ($FN/T_{\text{union}}$) | $\Delta_{\text{C50}}$ |
|---|---|---|---|---|---|---|
| **R01** | 27.59% | **27.59%** | 0.00% | 10.24% | **10.24%** | 0.00% |
| **R02** | 22.84% | **40.04%** | **+17.20%** | 18.70% | **32.78%** | **+14.08%** |
| **R03** | 12.58% | **12.72%** | +0.14% | 9.13% | **9.24%** | +0.11% |
| **R04** | 0.13% | **0.20%** | +0.07% | 0.16% | **0.24%** | +0.08% |
| **R05** | 13.52% | **13.52%** | 0.00% | 6.86% | **6.86%** | 0.00% |
| **R06** | 16.51% | **16.60%** | +0.09% | 10.94% | **11.00%** | +0.06% |
| **R07** | 10.66% | **10.75%** | +0.09% | 7.74% | **7.80%** | +0.06% |
| **R08** | 12.44% | **12.44%** | 0.00% | 12.24% | **12.24%** | 0.00% |
| **MACRO** | **15.22%** | **16.39%** | **+1.17%** | **8.75%** | **9.15%** | **+0.40%** |

### 3.2 Key Revelations:
1. **The R02 Discrepancy**:
   In `R02_dennis_skinner`, the legacy script reported a VAD Miss Rate of only $22.84\%$. Dividing by the correct union of speech ($162$s instead of $284$s) reveals that the true binary VAD Miss Rate on R02 was actually **$40.04\%$**! In R02, DeepFilterNet3 attenuated $64.87$ seconds of speech out of $162$ seconds of active dialogue.
2. **Impact on C50 (Mono Stream)**:
   On the mono stream, the true binary VAD Miss Rate in R02 drops from $40.04\%$ down to **$32.78\%$** (a **$-7.26\%$ absolute reduction** in missed speech).
3. **Macro VAD Metrics Reconciled**:
   Across the 8 recordings:
   - Baseline C00 Macro Miss Rate increases from $15.22\%$ to **$16.39\%$** ($+1.17\%$).
   - Raw Mono C50 Macro Miss Rate increases from $8.75\%$ to **$9.15\%$** ($+0.40\%$).
   - Baseline C00 Macro FA Rate increases from $13.39\%$ to **$13.68\%$** ($+0.29\%$).
   - Raw Mono C50 Macro FA Rate increases from $15.49\%$ to **$15.94\%$** ($+0.45\%$).

---

## 4. Separation of Denominators: VAD vs. DER

To prevent future confusion, the pipeline must strictly separate:
1. **Binary VAD Denominator**: Always $T_{\text{union}} = \text{duration of active speech union}$.
2. **Diarization Error Rate (NIST/pyannote) Denominator**: Always $T_{\text{sum}} = \text{total reference speaker time}$.

In DER, if two speakers talk simultaneously for 10 seconds, that constitutes 20 seconds of reference speaker-time. If a single-label diarization model assigns the segment to Speaker A, Speaker B's 10 seconds are counted as Missed Detection. This is structurally distinct from binary VAD, where any speech detection constitutes a true positive.
