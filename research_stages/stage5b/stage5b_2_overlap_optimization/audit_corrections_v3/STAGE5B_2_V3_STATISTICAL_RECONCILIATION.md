# Stage 5B.2 V3 Matched Statistical Reconciliation & Evaluation Audit

**Workspace:** `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v3/`  
**Data Sources:** [`STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v3/STAGE5B_2_CORRECTED_PER_RECORDING_RESULTS.csv), [`STAGE5B_2_CORRECTED_RESULTS.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v3/STAGE5B_2_CORRECTED_RESULTS.csv)  
**Evaluated Systems:** Baseline Single-Label C62 (`O02`) vs Integrated Upstream C62 + Conservative Abstention M4 (`O42-R2`)  
**Bootstrap Parameters:** $B = 5,000$ paired cluster resamples, seed = 42, cluster unit = recording ($N=8$).

---

## 1. Audit Finding: Mismatched Comparison in V2 Final Report

In the Stage 5B.2 V2 final corrective report, the recommended integrated configuration **O42-R2** was compared against the **O00 baseline** (C00 single-label VAD, no overlap):

- Reported claim: O42 achieves Macro DER of $45.42\%$, beating O00 ($46.42\%$) by $-1.00\%$ absolute DER.
- **The Audit Finding**: This was **not a matched comparison**.
  - O42 uses the **C62 Consensus VAD backbone** (with hysteresis and gap-filling).
  - The matched baseline for O42 is **O02** (C62 VAD without overlap handling).
  - Comparing O42 against O00 confounded the benefit of consensus VAD with the effect of overlap attribution.

---

## 2. Forensic Reconciliation: Matched Comparison (O02 vs O42-R2)

### 2.1 Aggregated Macro and Pooled Metrics

| Metric | Matched Baseline O02 (C62 VAD) | Integrated O42-R2 (C62 + M4) | Unmatched Baseline O00 (C00 VAD) | Matched Difference $\Delta(\text{O42} - \text{O02})$ |
| :--- | :---: | :---: | :---: | :---: |
| **Macro DER (c=0)** | **44.9012%** | **45.4233%** | 46.4174% | **+0.5221% (worse)** |
| **Pooled DER (c=0)** | **48.6807%** | **49.2787%** | 49.8678% | **+0.5980% (worse)** |
| **Macro JER** | **61.2698%** | **60.7327%** | 62.2756% | **-0.5371% (better)** |
| **R04 DER (c=0)** | **39.3486%** | **31.2634%** | 39.2641% | **-8.0852% (better)** |
| **R04 JER** | **48.3941%** | **42.5558%** | 48.3941% | **-5.8383% (better)** |

### 2.2 Recomputation Proof:
1. **Macro DER**:
   $$\text{Macro DER}_{\text{O02}} = \frac{1}{8}\sum_{i=1}^8 \text{DER}_{\text{O02}}(i) = 44.9012\%$$
   $$\text{Macro DER}_{\text{O42}} = \frac{1}{8}\sum_{i=1}^8 \text{DER}_{\text{O42}}(i) = 45.4233\%$$
   $$\Delta \text{Macro DER} = +0.5221 \text{ percentage points}$$
2. **Pooled DER** (total error time divided by total reference time, not average of percentages):
   $$\text{Pooled DER}_{\text{O02}} = \frac{\sum (\text{miss} + \text{fa} + \text{conf})}{\sum \text{total\_ref}} = \frac{632.85\text{s}}{1300.00\text{s}} = 48.6807\%$$
   $$\text{Pooled DER}_{\text{O42}} = \frac{\sum (\text{miss} + \text{fa} + \text{conf})}{\sum \text{total\_ref}} = \frac{640.62\text{s}}{1300.00\text{s}} = 49.2787\%$$
   $$\Delta \text{Pooled DER} = +0.5980 \text{ percentage points}$$

---

## 3. Per-Recording Paired Analysis ($N=8$)

The per-recording breakdown reveals the exact mechanism of the metric shift:

| Recording ID | Ground-Truth Overlap (%) | O02 DER (c=0) | O42-R2 DER (c=0) | DER Difference $\Delta$ | O02 JER | O42-R2 JER | JER Difference $\Delta$ | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `R01_sara_sharif` | 1.1% | 31.19% | 31.19% | +0.00% | 60.60% | 60.60% | +0.00% | TIE |
| `R02_dennis_skinner` | 27.2% | 64.67% | 64.67% | +0.00% | 68.77% | 68.77% | +0.00% | TIE |
| `R03_disabled_parking` | 1.3% | 34.29% | 36.92% | **+2.64%** | 62.06% | 62.12% | **+0.07%** | LOSS |
| `R04_piers_robinson` | **51.1%** | **39.35%** | **31.26%** | **-8.09%** | **48.39%** | **42.56%** | **-5.84%** | **WIN** |
| `R05_southwest_pilot` | 0.9% | 60.38% | 63.20% | **+2.82%** | 49.91% | 51.15% | **+1.25%** | LOSS |
| `R06_rich_karen` | 0.6% | 71.07% | 77.87% | **+6.81%** | 64.47% | 64.70% | **+0.23%** | LOSS |
| `R07_bwc_cheap_shot` | 0.8% | 41.98% | 41.98% | +0.00% | 78.35% | 78.35% | +0.00% | TIE |
| `R08_uk_parliament` | 0.4% | 16.28% | 16.28% | +0.00% | 57.61% | 57.61% | +0.00% | TIE |

### Summary Counts ($N=8$ Cluster Units):
- **DER**: 1 Win (R04), 3 Losses (R03, R05, R06), 4 Ties (R01, R02, R07, R08).
- **JER**: 1 Win (R04), 3 Losses (R03, R05, R06), 4 Ties (R01, R02, R07, R08).

---

## 4. Paired Recording-Cluster Bootstrap & Statistical Interpretation

To rigorously account for the recording-level clustering (preserving recordings as independent sample units), a paired bootstrap with $B = 5,000$ resamples and fixed seed 42 was executed:

- **Macro DER Difference $\Delta(\text{O42} - \text{O02})$**:
  - Point Estimate: **$+0.5221\%$**
  - 95% Bootstrap CI: **$[-2.5432\%, \; +3.2311\%]$** (spans zero)
  - Wilcoxon Signed-Rank Test: $W = 4.0, \; p = 0.8750$ (not significant)
- **Pooled DER Difference $\Delta(\text{O42} - \text{O02})$**:
  - Point Estimate: **$+0.5980\%$**
  - 95% Bootstrap CI: **$[-3.6174\%, \; +4.1523\%]$** (spans zero)
- **Macro JER Difference $\Delta(\text{O42} - \text{O02})$**:
  - Point Estimate: **$-0.5371\%$**
  - 95% Bootstrap CI: **$[-2.1520\%, \; +0.4915\%]$** (spans zero)
  - Wilcoxon Signed-Rank Test: $W = 4.0, \; p = 0.8750$ (not significant)

---

## 5. Honest Scientific Interpretation: Trade-Off, Not Corpus-Wide Superiority

1. **The Source of the -1.00% DER Reduction Claim**:
   The apparent reduction from $46.42\%$ (O00) to $45.42\%$ (O42) reported in V2 was driven entirely by **Consensus VAD (C62)**, which suppresses false alarms and fills pauses. C62 alone brought DER down to $44.90\%$ (O02).
2. **The Impact of Overlap Handling**:
   Adding overlap attribution (M4) on top of C62:
   - Recovers massive amounts of overlapping speech in heavy multi-speaker audio (**$-8.09\%$ DER on R04**, cutting missed speech by half).
   - Lowers Jaccard Error Rate across the recording set (**$-0.54\%$ Macro JER**, $-5.84\%$ on R04).
   - But introduces false alarms on low-overlap files (R03, R05, R06), increasing Macro DER by $+0.52\%$ and Pooled DER by $+0.60\%$.
3. **Conclusion for Stage 5B.4**:
   O42-R2 represents a **metric trade-off** between speaker turn recall and false-alarm speech, not an unqualified corpus-wide improvement over O02. Overlap handling should be gated selectively rather than applied uniformly to clean audio.
