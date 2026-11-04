# Audit A: Reconciliation of R04 Theoretical Overlap Miss Floor

## 1. Executive Summary

An audit of Stage 5B.2 identified apparent numerical inconsistencies in the reported ground-truth speech and overlap metrics for recording `R04_piers_robinson`:
- Reported total reference speaker-time: 284.00 s
- Reported union speech duration: 186.00 s
- Difference: $284.00 - 186.00 = 98.00\text{ s}$
- Reported overlap duration: 95.00 s (discrepancy of 3.00 s)
- Reported baseline missed detection rate: 32.00%
- Theoretical ratio: $95.00 / 284.00 = 33.45\%$, and $98.00 / 284.00 = 34.51\%$, neither of which equals 32.00%.

This forensic audit inspected the exact reference RTTM, metric evaluation source code, and pyannote data structures to reconcile every millisecond of difference.

### Core Discoveries:
1. **The 95.0s vs 98.0s Overlap Discrepancy**:
   - $T_{\text{overlap}} = \int \mathbb{1}[|Y(t)| \ge 2]\,dt = \mathbf{95.000\text{ s}}$. This is the temporal duration of time frames containing at least two active speakers (binary overlap presence).
   - $T_{\text{excess}} = \int \max(0, |Y(t)| - 1)\,dt = \mathbf{98.000\text{ s}}$. This is the total overlapping speaker-time.
   - The exact difference of **3.000 s** corresponds to time frames where **three speakers are simultaneously active** ($|Y(t)| = 3$):
     $$\int \mathbb{1}[|Y(t)| \ge 3]\,dt = 3.000\text{ s}$$
     During these 3.0 seconds, $|Y(t)| - 1 = 2$ simultaneous overlapping voices exist. Thus, $95.0\text{ s} + 3.0\text{ s} = 98.0\text{ s}$.
   - $T_{\text{union}} + T_{\text{excess}} = 186.000\text{ s} + 98.000\text{ s} = 284.000\text{ s}$, which exactly equals total ground-truth speaker-time.

2. **The 32.00% Baseline Miss Rate Root Cause**:
   - In `phase20_standalone/scripts/common.py` line 348, `diarization_metrics()` constructs `pyannote.core.Annotation` using:
     ```python
     ann[Segment(seg["start"], seg["end"])] = seg["speaker"]
     ```
     Because the `track` identifier was omitted, `pyannote.core.Annotation` keys by segment geometry.
   - In `research_stages/stage1_data_prep/annotations_rttm/R04_piers_robinson.rttm`:
     - Line 29: `SPEAKER R04_piers_robinson 1 183.000 11.000 <NA> <NA> Speaker_1 <NA> <NA>`
     - Line 30: `SPEAKER R04_piers_robinson 1 183.000 11.000 <NA> <NA> Speaker_2 <NA> <NA>`
   - Both segments have identical temporal boundaries `[183.000, 194.000]`. Adding Line 30 **silently overwrote** Line 29 in `ann`.
   - Consequently, 11.000 s of Speaker_1 speech was deleted from the reference annotation:
     $$\text{Total Reference Time Evaluated} = 284.000\text{ s} - 11.000\text{ s} = \mathbf{273.000\text{ s}}$$
   - Under this 273.0s reference, baseline C00 missed detection was 87.370 s:
     $$\text{Miss Rate} = \frac{87.370\text{ s}}{273.000\text{ s}} = 0.320037 = \mathbf{32.00\%}$$

3. **Under Corrected Track-Aware Evaluation**:
   - Preserving all tracks with distinct track IDs (`ann[Segment(s, e), f"track_{i}"] = spk`):
     $$\text{Total Reference Time} = \mathbf{284.000\text{ s}}$$
     $$\text{Baseline C00 Missed Detection} = \mathbf{98.370\text{ s}}$$
     $$\text{Corrected Baseline Miss Rate} = \frac{98.370\text{ s}}{284.000\text{ s}} = \mathbf{34.64\%}$$
   - The theoretical minimum single-label miss floor (assuming 100% perfect VAD on all 186.0s union speech) is:
     $$\text{Miss Floor}_{\text{single-label}} = \frac{T_{\text{excess}}}{T_{\text{total}}} = \frac{98.000\text{ s}}{284.000\text{ s}} = \mathbf{34.51\%}$$
   - The actual baseline miss of $34.64\%$ equals the theoretical overlap floor ($34.51\% = 98.00\text{ s}$) plus $0.37\text{ s}$ ($0.13\%$) of genuine VAD silence miss.

---

## 2. Mathematical Formalization & Reference Audit

### 2.1 Quantities and Definitions

Let $Y_k(t) \in \{0, 1\}$ denote the binary activity of speaker $k \in \{1, \dots, K\}$ at continuous time $t \in [0, T_{\text{rec}}]$.
The total instantaneous speaker count is:
$$|Y(t)| = \sum_{k=1}^K Y_k(t)$$

We define four fundamental reference quantities:
1. **Total Reference Speaker-Time ($T_{\text{total}}$)**:
   $$T_{\text{total}} = \sum_{k=1}^K \int_0^{T_{\text{rec}}} Y_k(t)\,dt = \int_0^{T_{\text{rec}}} |Y(t)|\,dt = 284.000\text{ s}$$
2. **Union Speech Duration ($T_{\text{union}}$)**:
   $$T_{\text{union}} = \int_0^{T_{\text{rec}}} \mathbb{1}[|Y(t)| \ge 1]\,dt = 186.000\text{ s}$$
3. **Overlap Presence Duration ($T_{\text{overlap}}$)**:
   $$T_{\text{overlap}} = \int_0^{T_{\text{rec}}} \mathbb{1}[|Y(t)| \ge 2]\,dt = 95.000\text{ s}$$
4. **Excess Overlapping Speaker-Time ($T_{\text{excess}}$)**:
   $$T_{\text{excess}} = \int_0^{T_{\text{rec}}} \max(0, |Y(t)| - 1)\,dt = 98.000\text{ s}$$

### 2.2 Speaker Multiplicity Breakdown in R04
Partitioning the audio timeline by instantaneous speaker count $|Y(t)|$:
- $|Y(t)| = 0$ (Silence / Inactive): $204.545\text{ s} - 186.000\text{ s} = \mathbf{18.545\text{ s}}$
- $|Y(t)| = 1$ (Single active speaker): $186.000\text{ s} - 95.000\text{ s} = \mathbf{91.000\text{ s}}$
- $|Y(t)| = 2$ (Exactly two simultaneous speakers): $95.000\text{ s} - 3.000\text{ s} = \mathbf{92.000\text{ s}}$
- $|Y(t)| = 3$ (Exactly three simultaneous speakers): $\mathbf{3.000\text{ s}}$
- $|Y(t)| \ge 4$: $\mathbf{0.000\text{ s}}$

### 2.3 Exact Duration Arithmetic
Summing the contributions to total speaker-time:
$$T_{\text{total}} = (1 \times 91.000) + (2 \times 92.000) + (3 \times 3.000) = 91.000 + 184.000 + 9.000 = \mathbf{284.000\text{ s}}$$
Summing the contributions to excess speaker-time:
$$T_{\text{excess}} = (0 \times 91.000) + (1 \times 92.000) + (2 \times 3.000) = 92.000 + 6.000 = \mathbf{98.000\text{ s}}$$
Notice:
$$T_{\text{overlap}} = 92.000 + 3.000 = \mathbf{95.000\text{ s}}$$
$$T_{\text{excess}} - T_{\text{overlap}} = 98.000 - 95.000 = \mathbf{3.000\text{ s}} = \int \mathbb{1}[|Y(t)| \ge 3]\,dt$$

---

## 3. Metric Evaluator Audit: The Annotation Overwrite Defect

### 3.1 Defect Mechanism in `common.py`
In `phase20_standalone/scripts/common.py` line 345:
```python
def load(path):
    ann = Annotation()
    for seg in rttm_to_segments(path):
        ann[Segment(seg["start"], seg["end"])] = seg["speaker"]
    return ann
```
In `pyannote.core.Annotation`, indexing with `ann[Segment(start, end)] = speaker` without specifying the second tuple element `track` assigns the segment to the default track `_`. If two records share identical start and end times, the subsequent assignment replaces the previous label.

### 3.2 Evidence from `R04_piers_robinson.rttm`
```text
Line 29: SPEAKER R04_piers_robinson 1 183.000 11.000 <NA> <NA> Speaker_1 <NA> <NA>
Line 30: SPEAKER R04_piers_robinson 1 183.000 11.000 <NA> <NA> Speaker_2 <NA> <NA>
```
Line 29 and Line 30 both represent the segment `Segment(183.000, 194.000)`.
Line 30 overwrote Line 29.
- Dropped duration: $11.000\text{ s}$.
- Reference duration evaluated: $284.000 - 11.000 = 273.000\text{ s}$.

### 3.3 Reconciled Evaluation Comparison

| Metric Component | Legacy Evaluator (`common.py` with overwrite) | Corrected Evaluator (Track-Aware Multi-Label) | Difference |
|---|---|---|---|
| Total Reference Speaker-Time | 273.000 s | 284.000 s | +11.000 s |
| Missed Detection Time (O00) | 87.370 s | 98.370 s | +11.000 s |
| False Alarm Time (O00) | 4.880 s | 4.880 s | 0.000 s |
| Speaker Confusion Time (O00) | 18.010 s | 8.260 s | -9.750 s |
| Correct Speaker Time (O00) | 167.620 s | 177.370 s | +9.750 s |
| **Miss Rate** | **32.00%** ($87.37 / 273.0$) | **34.64%** ($98.37 / 284.0$) | +2.64% |
| **False Alarm Rate** | **1.79%** ($4.88 / 273.0$) | **1.72%** ($4.88 / 284.0$) | -0.07% |
| **Speaker Confusion Rate**| **6.60%** ($18.01 / 273.0$) | **2.91%** ($8.26 / 284.0$) | -3.69% |
| **DER (collar 0.0s)** | **40.39%** ($110.26 / 273.0$) | **39.26%** ($111.51 / 284.0$) | -1.13% |

---

## 4. Rigorous Formulation of the Single-Label Miss Floor

Let $H(t) \in \{0, 1\}^K$ denote any diarization hypothesis.
In single-label diarization, by architectural definition:
$$|H(t)| = \sum_{k=1}^K H_k(t) \le 1 \quad \forall t$$

Let $M(t)$ denote the instantaneous missed speaker-time:
$$M(t) = \max(0, |Y(t)| - |H(t)|)$$

Since $|H(t)| \le 1$:
$$M(t) \ge \max(0, |Y(t)| - 1)$$

Integrating over the full recording duration:
$$T_{\text{miss}} = \int_0^{T_{\text{rec}}} M(t)\,dt \ge \int_0^{T_{\text{rec}}} \max(0, |Y(t)| - 1)\,dt = T_{\text{excess}} = \mathbf{98.000\text{ s}}$$

Dividing by total reference speaker-time $T_{\text{total}} = 284.000\text{ s}$:
$$\text{Miss Rate}_{\text{single-label}} = \frac{T_{\text{miss}}}{T_{\text{total}}} \ge \frac{98.000\text{ s}}{284.000\text{ s}} = \mathbf{34.507\%} \approx \mathbf{34.51\%}$$

### Conclusion on the Miss Floor Claim:
- The statement "single-label diarization is mathematically trapped at an overlap miss floor" is **MATHEMATICALLY VALID AND PROVEN**.
- The exact theoretical lower bound under the true reference is **$34.51\%$** (not $32.00\%$ or $33.45\%$).
- Under the legacy evaluator which lost 11.0s of overlap due to the track overwrite defect, the empirical miss rate was **$32.00\%$** because $T_{\text{excess}}$ was truncated to $87.0\text{ s}$ over $273.0\text{ s}$ ($87.0 / 273.0 = 31.87\%$).
- Both numbers reflect the same underlying physical constraint: single-label models mechanically discard simultaneous speakers.
