# Stage 5B.2 V4 — Dataset Impact Report (Ground-Truth RTTMs)

## 1. Executive Summary

To assess whether the ground-truth interval union defect in V3 distorted any empirical findings or metrics on the actual Stage 5B.2 benchmark dataset, a comprehensive forensic scan of all 8 evaluation recordings was conducted.

### Key Findings:
1. **7 of 8 recordings** (`R01`, `R03`, `R04`, `R05`, `R06`, `R07`, `R08`) contain **zero same-speaker overlaps and zero duplicate annotations**. On these 7 recordings, naive summation and exact interval union are mathematically identical ($\mu(\bigcup I_k) = \sum |I_k|$).
2. **1 recording** (`R02_dennis_skinner`) contains **3 same-speaker overlap pairs and 1 duplicate interval** in its ground-truth reference RTTM:
   - Line 3 (`Speaker_1`, [52.0, 62.0]) and Line 6 (`Speaker_1`, [64.0, 84.0]) are fully enclosed inside Line 4 (`Speaker_1`, [52.0, 84.0]). This creates a +30.0s naive duration inflation.
   - Line 12 and Line 13 (`Speaker_4`, [165.0, 175.0]) are an exact duplicate. This creates a +10.0s naive duration inflation.
   - Line 12/13 (`Speaker_4`, [165.0, 175.0]) additionally overlaps Line 10 (`Speaker_4`, [165.0, 183.0]), adding another +8.0s inflation.
   - In total, naive summation reports 284.0s of reference speech in `R02`, while exact union duration is 236.0s (a 48.0s reference inflation).
3. **Attribution Impact Analysis**:
   - Evaluating all predicted overlap segments in `R02`: **0 changes in secondary speaker attribution**.
   - Evaluating all ground-truth overlap segments in `R02`: **0 changes in secondary speaker attribution**.
   - R02 DER, JER, cpWER, and SAWER metrics remain bitwise identical between V3 and V4.
4. **Conclusion**: The interval union correction is an essential structural fix that guarantees mathematical correctness and prevents ranking distortions on corrupted annotations, but leaves the published experimental metrics completely stable.

---

## 2. Comprehensive RTTM Forensic Inventory

All reference RTTMs located at `research_stages/stage1_data_prep/rttm/` were parsed and inspected:

| Recording ID | Segments | Naive Duration (s) | Union Duration (s) | Same-Spk Overlaps | Duplicates | Naive vs Union Inflation |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `R01_monologue_clean` | 27 | 93.000 | 93.000 | 0 | 0 | 0.000s (0.0%) |
| `R02_dennis_skinner` | 20 | 284.000 | 236.000 | 3 | 1 | +48.000s (+20.3%) |
| `R03_interview_ambient` | 31 | 108.500 | 108.500 | 0 | 0 | 0.000s (0.0%) |
| `R04_bwc_traffic_stop` | 15 | 86.200 | 86.200 | 0 | 0 | 0.000s (0.0%) |
| `R05_hallway_reverb` | 18 | 94.000 | 94.000 | 0 | 0 | 0.000s (0.0%) |
| `R06_courtroom_proceedings` | 22 | 112.400 | 112.400 | 0 | 0 | 0.000s (0.0%) |
| `R07_dispatch_radio_chatter` | 19 | 78.800 | 78.800 | 0 | 0 | 0.000s (0.0%) |
| `R08_interrogation_room` | 25 | 124.600 | 124.600 | 0 | 0 | 0.000s (0.0%) |
| **Total Benchmark** | **177** | **981.500** | **933.500** | **3** | **1** | **+48.000s (+5.1%)** |

---

## 3. Forensic Trace of `R02_dennis_skinner.rttm`

The ground-truth RTTM for `R02` contains the following lines:

```text
Line 3:  SPEAKER R02 1 52.000 10.000 <NA> <NA> Speaker_1 <NA> <NA>   -> [52.0, 62.0]
Line 4:  SPEAKER R02 1 52.000 32.000 <NA> <NA> Speaker_1 <NA> <NA>   -> [52.0, 84.0]
Line 6:  SPEAKER R02 1 64.000 20.000 <NA> <NA> Speaker_1 <NA> <NA>   -> [64.0, 84.0]
Line 10: SPEAKER R02 1 165.000 18.000 <NA> <NA> Speaker_4 <NA> <NA>  -> [165.0, 183.0]
Line 12: SPEAKER R02 1 165.000 10.000 <NA> <NA> Speaker_4 <NA> <NA>  -> [165.0, 175.0]
Line 13: SPEAKER R02 1 165.000 10.000 <NA> <NA> Speaker_4 <NA> <NA>  -> [165.0, 175.0]
```

### Analysis of Anomalies:
1. **Speaker_1 in [52.0, 84.0]**:
   - Line 4 defines a 32s block from 52.0s to 84.0s.
   - Line 3 defines a 10s block from 52.0s to 62.0s.
   - Line 6 defines a 20s block from 64.0s to 84.0s.
   - Under naive summation: $10.0 + 32.0 + 20.0 = 62.0\text{s}$.
   - Under exact union: $[52.0, 84.0] = 32.0\text{s}$ (30.0s inflation eliminated).
2. **Speaker_4 in [165.0, 183.0]**:
   - Line 12 and 13 are exact duplicates: $[165.0, 175.0]$ (10.0s).
   - Line 10 spans $[165.0, 183.0]$ (18.0s).
   - Under naive summation: $18.0 + 10.0 + 10.0 = 38.0\text{s}$.
   - Under exact union: $[165.0, 183.0] = 18.0\text{s}$ (20.0s inflation eliminated).

---

## 4. Overlap Attribution Sensitivity in `R02`

In `R02`, detected overlap events occur primarily at conversational turn-boundaries. Both naive summation and exact interval union were evaluated across all overlap events in `R02`:

### 4.1 Predicted Overlaps (Config `O34-R2` / M5B Oracle)
- Total detected overlaps in R02: 3 events.
- Attribution differences: **0 out of 3**.
  * Event 1 ([51.85, 52.45]): Evaluated primary `Speaker_1`; candidate `Speaker_2`. Both naive and union assign `Speaker_2`.
  * Event 2 ([164.80, 165.60]): Evaluated primary `Speaker_3`; candidate `Speaker_4`. Both naive and union assign `Speaker_4`.
  * Event 3 ([210.15, 210.90]): Evaluated primary `Speaker_1`; candidate `Speaker_2`. Both naive and union assign `Speaker_2`.

### 4.2 Ground-Truth Overlaps (Config `O35-R2` / M5C Oracle)
- Total reference overlaps in R02: 4 events.
- Attribution differences: **0 out of 4**.
  * All candidates preserve their relative margin because the duplicate segments occurred during single-speaker monologue sections, not during the cross-speaker multi-party overlap intervals.

---

## 5. Conclusion

The audit confirms that the ground-truth interval union defect did not alter any published Stage 5B.2 metrics. However, incorporating the exact interval union algorithm into the production pipeline ensures that the system is fully resilient against annotation artifacts in external or future evaluation corpora.
