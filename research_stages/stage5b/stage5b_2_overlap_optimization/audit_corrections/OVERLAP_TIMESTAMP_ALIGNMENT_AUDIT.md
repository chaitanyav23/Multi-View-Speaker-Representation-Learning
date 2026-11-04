# Audit B: Overlap Posterior-to-Interval Conversion & Timestamp Alignment

## 1. Executive Summary

Audit B evaluated the temporal fidelity of the overlap posterior binarization pipeline.
The previous implementation in `binarize_overlap()` extracted timestamps from `feature.sliding_window[i].middle` and treated these center points directly as the start and end of detected speech segments.

### Key Audit Findings:
1. **Timestamp Convention**:
   The cached array `frame_times` in `overlap_cache/*.npz` contains frame **centers** ($t_{\text{mid}} = t_{\text{start}} + \frac{W}{2}$), NOT frame boundaries.
2. **Model Geometry**:
   The Pyannote PyanNet segmentation architecture (`pytorch_model.bin`) operates with:
   - Receptive field window duration $W = \mathbf{0.0619375\text{ s}}$ (61.94 ms).
   - Sliding window step/stride $S = \mathbf{0.0168750\text{ s}}$ (16.875 ms, exactly 270 samples at 16,000 Hz).
   - Frame 0 bounds: $[0.000000, 0.0619375]\text{ s}$, with midpoint $t_0 = 0.03096875\text{ s}$.
3. **Temporal Truncation in Legacy Binarizer**:
   Treating frame centers as interval boundaries introduced a systematic under-coverage artifact of approximately $31\text{ ms}$ at each boundary (~$62\text{ ms}$ per isolated burst). Furthermore, if overlap extended to the recording boundary, the legacy code terminated the interval at `frame_times[-1]`, dropping the final $31\text{ ms}$ of the recording.
4. **VAD vs Overlap Grid Alignment**:
   - Upstream VAD (Stage 5B.1) operates on a uniform $10.0\text{ ms}$ grid ($\Delta t = 0.010\text{ s}$).
   - PyanNet Overlap operates on a $16.875\text{ ms}$ grid ($S = 0.016875\text{ s}$).
   - Because both systems output time intervals on continuous physical seconds $[t_{\text{start}}, t_{\text{end}}]$, temporal alignment is properly achieved via continuous interval intersection rather than direct index-to-index array truncation.

---

## 2. Model Frame Geometry Analysis

| Parameter | Pyannote PyanNet Model Spec | Physical Meaning |
|---|---|---|
| Sample Rate ($f_s$) | 16,000 Hz | Input audio sampling rate |
| Stride / Hop ($S$) | 0.016875 s | 270 audio samples per output frame |
| Receptive Field ($W$) | 0.0619375 s | 991 audio samples per temporal output neuron |
| Offset ($t_0$) | 0.03096875 s | Midpoint of the initial window $[0.0, 0.0619375]$ |
| Frame $i$ Support | $[i \cdot S, \; \min(T_{\text{rec}}, i \cdot S + W)]$ | Exact physical temporal extent |

### Comparison Across All 8 Benchmark Recordings

Evaluating legacy vs corrected interval reconstruction at $\theta_{\text{ovl}} = 0.40$:

| Recording ID | Total Frames | Total Duration | Legacy Overlap Sec | Corrected Overlap Sec | Temporal Discrepancy ($\Delta$) | Segment Count Change |
|---|---|---|---|---|---|---|
| `R01_sara_sharif` | 5,983 | 100.97 s | 2.19 s | 2.48 s | +0.29 s | 3 -> 3 |
| `R02_dennis_skinner` | 10,446 | 176.28 s | 6.11 s | 6.20 s | +0.09 s | 3 -> 3 |
| `R03_disabled_parking` | 12,221 | 206.24 s | 13.43 s | 14.29 s | +0.86 s | 19 -> 19 |
| `R04_piers_robinson` | 12,122 | 204.54 s | 68.50 s | 70.14 s | +1.64 s | 28 -> 29 |
| `R05_southwest_pilot` | 9,627 | 162.47 s | 5.11 s | 5.55 s | +0.44 s | 5 -> 6 |
| `R06_rich_karen` | 38,648 | 652.21 s | 50.39 s | 52.99 s | +2.60 s | 47 -> 49 |
| `R07_bwc_cheap_shot` | 10,396 | 175.44 s | 0.40 s | 0.45 s | +0.05 s | 1 -> 1 |
| `R08_uk_parliament` | 12,761 | 215.35 s | 0.00 s | 0.00 s | 0.00 s | 0 -> 0 |

---

## 3. Corrected Algorithm: Physical Support Binarization

The corrected conversion `binarize_overlap_corrected()` enforces the following rules:

1. **Frame Index Clustering**:
   Contiguous frames where $p_{\text{overlap}}[i] \ge \theta$ are identified as integer index runs $[s_i, e_i]$.
2. **Temporal Support Mapping**:
   The continuous physical interval is computed using the true window geometry:
   $$t_{\text{start}} = \max(0.0, s_i \times S)$$
   $$t_{\text{end}} = \min(T_{\text{rec}}, e_i \times S + W)$$
3. **Gap Merging**:
   Any inactive gap between adjacent runs shorter than `min_duration_off` (default: 0.15 s) is filled before duration filtering.
4. **Minimum Duration Thresholding**:
   Merged intervals with $(t_{\text{end}} - t_{\text{start}}) < \text{min\_duration\_on}$ (default: 0.10 s) are pruned.
5. **Boundary Safety**:
   Intervals are strictly clamped to $[0.0, T_{\text{rec}}]$. No synthetic frames or artificial zero-padding shifts are introduced.
