# Stage 5B.1: Consensus Alignment & Timestamp Grid Audit

## 1. Objective and Audit Focus

This audit investigates the temporal alignment of the acoustic probability streams used to construct linear consensus Voice Activity Detection (VAD) models (`C52_consensus`, `C61_comb_consensus_hyst`, and `C62_comb_consensus_hyst_gap`).

Specifically, the audit verifies:
1. Exact timestamp equality between `dfn` (`_dfn_posteriors.npz`) and `mono` (`_16k_mono_posteriors.npz`) probability curves.
2. Frame stride and sample rate consistency.
3. Start and end temporal offsets.
4. Maximum absolute timestamp deviation over the arrays.
5. Whether array truncation in the original script (`min_len = min(len(p_dfn), len(p_mono))`) induced temporal distortion or frame dropping.

---

## 2. Empirical Verification Across All 8 Recordings

An exhaustive audit of the raw posterior caches stored in `research_stages/stage2_baseline/posteriors/` yielded the following findings across all 8 canonical recordings:

| Recording ID | DFN Frames | Mono Frames | Frame $\Delta$ | Start DFN (s) | Start Mono (s) | End DFN (s) | End Mono (s) | Mean Stride (s) | Max Timestamp Deviation (s) | Array Equal? |
|---|---|---|---|---|---|---|---|---|---|---|
| **R01_sara_sharif** | 5,983 | 5,983 | 0 | 0.030969 | 0.030969 | 100.977219 | 100.977219 | 0.01687500 | **0.000000e+00** | **True** |
| **R02_dennis_skinner** | 10,446 | 10,446 | 0 | 0.030969 | 0.030969 | 176.290344 | 176.290344 | 0.01687500 | **0.000000e+00** | **True** |
| **R03_disabled_parking** | 12,221 | 12,221 | 0 | 0.030969 | 0.030969 | 206.243469 | 206.243469 | 0.01687500 | **0.000000e+00** | **True** |
| **R04_piers_robinson** | 12,122 | 12,122 | 0 | 0.030969 | 0.030969 | 204.572845 | 204.572845 | 0.01687500 | **0.000000e+00** | **True** |
| **R05_southwest_pilot** | 9,627 | 9,627 | 0 | 0.030969 | 0.030969 | 162.469711 | 162.469711 | 0.01687500 | **0.000000e+00** | **True** |
| **R06_rich_karen** | 38,648 | 38,648 | 0 | 0.030969 | 0.030969 | 652.199097 | 652.199097 | 0.01687500 | **0.000000e+00** | **True** |
| **R07_bwc_cheap_shot** | 10,396 | 10,396 | 0 | 0.030969 | 0.030969 | 175.446594 | 175.446594 | 0.01687500 | **0.000000e+00** | **True** |
| **R08_uk_parliament** | 12,761 | 12,761 | 0 | 0.030969 | 0.030969 | 215.355972 | 215.355972 | 0.01687500 | **0.000000e+00** | **True** |

---

## 3. Analysis and Technical Deductions

1. **Exact Bitwise Equality**:
   Across all 8 benchmark files (spanning 112,204 total evaluated frames), `np.array_equal(t_dfn, t_mono)` is strictly `True`.
   - The maximum absolute timestamp deviation across all frames is **`0.0000000000e+00`**.
   - The frame lengths are identical (`diff = 0`).
   - The starting offsets are identical down to machine precision (`0.030969s`).
   - The end offsets are identical down to machine precision.
2. **Receptive Field and Stride Invariance**:
   The frame stride is precisely constant at $0.01687500$ seconds ($16.875$ ms), which corresponds to the standard Pyannote 3.1 PyanNet convolutional receptive stride (270 samples at 16,000 Hz). Because both `dfn` and `mono` audio files were resampled to exactly 16 kHz and passed through the exact same feature extraction pipeline, their temporal grids are structurally isomorphic.
3. **Array Truncation Guard**:
   In `run_stage5b_1_experiments.py`, the code used:
   ```python
   min_len = min(len(p_dfn), len(p_mono))
   p_joint = 0.5 * p_dfn[:min_len] + 0.5 * p_mono[:min_len]
   ```
   Because `len(p_dfn) == len(p_mono)` for 100% of the recordings, `min_len` was always equal to the full array length. Zero frames were dropped or truncated.
4. **Conclusion**:
   Interpolation or continuous time-grid resampling was unnecessary because the discrete sampling grids are already strictly identical. The consensus stream calculations in `C52`, `C61`, and `C62` are mathematically exact on their temporal grids.
