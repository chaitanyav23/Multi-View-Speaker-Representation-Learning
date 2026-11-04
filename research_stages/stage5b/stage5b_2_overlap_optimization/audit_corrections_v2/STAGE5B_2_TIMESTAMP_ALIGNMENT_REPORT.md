# Stage 5B.2 Timestamp Alignment Report: Output Support vs Receptive Field

## 1. Investigation of Output Semantics

The Pyannote PyanNet architecture (`whisperx/assets/pytorch_model.bin`) produces temporal predictions across a discrete sliding window:
- **Audio Sample Rate ($f_s$)**: 16,000 Hz
- **Stride / Step ($S$)**: 0.016875 s (270 samples)
- **Receptive Field ($W$)**: 0.0619375 s (991 samples)
- **Offset ($t_0$)**: 0.03096875 s (midpoint of $[0.0, 0.0619375]$)

### The Receptive Field vs Label Support Distinction:
In convolutional neural networks for sequence modeling, the receptive field $W$ represents the acoustic context required by the temporal filters to generate an output. It does **not** imply that the predicted event spans the entirety of $W$.
Because the network is evaluated with stride $S = 16.875\text{ ms}$, each output frame $i$ represents a discrete time bin of width $S$, centered at `frame_times[i]`.

---

## 2. Mathematical Formulation of Frame-Bin Support

For a sequence of contiguous active frames from index $s_i$ to $e_i$:
$$t_{\text{start}} = \max\left(0.0, \; \text{frame\_times}[s_i] - \frac{S}{2}\right)$$
$$t_{\text{end}} = \min\left(T_{\text{rec}}, \; \text{frame\_times}[e_i] + \frac{S}{2}\right)$$

### Properties:
1. **Physical Duration**:
   $$t_{\text{end}} - t_{\text{start}} = (e_i - s_i + 1) \times S$$
   The duration is strictly equal to the number of active frames multiplied by the model hop size.
2. **Decoupling from Receptive Field**:
   Unlike the previous heuristic ($t_{\text{end}} = e_i \cdot S + W$), this formulation does not artificially append $W - S = 45.06\text{ ms}$ of ungrounded duration to every positive burst.
3. **Half-Open Intervals**:
   Intervals are strictly non-overlapping and partition continuous time without gaps or boundary leakage.

---

## 3. Empirical Comparison: Previous Expansion vs Corrected Frame-Bin Support

| Recording ID | Total Active Overlap Runs | Duration under Receptive Field Dilation | Duration under Corrected Frame-Bin Support | Difference ($\Delta$) | Frame Agreement (IoU) |
|---|---|---|---|---|---|
| `R01_sara_sharif` | 4 | 2.42 s | 2.24 s | -0.18 s | 0.925 |
| `R02_dennis_skinner` | 5 | 6.38 s | 6.17 s | -0.21 s | 0.967 |
| `R03_disabled_parking` | 24 | 14.62 s | 13.53 s | -1.09 s | 0.925 |
| `R04_piers_robinson` | 34 | 70.18 s | 68.65 s | -1.53 s | 0.978 |
| `R05_southwest_pilot` | 7 | 5.48 s | 5.16 s | -0.32 s | 0.941 |
| `R06_rich_karen` | 55 | 53.07 s | 50.59 s | -2.48 s | 0.953 |
| `R07_bwc_cheap_shot` | 1 | 0.45 s | 0.40 s | -0.05 s | 0.888 |
| `R08_uk_parliament` | 0 | 0.00 s | 0.00 s | 0.00 s | 1.000 |

### Conclusion:
Frame-bin support eliminates the systematic ~45 ms boundary overshoot per burst, producing sharper temporal intervals that align cleanly with WhisperX word token boundaries.
