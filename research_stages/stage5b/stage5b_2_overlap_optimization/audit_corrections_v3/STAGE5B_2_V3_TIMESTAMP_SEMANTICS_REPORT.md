# Stage 5B.2 V3 Timestamp Semantics & Independent Model Validation Report

**Workspace:** `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v3/`  
**Inspected Models & Packages:** `pyannote.audio` 3.3.2, `pyannote.core` 5.0.0, PyanNet SincNet architecture  
**Target Audio:** `R04_piers_robinson_mono_overlap.npz` (204.60s duration, 12,122 frames)  
**Verification Status:** Formally Audited Against Independent Pyannote Reference Implementation.

---

## 1. Objective of Independent Timestamp Validation

Stage 5B.2 V2 introduced frame-bin support:
$$[t_{\text{mid}}(s_i) - S/2, \; t_{\text{mid}}(e_i) + S/2)$$
where $t_{\text{mid}}$ is the frame timestamp from the cached `.npz` file, $S = 16.875\text{ms}$ is the frame stride, and $W = 61.94\text{ms}$ is the physical receptive field.

The purpose of this audit is to:
1. Determine how output frame timestamps in `overlap_cache/` are mathematically defined by the upstream neural model.
2. Verify whether `frame_times` represents the window start, receptive field center, frame-bin center, or another convention.
3. Compare the reconstructed intervals against an independent reference implementation (`pyannote.audio.utils.signal.Binarize`).
4. Quantify exact duration and boundary differences between V1 ($W$-dilation), V2 (frame-bin support), and Pyannote's reference implementation.

---

## 2. Upstream Architecture & Receptive Field Forensic Audit

### 2.1 Model Architecture Inspection
In pyannote-audio 3.x, the segmentation model (`pyannote/segmentation-3.0`) uses PyanNet with a multi-convolution SincNet front-end:

- **Convolution Specifications** (from `pyannote.audio.models.blocks.sincnet.SincNet`):
  - `kernel_size = [251, 3, 5, 3, 5, 3]`
  - `stride = [10, 3, 1, 3, 1, 3]`
  - `padding = [0, 0, 0, 0, 0, 0]`
  - `dilation = [1, 1, 1, 1, 1, 1]`
- **Receptive Field Calculation** (evaluated via `pyannote.audio.utils.receptive_field.multi_conv_receptive_field_size`):
  $$W = 991 \text{ samples at } 16\text{kHz} = 0.0619375\text{s} = 61.9375\text{ms}$$
- **Frame Stride Calculation**:
  $$\text{Stride} = 10 \times 3 \times 1 \times 3 \times 1 \times 3 = 270 \text{ samples at } 16\text{kHz} = 0.0168750\text{s} = 16.875\text{ms}$$
- **Receptive Field Center of Frame 0**:
  $$\text{Center}(0) = \frac{991}{2 \times 16000} = 0.03096875\text{s}$$

### 2.2 Forensic Inspection of Cached `frame_times`
Inspecting `R04_piers_robinson_mono_overlap.npz`:
- `frame_times[0] = 0.03096875`
- `frame_times[1] = 0.04784375`
- `frame_times[2] = 0.06471875`
- Frame step: $\Delta t = 0.04784375 - 0.03096875 = 0.01687500\text{s}$

In [`experiments/phase20_dual_path_improved_vad/scripts/step03_vad.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/experiments/phase20_dual_path_improved_vad/scripts/step03_vad.py#L45-L47):
```python
frames = probabilities.sliding_window
times = np.asarray([frames[i].middle for i in range(len(values))], dtype=float)
```
Where `frames` is an instance of `pyannote.core.SlidingWindow(start=0.0, step=0.016875, duration=0.0619375)`.

**Mathematical Proof:**
`frame_times[i]` represents the **exact midpoint $t_{\text{mid}}(i) = i \times S + W/2$ of the PyanNet receptive field**.

---

## 3. Independent Reference Implementation Comparison

To test whether V2 frame-bin support accurately discretizes the posteriors, we compared three implementations on the exact cached posterior array of `R04_piers_robinson`:

1. **Independent Pyannote Reference**: `pyannote.audio.utils.signal.Binarize(onset=0.40, offset=0.40, min_duration_on=0.10, min_duration_off=0.15)` applied to `SlidingWindowFeature(p_ovl, sw)`.
2. **V2 Frame-Bin Support**: `binarize_overlap_frame_bin_support(frame_times, p_ovl, dur=204.60, threshold=0.40)` setting $[t_{\text{mid}}(s_i) - S/2, t_{\text{mid}}(e_i) + S/2)$.
3. **V1 Receptive Field Dilation**: `binarize_overlap_corrected(p_ovl, dur=204.60, threshold=0.40)` setting $[s_i \times S, e_i \times S + W]$.

### 3.1 Global Quantitative Comparison (R04 Overlap Posteriors)

| Implementation | Number of Segments | Total Detected Overlap Duration | Difference vs Pyannote Reference |
| :--- | :---: | :---: | :---: |
| **Pyannote Reference `Binarize`** | **28** | **68.534 s** | Baseline Reference (0.000 s) |
| **V2 Frame-Bin Support** | **28** | **68.525 s** | **$-0.009 \text{ s}$ ($-9\text{ms}$ across entire recording)** |
| **V1 Receptive Field Dilation** | **29** | **70.139 s** | **$+1.605 \text{ s}$ (+1,605ms dilation error)** |

### 3.2 Segment-Level Boundary Inspection (First 5 Segments)

```
Segment 1:
  Pyannote Reference: [14.611s, 16.248s] (duration = 1.637s)
  V2 Frame-Bin:       [14.603s, 16.239s] (duration = 1.636s)
  V1 Dilation:        [14.580s, 16.262s] (duration = 1.682s, +45ms dilated)

Segment 2:
  Pyannote Reference: [21.850s, 26.002s] (duration = 4.152s)
  V2 Frame-Bin:       [21.842s, 25.993s] (duration = 4.151s)
  V1 Dilation:        [21.819s, 26.016s] (duration = 4.197s, +45ms dilated)

Segment 3:
  Pyannote Reference: [37.797s, 38.388s] (duration = 0.591s)
  V2 Frame-Bin:       [37.789s, 38.379s] (duration = 0.590s)
  V1 Dilation:        [37.766s, 38.402s] (duration = 0.636s, +45ms dilated)

Segment 4:
  Pyannote Reference: [39.603s, 46.387s] (duration = 6.784s)
  V2 Frame-Bin:       [39.594s, 46.378s] (duration = 6.784s)
  V1 Dilation:        [39.572s, 46.401s] (duration = 6.829s, +45ms dilated)

Segment 5:
  Pyannote Reference: [51.128s, 52.495s] (duration = 1.367s)
  V2 Frame-Bin:       [51.120s, 52.487s] (duration = 1.367s)
  V1 Dilation:        [51.098s, 52.509s] (duration = 1.411s, +45ms dilated)
```

---

## 4. Analysis of Boundary Semantics

### 4.1 Why Duration Matches Exactly
For an active run of frames from index $s_i$ to $e_i$ ($N = e_i - s_i + 1$ active frames):
- **Pyannote `Binarize`**:
  Sets start at $t_{\text{mid}}(s_i)$ and terminates when the next frame $e_i + 1$ falls below threshold, setting end at $t_{\text{mid}}(e_i + 1)$.
  $$\text{Duration} = t_{\text{mid}}(e_i + 1) - t_{\text{mid}}(s_i) = (e_i + 1 - s_i) \times S = N \times S$$
- **V2 Frame-Bin Support**:
  Centers the support of each frame around its midpoint $[t_{\text{mid}} - S/2, t_{\text{mid}} + S/2)$:
  $$\text{Duration} = (t_{\text{mid}}(e_i) + S/2) - (t_{\text{mid}}(s_i) - S/2) = (e_i - s_i + 1) \times S = N \times S$$
- Both methods yield the exact physical duration: $N \times 16.875\text{ms}$.

### 4.2 Boundary Offset
- In Pyannote `Binarize`:
  $$\text{Start} = t_{\text{mid}}(s_i) = s_i \times S + W/2$$
- In V2 Frame-Bin:
  $$\text{Start} = t_{\text{mid}}(s_i) - S/2 = s_i \times S + (W - S)/2$$
- The boundary offset is:
  $$\Delta t_{\text{offset}} = S/2 = \frac{16.875\text{ms}}{2} = 8.4375\text{ms}$$
- In speech diarization, standard evaluation collars are $0.25\text{s} = 250\text{ms}$, and even with $collar = 0.0$, $8.44\text{ms}$ is smaller than a single 10ms evaluation frame.

---

## 5. Formal Audit Classification

Based on independent inspection of the PyanNet architecture and comparison with `pyannote.audio.utils.signal.Binarize`:

> **CLASSIFICATION**: The V2 timestamp convention is **INDEPENDENTLY SUPPORTED** by PyanNet sliding window midpoint semantics. It correctly decouples the receptive field ($W = 61.94\text{ms}$) from the label support step ($S = 16.875\text{ms}$), producing segment durations that match Pyannote's reference implementation to within $9\text{ms}$ over 204 seconds. The boundaries are centered on the frame bins $[t_{\text{mid}} - S/2, t_{\text{mid}} + S/2)$, which is shifted $8.44\text{ms}$ earlier than Pyannote's $[t_{\text{mid}}, t_{\text{mid}} + S)$ convention. Both conventions reject the V1 receptive field dilation error.
