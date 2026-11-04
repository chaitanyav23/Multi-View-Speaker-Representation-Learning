# Stage 5B.2 Oracle Mapping Report: Multi-Label Hungarian Alignment

## 1. Executive Summary

In multi-speaker conversational audio, overlapping speech intervals inherently contain multiple ground-truth speaker labels simultaneously active at the exact same physical timestamps.
Previous implementations discretized timelines into 1D scalar arrays:
```python
ref_frames[i] = s["speaker"]
```
When two speakers overlapped, the second segment in the RTTM file overwrote the first segment in `ref_frames[i]`. This corrupted the bipartite co-occurrence matrix, introducing order-dependency and dropping speaker co-occurrence evidence.

---

## 2. Multi-Label Formulation

To evaluate oracle diagnostics rigorously:

1. **Active Speaker Sets**:
   At each frame $i$ (resolution $\Delta t = 0.01\text{ s}$), the active reference and hypothesis speakers are stored as sets:
   $$\mathcal{R}_i = \{s \in \mathcal{K}_{\text{ref}} \mid \text{Speaker } s \text{ active at frame } i\}$$
   $$\mathcal{H}_i = \{h \in \mathcal{K}_{\text{hyp}} \mid \text{Cluster } h \text{ active at frame } i\}$$
2. **Co-occurrence Accumulation**:
   For each frame $i$ where $\mathcal{R}_i \neq \emptyset$ and $\mathcal{H}_i \neq \emptyset$:
   $$C[r, h] = \sum_{i=1}^{N_{\text{frames}}} \mathbb{1}[r \in \mathcal{R}_i] \cdot \mathbb{1}[h \in \mathcal{H}_i]$$
3. **One-to-One Hungarian Assignment**:
   The optimal assignment $\pi^* = \arg\max_\pi \sum_{r} C[r, \pi(r)]$ is solved using `scipy.optimize.linear_sum_assignment` on $-\mathbf{C}$.
4. **Handling Unequal Counts**:
   For $K_{\text{ref}} \neq K_{\text{hyp}}$, unmatched reference speakers or unmatched hypothesis clusters are explicitly isolated in `unmatched_refs` and `unmatched_hyps`.

---

## 3. Oracle Diagnostics Evaluation Results

### Diagnostic Definitions:
- **Diagnostic B (M5B-R2)**: Model-predicted overlap intervals ($p_{\text{ovl}} \ge 0.40$), with secondary speaker chosen from the true ground-truth active speakers mapped via Hungarian alignment.
- **Diagnostic C (M5C-R2)**: Full Oracle Overlap ($|Y(t)| \ge 2$) with true ground-truth secondary speaker mapped via Hungarian alignment.

### Empirical Upper Bounds:

| Configuration | Description | Macro DER (c=0) | Macro Miss | Macro FA | Macro Conf | Macro JER | R04 DER (c=0) | R04 Miss |
|---|---|---|---|---|---|---|---|---|
| **O00** | Baseline Single-Label | 46.42% | 25.37% | 13.30% | 7.75% | 62.28% | 39.26% | 34.64% |
| **O34-R2** | Diagnostic B (Pred Ovl + Hungarian Oracle) | **44.44%** | 22.53% | 14.24% | 7.67% | 59.17% | **23.76%** | **13.53%** |
| **O35-R2** | Diagnostic C (Full Oracle Ovl + Hungarian Oracle) | **41.14%** | 18.53% | 13.30% | 9.31% | 57.16% | **13.56%** | **1.25%** |

### Insights:
1. When secondary attribution is performed with oracle accuracy on predicted overlap (`O34-R2`), R04 DER falls from $39.26\%$ down to **$23.76\%$** ($-15.50\%$ absolute gain).
2. When both overlap detection and secondary attribution are perfect (`O35-R2`), R04 DER collapses to **$13.56\%$**, with Miss dropping to a near-zero **$1.25\%$**.
3. This proves that the multi-speaker segmentation model's detections are acoustic genuine, and secondary speaker attribution has a theoretical headroom of up to $-25.70\%$ DER reduction on high-overlap conversational audio.
