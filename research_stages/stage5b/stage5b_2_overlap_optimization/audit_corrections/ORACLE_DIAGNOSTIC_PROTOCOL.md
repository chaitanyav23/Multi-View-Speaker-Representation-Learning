# Audit F: Oracle Diagnostic Protocol & Hungarian Alignment

## 1. Executive Summary & Defect Identification

In the original Stage 5B.2 script, the oracle diagnostic M5 was implemented as:
```python
elif attribution_method == "M5_oracle":
    if gt_segs:
        gt_active = [s["speaker"] for s in gt_segs if s["start"] <= mid <= s["end"]]
        if len(gt_active) >= 2:
            sec_spk = f"oracle_{gt_active[1]}"
```
This implementation had two fatal defects:
1. **Unmapped Foreign Identity**: Prepending `"oracle_"` created a brand-new speaker string (e.g. `oracle_Speaker_1`) that never existed in the hypothesis speaker set. Pyannote's DER metric treated this as an extra, unclustered speaker, producing massive speaker confusion instead of demonstrating an upper bound. In R04, this caused the "oracle" DER to deteriorate from $40.39\%$ to $43.53\%$!
2. **Conflated Diagnostic Dimensions**: The script evaluated predicted overlap with unmapped oracle labels, without separating:
   - Oracle overlap detection vs predicted overlap detection.
   - Oracle attribution vs predicted attribution.

---

## 2. Formalization of Oracle Diagnostics

To isolate upstream detection errors from downstream attribution errors, three distinct oracle diagnostics are established:

### Diagnostic A: Oracle Overlap Detection + Predicted Model Attribution (M5A)
- **Overlap Intervals**: Exact ground-truth overlap segments where $|Y(t)| \ge 2$.
- **Attribution**: Deployable model attribution (e.g. M1 or M3).
- **Purpose**: Evaluates how well the deployable attribution strategy performs if the overlap detector were 100% perfect (Precision = 1.0, Recall = 1.0).

### Diagnostic B: Predicted Overlap Detection + Oracle Secondary Attribution (M5B)
- **Overlap Intervals**: Model-predicted overlap segments ($p_{\text{ovl}} \ge \theta$).
- **Attribution**: True secondary ground-truth speaker active at $t_{\text{mid}}$, mapped to hypothesis cluster space.
- **Purpose**: Evaluates the maximum achievable gain from the existing overlap detector if secondary attribution were perfect.

### Diagnostic C: Full Oracle Diarization Diagnostic (M5C)
- **Overlap Intervals**: Exact ground-truth overlap segments ($|Y(t)| \ge 2$).
- **Attribution**: True secondary ground-truth speaker active at $t_{\text{mid}}$, mapped to hypothesis cluster space.
- **Purpose**: Establishes the true theoretical upper bound of adding secondary speaker turns to the baseline single-label hypothesis.

---

## 3. Hungarian Hypothesis-Space Speaker Mapping

To ensure oracle speaker labels map into the hypothesis cluster space without creating artificial speaker confusion:

1. **Contingency Matrix Construction**:
   On the baseline single-label diarization hypothesis, construct the joint frame-level co-occurrence matrix $\mathbf{C} \in \mathbb{R}^{K_{\text{ref}} \times K_{\text{hyp}}}$:
   $$C_{i, j} = \int_0^{T_{\text{rec}}} \mathbb{1}[Y_{\text{ref}}(t) = i] \cdot \mathbb{1}[H_{\text{base}}(t) = j]\,dt$$

2. **Optimal Bijective Assignment**:
   Solve the maximum-weight bipartite matching problem via the Hungarian algorithm (scipy `linear_sum_assignment` on $-\mathbf{C}$):
   $$\pi^* = \arg\max_\pi \sum_{i=1}^{K_{\text{ref}}} C_{i, \pi(i)}$$
   Yielding mapping function $\mathcal{M}(g) = h$.

3. **Oracle Label Assignment**:
   If ground-truth overlap contains active speakers $\{g_1, g_2, \dots\}$:
   - Identify which ground-truth speaker corresponds to the primary hypothesis speaker:
     $$g_{\text{primary}} = \arg\min_g \text{distance}(\mathcal{M}(g), k_{\text{primary}})$$
   - The secondary oracle speaker is the remaining active ground-truth speaker $g_{\text{sec}} \neq g_{\text{primary}}$.
   - Assign the secondary hypothesis speaker as:
     $$s^* = \mathcal{M}(g_{\text{sec}})$$
   If no valid mapping exists or no second ground-truth speaker is active at that frame $\implies$ **ABSTAIN**.
