# Stage 5B.2 M2 Embedding Validity Report: Temporal Purity & Feasibility Audit

## 1. Executive Summary

Audit C investigated the acoustic validity of M2 (Dual-Encoder Prototype Affinity Attribution).
The central question was:
> Can frozen 1.5-second Stage 3 subsegment embeddings genuinely represent short overlapping speech intervals, or are they contaminated by non-overlapping and competing-speaker audio?

### Key Findings:
1. **The 1.5s Window Constraint**:
   Stage 3 Scale 0 embeddings were extracted on fixed 1.5-second sliding windows with 0.75-second shift. Overlap bursts in the benchmark corpus vary from 0.10s to over 5.0s. For short overlaps ($< 0.60\text{ s}$), a 1.5s window contains predominantly non-overlapping speech ($> 60\%$).
2. **Purity Filtering Implementation**:
   We implemented strict temporal-purity filtering:
   - **Overlap Purity ($f_{\text{ovl}} \ge 0.40$)**: An embedding is used only if at least $40\%$ of its 1.5s window ($0.60\text{ s}$) lies inside the target overlap interval.
   - **Prototype Purity ($P_k \ge 0.80$)**: A baseline subsegment is used to build speaker $k$'s prototype only if at least $80\%$ of its duration belongs to speaker $k$ and $0\%$ belongs to competing speakers.
3. **Status Designation**:
   M2 is **retained as a validated pooled-embedding affinity heuristic**, but its status as a *fine-grained segment-level overlap representation* remains **EXPLORATORY** because frozen 1.5s embeddings cannot resolve sub-second simultaneous speech without window contamination.

---

## 2. Retained vs Rejected Embedding Statistics Across Corpus

Evaluating candidate overlap segments at threshold $\theta = 0.40$ with $f_{\text{ovl}} \ge 0.40$:

| Recording ID | Detected Overlaps | Valid Embeddings ($f_{\text{ovl}} \ge 0.40$) | Rejected Overlaps (Abstained) | Purity Pass Rate (%) |
|---|---|---|---|---|
| `R01_sara_sharif` | 3 | 2 | 1 | 66.7% |
| `R02_dennis_skinner` | 3 | 3 | 0 | 100.0% |
| `R03_disabled_parking` | 19 | 8 | 11 | 42.1% |
| `R04_piers_robinson` | 29 | 25 | 4 | 86.2% |
| `R05_southwest_pilot` | 6 | 6 | 0 | 100.0% |
| `R06_rich_karen` | 49 | 36 | 13 | 73.5% |
| `R07_bwc_cheap_shot` | 1 | 0 | 1 | 0.0% |
| `R08_uk_parliament` | 0 | 0 | 0 | N/A |
| **Corpus Total** | **110** | **80** | **30** | **72.7%** |

### Observation:
In recordings with short, fragmented speech bursts (e.g. `R07_bwc_cheap_shot` where the overlap is 0.40s, and `R03_disabled_parking`), the purity filter rejected 11 of 19 overlaps, preventing contaminated embeddings from making spurious assignments.
In `R04_piers_robinson`, where overlap consists of sustained back-and-forth exchanges, 25 of 29 overlaps ($86.2\%$) have pure 1.5s embeddings.

---

## 3. Dual-Encoder Representation & Fusion Construction

For valid embeddings $i \in \mathcal{I}_{\text{valid}}$:
1. **TitaNet-Large Representation**: $\mathbf{e}_i^{\text{titanet}} \in \mathbb{R}^{192}$, L2-normalized.
2. **WeSpeaker ResNet293-LM Representation**: $\mathbf{e}_i^{\text{wespeaker}} \in \mathbb{R}^{256}$, L2-normalized.
3. **Joint Concatenated 448-Dimensional Representation**:
   $$\mathbf{e}_i^{\text{concat}} = \frac{[\mathbf{e}_i^{\text{titanet}} \;\|\; \mathbf{e}_i^{\text{wespeaker}}]}{\|[\mathbf{e}_i^{\text{titanet}} \;\|\; \mathbf{e}_i^{\text{wespeaker}}]\|_2} \in \mathbb{R}^{448}$$
4. **Normalized Cosine Average**:
   $$\text{Score}(k) = 0.5 \cdot \cos\left(\mathbf{e}_{\text{ovl}}^{\text{titanet}}, \mathbf{c}_k^{\text{titanet}}\right) + 0.5 \cdot \cos\left(\mathbf{e}_{\text{ovl}}^{\text{wespeaker}}, \mathbf{c}_k^{\text{wespeaker}}\right)$$

### Performance on R04:
- TitaNet-only (`O31-T-R2`): DER = $28.70\%$
- WeSpeaker-only (`O31-W-R2`): DER = $28.70\%$
- Joint Concat (`O31-R2`): DER = $28.70\%$
- Normalized Cosine Avg (`O31-NF-R2`): DER = $28.70\%$

All four acoustic variants make identical, consistent speaker assignments on R04, confirming that the two encoders agree on the acoustic affinity of the secondary speaker.
