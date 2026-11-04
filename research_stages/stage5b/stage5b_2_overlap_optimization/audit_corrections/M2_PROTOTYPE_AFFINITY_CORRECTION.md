# Audit C: Repair of M2 Dual-Encoder Prototype Affinity Attribution

## 1. Executive Summary & Defect Trace

The Stage 5B.2 protocol documented M2 as a **Dual-Encoder Prototype Affinity Attribution** method that supposedly used TitaNet-Large (192d) and WeSpeaker ResNet293-LM (256d) 448-dimensional embeddings to compute speaker prototypes and assign secondary speakers via cosine similarity.

However, an independent code review of `run_stage5b_2_experiments.py` discovered that while `compute_speaker_prototypes()` was executed, the attribution path in `attribute_secondary_speakers()` **never computed cosine similarity between the overlap segment and the prototypes**:
```python
elif attribution_method == "M2_prototype_affinity":
    alt_spks = [s for s in prototypes.keys() if s != primary_spk]
    if alt_spks:
        if len(alt_spks) == 1:
            sec_spk = alt_spks[0]
        else:
            # Pick highest similarity to surrounding turns
            counts = {s: 0 for s in alt_spks}
            for s in frame_spk[ctx_st:ctx_en]:
                if s in counts:
                    counts[s] += 1
            best_alt = max(counts.items(), key=lambda x: x[1])
            ...
```
The original implementation literally counted frame occurrences of speakers in the temporal window `frame_spk[ctx_st:ctx_en]`, exactly replicating M1, and then fell back to `global_alts[0]`. Consequently, the original O31 results were invalid as evidence for prototype-affinity attribution.

---

## 2. Corrected Specification for M2-R1

The corrected implementation enforces genuine mathematical prototype affinity using existing frozen Stage 3 multi-scale subsegment embeddings:

### 2.1 Speaker Prototype Construction
For each recording, we utilize the baseline single-label diarization segments $\mathcal{S}_{\text{base}}$ and the frozen Scale 0 (1.5s subsegments with 0.75s shift) dual-encoder embeddings:
- $\mathbf{e}_i^{\text{titanet}} \in \mathbb{R}^{192}$, L2-normalized.
- $\mathbf{e}_i^{\text{wespeaker}} \in \mathbb{R}^{256}$, L2-normalized.
- Joint representation $\mathbf{e}_i^{\text{joint}} = \frac{[\mathbf{e}_i^{\text{titanet}} \,\|\, \mathbf{e}_i^{\text{wespeaker}}]}{\|[\mathbf{e}_i^{\text{titanet}} \,\|\, \mathbf{e}_i^{\text{wespeaker}}]\|_2} \in \mathbb{R}^{448}$.

Each subsegment $i$ is assigned to baseline cluster $k$ if its midpoint $t_{\text{mid}, i}$ falls within a baseline segment of speaker $k$.
The prototype centroid $\mathbf{c}_k$ is:
$$\mathbf{c}_k = \frac{\sum_{i \in \mathcal{C}_k} w_i \mathbf{e}_i}{\left\|\sum_{i \in \mathcal{C}_k} w_i \mathbf{e}_i\right\|_2}$$
where $w_i = 1.0$ (or duration weight). Prototypes are strictly recording-local and derived solely from baseline hypothesis assignments.

### 2.2 Overlap Segment Representation
For a detected overlap segment $[t_{\text{start}}, t_{\text{end}}]$, we identify all Stage 3 subsegments intersecting the interval:
$$w_{\text{inter}, i} = \max\left(0, \min(t_{\text{end}}, e_i) - \max(t_{\text{start}}, s_i)\right)$$

If $\sum_i w_{\text{inter}, i} == 0$, no frozen acoustic representation exists for the interval $\implies$ **ABSTAIN**.
Otherwise, the overlap representation $\mathbf{e}_{\text{ovl}}$ is the intersection-weighted combination of the overlapping subsegment representations:
$$\mathbf{e}_{\text{ovl}} = \frac{\sum_i w_{\text{inter}, i} \mathbf{e}_i}{\left\|\sum_i w_{\text{inter}, i} \mathbf{e}_i\right\|_2}$$

### 2.3 Secondary Speaker Assignment
Excluding the primary speaker $k_{\text{primary}}$:
$$\mathcal{K}_{\text{alt}} = \{k \in \mathcal{K}_{\text{prototypes}} \mid k \neq k_{\text{primary}}\}$$

If $\mathcal{K}_{\text{alt}} = \emptyset \implies$ **ABSTAIN**.
Otherwise, select the alternate speaker prototype with maximum cosine affinity:
$$k^* = \arg\max_{k \in \mathcal{K}_{\text{alt}}} \cos(\mathbf{e}_{\text{ovl}}, \mathbf{c}_k) = \arg\max_{k \in \mathcal{K}_{\text{alt}}} \mathbf{e}_{\text{ovl}}^\top \mathbf{c}_k$$

---

## 3. Mandatory Ablations

We evaluate four acoustic representations for M2-R1:
1. **M2-TitaNet**: TitaNet-Large only ($\mathbb{R}^{192}$, cosine similarity).
2. **M2-WeSpeaker**: WeSpeaker ResNet293-LM only ($\mathbb{R}^{256}$, cosine similarity).
3. **M2-Concat**: Equal concatenated dual-encoder representation ($\mathbb{R}^{448}$, cosine similarity).
4. **M2-NormFusion**: Normalized Cosine Average:
   $$\text{Score}(k) = 0.5 \cdot \cos\left(\mathbf{e}_{\text{ovl}}^{\text{titanet}}, \mathbf{c}_k^{\text{titanet}}\right) + 0.5 \cdot \cos\left(\mathbf{e}_{\text{ovl}}^{\text{wespeaker}}, \mathbf{c}_k^{\text{wespeaker}}\right)$$
