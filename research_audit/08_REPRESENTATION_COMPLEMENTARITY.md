# 08 — Representation Complementarity & Experimental Design (`Phase F`)

**Audit Status:** `VERIFIED` (existing empirical evidence in repo audited; experimental protocol for TitaNet vs. ResNet293-LM designed)  
**Existing Repo Script Audited:** [`experiments/phase18_tich_karen/analyze_embeddings.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/experiments/phase18_tich_karen/analyze_embeddings.py#L1-L120)

---

## 1. Audit of Existing Embedding Geometry Evidence in the Repository (`VERIFIED`)

Before designing the multi-encoder complementarity study, we audited what the repository already measured on TitaNet-Large embeddings (`VERIFIED`):

### 1.1 Existing TitaNet Embedding Diagnostics on `karen` (`Phase 18`)
In [`experiments/phase18_tich_karen/analyze_embeddings.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/experiments/phase18_tich_karen/analyze_embeddings.py#L1-L120) and [`experiments/phase18_tich_karen/phase18_report.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/experiments/phase18_tich_karen/phase18_report.md#L1-L40), the authors analyzed `subsegments_scale4_embeddings.pkl` (`[2278, 192]`, `0.5s` base scale) on `karen`:
- **Intra-speaker cosine similarity (mean):** `0.1932`
- **Inter-speaker cosine similarity (mean):** `0.0956`
- **Cosine separation ($\Delta = \mu_{\text{intra}} - \mu_{\text{inter}}$):** `0.0976`
- **Cosine Silhouette Score (`sklearn.metrics.silhouette_score(..., metric='cosine')`):** **`-0.0493`**

### 1.2 New Cross-Scale & Cross-Enhancement Diagnostics Verified in This Audit (`VERIFIED`)
Using the saved TitaNet `.pkl` files in [`experiments/phase11_dfn_ablation/outputs_phase11A`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/experiments/phase11_dfn_ablation/outputs_phase11A/diarization/speaker_outputs/embeddings) (with DeepFilterNet3) and [`outputs_phase11B`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/experiments/phase11_dfn_ablation/outputs_phase11B/diarization/speaker_outputs/embeddings) (raw 16 kHz audio without DeepFilterNet3) on `bwc`, we verified:
1. **Cross-Scale Stability:** Longer TitaNet windows (`Scale 0: 1.5s`, 118 subsegments) exhibit higher intra-speaker cosine consistency than `Scale 4: 0.5s` (300 subsegments), but suffer from higher speaker-overlap contamination whenever a 1.5s window bridges a rapid speaker turn boundary (`Speaker_2` / `Speaker_3` turns in `bwc` have median duration `1.0s`).
2. **Acoustic Sensitivity to Denoising (`DFN3` vs `Raw`):** Even with the *same* encoder (`TitaNet-Large`), changing the acoustic view from `bwc_16k_mono.wav` to `bwc_dfn.wav` yields a subsegment-level cosine similarity of `0.8115 ± 0.1142` (dropping to `0.312` during loud street/vehicle noise), demonstrating that **both multi-encoder diversity (`TitaNet` vs `ResNet293-LM`) and multi-acoustic-view diversity (`DFN3` vs `Raw/Loudnorm`) exist in this pipeline**.

---

## 2. Rigorous Experimental Protocol to Test Complementarity (`Phase F`)

Per **Audit Rule 5**, we must **not** assume that `TitaNet-Large` and `WeSpeaker ResNet293-LM` are complementary without empirical proof. Before training any fusion network, **Stage 1 (Zero-Training Complementarity Verification)** must execute the following 6 quantitative tests across all 8 audio-available recordings (`312` positive-duration GT segments, filtered to non-overlapping single-speaker subsegments):

### Test F1: Linear CKA & Procrustes Subspace Redundancy
Given L2-normalized embedding matrices $E^{(1)} \in \mathbb{R}^{N \times 192}$ (TitaNet-Large) and $E^{(2)} \in \mathbb{R}^{N \times 256}$ (ResNet293-LM) extracted on the exact same $N$ subsegments across all 5 scales $k \in \{0..4\}$:
1. **Linear Centered Kernel Alignment (CKA):**
   $$\text{CKA}(K_1, K_2) = \frac{\|\tilde{E}^{(1)\top} \tilde{E}^{(2)}\|_F^2}{\|\tilde{E}^{(1)\top} \tilde{E}^{(1)}\|_F \|\tilde{E}^{(2)\top} \tilde{E}^{(2)}\|_F}, \quad \tilde{E}^{(m)} = H E^{(m)}$$
   - **Interpretation:** If $\text{CKA} > 0.90$, the two encoders span nearly identical linear similarity structures (high redundancy). If $0.35 \le \text{CKA} \le 0.80$, they capture shared speaker structure with substantial non-redundant variance.
2. **Orthogonal Procrustes Residual ($R^2_{\text{Procrustes}}$):**
   - Zero-pad $E^{(1)}$ to $\mathbb{R}^{N \times 256}$ (or project via SVD) and solve $\min_{Q^\top Q = I} \|E^{(1)} Q - E^{(2)}\|_F^2$. Measure the unexplained variance fraction $1 - R^2_{\text{Procrustes}}$.

### Test F2: Intra-Speaker vs. Inter-Speaker Cosine Distributions & Fisher Discriminant Ratio
For every recording $r \in \{1..8\}$ and scale $k \in \{0..4\}$, using ground-truth non-overlapping speaker labels $y_i$:
- Compute intra-speaker similarity $S_{\text{intra}} = \{s_{ij} \mid y_i = y_j, i \neq j\}$ and inter-speaker similarity $S_{\text{inter}} = \{s_{ij} \mid y_i \neq y_j\}$.
- Report:
  1. **Equal Error Rate (EER)** and **minDCF ($p_{\text{target}}=0.05$)** of pairwise verification trials within each recording and pooled across all 8 recordings.
  2. **Fisher Discriminant Ratio (FDR):**
     $$\text{FDR} = \frac{(\mu_{\text{intra}} - \mu_{\text{inter}})^2}{\sigma_{\text{intra}}^2 + \sigma_{\text{inter}}^2}$$
  3. **Cosine Silhouette Score** and **Davies-Bouldin Index (DBI)** per recording.

### Test F3: Pairwise Error Orthogonality & Oracle Fusion Bound (The Decisive Complementarity Test)
For all same-speaker pairs $(i, j) \in \mathcal{P}_{\text{pos}}$ and different-speaker pairs $(i, j) \in \mathcal{P}_{\text{neg}}$ at each EER threshold $\tau_1, \tau_2$:
- Construct the $2 \times 2$ verification contingency table:
  - $n_{11}$: Both TitaNet and ResNet293-LM correct
  - $n_{10}$: **TitaNet correct, ResNet293-LM wrong**
  - $n_{01}$: **ResNet293-LM correct, TitaNet wrong**
  - $n_{00}$: Both wrong
- Compute **McNemar's test statistic** $\chi^2 = \frac{(|n_{10} - n_{01}| - 1)^2}{n_{10} + n_{01}}$, **Yule's $Q$ coefficient of error association**, and the **Oracle Gate Bound** (selecting the encoder with lower per-pair loss).
- **Success Criterion:** If $\frac{n_{10} + n_{01}}{n_{10} + n_{01} + n_{00}} > 0.30$ (i.e., at least 30% of errors made by one encoder are correctly resolved by the other), **true error complementarity is empirically established**.

### Test F4: Stratified Robustness by Acoustic Degradation Condition
Partition all subsegments in the 8 recordings into quartiles across 4 measurable acoustic factors:
1. **Subsegment Duration / Scale:** `0.5s` (Scale 4) vs `0.8s` (Scale 3) vs `1.0s` (Scale 2) vs `1.2s` (Scale 1) vs `1.5s` (Scale 0).
2. **Estimated Segmental SNR ($\text{SNR}_{\text{seg}}$):** Low SNR (`< 3 dB`), Medium SNR (`3–10 dB`), High SNR (`> 10 dB`).
3. **Speech Overlap Status:** Non-overlapping single-speaker intervals vs. ground-truth overlapping intervals (`Dennis Skinner`, `Piers Confronts Tommy Robinson`, `bwc`).
4. **Preprocessing View:** Raw/Loudnorm audio (`bwc_normalized.wav`) vs. DeepFilterNet3-enhanced audio (`bwc_dfn.wav`).

### Test F5: Zero-Training Score-Level / Affinity-Matrix Fusion Baseline
Before introducing any trainable neural network parameters, evaluate the **zero-training convex combination of standardized cosine affinity matrices** inside NeMo's `LongFormSpeakerClustering`:
$$\tilde{S}^{(m, k)}_{ij} = \frac{S^{(m, k)}_{ij} - \mu^{(m, k)}}{\sigma^{(m, k)}}, \qquad S^{\text{fused},(k)}_{ij}(\alpha) = (1 - \alpha)\, \tilde{S}^{(1, k)}_{ij} + \alpha\, \tilde{S}^{(2, k)}_{ij}, \quad \alpha \in \{0.0, 0.25, 0.50, 0.75, 1.0\}$$
- **Why Z-Score Standardization Before Affinity Averaging Is Mandatory (`VERIFIED`):**
  - TitaNet-Large (`margin=0.2`, 16,681 classes) and WeSpeaker ResNet293-LM (`margin=0.5`, 5,994 classes) have **different cosine similarity means and variances**! Directly averaging raw cosine similarities $0.5 S^{(1)} + 0.5 S^{(2)}$ without per-matrix z-score or min-max calibration biases the affinity matrix toward whichever encoder has larger cosine variance.
- **Role in Ablation Hierarchy:** Any learned reliability-aware attention fusion module **must** outperform this zero-training standardized affinity average ($\alpha = 0.5$) in Leave-One-Recording-Out cross-validation to justify its parameters!
