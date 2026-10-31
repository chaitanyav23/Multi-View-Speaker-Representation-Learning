# 15 — Research Gaps, Failure Modes & Novelty Analysis (`Phase M: Q21–Q24`)

**Audit Status:** `VERIFIED`  
**Base Paper Audited:** [`papers/2607.24958v1.pdf`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/papers/2607.24958v1.pdf) (*Towards Operational Conversational Intelligence: A Speech Intelligence Framework*, Section 6.2, p. 21)

---

## 1. Why Existing Diarization & Multi-Encoder Methods Fail on BWC Audio (`Q21`)

Our code and empirical audit of Phases 5–20 reveals **four concrete failure mechanisms** of standard speaker diarization on body-worn camera (BWC) audio:

1. **Failure Mode 1 — Single-Encoder Inductive Bias & Short-Window Collapse (`VERIFIED`):**
   - Standard diarization relies on a single pretrained encoder (`TitaNet-Large`, 1D depthwise-separable convolutions with global Squeeze-and-Excitation and per-feature variance normalization).
   - On `0.5s` base-scale BWC windows (`subsegments_scale4`), sudden transient noise (sirens, wind, clothing rustle, radio clicks) distorts both the per-feature variance normalizer and the global SE channel weights, driving intra-speaker cosine similarity down to `0.1932` and overall cosine Silhouette Score to **`-0.0493`** (`experiments/phase18_tich_karen/analyze_embeddings.py`).
2. **Failure Mode 2 — Dominant-Speaker Spectral Eigengap Collapse (`VERIFIED`):**
   - BWC recordings have extreme speaker skew (the camera-wearing officer accounts for `57%–87%` of all speech, while civilians/bystanders utter `0.5s–2.0s` responses).
   - When intra-speaker similarity is noisy, NeMo's `LongFormSpeakerClustering` (`max_rp_threshold=0.25`) collapses all subsegments into a **single cluster (`speaker_0` on `bwc`, `JER = 84.24%`)** or merges minority speakers into the dominant speaker.
3. **Failure Mode 3 — Uniform Multiscale & Prototype Weighting Under Acoustic Degradation (`VERIFIED`):**
   - NeMo's `ClusteringDiarizer` uses static, recording-agnostic multiscale weights (`[1, 2, 3, 3, 3]`) across all subsegments regardless of local SNR, window duration, or speech overlap.
   - Worse yet, `ClusterEmbedding.get_cluster_avg_embs()` (`msdd_models.py` L829) computes MSDD speaker prototypes using an **unweighted arithmetic mean** `torch.mean(selected_embs, dim=0)`, allowing overlapping or heavily corrupted `0.5s` subsegments to contaminate the speaker reference vectors fed to the MSDD neural decoder!
4. **Failure Mode 4 — Single Acoustic Preprocessing Trade-Off (`DFN3` vs `Raw`, `VERIFIED`):**
   - DeepFilterNet3 (`Path A`) removes stationary background noise for VAD, but distorts speaker formants on low-SNR speech (changing TitaNet embeddings by up to `cos_sim = 0.312` on noisy segments in `bwc`). Relying exclusively on a single encoder over a single enhanced stream loses acoustic cues preserved in the unenhanced/loudness-normalized view.

---

## 2. Why Naive Embedding Concatenation or Static Score Averaging Is Insufficient (`Q22`)

Simply concatenating L2-normalized embeddings $[\tilde{\mathbf{e}}_1; \tilde{\mathbf{e}}_2] \in \mathbb{R}^{448}$ (which is mathematically identical to unweighted cosine similarity averaging $\frac{1}{2} S^{(1)}_{ij} + \frac{1}{2} S^{(2)}_{ij}$) fails in three ways:
1. **Uncalibrated Cosine Geometry Mismatch:**
   - `TitaNet-Large` (`192-D`, trained on 16,681 speakers with `margin=0.2` on `3.0s` chunks) and `WeSpeaker ResNet293-LM` (`256-D`, trained on 5,994 speakers with Large-Margin `margin=0.5` on `6.0s` chunks) have systematically different cosine similarity means, variances, and angular margins across the 5 temporal scales (`1.5s` down to `0.5s`). Uncalibrated concatenation biases the inner product toward the encoder with wider similarity variance.
2. **Scale-Dependent Encoder Reliability Inversion:**
   - Because `ResNet293-LM` was fine-tuned on `6.0s` windows with a large angular margin (`m=0.5`), its embeddings are sharper on longer scales (`Scale 0: 1.5s`, `Scale 1: 1.2s`) but degrade faster on ultra-short `0.5s` windows (`Scale 4`) than TitaNet-Large (`3.0s`, `m=0.2`). Static 50/50 weighting across all 5 scales cannot adapt to subsegment duration or scale index!
3. **Blindness to Local Acoustic Corruptions & Overlap:**
   - Static concatenation weights a `0.5s` subsegment containing 80% overlapping speech or narrowband siren noise identically to a clean `1.5s` single-speaker segment, both when constructing the affinity graph $A_{ij}$ and when pooling MSDD cluster prototypes $\bar{\mathbf{e}}_s^{(k)}$.

---

## 3. What Makes This Direction Scientifically Novel & Publishable (`Q23 & Q24`)

### 3.1 Genuine Scientific Contributions (`Q23`)
1. **Structural + Acoustic Multi-View Complementarity on Operational BWC Audio:**
   - Bridges **architectural diversity** (1D cross-frequency depthwise-separable CNN with global SE vs. 2D localized time-frequency bottleneck ResNet-293) and **frontend normalization diversity** (per-feature variance normalization vs. mean-only CMN) across NeMo's 5 multiscale temporal resolutions.
2. **Dual-Level Reliability-Aware Fusion (Inter-Encoder + Intra-Cluster Prototype Gating):**
   - Formulates a **Precision-Weighted / Reliability-Gated Multi-View Fusion** mechanism conditioned on zero-cost pipeline signals already present in the dual-path architecture:
     - **DFN3 local noise-reduction energy ratio** ($\Delta E_{\text{DFN}}$, computed directly from `16k_mono.wav` vs `dfn.wav`),
     - **Pyannote `segmentation-3.0` frame-level VAD confidence and powerset overlap posterior** ($\bar{p}_{\text{VAD}}, \bar{p}_{\text{overlap}}$),
     - **Pre-normalization embedding magnitude** ($\|\mathbf{e}_m\|_2$),
     - **Cross-scale self-consistency** ($c_{\text{scale}, m}(i) = \cos(\mathbf{e}^{(k)}_{m,i}, \mathbf{e}^{(k-1)}_{m,\pi(i)})$), and
     - **Subsegment duration / scale index** ($d_i, k$).
   - Applies reliability weights **both** to fuse encoder views ($\alpha_{1,i}, \alpha_{2,i}$) **and** to weight subsegments ($\beta_i$) during spectral affinity graph construction and MSDD cluster-average prototype pooling (`ClusterEmbedding.get_cluster_avg_embs`).
3. **Coordinate-Aligned Residual Integration with Frozen Neural Diarizers (`Procrustes + Residual Gate`):**
   - Solves a general, practical problem in speech processing: how to inject a higher-dimensional external speaker encoder (`256-D ResNet293-LM`) into a frozen downstream neural diarization decoder (`192-D diar_msdd_telephonic`) without retraining the neural diarizer, via closed-form Orthogonal Procrustes alignment + residual reliability gating.
4. **Exposing & Resolving Dominant-Speaker Collapse in BWC Evaluation:**
   - Demonstrates empirically how micro-averaged Word Attribution Error (`WAE = 9.73%` / `Speaker Accuracy = 90.27%`) masks complete 1-speaker collapse (`JER = 84.24%`) under BWC dominant-speaker skew, and establishes a rigorous 8-recording Leave-One-Recording-Out benchmark with `JER`, `Macro-SACC`, `cpWER`, and `saWER`.

### 3.2 What Must Be Proven Experimentally to Justify Publication (`Q24`)
To meet top-tier speech venue standards (Interspeech / ICASSP / IEEE TASLP / Computer Speech & Language), the experimental results must prove four sequential hypotheses:
1. **H1 (Complementarity):** `TitaNet-Large` and `WeSpeaker ResNet293-LM` exhibit statistically significant error orthogonality on BWC subsegments (McNemar's test $p < 0.01$, oracle gate EER strictly lower than both individual encoders).
2. **H2 (Multi-View over Single-View):** Even zero-training standardized multiscale affinity fusion of `TitaNet-Large + ResNet293-LM` outperforms both single-encoder `TitaNet-Large` and single-encoder `ResNet293-LM` on EER, Silhouette Score, JER, DER, and `cpWER`/`saWER` across the 8 BWC recordings.
3. **H3 (Reliability-Aware Fusion over Static Concatenation/Averaging):** Conditioning the view and prototype weights on the 8-D acoustic/view reliability vector $\mathbf{q}_i$ outperforms static 50/50 affinity averaging and unweighted concatenation under 8-fold Leave-One-Recording-Out cross-validation, specifically recovering minority speakers (reducing `JER` and speaker count error $|\hat{K} - K_{\text{ref}}|$).
4. **H4 (Component Attribution via Ablation):** Ablating individual reliability signals (removing $\bar{p}_{\text{overlap}}$, removing $\Delta E_{\text{DFN}}$, removing scale/duration $d_i$, or removing cross-scale consistency $c_{\text{scale}}$) measurably degrades diarization and speaker attribution performance.
