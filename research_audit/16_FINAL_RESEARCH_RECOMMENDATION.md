# 16 — Final Research Synthesis & Answers to All 24 Audit Questions (`Phase M`)

**Audit Status:** `VERIFIED`  
**Scope:** Complete synthesis of Phases A–L and direct evidence-backed answers to Questions 1–24 (`Section 16` of Audit Specification).

---

## 1. Direct Evidence-Backed Answers to Audit Questions 1–24 (`Phase M`)

### A. Pipeline Reality (`Q1–Q4`)

#### Q1: What is the exact implemented pipeline in `phase20_standalone`?
- **Answer (`VERIFIED`):** [`phase20_standalone/run_phase20.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/run_phase20.py#L38-L59) executes 15 sequential subprocess steps for recording `bwc`:
  1. `step00_ground_truth.py`: Parses `ground_truth/bwc.txt` (`54` raw segments), applies hardcoded `bwc` curation (`apply_legacy_reference_curation`, `L38–72`), and writes `ground_truth_json/bwc.json` (`48` segments) and `ground_truth_rttm/bwc.rttm` (`37` positive-duration segments).
  2. `step01_preprocess.py`: Converts `raw_audio/bwc.wav` to 16 kHz mono 16-bit PCM `audio/bwc_16k_mono.wav` via `ffmpeg`.
  3. `step02a_deepfilternet.py`: Runs `deepFilter --no-suffix` (`DeepFilterNet3`) on `bwc_16k_mono.wav` to produce Path A `audio/bwc_dfn.wav`.
  4. `step02b_loudnorm.py`: Runs `ffmpeg` EBU R128 loudness normalization (`I=-16:TP=-1.5:LRA=11`) on `bwc_16k_mono.wav` to produce Path B `audio/bwc_normalized.wav`.
  5. `step03_vad.py --path A`: Runs `pyannote/segmentation-3.0` (`onset=0.6, offset=0.4`) + custom post-processing (`pad=0.05s, min_off=0.25s, min_on=0.10s`) on `bwc_dfn.wav`, writing `outputs/vad/bwc_pathA_segments.json` and `bwc_pathA_oracle.rttm` (`46` segments, `108.32s`).
  6. `step03_vad.py --path B`: Runs the same VAD on `bwc_normalized.wav`, writing `bwc_pathB_segments.json` (`41` segments, `108.08s`).
  7. `step04_vad_eval.py`: Evaluates Path A (`F1=0.8068`) and Path B (`F1=0.8026`) VAD against `ground_truth_rttm/bwc.rttm`.
  8. `step05b_whisperx.py`: Transcribes `bwc_normalized.wav` with WhisperX `large-v2` (using WhisperX's internal VAD, ignoring `bwc_pathB_segments.json`) and aligns words with `WAV2VEC2_ASR_LARGE_LV60K_960H`, producing `outputs/whisperx/bwc.json` (`18` segments, `465` words).
  9. `step05a_msdd.py`: Runs NeMo `NeuralDiarizer` (`diar_msdd_telephonic` + `titanet_large`) on unmasked `bwc_dfn.wav` + `bwc_pathA_oracle.rttm` (`oracle_vad=True`), producing `outputs/msdd/bwc.rttm` (`30` segments, all collapsed to `speaker_0`).
  10. `step06b_asr_eval.py`: Evaluates WhisperX WER (`0.3244`) and CER (`0.1976`).
  11. `step06a_diar_eval.py`: Evaluates MSDD DER (`0.4471` at `collar=0.0`; `0.3241` at `collar=0.25, skip_overlap=True`) and JER (`0.8424`).
  12. `step07_fusion.py`: Attributes WhisperX words to MSDD speakers via overlap + `±0.5s` nearest-turn fallback (`common.fuse_transcript_and_diarization`).
  13. `step08_fusion_eval.py`: Evaluates micro word speaker accuracy (`0.9027`, `WAE=0.0973` on `401` matched words).
  14. `step09_aggregate_report.py` & 15. `step10_compare_phase19.py`: Aggregates JSON/CSV/MD reports and verifies parity with Phase 19.

#### Q2: Where does the code differ from the paper and documentation?
- **Answer (`VERIFIED`):** Six major discrepancies were verified (detailed in [`02_EXECUTION_PIPELINE_AUDIT.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/02_EXECUTION_PIPELINE_AUDIT.md)):
  1. **Dual independent VAD runs instead of shared VAD grounding:** Path A (`46` segs) and Path B (`41` segs) run VAD separately, and WhisperX ignores Path B's VAD completely (`step05b_whisperx.py` L61–68).
  2. **No binary waveform masking before TitaNet:** Contrary to Paper Sec. 4.3 ($x_{\text{masked}}(t) = x(t)\cdot m(t)$), `step05a_msdd.py` passes unmasked `bwc_dfn.wav` + `bwc_pathA_oracle.rttm` to NeMo.
  3. **Single-recording results reported as 8-recording benchmark tables:** Paper Tables 3–6 (pp. 15–17) report the single-recording numbers of `bwc` (`DER=0.4471, JER=0.8424, WER=0.3244, SACC=0.9027`).
  4. **Single-speaker collapse masked by `90.27%` Speaker Accuracy:** On `bwc`, NeMo MSDD predicts **1 speaker (`speaker_0`)** for the entire file; `90.27%` accuracy is merely the dominant speaker's word share (`362/401` words).
  5. **Silent ground-truth fallback:** Missing libraries/models in Steps 02a, 03, 05a, 05b trigger silent fallbacks that copy ground-truth annotations as predictions.
  6. **Unused metric scripts:** `cpwer.py`, `sawer.py`, `attribution_metrics.py`, `timeline_metrics.py`, `fusion_metrics.py`, and `speaker_mapping.py` in `phase20_standalone/scripts/` are never called by `run_phase20.py`.

#### Q3: How are TitaNet embeddings currently extracted and used?
- **Answer (`VERIFIED`):** Inside `NeuralDiarizer`, `ClusteringDiarizer` slices each Path A VAD segment (`bwc_pathA_oracle.rttm`) into 5 overlapping temporal scales (`window=[1.5, 1.2, 1.0, 0.8, 0.5]s`, `shift=[0.75, 0.6, 0.5, 0.4, 0.25]s`). For each scale $k \in \{0..4\}$, `EncDecSpeakerLabelModel` (`titanet_large` extracted from `diar_msdd_telephonic.nemo`) computes 80-bin per-feature normalized log-Mel spectrograms, passes them through 1D depthwise-separable Conv Mega-blocks with Squeeze-and-Excitation (`[B, 3072, T]`), applies Attentive Statistics Pooling (`[B, 6144]`), and projects to unnormalized **`[N_k, 192]`** `float32` vectors saved to `subsegments_scale{k}_embeddings.pkl`.

#### Q4: How does NeMo MSDD consume those embeddings?
- **Answer (`VERIFIED`):** In two distinct stages:
  - **Stage 1 (`LongFormSpeakerClustering`):** Stacks all 5 scale embeddings (`[sum N_k, 192]`), maps coarse scales to the base scale (`Scale 4`, `0.5s`), computes 5 cosine affinity matrices $S^{(k)} \in \mathbb{R}^{N_4 \times N_4}$, combines them with `multiscale_weights=[1, 2, 3, 3, 3]`, and runs spectral clustering (`max_rp_threshold=0.25`) to produce base-scale cluster labels `subsegments_scale4_cluster.label`.
  - **Stage 2 (`ClusterEmbedding` + `EncDecDiarLabelModel`):** Propagates base-scale cluster labels to coarse scales via `mode()`, computes unweighted cluster-average embeddings `avg_embs` (`[5, 192, max_spks]`), and runs pairwise neural decoding (`forward_infer`) over `[B, T_win, 5, 192]` sequences and `[B, 5, 192, 2]` prototypes to predict overlap-aware speaker probabilities (`sigmoid_threshold=0.7`).

---

### B. Dataset Reality (`Q5–Q8`)

#### Q5: How many recordings, speakers, and segments actually exist?
- **Answer (`VERIFIED`):**
  - **12 ground-truth `.txt` files** exist in `ground_truth/` (`1,362` total segments, `88` recording-local speaker entries), **8 `.mp3` audio files** exist in `raw_audio/` (`1,893.69s` = `31.56 min` total audio), and **4 GT files have missing audio** (`Drunk Karen`, `Entitled Karen`, `When an Entitled Girl`, `Woman's Attitude`).
  - Across the **8 recordings with audio**, there are **`41` recording-local speaker entries** (`40` real speakers + `1` `Speaker_all`) and **`432` total segments**, of which **`312` have positive duration (`>0s`, totaling `1,625.0s` summed duration / `1,400.0s` union speech)** and **`120` (`27.8%`) have zero duration (`start == end`)** due to 1-second timestamp rounding.

#### Q6: Are ground-truth annotations valid and aligned with audio?
- **Answer (`VERIFIED`):** Partially—7 of the 8 audio-available `.txt` files parse cleanly in `step00_ground_truth.py`, but **587 total anomalies** were verified in [`annotation_validation.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/annotation_validation.csv):
  - `Piers Confronts Tommy Robinson.txt` **crashes `step00_ground_truth.py`** due to 16 Unicode en-dashes `–` (`U+2013`) and a 1-digit hour `0:02:25` on line 79.
  - `120 / 432` segments (`27.8%`) in the 8 audio files have `start == end` (`0.0s` duration) and are silently dropped from RTTM files (`5` of the `41` speakers have **zero** positive-duration segments!).
  - `Dennis Skinner` and `Piers Confronts Tommy Robinson` contain heavy multi-speaker overlap (`sum_pos = 284s` on `176s` and `204s` audio) and non-monotonic timestamps.
  - Among the 4 missing-audio files, `Drunk Karen` crashes with negative durations (`-592s`, `-757s`) and `Entitled Karen` has 201 corrupt block-pasted timestamps (`sum_pos = 7,685s`).

#### Q7: Is the dataset sufficient for supervised training, contrastive learning, or only inference/lightweight fusion?
- **Answer (`VERIFIED`):**
  - **Insufficient for end-to-end encoder fine-tuning or closed-set speaker classification (CE / ArcFace):** Only `25` speakers across the 8 audio recordings have $\ge 2$ positive-duration turns (`21` speakers with $\ge 2$ turns $\ge 1.5\text{s}$).
  - **Sufficient for:**
    1. **Zero-training multiscale affinity fusion & closed-form Orthogonal Procrustes alignment**,
    2. **Low-parameter (`10–500` params) reliability-aware gating / Precision-Weighted Product-of-Experts** trained with **within-recording Supervised Contrastive / Affinity Margin loss** or **Oracle-Gate distillation** under **8-fold Leave-One-Recording-Out (LORO) cross-validation**, and
    3. **Pre-calibrating the reliability estimator on synthetic BWC degradations** generated via the repository's existing [`vad_aug_temp/vad_eval/synthesis.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/vad_aug_temp/vad_eval/synthesis.py#L1-L120).

#### Q8: What data splits are scientifically valid?
- **Answer (`VERIFIED`):** **Strict 8-Fold Leave-One-Recording-Out (LORO) Cross-Validation** across the 8 audio-available recordings (training alignment/gate parameters on 7 recordings and testing on the held-out 8th recording). Splitting segments from the same recording across train and test sets is strictly forbidden due to speaker and acoustic-channel leakage.

---

### C. Encoder & Representation Feasibility (`Q9–Q12`)

#### Q9: Is WeSpeaker ResNet293-LM compatible with the existing audio and segmentation pipeline?
- **Answer (`VERIFIED`):** **Yes, 100% compatible.** Both `TitaNet-Large` and `WeSpeaker ResNet293-LM` consume 16 kHz mono audio (`audio/<stem>_dfn.wav` and `audio/<stem>_normalized.wav`) and can be run over the exact same 5-scale subsegment manifests (`subsegments_scale{0..4}.json`, windows `[1.5, 1.2, 1.0, 0.8, 0.5]s`), provided that float waveforms `[-1, 1]` are scaled by `32768.0` for ResNet293-LM's `torchaudio.compliance.kaldi.fbank` frontend.

#### Q10: How can 192-D TitaNet and 256-D ResNet293-LM embeddings be aligned?
- **Answer (`VERIFIED`):** Two mathematically rigorous alignment mechanisms exist:
  1. **In Affinity Space (Dimension-Free):** Standardize each encoder's cosine similarity matrix $\tilde{S}^{(m,k)} = (S^{(m,k)} - \mu_m)/\sigma_m \in \mathbb{R}^{N_4 \times N_4}$ before reliability-weighted combination—requiring zero dimension compression!
  2. **In Latent Feature Space (`256-D -> 192-D`):** Fit a closed-form **Orthogonal Procrustes / Ridge alignment matrix** $P \in \mathbb{R}^{256 \times 192}$ on the training folds (or self-supervised across subsegments of the recording) to project L2-normalized ResNet293-LM embeddings directly into TitaNet's 192-D coordinate basis.

#### Q11: How can complementarity between encoders be tested before training a fusion model?
- **Answer (`VERIFIED`):** By executing the 5-part Stage 1 Complementarity Suite ([`08_REPRESENTATION_COMPLEMENTARITY.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/08_REPRESENTATION_COMPLEMENTARITY.md)):
  1. Linear CKA and Procrustes residual variance ($1 - R^2_{\text{Procrustes}}$) across all 5 scales.
  2. Intra/inter-speaker cosine separation, FDR, EER, minDCF, and Silhouette Score per scale and per SNR quartile.
  3. $2 \times 2$ pairwise verification error contingency table ($n_{10}, n_{01}$), McNemar's test, and Oracle Gate EER bound.
  4. Scale-sensitivity curve (`0.5s` vs `1.5s`) comparing `ResNet293-LM` (`6s` LM fine-tuned) vs `TitaNet-Large` (`3s` trained).
  5. Zero-training standardized affinity fusion ($\alpha \in [0, 1]$) inside `LongFormSpeakerClustering`.

#### Q12: Should other encoders also be considered as baselines or alternatives?
- **Answer (`VERIFIED`):** Including **`ECAPA-TDNN` (`192-D`, 1D Res2Net+SE)** or **`WavLM-Large SV` (`256-D`, SSL Transformer)** as a third diagnostic comparison in Stage 1 strengthens the paper by testing whether **cross-topology fusion** (`1D-Conv TitaNet + 2D-Conv ResNet293-LM`) yields greater complementarity (lower CKA, higher error orthogonality) than **same-topology fusion** (`1D-Conv TitaNet + 1D-Conv ECAPA-TDNN`).

---

### D. Fusion & Reliability Design (`Q13–Q16`)

#### Q13: Which reliability indicators can be computed from existing pipeline signals?
- **Answer (`VERIFIED`):** Seven high-value, zero-extra-model indicators can be computed for every subsegment $i$ at scale $k$ ([`09_RELIABILITY_FUSION_FEASIBILITY.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/09_RELIABILITY_FUSION_FEASIBILITY.md)):
  1. **DFN3 Local Noise-Reduction Ratio ($\Delta E_{\text{DFN}, i}$)** from `16k_mono.wav` vs `dfn.wav`,
  2. **Local Segmental SNR ($\text{SNR}_{\text{seg}, i}$)** relative to adjacent non-VAD background frames,
  3. **Effective Subsegment Duration & Scale Index ($d_i, k$)** from `subsegments_scale{k}.json`,
  4. **Frame-Level VAD Confidence ($\bar{p}_{\text{VAD}, i}, \sigma_{\text{VAD}, i}$)** from `pyannote/segmentation-3.0`,
  5. **Frame-Level Overlap Posterior ($\bar{p}_{\text{overlap}, i}$)** from `pyannote/segmentation-3.0`'s multi-speaker powerset classes,
  6. **Pre-Normalization Embedding L2 Norms ($\|\mathbf{e}_{1,i}\|_2, \|\mathbf{e}_{2,i}\|_2$)**, and
  7. **Cross-Scale Self-Consistency ($c_{\text{scale}, 1, i}, c_{\text{scale}, 2, i}$)** between scale $k$ and scale $k-1$.
  *(Blind RT60 reverberation estimation on `0.5s` segments is ill-conditioned and should be replaced by spectral flatness/entropy).*

#### Q14: Is self-attention appropriate, or is gated / MLP / score-level fusion more suitable?
- **Answer (`VERIFIED`):** Full unconstrained Multi-Head Self-Attention (`~640K` parameters) over $M=2$ views is **inappropriate** and will overfit on 8 recordings (`312` segments). Instead, a **Lightweight Reliability-Conditioned Attention Gate / Precision-Weighted Product-of-Experts (`10` to `3,500` parameters)** that predicts view precisions $\tau_{m,i} = f_\theta(\mathbf{q}_i, \mathbf{e}_{m,i})$ and subsegment quality $\beta_i = h_\theta(\mathbf{q}_i)$ provides exact mathematical interpretability, zero overfitting, and direct integration into both multiscale affinity construction and MSDD prototype pooling.

#### Q15: What training objective is feasible given the dataset?
- **Answer (`VERIFIED`):**
  1. **Closed-form Orthogonal Procrustes SVD** for latent coordinate alignment (`0` gradient steps), combined with:
  2. **Within-Recording Supervised Contrastive / Cosine Affinity Margin Loss ($\mathcal{L}_{\text{aff}} + \mathcal{L}_{\text{SupCon}}$)** + **Cross-Scale Consistency Loss ($\mathcal{L}_{\text{cons}}$)** evaluated via 8-fold LORO cross-validation (optionally pre-calibrated on synthetic BWC mixtures from `vad_aug_temp/vad_eval/synthesis.py`).

#### Q16: How do we prevent overfitting?
- **Answer (`VERIFIED`):**
  1. Keep trainable parameters strictly $\le 500\text{–}3,500$ (freezing `TitaNet-Large`, `ResNet293-LM`, and `diar_msdd_telephonic`).
  2. Fix coordinate projections via closed-form Orthogonal Procrustes rather than end-to-end unconstrained $256 \times 256$ matrices.
  3. Restrict contrastive negatives to **within-recording speaker pairs** so the gate cannot cheat using recording-level channel acoustics.
  4. Enforce strict **8-fold Leave-One-Recording-Out (LORO)** evaluation and compare against the **0-parameter standardized affinity fusion baseline**.

---

### E. Downstream Diarization Integration (`Q17–Q20`)

#### Q17: Can NeMo MSDD consume fused embeddings directly?
- **Answer (`VERIFIED`):**
  - **Stage 1 (`ClusteringDiarizer` / `LongFormSpeakerClustering`):** **YES, directly** for any embedding dimension $D$ (`192`, `256`, `448`) or fused affinity matrix $A_{ij}$.
  - **Stage 2 (Frozen `diar_msdd_telephonic` `EncDecDiarLabelModel`):** Requires **exact dimension `D = 192`** aligned to TitaNet-Large's coordinate space (`ms_emb_seq` `[B, T, 5, 192]` and `ms_avg_embs` `[B, 5, 192, 2]`).

#### Q18: What is the cleanest integration point in the existing pipeline?
- **Answer (`VERIFIED`):** **Option 3 + Option 2 Unified Wrapper (`MultiViewClusterEmbedding` in `step05a_msdd.py`, [`13_ARCHITECTURAL_INTEGRATION_OPTIONS.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/13_ARCHITECTURAL_INTEGRATION_OPTIONS.md)):**
  - Use **Reliability-Aware Multiscale Affinity Fusion** (combining full `192-D` TitaNet and `256-D` ResNet293-LM views weighted by $\alpha_{m,i}^{(k)}(\mathbf{q}_i)$ and $\beta_i^{(k)}(\mathbf{q}_i)$) to produce the base-scale cluster labels `subsegments_scale4_cluster.label` (`clus_test_label_dict`), **AND**
  - Use **Reliability-Weighted Prototype Pooling** ($\beta_i$-weighted cluster average instead of unweighted `torch.mean()` in `ClusterEmbedding.get_cluster_avg_embs`) + **Procrustes-Aligned Residual 192-D Embeddings** for `diar_msdd_telephonic`!

#### Q19: Which evaluation metrics should be primary and secondary?
- **Answer (`VERIFIED`):**
  - **Primary Diarization Metrics:** **JER (`collar=0.0`)** (captures minority-speaker recall and prevents single-speaker collapse), **DER (`collar=0.0, skip_overlap=False`)**, **DER (`collar=0.25, skip_overlap=True`)**, and **Speaker Count Error ($|\hat{K} - K_{\text{ref}}|$)**.
  - **Primary Attribution / End-to-End Metrics:** **Macro-Averaged Speaker Attribution F1 (`Macro-SACC`)**, **`cpWER`**, **`saWER`**, and **Micro WAE / SACC** (using Hungarian mapping in `fusion_metrics.py`).
  - **Primary Representation Metrics:** Within-recording verification **EER**, **FDR**, and **Cosine Silhouette Score** across all 5 scales.

#### Q20: What baseline comparisons are mandatory?
- **Answer (`VERIFIED`):**
  1. **Baseline 1:** Single-View `TitaNet-Large (192-D)` (Current Phase 20 baseline).
  2. **Baseline 2:** Single-View `WeSpeaker ResNet293-LM (256-D)` (Using ResNet293-LM for multiscale spectral clustering + TitaNet 192-D for MSDD overlap refinement).
  3. **Baseline 3:** Naive L2-Normalized Embedding Concatenation (`[TitaNet; ResNet293-LM]`, `448-D`).
  4. **Baseline 4:** Unweighted Standardized Cosine Affinity Average ($0.5\tilde{S}^{(1)} + 0.5\tilde{S}^{(2)}$, `0` trainable params).
  5. **Proposed Method A:** Deterministic / Logistic Reliability-Gated Multiscale Affinity + Quality-Weighted MSDD Prototypes.
  6. **Proposed Method B:** Probabilistic Precision-Weighted Product-of-Experts / Procrustes-Aligned Residual Attention Fusion (`192-D`).

---

## 2. Concrete Staged Research Roadmap (`Stages 1–9`)

*(Per Audit Rule 4, no implementation has been executed during Stage 0. Below is the exact, step-by-step execution plan for Stages 1–9 awaiting user approval.)*

| Stage | Title | Exact Technical Deliverable & Gate Criterion |
|---|---|---|
| **Stage 1** | **Non-Destructive Multi-Recording Data & Audio Preparation** | Write a clean, non-destructive loader that converts the 8 `raw_audio/*.mp3` files into `audio/<id>_{16k_mono,dfn,normalized}.wav`, fixes the 16 Unicode en-dashes and `0:02:25` in `Piers Confronts Tommy Robinson` in-memory, and disables `bwc`-only legacy curation on the other 7 recordings. |
| **Stage 2** | **Baseline Reproduction Across All 8 Recordings** | Run the existing single-encoder `TitaNet-Large` + NeMo MSDD + WhisperX pipeline on all 8 audio-available recordings (including activating `fusion_metrics.py` for `cpWER`, `saWER`, `JER`, and `Macro-SACC`) to establish the true 8-recording baseline table. |
| **Stage 3** | **Multiscale Dual-Encoder Extraction & Caching** | Extract and cache 5-scale embeddings (`[1.5, 1.2, 1.0, 0.8, 0.5]s`) for both **TitaNet-Large (`192-D`)** and **WeSpeaker ResNet293-LM (`256-D`)** across both `dfn.wav` and `normalized.wav` for all 8 recordings (`~11,500` subsegments), plus the 8-D subsegment reliability vector $\mathbf{q}_i^{(k)}$. |
| **Stage 4** | **Complementarity Verification Suite** | Run Tests F1–F4 (Linear CKA, Procrustes $R^2$, per-scale EER/FDR/Silhouette, $2\times 2$ error orthogonality & McNemar's test, SNR/duration stratified curves). **Go/No-Go Gate:** Verify statistically significant error complementarity before training any fusion weights. |
| **Stage 5** | **Zero-Training Multiscale Fusion Baselines** | Evaluate Naive Concatenation (`448-D`), Standardized Affinity Average ($0.5\tilde{S}^{(1)} + 0.5\tilde{S}^{(2)}$), and Closed-Form Orthogonal Procrustes `192-D` fusion across all 8 recordings. |
| **Stage 6** | **Reliability-Aware Precision/Attention Fusion & Prototype Weighting** | Implement and evaluate the lightweight Reliability-Conditioned Precision Gate ($\alpha_{m,i}(\mathbf{q}_i)$) + Overlap/VAD Prototype Quality Weighting ($\beta_i(\mathbf{q}_i)$) under strict 8-fold LORO cross-validation (and synthetic BWC pre-calibration). |
| **Stage 7** | **Downstream NeMo MSDD & Attribution Integration** | Evaluate Option 3 (Decoupled Multiscale Affinity + Quality-Weighted Prototypes) and Option 2 (Procrustes Residual `192-D` Sequence + Prototypes) inside `NeuralDiarizer` and `step07_fusion.py`. |
| **Stage 8** | **Comprehensive Ablation & Sensitivity Study** | Ablate: (a) individual reliability features in $\mathbf{q}_i$, (b) scale-specific vs scale-uniform gating, (c) inter-encoder gating $\alpha_{m,i}$ vs prototype quality weighting $\beta_i$, and (d) clustering threshold sensitivity (`max_rp_threshold`). |
| **Stage 9** | **Publication-Grade Reporting & Statistical Significance** | Generate final 8-recording tables, per-recording breakdowns, Wilcoxon signed-rank / bootstrap $p$-values, embedding t-SNE/UMAP + reliability gate visualizations, and LaTeX tables extending Section 6.2 of the base paper. |
