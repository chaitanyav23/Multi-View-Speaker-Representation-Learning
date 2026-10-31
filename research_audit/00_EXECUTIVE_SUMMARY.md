# 00 — Executive Summary: Complete Source-Code & Dataset Audit (Stage 0)

**Project:** Multi-View Speaker Representation Learning for Noisy Body-Worn Camera (BWC) Audio  
**Base Paper:** *Towards Operational Conversational Intelligence: A Speech Intelligence Framework* ([`papers/2607.24958v1.pdf`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/papers/2607.24958v1.pdf), Section 6.2)  
**Repository:** `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning` (`HEAD` = `f2ac2e4`, clean parity with `origin/main`)  
**Audit Date:** 2026-10-31  

---

## A. What Is Currently Implemented (`VERIFIED`)

1. **Dual-Path Asynchronous Diarization & ASR Pipeline (`phase20_standalone/`):**
   - Orchestrated by [`phase20_standalone/run_phase20.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/run_phase20.py#L1-L63) across 15 sequential subprocess steps (`step00_ground_truth.py` through `step10_compare_phase19.py`) backed by [`phase20_standalone/scripts/common.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/common.py#L1-L853).
   - **Path A (Diarization Stream):** Resamples audio to 16 kHz mono PCM (`step01_preprocess.py`), applies `DeepFilterNet3` spectral enhancement (`step02a_deepfilternet.py` -> `audio/<stem>_dfn.wav`), runs `pyannote/segmentation-3.0` VAD (`onset=0.6, offset=0.4`) with custom padding/merging (`pad=0.05s, min_off=0.25s, min_on=0.10s` in `step03_vad.py`), and executes NVIDIA NeMo `NeuralDiarizer` (`step05a_msdd.py`) using `titanet_large` (`192-D` embeddings across 5 multiscale windows `[1.5, 1.2, 1.0, 0.8, 0.5]s`), multiscale spectral clustering, and the `diar_msdd_telephonic` overlap-aware neural decoder.
   - **Path B (ASR Stream):** Applies `ffmpeg` EBU R128 loudness normalization (`I=-16 LUFS`, `step02b_loudnorm.py` -> `audio/<stem>_normalized.wav`), runs Faster-Whisper `large-v2` transcription + `WAV2VEC2_ASR_LARGE_LV60K_960H` forced word alignment (`step05b_whisperx.py`), and attributes words to MSDD speaker intervals via overlap + `±0.5s` drift-aware nearest-neighbor fallback (`step07_fusion.py`).
2. **Saved Multiscale TitaNet Embedding Artifacts (`35` `.pkl` files):**
   - Although `.wav` and `.nemo` files are git-ignored, **35 multiscale TitaNet embedding pickle files** (`subsegments_scale{0..4}_embeddings.pkl`) are preserved in `experiments/phase11_dfn_ablation/outputs_phase11A`, `outputs_phase11B`, `experiments/karen_11A_audit`, `experiments/phase9_production`, and `experiments/phase5_grounded_diarization`.
   - Direct inspection verified that TitaNet returns **unnormalized `torch.float32` tensors of shape `[N_k, 192]`** (mean L2 norm `0.737–0.854`), and that on `karen` (`[2278, 192]` at `0.5s` Scale 4), intra-speaker cosine similarity is `0.1932` vs. inter-speaker `0.0956`, yielding a **negative Cosine Silhouette Score of `-0.0493`** ([`experiments/phase18_tich_karen/analyze_embeddings.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/experiments/phase18_tich_karen/analyze_embeddings.py#L1-L120)).

---

## B. Critical Discrepancies & Risks Discovered (`VERIFIED`)

| ID | Finding (`VERIFIED` in Source Code & Artifacts) | Impact on Research |
|---|---|---|
| **1. Single-Speaker Collapse on `bwc` Masked by `90.27%` Accuracy** | In [`phase20_standalone/outputs/msdd/bwc.rttm`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/outputs/msdd/bwc.rttm#L1-L30) and [`metrics/bwc_fusion_metrics.json`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/metrics/bwc_fusion_metrics.json#L1-L15), NeMo MSDD collapsed the 4-speaker `bwc` recording into **a single speaker (`speaker_0`)**! Because `Speaker_1` utters `362/401` evaluated words (`90.27%`), predicting 1 speaker for the whole file yields `Speaker Accuracy = 90.27%` (`WAE = 9.73%`), while **`JER = 0.8424`** reveals that 3 of 4 speakers have `0.0` recall! | Proves why the single-encoder TitaNet pipeline fails on skewed BWC audio—and why **JER, Macro-SACC, cpWER, and Speaker Count Error** must be primary metrics alongside DER. |
| **2. Paper Tables 3–6 Report Single-Recording (`bwc`) Numbers** | While Base Paper Sec. 3.2 (Tables 1–2, pp. 4–5) describes the 8-recording corpus (`31m31s`, `41` speakers), Paper Sec. 5.1 (Tables 3–6, pp. 15–17) reports the exact single-recording metrics of `bwc` (`175.46s`, `DER=0.4471`, `JER=0.8424`, `WER=0.3244`, `SACC=0.9027`) from [`bwc_all_metrics.json`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/metrics/bwc_all_metrics.json#L1-L43). | Running and reporting the baseline + proposed multi-view architecture across **all 8 audio-available recordings** is an immediate major upgrade over the base paper. |
| **3. Paper vs. Code Pipeline Divergences** | (a) Paper claims a shared VAD timeline, but `run_phase20.py` runs `step03_vad.py` twice independently (`46` segs on Path A vs `41` segs on Path B), and `step05b_whisperx.py` ignores `bwc_pathB_segments.json` entirely.<br>(b) Paper Sec. 4.3 claims binary waveform zero-masking ($x(t)\cdot m(t)$) before TitaNet, whereas `step05a_msdd.py` passes unmasked `bwc_dfn.wav` + `bwc_pathA_oracle.rttm` to NeMo. | Must be explicitly documented in our methodology so experiments reflect actual code execution. |
| **4. Dataset & Ground-Truth Anomalies (`587` total in `annotation_validation.csv`)** | • **8 `.mp3` files exist** in `raw_audio/` (`1,893.7s` = `31.56 min`), while **4 of 12 `.txt` files have no audio**.<br>• Across the 8 audio files (`432` segments, `41` speaker entries), **`120` segments (`27.8%`) have zero duration (`start == end`)** due to `1s` integer timestamp rounding; **`5` of `41` speakers have `0.0s` of positive duration**, and **`1` is `Speaker_all`**.<br>• `Piers Confronts Tommy Robinson.txt` crashes `step00_ground_truth.py` due to 16 Unicode en-dashes `–` (`U+2013`) and `0:02:25` (`L79`). | Requires a non-destructive ground-truth sanitizer and strict **8-fold Leave-One-Recording-Out (LORO)** evaluation. |
| **5. Frozen `diar_msdd_telephonic` Coordinate-Space Lock (`D = 192`)** | While NeMo's `LongFormSpeakerClustering` is dimension-agnostic and rotation-invariant (operating on $N_4 \times N_4$ cosine similarity matrices), the frozen `EncDecDiarLabelModel` (`diar_msdd_telephonic`) has weights hardcoded to `emb_dim = 192` in TitaNet's exact coordinate basis, and computes cluster prototypes using an **unweighted `torch.mean()`** (`msdd_models.py` L829). | Fused representations must either drive multiscale spectral clustering + quality-weighted `192-D` prototypes (**Option 3**), or align to TitaNet's `192-D` basis via Orthogonal Procrustes + residual gating (**Option 2**). |

---

## C. What Is Feasible vs. Not Recommended (`VERIFIED`)

| Category | Feasible & Scientifically Sound (`RECOMMENDED`) | Not Feasible or Scientifically Flawed (`REJECTED`) |
|---|---|---|
| **Pretrained Encoders** | **NVIDIA TitaNet-Large (`192-D`, 1D-Conv + Global SE + Per-Feature Var Norm)** + **WeSpeaker ResNet293-LM (`256-D`, 2D-Conv Bottleneck + Mean-Only CMN)** extracted across the identical 5 multiscale windows (`[1.5, 1.2, 1.0, 0.8, 0.5]s`). | Fine-tuning the 25M–29M backbone weights on 8 BWC recordings (guaranteed catastrophic overfitting). |
| **Latent Alignment** | **Closed-Form Orthogonal Procrustes SVD ($P: \mathbb{R}^{256} \to \mathbb{R}^{192}$)** and **Z-Score Standardized Multiscale Cosine Affinity Calibration ($\tilde{S}^{(m,k)}$)**. | Unconstrained high-capacity MLP / Transformer projection layers (`>100K` params) trained from scratch on 312 segments. |
| **Reliability Indicators ($\mathbf{q}_i \in \mathbb{R}^8$)** | Zero-extra-model features already available in the pipeline:<br>1. **DFN3 Noise-Reduction Power ($\Delta E_{\text{DFN}}$)** from `16k_mono.wav` vs `dfn.wav`<br>2. **Segmental SNR ($\text{SNR}_{\text{seg}}$)**<br>3. **Subsegment Duration & Scale Index ($d_i, k$)**<br>4. **Pyannote `segmentation-3.0` Frame VAD Confidence ($\bar{p}_{\text{VAD}}$)**<br>5. **Pyannote `segmentation-3.0` Powerset Overlap Posterior ($\bar{p}_{\text{overlap}}$)**<br>6. **Pre-Normalization Embedding Norms ($\|\mathbf{e}_1\|_2, \|\mathbf{e}_2\|_2$)**<br>7. **Cross-Scale Self-Consistency ($c_{\text{scale},1}, c_{\text{scale},2}$)**. | Blind RT60 reverberation estimation on `0.5s` subsegments (mathematically ill-conditioned on sub-second moving BWC audio). |
| **Fusion Mechanism** | **Dual-Level Reliability-Aware Precision / Attention Fusion (`10–3,500` params):**<br>1. **Inter-Encoder View Precision Gate ($\alpha_{1,i}, \alpha_{2,i}$)** combining TitaNet and ResNet293-LM.<br>2. **Intra-Cluster Subsegment Quality Weight ($\beta_i$)** down-weighting overlapping/low-VAD subsegments in both spectral affinity construction and MSDD cluster-average prototype pooling (`get_cluster_avg_embs`). | Unconstrained Multi-Head Self-Attention + FFN (`~640K` params) over 2 views, or generative diffusion waveform restoration (which hallucinates vocal tracts and degrades biometric speaker identity). |
| **Training & Validation** | **Within-Recording Supervised Contrastive / Affinity Margin Loss** + **Synthetic BWC Degradation Pre-Calibration (`vad_aug_temp/synthesis.py`)**, evaluated under **Strict 8-Fold Leave-One-Recording-Out (LORO) Cross-Validation**. | Closed-set speaker classification (CE / ArcFace) over recording-local speaker IDs, or random segment splits that leak same-recording segments between train and test. |

---

## D. Recommended Research Architecture (`Stage 1+ Candidate`)

Our audit establishes that the strongest, most defensible architecture for **Section 6.2 (Multi-View Speaker Representation Learning)** is:

> **Multiscale Complementary Speaker Representation Learning with Precision-Weighted Reliability Gating and Quality-Aware MSDD Prototype Pooling**
> - **View 1:** `NVIDIA TitaNet-Large` (`192-D`, 1D depthwise-separable CNN + SE, per-feature mean/std norm).
> - **View 2:** `WeSpeaker ResNet293-LM` (`256-D`, 2D time-frequency ResNet-293, Kaldi fbank + mean-only CMN).
> - **Reliability Estimator ($\mathbf{q}_i^{(k)} \in \mathbb{R}^8$):** Extracts $\Delta E_{\text{DFN}}, \text{SNR}_{\text{seg}}, d_i, k, \bar{p}_{\text{VAD}}, \bar{p}_{\text{overlap}}, \|\mathbf{e}_m\|_2, c_{\text{scale},m}$ per subsegment.
> - **Dual-Stage Integration into NeMo `NeuralDiarizer`:**
>   1. **Stage 1 (Multiscale Spectral Clustering):** Reliability-gated fusion of standardized multiscale affinity matrices $\tilde{S}^{(1,k)}$ (`192-D`) and $\tilde{S}^{(2,k)}$ (`256-D`) weighted by view reliability $\alpha_{m,i}^{(k)}(\mathbf{q}_i)$ and subsegment quality $\beta_i^{(k)}(\mathbf{q}_i)$, preventing single-speaker eigengap collapse.
>   2. **Stage 2 (MSDD Overlap Decoder):** Procrustes-aligned residual `192-D` embeddings + $\beta_i$-weighted cluster prototype pooling in `ClusterEmbedding.get_cluster_avg_embs()`, eliminating overlap/noise prototype contamination in frozen `diar_msdd_telephonic`.

---

## E. Step-by-Step Roadmap (`Stages 1–9`)

1. **Stage 1 — Non-Destructive Multi-Recording Data & Audio Preparation:** Convert the 8 `raw_audio/*.mp3` files to `16k_mono.wav`, `dfn.wav`, and `normalized.wav`; sanitize `Piers Confronts Tommy Robinson.txt` Unicode en-dashes in-memory; generate clean single-speaker evaluation intervals.
2. **Stage 2 — True 8-Recording Baseline Reproduction:** Execute the baseline single-encoder TitaNet + MSDD + WhisperX pipeline across all 8 recordings and activate `fusion_metrics.py` (`DER`, `JER`, `Speaker Count Error`, `cpWER`, `saWER`, `Macro-SACC`).
3. **Stage 3 — Multiscale Dual-Encoder & Reliability Feature Extraction:** Extract and cache 5-scale TitaNet (`192-D`), WeSpeaker ResNet293-LM (`256-D`), and 8-D reliability vectors $\mathbf{q}_i^{(k)}$ across all 8 recordings (`~11,500` subsegments; `< 30 seconds` total GPU time on local RTX 4090).
4. **Stage 4 — Complementarity Verification Gate:** Run Linear CKA, Procrustes residual analysis, stratified EER/FDR/Silhouette across scales and SNR quartiles, and McNemar's $2\times 2$ error orthogonality test.
5. **Stage 5 — Zero-Training Fusion Baselines:** Evaluate Naive Concatenation (`448-D`), Standardized Cosine Affinity Averaging, and Closed-Form Orthogonal Procrustes `192-D` projection.
6. **Stage 6 — Reliability-Aware Precision Gate & Prototype Pooling:** Train/calibrate the lightweight reliability gate (`10–500` params) via 8-fold LORO cross-validation (and `vad_aug_temp` synthetic BWC mixtures).
7. **Stage 7 — End-to-End NeMo MSDD & WhisperX Attribution Integration:** Evaluate Option 3 (Decoupled Affinity + Quality-Weighted Prototypes) and Option 2 (Procrustes Residual `192-D` MSDD) on full diarization and word attribution.
8. **Stage 8 — Systematic Ablation & Sensitivity Analysis:** Ablate each reliability indicator in $\mathbf{q}_i$, view gating $\alpha_{m,i}$ vs. prototype weighting $\beta_i$, scale weights, and spectral clustering thresholds.
9. **Stage 9 — Publication Package Generation:** Compile final 8-recording tables, statistical significance tests, and figures.

---

## Index of Audit Deliverables in `research_audit/`

1. [`00_EXECUTIVE_SUMMARY.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/00_EXECUTIVE_SUMMARY.md)
2. [`01_REPOSITORY_INVENTORY.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/01_REPOSITORY_INVENTORY.md)
3. [`02_EXECUTION_PIPELINE_AUDIT.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/02_EXECUTION_PIPELINE_AUDIT.md)
4. [`03_TITANET_EMBEDDING_AUDIT.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/03_TITANET_EMBEDDING_AUDIT.md)
5. [`04_NEMO_MSDD_INTEGRATION_AUDIT.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/04_NEMO_MSDD_INTEGRATION_AUDIT.md)
6. [`05_DATASET_INVENTORY.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/05_DATASET_INVENTORY.md)
7. [`06_GROUND_TRUTH_VALIDATION.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/06_GROUND_TRUTH_VALIDATION.md)
8. [`07_ENCODER_COMPARISON.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/07_ENCODER_COMPARISON.md)
9. [`08_REPRESENTATION_COMPLEMENTARITY.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/08_REPRESENTATION_COMPLEMENTARITY.md)
10. [`09_RELIABILITY_FUSION_FEASIBILITY.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/09_RELIABILITY_FUSION_FEASIBILITY.md)
11. [`10_TRAINING_OBJECTIVES_AND_DATA_FEASIBILITY.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/10_TRAINING_OBJECTIVES_AND_DATA_FEASIBILITY.md)
12. [`11_GENERATIVE_AI_FEASIBILITY.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/11_GENERATIVE_AI_FEASIBILITY.md)
13. [`12_EVALUATION_FRAMEWORK_AUDIT.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/12_EVALUATION_FRAMEWORK_AUDIT.md)
14. [`13_ARCHITECTURAL_INTEGRATION_OPTIONS.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/13_ARCHITECTURAL_INTEGRATION_OPTIONS.md)
15. [`14_COMPUTATIONAL_REQUIREMENTS.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/14_COMPUTATIONAL_REQUIREMENTS.md)
16. [`15_RESEARCH_GAPS_AND_NOVELTY.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/15_RESEARCH_GAPS_AND_NOVELTY.md)
17. [`16_FINAL_RESEARCH_RECOMMENDATION.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/16_FINAL_RESEARCH_RECOMMENDATION.md)
18. [`dataset_inventory.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/dataset_inventory.csv)
19. [`speaker_statistics.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/speaker_statistics.csv)
20. [`recording_statistics.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/recording_statistics.csv)
21. [`annotation_validation.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/annotation_validation.csv)
22. [`evidence_index.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/evidence_index.md)
