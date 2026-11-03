# Stage 5B.2: Multi-Speaker Overlap Optimization & Attribution Protocol

## 1. Experimental Rationale & Objectives

In single-label speaker diarization, each time frame $t$ is constrained to at most one speaker label:
$$\hat{y}(t) \in \{\emptyset, S_1, S_2, \dots, S_K\}$$

When two or more speakers speak simultaneously ($|Y(t)| \ge 2$), assigning only a single speaker mechanically incurs a Missed Detection error for all secondary speakers:
$$\text{Miss}_{\text{overlap}} = \int_{\mathcal{T}} \max(0, |Y(t)| - 1) dt$$

In our 8-recording research corpus, ground-truth overlap accounts for:
- **51.08%** of active speech union in `R04_piers_robinson` (95.0s overlap out of 186.0s speech).
- **45.68%** of active speech union in `R02_dennis_skinner` (74.0s overlap out of 162.0s speech).
- In total, **169.0 seconds (97.1% of all overlap in the corpus)** is concentrated in R02 and R04.
- In single-label clustering (Condition J / Stage 5B.3 M1 / Stage 5B.1), Missed Detection in R04 is mathematically bounded below **$32.0\%$**, and in R02 below **$26.1\%$**.

The core objective of Stage 5B.2 is to investigate whether detecting overlapping speech and attributing secondary speakers can penetrate this structural miss floor without triggering unacceptable false alarms or speaker confusion in clean, low-overlap recordings (R01, R03, R05, R06, R07, R08).

---

## 2. Experimental Safeguards & Immutability Rules

1. **Frozen Artifacts**: All 298 artifacts from Stages 1–5A.1.2 and all Stage 5B.0, Stage 5B.3, and Stage 5B.1 audit-correction artifacts are strictly immutable and protected.
2. **Local Workspace**: All Stage 5B.2 code, configs, cache, metrics, plots, and reports reside strictly within:
   `research_stages/stage5b/stage5b_2_overlap_optimization/`
3. **Candidate VAD Integration**: Evaluates overlap on the three candidate VAD backbones established in Stage 5B.1 (`C00_baseline`, `C50_stream_mono`, `C62_comb_consensus_hyst_gap`).
4. **Recording-Local Identities**: No cross-recording speaker clustering is performed. All speaker labels are strictly recording-local.
5. **No Zero-Valued Fallbacks**: Every metric is explicitly computed from valid ground truth and aligned words.

---

## 3. Methodological Architecture

### 3.1 Overlap Detection Pipeline
We leverage the local Pyannote 3.1 PyanNet multi-speaker segmentation checkpoint (`pytorch_model.bin` located in `whisperx/assets/`).
- The model outputs multi-label frame activations for 3 speaker channels: `['speaker#1', 'speaker#2', 'speaker#3']`.
- We construct a continuous overlap sliding window aggregator via:
  $$\text{pre\_aggregation\_hook}(S) = \text{Top-2}(S)_{\text{second highest}}$$
  This produces a calibrated, continuous overlap posterior curve $p_{\text{overlap}}(t) \in [0, 1]$ aligned to the exact discrete time grid of VAD ($16.875$ ms stride).
- Posteriors are cached in `overlap_cache/{rec_id}_{stream}_overlap.npz` for both `mono` and `dfn` streams.
- Overlap intervals are binarized using threshold $\theta_{\text{ovl}}$, minimum overlap turn duration ($\text{min\_on} = 0.10$s), and gap merging ($\text{min\_off} = 0.15$s).

### 3.2 Secondary Speaker Attribution Strategies Evaluated

When an overlap interval $[t_{\text{start}}, t_{\text{end}}]$ is detected, the primary speaker $S_{\text{primary}}$ is inherited from the baseline clustering. The secondary speaker $S_{\text{secondary}}$ is attributed using one of five candidate strategies:

1. **M0 (Single-Label Baseline)**:
   No secondary speaker is assigned. Baseline single-label RTTM is retained.
2. **M1 (Local Temporal Context Attribution)**:
   In conversational dialogue, interruptions occur between active participants. We examine a temporal context window $[t_{\text{start}} - W, t_{\text{end}} + W]$ ($W=3.0$s) and identify the most prevalent active speaker $S \neq S_{\text{primary}}$.
3. **M2 (Dual-Encoder Prototype Affinity Attribution)**:
   Using the frozen 448-dimensional multi-scale dual-encoder embeddings (TitaNet-Large 192d + WeSpeaker 256d), we construct L2-normalized speaker centroids $\mathbf{c}_k = \frac{\sum_{i \in C_k} \mathbf{e}_i}{\|\sum_{i \in C_k} \mathbf{e}_i\|}$. For each overlap segment, we compute cosine similarity against all alternate prototypes $k \neq S_{\text{primary}}$ and assign the highest-affinity speaker.
4. **M3 (Turn-Alternation Ping-Pong Heuristic)**:
   Assigns the immediately preceding speaker ($S_{\text{prev}} \neq S_{\text{primary}}$) if active within the turn horizon, otherwise subsequent speaker ($S_{\text{next}}$).
5. **M4 (Conservative Abstention / Consensus Agreement)**:
   Requires agreement between preceding and subsequent alternating speakers ($S_{\text{prev}} == S_{\text{next}}$); if confidence is ambiguous or speakers disagree, the system explicitly **abstains** from adding a secondary speaker.
6. **M5 (Oracle Secondary Speaker Diagnostic)**:
   Diagnostic upper bound: assigns the true ground-truth secondary speaker to detected overlap regions. Isolates detection accuracy from attribution errors.

---

## 4. Evaluation Metrics & Standards

1. **Diarization Error Rate (NIST / pyannote)**:
   - Collar = 0.0s (strict boundary evaluation)
   - Collar = 0.25s (standard tolerance evaluation)
   - Component decomposition: Missed Detection Rate, False Alarm Rate, Speaker Confusion Rate
   - Jaccard Error Rate (JER)
2. **Overlap-Specific Metrics**:
   - Ground truth overlap duration ($T_{\text{ovl, ref}}$)
   - Hypothesis overlap duration ($T_{\text{ovl, hyp}}$)
   - Overlap Precision, Recall, and F1
3. **Conversational Intelligence Metrics**:
   - Concatenated Permutation Word Error Rate (cpWER)
   - Speaker-Attributed Word Error Rate (saWER)
   - Evaluated via `fusion_metrics.py` with dictionary schema `{"segments": segs_attr}`.
