# 10 — Training Objectives & Data Sufficiency Feasibility (`Phase H`)

**Audit Status:** `VERIFIED`  
**Supporting Data:** [`speaker_statistics.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/speaker_statistics.csv), [`recording_statistics.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_audit/recording_statistics.csv), [`vad_aug_temp/vad_eval/synthesis.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/vad_aug_temp/vad_eval/synthesis.py#L1-L120)

---

## 1. Exact Quantitative Data Sufficiency Audit (`Phase H1`)

From our verified audit of the 8 audio-available recordings in [`raw_audio/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/raw_audio) and [`ground_truth/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/ground_truth):

| Corpus Metric | Exact Verified Count (All 8 Audio Recordings) | Exact Count After Removing Overlapping & `<0.5s` Intervals | Sufficiency Assessment for Deep Metric Learning |
|---|---|---|---|
| **Recordings with Physical Audio** | `8` recordings (`31.56 min` total audio) | `8` recordings (`23.33 min` union speech) | Extremely small for recording-level split unless evaluated via Leave-One-Recording-Out (8-fold LORO CV). |
| **Annotated Speaker IDs (Recording-Local)** | `41` (`40` real speakers + `1` `Speaker_all`) | `35` speakers with `>0s` speech | **Insufficient for closed-set speaker classification (CE / AAM-Softmax)** without severe overfitting. |
| **Speakers with $\ge 2$ Positive-Duration Segments** | `25` speakers | `21` speakers (with $\ge 2$ clean non-overlapping segments $\ge 1.0\text{s}$) | Only `21–25` speakers can form genuine cross-turn positive pairs $(i, j)$ from distinct speaker turns! |
| **Positive-Duration Annotated Segments** | `312` segments (`1,625.0s` sum) | `~240` non-overlapping single-speaker segments | Too small to train high-capacity projection matrices ($>50\text{K}$ params) from random initialization. |
| **Multiscale Subsegments (`5` scales per recording)** | `~11,500` subsegments across all 8 recordings (`899` on `bwc`, `6,871` on `karen`) | `~8,200` subsegments mapped to single-speaker GT intervals | Subsegments within the same turn are highly correlated (50% overlap), so effective independent sample size is still bounded by the `~240–312` speaker turns. |

---

## 2. Feasibility Analysis of Candidate Training Objectives (`Phase H2`)

We evaluated six candidate training objectives against the exact data constraints above:

| Training Objective | Mathematical Loss Function | Requires Speaker Class Head? | Works with 21–25 Training Speakers? | Feasibility & Failure Modes on Local BWC Dataset |
|---|---|---|---|---|
| **1. Closed-Set Speaker Classification (CE / ArcFace / AAM-Softmax)** | $\mathcal{L}_{\text{AAM}} = -\log \frac{e^{s\cos(\theta_{y_i} + m)}}{e^{s\cos(\theta_{y_i} + m)} + \sum_{c \neq y_i} e^{s\cos\theta_c}}$ | **YES** ($W_{\text{cls}} \in \mathbb{R}^{d \times C}$) | **NO (`INFEASIBLE` on BWC alone)** | With only $C \approx 21\text{–}35$ recording-local speakers across 7 training recordings (in 8-fold LORO), a classification head learns recording-channel shortcuts (identifying which of the 7 recordings a segment came from via background acoustics rather than speaker identity!). |
| **2. Within-Recording Supervised Contrastive Loss (`SupCon`)** | $\mathcal{L}_{\text{SupCon}} = \sum_{i} \frac{-1}{|P(i)|}\sum_{p \in P(i)} \log \frac{\exp(\langle \mathbf{z}_i, \mathbf{z}_p \rangle / \tau)}{\sum_{a \in A_{\text{rec}}(i)} \exp(\langle \mathbf{z}_i, \mathbf{z}_a \rangle / \tau)}$ | **NO** | **YES (`HIGHLY FEASIBLE`)** | **Crucial Design Rule:** Restrict the denominator $A_{\text{rec}}(i)$ to **same-recording negatives** (or hard negatives from the same acoustic environment)! Why? If negatives come from other recordings, the model can trivially separate pairs using background noise/channel differences instead of speaker voice characteristics. Within-recording SupCon forces the fusion gate to separate speakers sharing the exact same BWC microphone and acoustic environment! |
| **3. Affinity Matrix / Cosine Margin Alignment Loss** | $\mathcal{L}_{\text{aff}} = \frac{1}{|\mathcal{P}_+|} \sum_{(i,j)\in\mathcal{P}_+} (1 - S^{\text{fused}}_{ij})^2 + \frac{1}{|\mathcal{P}_-|} \sum_{(i,j)\in\mathcal{P}_-} \max(0, S^{\text{fused}}_{ij} - m_{\text{neg}})^2$ | **NO** | **YES (`OPTIMAL FOR CLUSTERING`)** | Directly optimizes the exact quantity consumed by NeMo's `LongFormSpeakerClustering`—the multiscale cosine affinity matrix $S^{\text{fused}}_{ij}$! Works seamlessly with low-parameter reliability gates (`9–150` parameters). |
| **4. Self-Supervised Cross-View Contrastive Alignment (InfoNCE / CLIP-style)** | $\mathcal{L}_{\text{align}} = -\log \frac{\exp(\langle \hat{\mathbf{e}}_{1,i}, \hat{\mathbf{e}}_{2,i} \rangle / \tau)}{\sum_{j} \exp(\langle \hat{\mathbf{e}}_{1,i}, \hat{\mathbf{e}}_{2,j} \rangle / \tau)}$ | **NO** (Requires **zero** speaker labels!) | **YES (`SELF-SUPERVISED`)** | Aligns ResNet293-LM's 256-D space and TitaNet's 192-D space into a shared coordinate basis across all `~11,500` multiscale subsegments without needing ground-truth speaker labels! Even stronger when initialized in closed form via **Orthogonal Procrustes SVD**. |
| **5. Multi-Scale / Denoising Consistency Loss** | $\mathcal{L}_{\text{cons}} = 1 - \cos(\mathbf{z}^{(k)}_i, \mathbf{z}^{(k-1)}_{\pi(i)}) + 1 - \cos(\mathbf{z}_i(x_{\text{dfn}}), \mathbf{z}_i(x_{\text{norm}}))$ | **NO** (Self-supervised) | **YES (`FEASIBLE`)** | Penalizes view/scale instability between `0.5s` base-scale embeddings and `1.0s/1.5s` coarse-scale embeddings, and between Path A (`dfn`) and Path B (`normalized`) audio views. |
| **6. Teacher-Student Oracle-Gate Distillation** | $\mathcal{L}_{\text{gate}} = \text{BCE}\big(g_\theta(\mathbf{q}_i, \mathbf{q}_j),\, y^*_{\text{oracle}}(i,j)\big)$ where $y^*_{\text{oracle}} = \mathbb{I}[\ell_2(i,j) < \ell_1(i,j)]$ | **NO** | **YES (`HIGHLY INTERPRETABLE`)** | Trains the lightweight reliability estimator $g_\theta(\mathbf{q})$ directly to predict which encoder (TitaNet vs. ResNet293-LM) achieves higher ground-truth speaker separation on subsegment pair $(i, j)$! |

---

## 3. Prevention of Data Leakage & Overfitting Protocol (`Phase H3`)

Because the local corpus has 8 audio recordings (`1,893.7s`), **any evaluation that trains and tests on segments from the same recording is scientifically invalid (severe speaker and acoustic channel leakage).**

### Mandatory Cross-Validation & Anti-Leakage Protocol:
1. **Strict Leave-One-Recording-Out (8-Fold LORO) Cross-Validation:**
   - For each held-out test recording $r \in \{1, \dots, 8\}$, all fitting of Procrustes alignment $P$, z-score standardization statistics $(\mu, \sigma)$, reliability gate weights $\boldsymbol{\theta}$, and clustering hyperparameters must use **only the remaining 7 recordings** $\mathcal{D}_{\text{train}}^{(-r)} = \{1, \dots, 8\} \setminus \{r\}$.
   - Never split segments of the same recording across train and test sets.
2. **Ground-Truth Segment Hygiene for Training Folds:**
   - Exclude `Speaker_all` (`UK Parliament Debate`).
   - Subtract any time interval where $\ge 2$ ground-truth speakers overlap (`Dennis Skinner`, `Piers Confronts Tommy Robinson`, `bwc`) before slicing training subsegments for supervised contrastive / affinity losses, so that overlapping speech is never labeled as a clean single speaker.
   - Intersect integer-rounded GT intervals `[start, end]` with Path A VAD intervals so silence/noise margins from 1-second timestamp rounding are excluded from speaker prototypes.
3. **Exploiting the Existing Synthetic BWC Augmentation Pipeline (`vad_aug_temp/`) (`VERIFIED` Opportunity):**
   - Inspecting [`vad_aug_temp/vad_eval/synthesis.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/vad_aug_temp/vad_eval/synthesis.py#L1-L120) reveals that the repository **already contains a complete, seed-controlled BWC acoustic degradation simulator** (`wind`, `siren`, `crowd`, `traffic`, `rustle`, `clipping`, `muffle`, `reverb` at calibrated SNRs `[-10, -5, 0, +5, +10, +20] dB`)!
   - If a higher-capacity reliability network ($>500$ parameters) is desired, it can be pre-trained on controlled synthetic BWC degradations (where true SNR, true degradation type, and clean reference embeddings are known exactly) and evaluated **zero-shot** on the 8 real BWC recordings!
