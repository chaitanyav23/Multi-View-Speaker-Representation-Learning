# Stage 5A.1.2: Exact Reliability-Gate Feature Vector Verification

**Audit Date**: November 3, 2026  
**Auditor**: Independent Senior Machine Learning Auditor & Speech Processing Researcher  
**Target Subsystem**: Reliability-Aware Affinity Fusion Gate Model  
**Audit Status**: RESOLVED AND VERIFIED  
**Repository State**: Git Commit `ddf089de986f9e7cf026aa507324f3fccaf0cd8f`

---

## 1. Executive Resolution of Feature Vector Discrepancy

Prior project documentation contained two conflicting feature descriptions:
1. **Earlier Three-Feature Description**: ResNet embedding L2 norm, DeepFilterNet noise reduction proxy, and VAD confidence.
2. **Later Four-Feature Description**: WADA SNR, VAD confidence, TitaNet embedding norm, and ResNet embedding norm (appearing in the Stage 5A.1 Final Audit Report prose and freeze record).

An exhaustive source-code trace across extraction, training, serialization, and inference establishes conclusively that **the executed gate model strictly uses the Three-Feature Vector**:

$$\mathbf{x}_{rel} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} \text{resnet293\_l2\_norm} \\ \text{dfn\_noise\_reduction\_db} \\ \text{vad\_conf\_mean} \end{bmatrix}$$

The four-feature description in the previous report was an erroneous narrative description that mistakenly conflated exploratory feature analysis with the active pipeline. WADA SNR and TitaNet embedding norm **never entered the executed gate model**.

---

## 2. Source-Code and Pipeline Trace

### 2.1 Feature Definitions and Origins
- **$x_1$: `resnet293_l2_norm`**
  - **Extraction Source**: [`research_stages/stage3_dual_encoder/run_stage3_extraction.py:560`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage3_dual_encoder/run_stage3_extraction.py#L560)
  - **Transformation**: Raw Euclidean norm computed on the linear projection output of WeSpeaker ResNet293-LM *before* unit L2-normalization: `np.linalg.norm(res_raw, axis=-1)`.
  - **Role**: Serves as an internal encoder confidence metric (higher norm indicates stronger speaker feature activation).

- **$x_2$: `dfn_noise_reduction_db`**
  - **Extraction Source**: [`research_stages/stage1_data_prep/scripts/run_stage1_prep.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage1_data_prep/scripts/run_stage1_prep.py) and cached in Stage 3 metadata.
  - **Transformation**: Energy ratio in decibels between original mono audio and DeepFilterNet residual:
    $$\text{SNR}_{\text{reduction}} = 10 \cdot \log_{10} \left( \frac{\mathcal{P}_{\text{mono}}}{\max(\mathcal{P}_{\text{mono} - \text{dfn}},\, 10^{-12})} \right)$$
  - **Role**: Measures local environmental noise suppression by DeepFilterNet3.

- **$x_3$: `vad_conf_mean`**
  - **Extraction Source**: [`research_stages/stage3_dual_encoder/run_stage3_extraction.py:270-285`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage3_dual_encoder/run_stage3_extraction.py#L270-L285)
  - **Transformation**: Frame-level posterior speech probabilities from pyannote VAD averaged over each subsegment window.
  - **Role**: Reflects speech activity certainty within the subsegment window.

### 2.2 Gate Training and Inference Code Evidence
In [`research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py):
- **Feature Vector Definition (Line 307)**:
  ```python
  gate_features = ["resnet293_l2_norm", "dfn_noise_reduction_db", "vad_conf_mean"]
  ```
- **Training Input Assembly (Lines 298–302)**:
  ```python
  loto_subseg_items.append({
      "rec_id": rec_id,
      "subseg_idx": i,
      "turn_id": t_id,
      "speaker": s,
      "delta_margin": float(delta_m),
      "res_margin": float(rm),
      "tita_margin": float(tm),
      "resnet293_l2_norm": float(meta_mono[i]["resnet293_l2_norm"]),
      "dfn_noise_reduction_db": float(meta_dfn[i]["dfn_noise_reduction_db"]),
      "vad_conf_mean": float(meta_mono[i]["vad_conf_mean"]),
      "res_wins": int(rm > tm),
  })
  ```
- **Model Fitting under LORO (Lines 310–314)**:
  ```python
  for held_rec in recordings:
      tr_items = [x for x in loto_subseg_items if x["rec_id"] != held_rec]
      gate = ReliabilityGate(features=gate_features, regularization_lambda=0.05, clip_min=0.15, clip_max=0.85, gain_factor=2.0)
      gate.fit(tr_items)
      gate_models_linear[held_rec] = gate
  ```
- **Inference Input Assembly (Lines 376–385)**:
  ```python
  meta_gate_items = [
      {
          "resnet293_l2_norm": float(meta_s4_mono[i]["resnet293_l2_norm"]),
          "dfn_noise_reduction_db": float(meta_s4_dfn[i]["dfn_noise_reduction_db"]),
          "vad_conf_mean": float(meta_s4_mono[i]["vad_conf_mean"]),
      }
      for i in range(len(meta_s4_dfn))
  ]
  g_model = gate_models_linear[rec_id]
  G_mat_linear = g_model.build_affinity_gate_matrix(meta_gate_items, device=torch.device("cuda"))
  ```

Identical 3-feature configurations exist in Stage 5A ([`research_stages/stage5a/scripts/run_stage5a_eval.py:307`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/scripts/run_stage5a_eval.py#L307)).

---

## 3. Resolution Table for All Candidate Features

| Candidate Feature | Status in Executed Pipeline | Evidence Path & Line Numbers | Reason / Role |
| :--- | :---: | :--- | :--- |
| **ResNet embedding norm** (`resnet293_l2_norm`) | **USED** (Index 0) | `stage5a_1/scripts/run_stage5a_1_controlled_eval.py:298, 307` | Captures unnormalized ResNet feature magnitude. |
| **DeepFilterNet noise reduction** (`dfn_noise_reduction_db`) | **USED** (Index 1) | `stage5a_1/scripts/run_stage5a_1_controlled_eval.py:299, 307` | Measures local acoustic enhancement degree. |
| **VAD confidence** (`vad_conf_mean`) | **USED** (Index 2) | `stage5a_1/scripts/run_stage5a_1_controlled_eval.py:300, 307` | Measures speech segment posterior probability. |
| **WADA SNR** (`wada_snr`) | **NOT USED** | `stage5a_1/scripts/run_stage5a_1_controlled_eval.py:307` | Erroneously reported in prior audit text; not in active feature vector. |
| **TitaNet embedding norm** (`titanet_embedding_norm`) | **NOT USED** | `stage5a_1/gate_audit/gate_audit_results.json` | NeMo TitaNet unit-normalizes internally ($1.0 \pm 10^{-6}$), yielding zero variance. |

---

## 4. Training and Inference Parity

The gate implementation in [`research_stages/stage5a/scripts/fusion_models.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a/scripts/fusion_models.py) guarantees strict parity:
1. **Feature Ordering**: In `ReliabilityGate.fit()`, features are mapped to columns via `X = np.array([[item[f] for f in self.features] for item in training_items])`. At inference (`predict_delta_margin`), the identical comprehension `[[item[f] for f in self.features] ...]` is evaluated.
2. **Standardization**: Training mean $\boldsymbol{\mu}_x$ and standard deviation $\boldsymbol{\sigma}_x$ are saved as instance attributes during `fit()` and applied identically during inference:
   $$\tilde{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}_x}{\boldsymbol{\sigma}_x + \epsilon}$$
3. **Weight Application**: The predicted delta margin is:
   $$\Delta_{\text{pred}} = \mathbf{w}^T \tilde{\mathbf{x}} + b$$
   The subsegment gate weight is:
   $$g_{\text{res}} = \text{clip}(0.5 + 2.0 \cdot \Delta_{\text{pred}},\, 0.15,\, 0.85)$$
   The pairwise affinity gate matrix is symmetric:
   $$G_{ij} = \frac{g_{\text{res}, i} + g_{\text{res}, j}}{2}$$
   $$A_{\text{fused}} = (1 - G) \odot \tilde{A}_{\text{tita}} + G \odot \tilde{A}_{\text{res}}$$

**Audit Conclusion**: The exact executed gate feature vector is definitively established as a 3-dimensional vector $[\text{resnet293\_l2\_norm}, \text{dfn\_noise\_reduction\_db}, \text{vad\_conf\_mean}]$.
