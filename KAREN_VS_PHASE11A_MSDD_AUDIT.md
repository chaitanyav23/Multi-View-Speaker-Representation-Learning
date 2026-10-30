# LINE-BY-LINE MSDD CONFIGURATION AUDIT: KAREN VS. PHASE 11A

This document identifies every parameter delta between the Karen Audit script (`run_karen_audit.py`) and the Phase 11A Baseline (`experiments/phase11_dfn_ablation/step3_run_pipelines.py`).

## 1. MANIFEST GENERATION (Data Metadata)

| Line Context | Phase 11A (Baseline) | Karen Audit | Change Type |
| :--- | :--- | :--- | :--- |
| `"num_speakers"` | `2` | `4` | **INCREASE** |

**Rationale:** The BWC dataset used in Phase 11A was restricted to 1-on-1 interactions. The Karen dataset is a multi-speaker forensic event (Officer, Subject, and background voices/dispatch), requiring a higher manifest capacity.

---

## 2. MSDD MODEL PARAMETERS (`msdd_model.parameters`)

| Parameter | Phase 11A (Baseline) | Karen Audit | Change Type |
| :--- | :--- | :--- | :--- |
| `overlap_infer_spk_limit` | `3` | `5` | **INCREASE** |

**Rationale:** This parameter defines the maximum number of speakers the model will attempt to resolve in an overlapping speech region. In the chaotic Karen audio (specifically the arrest and shouting segments), the probability of $N > 3$ speakers overlapping is significantly higher than in the controlled BWC baseline.

---

## 3. CLUSTERING PARAMETERS (`clustering.parameters`)

| Parameter | Phase 11A (Baseline) | Karen Audit | Change Type |
| :--- | :--- | :--- | :--- |
| `max_num_speakers` | `2` | `5` | **INCREASE** |
| `oracle_num_speakers` | `True` | `False` | **LOGIC FLIP** |

**Rationale (Max Speakers):** Allows the clustering engine to discover up to 5 distinct acoustic identities.
**Rationale (Oracle Logic):** In Phase 11A, the model was **FORCED** to produce exactly 2 speakers. In the Karen audit, the model was set to **INFERENCE** mode. By setting `oracle_num_speakers: False`, the system attempts to determine the true speaker count automatically based on the variance in the TitaNet embeddings.

---

## 4. UNCHANGED CORE PARAMETERS (Consistency Audit)

The following parameters were held **CONSTANT** to maintain scientific comparability:
* **`model_path`**: Both use `diar_msdd_telephonic`.
* **`oracle_vad`**: Both use `True` (System VAD injection via RTTM).
* **`multiscale_weights`**: Both use `[1, 2, 3, 3, 3]` (High-resolution embedding weights).
* **`window_length_in_sec`**: Both use multi-scale windows ranging from `0.5s` to `1.5s`.
* **`max_rp_threshold`**: Both use `0.25` (Sensitivity for speaker re-identification).

## SUMMARY OF BREAKTHROUGH
The Karen Audit script represents a "production-generalized" version of the Phase 11A script. It transitions from a **constrained, hardcoded research tool** (Fixed-2) into a **flexible forensic inference engine** capable of automatic speaker discovery in high-complexity acoustic environments.
