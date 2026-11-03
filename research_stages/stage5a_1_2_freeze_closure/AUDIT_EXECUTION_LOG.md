# Stage 5A.1.2: Audit Execution Log

**Execution Date**: November 3, 2026  
**Auditor**: Independent Senior Machine Learning Auditor & Speech Processing Researcher  
**Git Commit**: `ddf089de986f9e7cf026aa507324f3fccaf0cd8f`  
**Execution Environment**: Linux x86_64, Python 3.10.12, PyTorch 2.5.1+cu124, NeMo 2.7.3 (`.venv_phase20`)

---

## 1. Environment and Integrity Baseline Initialization

1. **Working Tree and Commit Verification**:
   - `git rev-parse HEAD` returned `ddf089de986f9e7cf026aa507324f3fccaf0cd8f`.
   - `git status` confirmed clean working tree.
2. **Pre-Audit Baseline Inventory**:
   - Executed recursive SHA-256 calculation for all files under `research_stages/stage5a`, `research_stages/stage5a_1`, and `research_stages/stage5a_1_final_audit`.
   - 298 files hashed and stored in [`frozen_artifact_hashes_before.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/frozen_artifact_hashes_before.csv).

---

## 2. Audit A Execution: Verification Macro and Held-Out Predictions

- **Script**: [`recompute_heldout_verification.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/recompute_heldout_verification.py)
- **Execution Command**: `python3 research_stages/stage5a_1_2_freeze_closure/recompute_heldout_verification.py`
- **Output Artifact**: [`macro_reconciliation.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/macro_reconciliation.csv)
- **Key Findings**:
  - Raw held-out pair-level predictions were not serialized to disk; classification assigned: `AGGREGATE_CONSISTENCY_ONLY`.
  - Recomputed 6-rec eligible macro: TitaNet = 0.8238, ResNet = 0.8230, Equal Fusion = 0.8450, Gate = 0.8458.
  - Recomputed 5-rec sensitivity macro: TitaNet = 0.8172, ResNet = 0.8304, Equal Fusion = 0.8426, Gate = 0.8436.
  - Arithmetic complementarity confirmed: $|E_T \triangle E_R| = 2077$, $|E_T \cup E_R| = 3579$, fraction = 58.03%, Jaccard = 0.4197.

---

## 3. Audit B Execution: Exact Reliability-Gate Feature Vector Trace

- **Script**: [`trace_gate_features.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/trace_gate_features.py)
- **Execution Command**: `python3 research_stages/stage5a_1_2_freeze_closure/trace_gate_features.py`
- **Output Artifacts**:
  - [`gate_feature_trace.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/gate_feature_trace.csv)
  - [`GATE_FEATURE_VECTOR_VERIFICATION.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/GATE_FEATURE_VECTOR_VERIFICATION.md)
- **Key Findings**:
  - Executed feature vector strictly contains 3 features: `resnet293_l2_norm`, `dfn_noise_reduction_db`, `vad_conf_mean`.
  - WADA SNR and TitaNet embedding norm were never passed to the active gate model.

---

## 4. Audit C Execution: NME-SC Diagonal Invariance Verification

- **Script**: [`verify_nmesc_diagonal_invariance.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/verify_nmesc_diagonal_invariance.py)
- **Execution Command**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python research_stages/stage5a_1_2_freeze_closure/verify_nmesc_diagonal_invariance.py`
- **Output Artifact**: [`NMESC_DIAGONAL_INVARIANCE_AUDIT.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/NMESC_DIAGONAL_INVARIANCE_AUDIT.md)
- **Key Findings**:
  - `offline_clustering.py` includes diagonal in row-wise `torch.argsort`.
  - Sigmoid maps off-diagonals to $(0, 1)$, ensuring all off-diagonals are strictly $< 1.0$.
  - Resetting diagonal to $1.0$ guarantees it is maximal (rank 0 descending).
  - ScalerMinMax preserves rank 0 and off-diagonal ordering.
  - Neighborhood masks and symmetrized graphs match 100% on synthetic and real recordings.

---

## 5. Audit D Execution: Paired Cluster Bootstrap Reproduction

- **Script**: [`reproduce_paired_cluster_bootstrap.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/reproduce_paired_cluster_bootstrap.py)
- **Execution Command**: `python3 research_stages/stage5a_1_2_freeze_closure/reproduce_paired_cluster_bootstrap.py`
- **Output Artifacts**:
  - [`bootstrap_reproduction.csv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/bootstrap_reproduction.csv)
  - [`BOOTSTRAP_REPRODUCTION_REPORT.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_2_freeze_closure/BOOTSTRAP_REPRODUCTION_REPORT.md)
- **Key Findings**:
  - Reconciled $B=2,000$ (retained CSV, $[-2.07\%, -0.10\%]$) vs $B=10,000$ (sensitivity run, $[-2.11\%, -0.10\%]$).
  - Gate vs Equal Fusion DER upper bound touches $0.0000$ exactly because 7 of 8 recordings have zero difference ($34.4\%$ zero resamples).
  - Equal fusion (37.05% cpWER) does not outperform ResNet (36.44%).
  - Results strictly qualified as nominal exploratory improvements over 8 clusters.

---

## 6. Post-Audit File Integrity Check

1. Recomputed SHA-256 for all 298 files in protected directories and saved to `frozen_artifact_hashes_after.csv`.
2. Verified bitwise identity against `frozen_artifact_hashes_before.csv`. Result: 0 modified, 0 missing. Recorded in `frozen_artifact_integrity_check.txt`.
