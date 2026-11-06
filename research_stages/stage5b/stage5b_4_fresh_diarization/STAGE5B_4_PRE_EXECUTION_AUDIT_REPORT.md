# Stage 5B.4 — Pre-Execution Audit & Technical Readiness Report

**Authoritative Recommendation**: **`CONDITIONAL GO`**  
**Audit Date**: 2026-11-06  
**Auditor**: Senior ML Research Engineer & Diarization Systems Auditor  
**Repository**: [Multi-View-Speaker-Representation-Learning](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning)  
**Deliverable Path**: `research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_PRE_EXECUTION_AUDIT_REPORT.md`

---

## 1. Executive Summary

This report delivers the authoritative **Pre-Execution Audit** for **Stage 5B.4 (Fresh Diarization Evaluation)**. 

Stage 5B.4 is designed to evaluate whether the Voice Activity Detection (VAD) optimizations from Stage 5B.1 and the multi-speaker overlap attribution methods from Stage 5B.2 translate into genuine performance gains when integrated into an **end-to-end, fresh diarization pipeline**. The proposed protocol tests a $3 \times 2$ factorial matrix comprising six experimental conditions (`E00`, `E01`, `E10`, `E11`, `E20`, `E21`) across all eight conversational benchmark recordings (R01–R08).

### Authoritative Finding:
The experimental protocol is scientifically justified, the theoretical formulation is rigorous, and the physical execution environment (hardware, libraries, neural network checkpoints) is fully functional and compliant. 

However, forensic audit of the repository revealed **two critical technical blockers**:
1. **Absence of an Integrated Fresh Diarization Script**: Existing scripts in `stage2_baseline`, `stage3_dual_encoder`, `stage5a_1`, and `stage5b_1` are decoupled. Stage 5B.1 evaluated VAD purely via transductive nearest-neighbor projection of frozen Stage 5A Condition J labels, and Stage 5B.2 evaluated overlap on those projected RTTMs. No script currently connects candidate VAD generation $\to$ dynamic subsegmentation $\to$ dual-encoder embedding extraction $\to$ Condition J equal fusion $\to$ fresh NME-SC clustering.
2. **Embedding Cache Invalidation Flaw**: The existing Stage 3 embedding cache schema lacks VAD configuration identifiers and subsegment timestamp hashes. Executing fresh conditions using the existing cache structure would either cause cache collisions, contaminate frozen Stage 3 artifacts, or mistakenly load stale `C00` embeddings for `C50` and `C62`.

Therefore, the final recommendation for Stage 5B.4 is **`CONDITIONAL GO`**. Execution is paused until a verified orchestration adapter and collision-free cache schema are deployed and validated via a single-recording dry run.

---

## 2. Experimental Design & The Six Conditions

The Stage 5B.4 factorial matrix evaluates three VAD operating points crossed with two overlap attribution strategies:

```
+--------------------------------------------------------------------------------------------------------------------+
| Condition | VAD Operating Point      | Overlap Strategy                 | Primary Scientific Purpose               |
+--------------------------------------------------------------------------------------------------------------------+
| E00       | C00: Path A DFN VAD      | O02: Single-Label Baseline (M0)  | Primary reference benchmark              |
| E01       | C00: Path A DFN VAD      | O42-R2: Conservative Abstain(M4) | Isolated overlap attribution effect      |
| E10       | C50: Raw Mono 16k VAD    | O02: Single-Label Baseline (M0)  | Isolated high-recall VAD effect          |
| E11       | C50: Raw Mono 16k VAD    | O42-R2: Conservative Abstain(M4) | Combined high-recall VAD + overlap       |
| E20       | C62: Consensus VAD       | O02: Single-Label Baseline (M0)  | Isolated robust consensus VAD effect     |
| E21       | C62: Consensus VAD       | O42-R2: Conservative Abstain(M4) | Full proposed integration pipeline       |
+--------------------------------------------------------------------------------------------------------------------+
```

### Component Clarifications & Disambiguations:
- **`O02` Nomenclature**: In Stage 5B.2, `O02` specifically denoted `C62 x M0_no_overlap`. In Stage 5B.4, `O02` is used generically to denote the **`M0_no_overlap` single-label baseline** across *all* three VAD conditions. The implementation orchestrator must bind directly to `M0_no_overlap` to avoid configuration lookup errors.
- **`O42-R2` Compatibility**: Confirmed **fully compatible with fresh diarization**. The underlying function `attribute_m3_m4_corrected` operates solely on the rasterized single-speaker hypothesis RTTM and local mono overlap posteriors. It is completely agnostic to cluster label semantics and requires zero knowledge of ground-truth identities.

---

## 3. Forensic Codebase Audit & The Missing Pipeline Path

A line-by-line inspection of repository scripts established the following findings:

1. **`research_stages/stage3_dual_encoder/run_stage3_extraction.py`**:
   - Lines 413–424 hardcode loading subsegments from `stage2_baseline/msdd/{rec_id}_msdd_outputs/speaker_outputs/subsegments_scale{k}.json`.
   - These subsegments represent `C00` (Path A DFN) only. The script cannot dynamically segment audio for `C50` or `C62`.
2. **`research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py`**:
   - Hardcodes loading embeddings from `research_stages/stage3_dual_encoder/cache/`.
   - Cannot cluster fresh segmentations.
3. **`research_stages/stage5b/stage5b_1_vad_optimization/`**:
   - Used nearest-neighbor label propagation (`project_vad_to_diarization_audited`) instead of spectral clustering.
   - For recordings R01, R02, and R05, up to 20% of speech received nearest-neighbor propagated labels.
4. **`research_stages/stage5b/stage5b_2_overlap_optimization/`**:
   - Applied overlap attribution on top of Stage 5B.1 projected RTTMs.

**Conclusion**: The repository currently contains all individual mathematical primitives (subsegmentation, feature extraction, affinity construction, z-score normalization, NME-SC clustering, overlap attribution), but lacks a **unified orchestrator** to execute them in sequence for fresh VAD segmentations.

---

## 4. Embedding Cache Architecture Audit

The existing Stage 3 cache in `research_stages/stage3_dual_encoder/cache/` organizes files as:
`{cond}/{rec_id}/titanet_large__{cond}__scale{scale_k}.npz`

### Identified Vulnerability:
1. The cache key contains no reference to the VAD operating point (`C00`, `C50`, `C62`).
2. The subsegment metadata contains timestamps but no cryptographic hash of the input speech intervals.
3. Writing fresh embeddings into the Stage 3 directory would overwrite frozen baseline artifacts, violating repository preservation rules.
4. Reading from the Stage 3 directory for `C50` or `C62` would result in cache hits on stale `C00` embeddings, corrupting downstream evaluations.

### Corrective Specification:
Stage 5B.4 must enforce a self-contained, collision-free cache under:
`research_stages/stage5b/stage5b_4_fresh_diarization/cache/{vad_id}/{rec_id}/`

Using the composite naming convention:
`{encoder}__{cond}__scale{scale_k}__{vad_intervals_sha256[:16]}.npz`

---

## 5. Downstream Evaluation & Metrics Verification

Evaluation procedures were audited to guarantee scientific rigor:
- **DER Evaluation**: Both primary benchmark (`collar=0.0s`, overlap evaluated) and secondary NIST benchmark (`collar=0.25s`, overlap evaluated) will be computed and reported side-by-side using `pyannote.metrics`.
- **Component Decomposition**: Missed speech rate, false alarm rate, and speaker confusion rate will be tracked per condition.
- **Speaker Count MAE**: Eigengap-estimated $\hat{K}$ vs ground-truth $K^*$ will be evaluated for all 48 evaluations ($6 \text{ conditions} \times 8 \text{ recordings}$).
- **Conversational cpWER**: Word attribution will strictly wrap segments in the canonical schema `{"segments": segs_attr}` required by `phase20_standalone/scripts/fusion_metrics.py`, avoiding the schema defect identified in Stage 5B.1.
- **Statistical Aggregation**: Macro-averages across $N=8$ recordings will be reported alongside duration-weighted pooled averages, accompanied by paired Wilcoxon signed-rank tests and 10,000-iteration clustered bootstrap confidence intervals.

---

## 6. Dependency, Checkpoint & Hardware Readiness

All technical prerequisites were inspected and validated:
- **GPU**: NVIDIA GeForce RTX 4090 (24,564 MiB VRAM), Driver 570.86.16, CUDA 12.8.
- **Software Stack**: Python 3.10.20, PyTorch 2.8.0+cu128, NeMo 2.7.3, pyannote-audio 4.0.4, whisperx 3.8.5, onnxruntime-gpu 1.23.2.
- **Local Checkpoints**:
  - Pyannote PyanNet: `research_stages/.venv_phase20/lib/python3.10/site-packages/whisperx/assets/pytorch_model.bin` (17,719,103 bytes, SHA-256: `0b5b3216...`)
  - WeSpeaker ResNet293-LM: `avg_model.pt` (133,998,630 bytes, SHA-256: `e7b8447e...`) and `voxceleb_resnet293_LM.onnx` (114,336,285 bytes, SHA-256: `dbb1ccc7...`)
  - NVIDIA TitaNet-Large: `diar_msdd_telephonic.nemo` (107,609,008 bytes, SHA-256: `450f6bd2...`)
- **Computational Budget**: Total estimated execution time across all 6 conditions is **12 to 15 minutes**, requiring $< 4.5$ GiB GPU VRAM and $< 6$ GiB host RAM.

---

## 7. Critical Blockers & Remediation Plan

```
+------------------------------------------------------------------------------------------------------------------+
| Blocker Identified                         | Severity | Impact on Execution       | Required Remediation         |
+------------------------------------------------------------------------------------------------------------------+
| 1. Missing Integrated Fresh Diarization    | Critical | Cannot execute fresh      | Author and test modular      |
|    Orchestration Script                    |          | clustering or attribution | pipeline orchestrator script |
+------------------------------------------------------------------------------------------------------------------+
| 2. Unkeyed Stage 3 Embedding Cache Schema  | High     | Cache collision or stale  | Implement VAD-aware isolated |
|    (Missing VAD hash & ID)                 |          | embedding leakage         | cache in stage5b_4 workspace |
+------------------------------------------------------------------------------------------------------------------+
| 3. Overlap Condition Nomenclature Ambiguity| Moderate | Incorrect parameter lookup| Explicitly bind conditions   |
|    (O02 vs M0_no_overlap)                  |          | during attribution        | to semantic method names     |
+------------------------------------------------------------------------------------------------------------------+
```

---

## 8. Authoritative Recommendation & Pre-Execution Conditions

### Recommendation: **`CONDITIONAL GO`**

Execution of Stage 5B.4 is authorized **ONLY AFTER** the following mandatory pre-execution conditions are satisfied:

1. **Precondition 1 (Orchestration Module)**:
   Author a clean, standalone execution script `research_stages/stage5b/stage5b_4_fresh_diarization/scripts/run_stage5b_4_fresh_diarization.py` that implements the complete fresh pipeline without reusing stale cluster labels.
2. **Precondition 2 (Isolated Cache Implementation)**:
   Ensure the orchestrator writes all intermediate subsegment manifests and embeddings exclusively to `stage5b_4_fresh_diarization/cache/`, keyed by recording ID, VAD configuration ID, and VAD interval SHA-256 hash.
3. **Precondition 3 (Phase B Dry Run)**:
   Execute a single-recording dry run (on `R01_sara_sharif`) across conditions `E00`, `E01`, and `E21` to verify that:
   - Fresh subsegments are properly generated from candidate VAD intervals.
   - TitaNet and WeSpeaker feature extraction runs without shape mismatches.
   - Multi-scale cosine affinity and off-diagonal z-score fusion execute correctly.
   - NME-SC generates a valid single-speaker hypothesis RTTM.
   - Conservative overlap attribution (`M4`) inserts valid secondary speaker segments.
   - DER and cpWER scoring execute cleanly without schema errors.
4. **Precondition 4 (Zero Modification of Prior Stages)**:
   Confirm that all 298 frozen Stage 1–5A.1.2 artifacts, Stage 5B.1 artifacts, Stage 5B.2 artifacts, and Stage 5B.3 artifacts remain 100% untouched.

---

## 9. Comprehensive Audit Deliverables Index

The complete audit suite comprises seven deliverables located in `research_stages/stage5b/stage5b_4_fresh_diarization/`:

1. [STAGE5B_4_PROTOCOL_REVIEW.md](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_PROTOCOL_REVIEW.md): Experimental matrix, condition definitions, factorial decomposition plan.
2. [STAGE5B_4_FRESH_DIARIZATION_PATH_AUDIT.md](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_FRESH_DIARIZATION_PATH_AUDIT.md): Pipeline architecture, data flow diagram, script gap analysis.
3. [STAGE5B_4_EMBEDDING_CACHE_AUDIT.md](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_EMBEDDING_CACHE_AUDIT.md): Dual-encoder checkpoint verification and collision-free cache schema specification.
4. [STAGE5B_4_EVALUATION_AUDIT.md](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_EVALUATION_AUDIT.md): Diarization collars, cpWER JSON schema verification, statistical significance protocols.
5. [STAGE5B_4_DEPENDENCY_AND_ENVIRONMENT_AUDIT.md](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_DEPENDENCY_AND_ENVIRONMENT_AUDIT.md): Hardware, software versions, CUDA verification, and computational budget.
6. [STAGE5B_4_PRE_EXECUTION_AUDIT_REPORT.md](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_PRE_EXECUTION_AUDIT_REPORT.md): This authoritative report and recommendation.
7. [STAGE5B_4_AUDIT_MANIFEST.json](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_AUDIT_MANIFEST.json): SHA-256 cryptographic manifest of all audit deliverables.
