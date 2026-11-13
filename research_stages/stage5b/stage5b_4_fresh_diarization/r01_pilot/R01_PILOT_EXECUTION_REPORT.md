# Stage 5B.4 — Authorized R01 Controlled Pilot: Preflight & Blocker Report
**Document ID**: `R01_PILOT_EXECUTION_REPORT.md`  
**Date**: November 10, 2026  
**Auditor**: Senior ML Systems Engineer & Independent Research Code Auditor  
**Repository**: `https://github.com/chaitanyav23/Multi-View-Speaker-Representation-Learning`  
**Current HEAD**: `03b26c6` (Clean working tree)  
**Authorized Scope**: Exactly one controlled R01 comparison — Condition E20 (C62 $\times$ M0) vs Condition E21 (C62 $\times$ M4 using verified DFN posterior)  

> [!NOTE] Preflight Audit State & Post-Run Reconciliation Notice
> This document records the **preflight audit state** prior to authorized production integration and execution.
> Subsequent to this preflight audit, the project owner authorized bounded production integration and pilot execution, which completed successfully in commit `50940f5`.
> Post-run reconciliation complete in the working tree; pending project-owner review and commit.
> For the final empirical results and post-run audit reconciliation, refer to:
> - Final Report: [R01_PILOT_FINAL_REPORT.md](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/r01_pilot/R01_PILOT_FINAL_REPORT.md)
> - Post-Run Audit Addendum: [R01_POST_RUN_AUDIT_ADDENDUM.md](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/r01_pilot/R01_POST_RUN_AUDIT_ADDENDUM.md)

---

## Executive Summary

The project owner authorized execution of exactly one controlled pilot on recording `R01_sara_sharif`:
- **E20**: C62 Linear Consensus VAD $\times$ M0 (Single-label baseline, no overlap attribution).
- **E21**: C62 Linear Consensus VAD $\times$ M4 (Conservative overlap attribution using only the provenance-verified R01 DFN overlap posterior).

**Registry & Condition Clarification**:  
The project owner explicitly corrected the prior prompt's mention of C50. The repository's frozen condition registry (`condition_registry.py`) maps **E20 and E21 strictly to C62** (Linear Consensus VAD). **C50** (Raw 16k Mono VAD) belongs exclusively to **E10 and E11**. The authorized comparison remains strictly E20 versus E21; no substitution or reinterpretation is permitted.

### Mandatory Preflight & Execution Status
In strict accordance with Section 2 (Mandatory Preflight) and the Mandatory Stop Conditions, preflight inspections were completed without modifying production source code or launching unapproved runs.

```
================================================================================
PILOT STATUS: NOT EXECUTED
CONDITION E20 STATUS: BLOCKED / NOT RUN
CONDITION E21 STATUS: BLOCKED / NOT RUN
DFN POSTERIOR PROVENANCE: VERIFIED PRODUCTION-ELIGIBLE
PRODUCTION PIPELINE: NOT READY FOR EXECUTION (UNRESOLVED INTERFACE BLOCKERS)
EXPERIMENTAL METRICS: NONE (0 METRICS GENERATED)
NEURAL INFERENCE / EMBEDDING EXTRACTION: NONE (0 OCCURRED)
================================================================================
```

Execution was safely and immediately halted prior to any neural inference, embedding extraction, or RTTM writing.

---

## 1. Comprehensive Preflight Audit (9 Mandatory Findings)

### Finding 1: Frozen E20/E21 Configuration (C62 $\times$ M0/M4)
- **Authoritative Source**: `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/scripts/condition_registry.py` lines 70–82, 117–124.
- **E20 Definition**: `ExperimentCondition("E20", VAD_C62, OVERLAP_M0, "C62 x M0", "Isolated robust consensus VAD effect")`.
- **E21 Definition**: `ExperimentCondition("E21", VAD_C62, OVERLAP_M4, "C62 x M4", "Full proposed integration system")`.
- **VAD C62 Definition**: `audio_source="consensus"`, linear combination ($0.5 \cdot \text{DFN} + 0.5 \cdot \text{Mono}$), hysteresis thresholds (`onset=0.600`, `offset=0.350`), minimum turn duration `min_duration_on=0.10s`, gap-fill merging `min_duration_off=0.30s`.
- **M0 Definition**: `method_id="M0_no_overlap"`, single-speaker baseline, overlap regions assigned to primary speaker.
- **M4 Definition**: `method_id="M4_conservative_abstention"`, conservative contiguous turn alternation with abstention, `tau_overlap=0.40`, `min_overlap_dur=0.10s`, `max_gap_sec=5.0s`, `min_turn_dur_sec=0.20s`, `frame_resolution=0.01s`.

### Finding 2: C50 Belongs to E10/E11, Not E20/E21
- **Authoritative Source**: `condition_registry.py` lines 60–68, 119–122.
- **VAD C50 Definition**: `audio_source="16k_mono"`, single-stream raw mono VAD, `onset=0.481`, `offset=0.481`, `min_duration_on=0.10s`, `min_duration_off=0.15s`.
- **Mapping**: `C50` is mapped exclusively to `E10` (C50 $\times$ M0) and `E11` (C50 $\times$ M4).
- **Confirmation**: E20/E21 does not use C50. The authorized pilot evaluates C62 Consensus VAD.

### Finding 3: R01 DFN Posterior Provenance & PyanNet Checkpoint Path Resolution
- **Posterior File**: `research_stages/stage5b/stage5b_2_overlap_optimization/overlap_cache/R01_sara_sharif_dfn_overlap.npz`
  - Computed SHA-256: `4ffda87cb2525563b90ff8b9b7fdae90bfb944b59d0958e1f8c41cfef60afa78`
  - Size: 40,207 bytes; Shape: 5,983 frames; Step: $0.016875$s; First center: $0.03096875$s.
  - Duration: $100.9772$s (matches R01 audio duration $100.9604$s within standard Pyannote boundary padding $<0.02$s).
- **Authoritative Manifest Match**: Matches `manifests/overlap_posterior_provenance_manifest.json` `R01_sara_sharif__dfn` entry across all 10 mandatory fields with zero unverified fields (`unverified_fields = []`).
- **Exact PyanNet Checkpoint Path Dependency**:
  - `OverlapProvenanceVerifier` in `overlap_provenance.py` lines 175–177 defaults to `get_repo_root() / "whisperx" / "assets" / "pytorch_model.bin"`, which does not exist in the root checkout.
  - The physical checkpoint is located in the active Python virtualenv:
    `research_stages/.venv_phase20/lib/python3.10/site-packages/whisperx/assets/pytorch_model.bin`
    SHA-256: `0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea` (exact match).
  - When `OverlapPosteriorLoaderStrict` is initialized with this virtualenv path, validation **passes with 100% compliance** (`production_eligible = True`).

### Finding 4: Orchestrator Hardcoded `stream="mono"` Prevents Authorized DFN Posterior Ingestion
- **File & Line**: `orchestrator_remediated.py` line 392 (and line 357 in override branch).
- **Code**:
  ```python
  if overlap_method == "M4_conservative_abstention":
      container = self.overlap_loader.load_and_validate(
          recording_id=rec_id,
          stream="mono",   # <-- HARDCODED TO "mono"
          expected_duration=expected_duration,
          source_audio_override=source_audio_override,
          pyannet_checkpoint_override=pyannet_checkpoint_override,
      )
  ```
- **Why It Fails Closed**:
  - `FreshDiarizationOrchestratorRemediated.process_recording()` does not take an overlap stream argument.
  - Querying `stream="mono"` in production mode looks up `R01_sara_sharif__mono`, which has `source_audio_sha256 = "UNRECORDED_IN_HISTORICAL_MANIFEST"` and `unverified_fields = ["source_audio_sha256"]`.
  - The multi-state validator sets `production_eligible = False`.
  - `orchestrator_remediated.py` lines 409–413 immediately abort with:
    `OverlapProvenanceError: Production M4 rejected for 'R01_sara_sharif': Posterior is not production eligible.`
  - The authorized DFN posterior cannot be loaded without modifying `orchestrator_remediated.py`.

### Finding 5: TitaNet Checkpoint Discovery — Missing Standalone Model vs MSDD Archive
- **Filesystem Search Results**:
  - **Standalone TitaNet Checkpoint (`titanet-l.nemo` / `EncDecSpeakerLabelModel`)**: **MISSING ON DISK**. No standalone checkpoint file exists in the repository, virtualenvs, or user cache.
  - **MSDD Diarization Archive**: Available at `/home/priyanka/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo` (SHA-256: `450f6bd279a3f78348f6e883359f9b3b34687dbd15814fd0f10e446ca0f708f6`, size: 104,181,957 bytes).
- **Architectural Distinction**:
  - `diar_msdd_telephonic.nemo` is a complete multi-scale diarization model (`EncDecDiarLabelModel`).
  - It contains a speaker model internally as a submodule (`msdd._speaker_model`), but the archive itself is NOT a standalone speaker embedding model and cannot be directly passed to a speaker label loader without unpacking.

### Finding 6: TitaNet Model Construction & Forward Interface Mismatch
- **File & Lines**: `model_factory.py` lines 322–326 vs `feature_extraction_interface.py` line 143.
- **Mechanism**:
  - In `model_factory.py`:
    ```python
    model_instance = EncDecSpeakerLabelModel.restore_from(restore_path=proc_path, map_location=device)
    ```
  - When NeMo restores `diar_msdd_telephonic.nemo`, the tarball metadata causes NeMo to instantiate `EncDecDiarLabelModel`.
  - In `feature_extraction_interface.py`:
    ```python
    out = model.forward(input_signal=wav_t, input_signal_length=len_t)
    ```
  - `EncDecDiarLabelModel.forward()` requires 7 mandatory parameters (`input_signal`, `input_signal_length`, `emb`, `emb_length`, `scale_weights`, `targets`, `targets_length`). Passing 2 arguments fails with:
    ```
    TypeError: Number of input arguments provided (2) is not as expected. Function has 7 total inputs with 7 mandatory inputs.
    ```
  - In Stage 3 (`stage3_dual_encoder/run_stage3_extraction.py` lines 364–367), the project resolved this by extracting `spk_sd = {k.replace("msdd._speaker_model.", ""): v for k, v in msdd_full.state_dict().items() if k.startswith("msdd._speaker_model.")}` and instantiating `EncDecSpeakerLabelModel.from_config_dict()`. `model_factory.py` does not perform this extraction.

### Finding 7: WeSpeaker ONNX Runtime Interface Mismatch (ONNX Session vs PyTorch Module)
- **File & Lines**: `model_factory.py` lines 490–493 vs `feature_extraction_interface.py` lines 205–213.
- **Checkpoint on Disk**: `/home/priyanka/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet293-LM/snapshots/6e6bffe5bf3d772a1f143dc6dbfea58a0799ea83/voxceleb_resnet293_LM.onnx` (SHA-256: `dbb1ccc7754caff552ebc46347a51aaee2669bb24efc740e665d1a1133d20e98`, size: 224,195,496 bytes).
- **Mechanism**:
  - `ControlledModelFactory.build_production_model()` builds `ort.InferenceSession`.
  - In `extract_wespeaker_embedding()`:
    ```python
    if hasattr(model, "forward"):
        out = model.forward(feats)
    elif callable(model):
        out = model(feats)
    else:
        raise ModelExecutionError(f"Model {type(model)} is neither callable nor has forward().")
    ```
  - `onnxruntime.InferenceSession` has neither `.forward()` nor `__call__()` (it requires `session.run(output_names, input_feed)`).
  - Passing this session into `extract_wespeaker_embedding()` immediately raises:
    ```
    ModelExecutionError: Model <class 'onnxruntime.capi.onnxruntime_inference_collection.InferenceSession'> is neither callable nor has forward().
    ```

### Finding 8: Production Pipeline Readiness & Incomplete Interfaces
- `research_stages/stage5b/stage5b_4_fresh_diarization/scripts/run_stage5b_4_fresh_diarization.py` mode `run` contains only a stub (`# Inference logic will execute here once authorized in Stage 5B.4-B / Phase B; return 0`).
- The production orchestrator `FreshDiarizationOrchestratorRemediated` cannot execute because of Findings 4, 6, and 7.
- Therefore, the production pipeline is **NOT READY** for execution.

### Finding 9: Audit of Completed, Failed, and Not-Attempted Checks

| Category | Checks Completed & Verified (PASS) | Checks Incompatible / Blocked (FAIL) | Checks Not Attempted (HALTED) |
|---|---|---|---|
| **Git & Baselines** | HEAD `03b26c6`; working tree clean; 13 of 13 historical baseline manifests verified. | None | None |
| **Audio Inputs** | R01 DFN audio (`90520b89...`, 3,230,810 bytes); R01 reference RTTM; aligned words JSON. | R01 mono audio hash unrecorded in historical Stage 1 manifest. | None |
| **Overlap Posteriors** | R01 DFN posterior (`4ffda87c...`, 5,983 frames); PyanNet checkpoint in venv (`0b5b3216...`). | Orchestrator hardcodes `stream="mono"`, blocking production DFN posterior. | None |
| **Models & Checkpoints** | Physical files verified on disk for WeSpeaker ONNX (`dbb1ccc7...`) and NeMo MSDD (`450f6bd2...`). | 1. Standalone TitaNet checkpoint missing on disk.<br>2. MSDD archive has 7-arg forward().<br>3. WeSpeaker ONNX has no forward() or __call__(). | Neural forward inference; GPU execution. |
| **Algorithms** | Stage 5A affinity fusion weights and formulas; Stage 5B.3 NeMo NME-SC parameters. | Top-level CLI script mode `run` is a stub. | Embedding extraction; clustering execution; RTTM writing. |
| **Evaluation** | Reference RTTM and evaluation adapter formulas. | None | DER / JER / cpWER computation. |

---

## 2. Summary of Concrete Blockers

1. **Blocker 1**: `orchestrator_remediated.py` line 392 hardcodes `stream="mono"`, which rejects the authorized DFN posterior and fails closed.
2. **Blocker 2**: `model_factory.py` loads `diar_msdd_telephonic.nemo` as `EncDecDiarLabelModel`, whose 7-argument `forward()` is incompatible with `extract_titanet_embedding()`.
3. **Blocker 3**: `model_factory.py` loads `voxceleb_resnet293_LM.onnx` as `ort.InferenceSession`, which lacks `.forward()` and `__call__()`, breaking `extract_wespeaker_embedding()`.
4. **Blocker 4**: `run_stage5b_4_fresh_diarization.py` mode `run` contains only a placeholder stub.

---

## 3. Final Determination

> **Can the existing production implementation support the authorized E20/E21 comparison without code changes?**  
> **NO.** The existing production code contains four blocking interface and parameter mismatches that cannot be resolved without engineering code modifications.

Execution is strictly halted. No neural inference, embedding extraction, or experimental evaluation occurred. Awaiting separate authorization and instructions.
