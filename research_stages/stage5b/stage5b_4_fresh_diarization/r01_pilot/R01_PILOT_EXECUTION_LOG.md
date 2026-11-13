# Stage 5B.4 Authorized R01 Controlled Pilot — Execution Log
**Date**: November 10, 2026  
**Host Environment**: Linux 6.6.137+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.137-1+rpt1 (2025-05-07) aarch64 / x86_64 compatible  
**Python Environment**: `research_stages/.venv_phase20/bin/python` (Python 3.10.20, PyTorch 2.6.0+cu124, NeMo 2.7.3, CUDA available: True, RTX 4090 24GB)  
**Git HEAD**: `03b26c6` (Clean working tree)  

> [!NOTE] Post-Run Reconciliation Notice
> Production integration and pilot execution were successfully completed in commit `50940f5`.
> Post-run reconciliation complete in the working tree; pending project-owner review and commit.
> For the comprehensive post-run audit reconciliation (including authoritative C62 VAD parameter confirmation, RTTM record vs word-turn accounting, and cryptographic integrity verification), refer to [R01_POST_RUN_AUDIT_ADDENDUM.md](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_4_fresh_diarization/r01_pilot/R01_POST_RUN_AUDIT_ADDENDUM.md).

---

## Preflight Execution Log

### 1. Repository State & Baseline Integrity
```bash
$ cd /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning
$ git status --short
?? research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_FINAL_HUMAN_SIGNOFF_PREPARATION.md
?? research_stages/stage5b/stage5b_4_fresh_diarization/r01_pilot/
$ git log -n 2 --oneline
03b26c6 (HEAD -> main, origin/main, origin/HEAD) Stage 5B.4-A.1.3.2.2 — Security Verification & Final Closure
2c7c185  Stage 5B.4-A.1.3.2.1 — Final Security and Integration Closure
```
*Result*: HEAD verified; all 13 protected baseline manifests verified 100% byte-identical.

### 2. Audio Input Hash Verification
```bash
DFN Audio: research_stages/stage1_data_prep/audio/R01_sara_sharif/R01_sara_sharif_dfn.wav
  SHA-256: 90520b893a11c9690e44c88ca5b9e94be49f36aa9e4eae1db88570b4bb051ced
  Size: 3230810 bytes (Matches Stage 1 recording_manifest.json exactly)

Mono Audio: research_stages/stage1_data_prep/audio/R01_sara_sharif/R01_sara_sharif_16k_mono.wav
  SHA-256: 96f9af8d250ed8e26749433e2da6e1928a6d78eb6caad002cc1587fffcc0e45d
  Size: 3230810 bytes (Unrecorded in historical Stage 1 manifest)
```

### 3. Checkpoint Physical Byte & Provenance Verification
```bash
PyanNet Checkpoint:
  Path: research_stages/.venv_phase20/lib/python3.10/site-packages/whisperx/assets/pytorch_model.bin
  SHA-256: 0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea (PASS)

WeSpeaker Checkpoint:
  Path: /home/priyanka/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet293-LM/snapshots/6e6bffe5bf3d772a1f143dc6dbfea58a0799ea83/voxceleb_resnet293_LM.onnx
  SHA-256: dbb1ccc7754caff552ebc46347a51aaee2669bb24efc740e665d1a1133d20e98 (PASS)

NeMo MSDD Diarization Checkpoint:
  Path: /home/priyanka/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo
  SHA-256: 450f6bd279a3f78348f6e883359f9b3b34687dbd15814fd0f10e446ca0f708f6 (PASS)

Standalone TitaNet-Large Checkpoint:
  Result: MISSING ON DISK across repository and cache.
```

### 4. Overlap Posterior Validation
```python
loader = OverlapPosteriorLoaderStrict(
    execution_mode=ExecutionMode.PRODUCTION,
    pyannet_checkpoint_path=Path("research_stages/.venv_phase20/lib/python3.10/site-packages/whisperx/assets/pytorch_model.bin")
)
c_dfn = loader.load_and_validate('R01_sara_sharif', stream='dfn')
# Result: DFN eligible: True ['Structural and cryptographic provenance verified.']
# Shape: (5983,) probabilities, (5983,) timestamps.
```

### 5. Production Pipeline Compatibility & Interface Audit
- **Audit Step 5.1 (VAD Registry Mapping Confirmation)**:
  `condition_registry.py` assigns `VAD_C62` to `E20` and `E21`. `VAD_C50` is assigned to `E10` and `E11`. The comparison is confirmed as E20 (C62 $\times$ M0) vs E21 (C62 $\times$ M4).
- **Audit Step 5.2 (Orchestrator Overlap Stream Parameterization)**:
  `orchestrator_remediated.py` line 392 hardcodes `stream="mono"`. In production mode, querying `mono` invokes fail-closed rejection because `R01_sara_sharif__mono` has `source_audio_sha256 = UNRECORDED_IN_HISTORICAL_MANIFEST`. Passing `dfn` is not exposed via `process_recording` arguments.
- **Audit Step 5.3 (TitaNet Forward Interface Contract)**:
  `load_titanet_model_from_verified_bytes` restores `diar_msdd_telephonic.nemo` as an `EncDecDiarLabelModel`. In `feature_extraction_interface.py`, `extract_titanet_embedding` calls `model.forward(input_signal=..., input_signal_length=...)` (2 arguments), which fails with:
  `TypeError: Number of input arguments provided (2) is not as expected. Function has 7 total inputs with 7 mandatory inputs.`
- **Audit Step 5.4 (WeSpeaker Inference Contract)**:
  `model_factory.py` loads `.onnx` models into `ort.InferenceSession`. In `feature_extraction_interface.py`, `extract_wespeaker_embedding` calls `model.forward(feats)` or `model(feats)`, which fails with:
  `ModelExecutionError: Model <class 'onnxruntime.capi.onnxruntime_inference_collection.InferenceSession'> is neither callable nor has forward().`
- **Audit Step 5.5 (CLI Entry Point Readiness)**:
  `run_stage5b_4_fresh_diarization.py` mode `run` contains only a stub.

### 6. Mandatory Preflight Verdict & Stop
All 4 blocking interface discrepancies initially triggered Mandatory Stop Conditions. Preflight halted prior to neural inference to prevent unapproved code modifications or silent parameter substitutions.
- Initial Preflight status: BLOCKED_MANDATORY_STOP_CONDITIONS

---

## Production Integration and Authorized Pilot Execution

### 7. Resolution of Blockers and C62 Provenance
1. **Fix 1 (Overlap Stream Selection)**: Added `overlap_stream: str = "dfn"` to `orchestrator_remediated.py` and wired through to `load_and_validate()`.
2. **Fix 2 (TitaNet Submodule Interface)**: Extracted `msdd._speaker_model.` into `EncDecSpeakerLabelModel` adhering to 2-argument `.forward(input_signal, input_signal_length)` interface.
3. **Fix 3 (WeSpeaker ONNX Wrapper)**: Implemented `WeSpeakerONNXWrapper` exposing `.forward()` and `__call__()` returning 256d PyTorch tensors from ONNX Runtime inference sessions.
4. **Fix 4 (Production Runner CLI)**: Created `run_production_pilot.py` wiring CLI execution to `FreshDiarizationOrchestratorRemediated` and frozen condition registry.
5. **C62 Audio Provenance**: Established canonical SHA-256 for `R01_sara_sharif_16k_mono.wav` (`96f9af8d...`) from immutable Stage 1 audit reports committed in `0a6da21` (Oct 31, 2026), documented in `R01_C62_MONO_PROVENANCE_DETERMINATION.md`.
6. **Synthetic Tests**: 12/12 unit tests passed in `test_production_integration.py`.

### 8. Pilot Execution Commands and Telemetry
```bash
$ research_stages/.venv_phase20/bin/python \
  research_stages/stage5b/stage5b_4_fresh_diarization/production_integration/scripts/run_production_pilot.py \
  run --authorize-inference --recordings R01_sara_sharif --conditions E20 E21 \
  --output-pilot-dir research_stages/stage5b/stage5b_4_fresh_diarization/production_integration/r01_pilot_outputs
```
- Hardware: NVIDIA GeForce RTX 4090 24GB (Driver: 595.71.05, CUDA: 13.2)
- Multi-scale embedding extraction: 5 temporal scales (1.5s, 1.25s, 1.0s, 0.75s, 0.5s)
- Clustering: NeMo `SpeakerClustering` on CUDA GPU
- Publication: Atomic descriptor-relative rename to `outputs/E20/R01_sara_sharif.rttm` and `outputs/E21/R01_sara_sharif.rttm`
- Publication verification: SHA-256 `109f49a6bd3da3503721c7f6ddc9b1c757985f316be0b5d95cb4995b47b9712a` (100% bit-exact across all copies)

### 9. Final Results Summary
- E20 (C62 $\times$ M0): DER(0ms) = 32.91%, DER(250ms) = 31.77%, JER = 62.18%, cpWER = 52.30%, $\hat{K} = 3$ (True $K = 5$)
- E21 (C62 $\times$ M4): DER(0ms) = 32.91%, DER(250ms) = 31.77%, JER = 62.18%, cpWER = 52.30%, $\hat{K} = 3$ (True $K = 5$)
- Difference ($\Delta$): 0.00% across all metrics.
- Findings: R01 has 0.0s ground-truth overlap; PyAnNet DFN posterior max is 0.3804 (below threshold $\tau = 0.40$); Method M4 correctly abstained, producing 0 false alarms and preserving baseline diarization.
- Pilot status: **COMPLETED AND VERIFIED**
