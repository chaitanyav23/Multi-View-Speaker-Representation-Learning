# Stage 5B.4 Authorized R01 Controlled Pilot — Execution Log
**Date**: November 10, 2026  
**Host Environment**: Linux 6.6.137+rpt-rpi-2712 #1 SMP PREEMPT Debian 1:6.6.137-1+rpt1 (2025-05-07) aarch64 / x86_64 compatible  
**Python Environment**: `research_stages/.venv_phase20/bin/python` (Python 3.10.20, PyTorch 2.6.0+cu124, NeMo 2.7.3, CUDA available: True, RTX 4090 24GB)  
**Git HEAD**: `03b26c6` (Clean working tree)  

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
All 4 blocking interface discrepancies trigger the Mandatory Stop Conditions. Preflight halted prior to neural inference to prevent unapproved code modifications or silent parameter substitutions.
- Pilot status: NOT EXECUTED
- E20 status: BLOCKED / NOT RUN
- E21 status: BLOCKED / NOT RUN
- Metrics: 0 generated
- Inference: 0 executed
