# Stage 5B.4 — Production Integration Workspace

## Mission
This workspace (`research_stages/stage5b/stage5b_4_fresh_diarization/production_integration/`) isolates the four engineering remediations required to execute the Stage 5B.4 authorized R01 controlled pilot:
1. **Fix 1**: Parameterized overlap stream routing supporting explicit `dfn` selection for Method M4 under zero-backfill security rules.
2. **Fix 2**: NVIDIA TitaNet speaker encoder extraction from NeMo MSDD (`msdd._speaker_model.`) matching Stage 3 methodology and 2-argument forward contract.
3. **Fix 3**: WeSpeaker ResNet293-LM ONNX Runtime wrapper providing standard `.forward()` and `__call__()` returning 256d normalized speaker embeddings.
4. **Fix 4**: Production runner CLI (`run_production_pilot.py`) wiring CLI arguments to `FreshDiarizationOrchestratorRemediated` and the frozen condition registry.

## Directory Structure
- `scripts/`: Production orchestrator, model factory, ONNX wrapper, VAD adapter, cache manager, and pilot CLI.
- `tests/`: 12 focused synthetic integration tests (`test_production_integration.py`).
- `reports/`:
  - `STAGE5B_4_PRODUCTION_INTEGRATION_REPORT.md`
  - `STAGE5B_4_SYNTHETIC_INTEGRATION_TEST_REPORT.md`
  - `R01_C62_MONO_PROVENANCE_DETERMINATION.md`
  - `R01_PILOT_FINAL_REPORT.md`
- `manifests/`: Cryptographic deliverable manifest `STAGE5B_4_PRODUCTION_INTEGRATION_MANIFEST.json`.
- `r01_pilot_outputs/`: Secondary verified RTTM outputs for conditions E20 and E21.
