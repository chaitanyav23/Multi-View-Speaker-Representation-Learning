# Stage 5B.4-A.1.3 — Model Identity and Overlap Provenance Closure

## Executive Overview
Stage 5B.4-A.1.3 is a narrowly scoped corrective engineering release for the Stage 5B.4 Fresh Diarization architecture, resolving the four remaining production-readiness defects identified during the independent review of Stage 5B.4-A.1.2:
1. **Model Identity Binding**: Binds production model construction strictly to cryptographically verified checkpoints via `ControlledModelFactory` and `VerifiedModelHandle`. Eliminates arbitrary preloaded model injection (`titanet_model`/`wespeaker_model`). Enforces pre-load physical byte TOCTOU revalidation.
2. **Canonical Output Containment**: Enforces strict containment of production RTTM writes to `research_stages/stage5b/stage5b_4_fresh_diarization/outputs/{condition_id}/{recording_id}.rttm`. Eliminates custom output roots in production mode, directory traversal, null bytes, and symlink escapes.
3. **Cryptographic Overlap Provenance**: Replaces unverified placeholder metadata with authoritative provenance tracking via `manifests/overlap_posterior_provenance_manifest.json` and `OverlapProvenanceVerifier`. Explicitly documents historical limits (e.g. unrecorded `source_audio_sha256` for `mono` streams).
4. **State Separation & Production Eligibility**: Strictly decouples `structurally_valid`, `provenance_verified`, and `production_eligible`. Condition M4 (`M4_conservative_abstention`) strictly fails closed on unverified provenance, while Condition M0 (`M0_no_overlap`) executes normally without consuming overlap posteriors.

## Workspace Layout
```
research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_corrections/
├── README.md
├── manifests/
│   ├── overlap_posterior_provenance_manifest.json
│   └── A1_3_MANIFEST.json
├── reports/
│   ├── STAGE5B_4_A1_3_IMPLEMENTATION_REPORT.md
│   ├── STAGE5B_4_A1_3_TEST_AND_VALIDATION_REPORT.md
│   ├── STAGE5B_4_A1_3_LIMITATIONS_AND_READINESS.md
│   └── STAGE5B_4_A1_3_PROTECTED_ARTIFACTS_INTEGRITY.md
├── scripts/
│   ├── __init__.py
│   ├── path_utils.py
│   ├── checkpoint_preflight.py
│   ├── model_factory.py
│   ├── output_path_guard.py
│   ├── overlap_provenance.py
│   ├── overlap_posterior_loader.py
│   ├── cache_manager_remediated.py
│   ├── cpwer_adapter.py
│   ├── evaluation_adapters_remediated.py
│   ├── orchestrator_remediated.py
│   └── verify_manifest.py
└── tests/
    ├── test_checkpoint_and_model_factory.py
    ├── test_output_containment.py
    ├── test_overlap_provenance.py
    ├── test_phase20_cpwer_parity.py
    ├── test_pipeline_validation_state.py
    ├── test_cross_component_scenarios.py
    └── test_manifest_validation.py
```

## Running the Verification Suite
To execute the complete 70-test A.1.3 test suite:
```bash
research_stages/.venv_phase20/bin/pytest research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_corrections/tests/ -v
```

To verify deliverables and 9 protected baseline manifests:
```bash
research_stages/.venv_phase20/bin/python research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_corrections/scripts/verify_manifest.py
```
