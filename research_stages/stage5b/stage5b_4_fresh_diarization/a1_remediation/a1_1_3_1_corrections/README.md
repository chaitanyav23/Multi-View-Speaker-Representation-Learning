# Stage 5B.4-A.1.3.1: Security, Checkpoint Identity, and Provenance Closure

## Overview

Stage 5B.4-A.1.3.1 is a targeted, narrowly scoped security, identity-binding, path-containment, and provenance-closure release in the Marston Speaker Diarization project. It resolves six critical findings identified during the independent source-code audit of Stage 5B.4-A.1.3:

1. **Finding 1 (Checkpoint Root Allowlist Containment)**: Caller-expanded roots and traversal attacks in production mode are eliminated. Production checkpoints are restricted to an immutable standard allowlist (`stage3_dual_encoder/models`, `~/.cache/torch`, `~/.cache/huggingface`, `whisperx/assets`). Any custom root attempting to expand authorization or act as a parent root is strictly rejected.
2. **Finding 2 (Physical Byte Hashing of Source Audio and Model Checkpoint)**: Manifest string checks are replaced by direct physical byte hashing of source audio and the PyanNet model checkpoint (`0b5b3216...`). Historical mono source audio is explicitly recorded as `UNRECORDED_IN_HISTORICAL_MANIFEST` and fails verification safely.
3. **Finding 3 (Trusted Configuration Specification)**: Introduced `TRUSTED_PYANNET_POSTERIOR_SPEC` and validated all 10 posterior semantic and generation parameters against this specification.
4. **Finding 4 (WeSpeaker Architecture & TOCTOU Elimination)**: Implemented official WeSpeaker ResNet293 (`num_blocks=[10, 20, 64, 3]`, `feat_dim=80`, `embed_dim=256`), projection head key filtering (`projection.*`), and in-memory byte buffer verification reading with `O_NOFOLLOW` directly into `io.BytesIO`, making file modification between verification and loading physically impossible.
5. **Finding 5 (RTTM Destination Symlink Race Defeated)**: Prevented destination symlink replacement attacks by opening files with `O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW`, verifying regular file descriptors via `fstat`, and confirming containment post-write.
6. **Finding 6 (M0 vs M4 Eligibility Formalization)**: Formalized the architectural distinction between M0 execution eligibility (does not consume overlap posteriors; lack of overlap provenance does not block M0) versus overlap provenance eligibility. M4 consumes overlap posteriors and strictly fails closed when provenance is unverified.

---

## Directory Structure

```
research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_1_corrections/
├── README.md
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
├── manifests/
│   ├── overlap_posterior_provenance_manifest.json
│   └── A1_3_1_MANIFEST.json
├── tests/
│   ├── test_checkpoint_roots.py
│   ├── test_source_and_checkpoint_hashes.py
│   ├── test_provenance_schema_and_config.py
│   ├── test_wespeaker_and_toctou.py
│   ├── test_rttm_symlink_race.py
│   ├── test_m0_m4_eligibility.py
│   ├── test_cross_component_scenarios.py
│   └── test_manifest_validation.py
└── reports/
    ├── STAGE5B_4_A1_3_1_IMPLEMENTATION_REPORT.md
    ├── STAGE5B_4_A1_3_1_SECURITY_AUDIT_REPORT.md
    ├── STAGE5B_4_A1_3_1_PROVENANCE_VALIDATION_REPORT.md
    ├── STAGE5B_4_A1_3_1_TEST_AND_VALIDATION_REPORT.md
    ├── STAGE5B_4_A1_3_1_LIMITATIONS_AND_READINESS.md
    ├── STAGE5B_4_A1_3_1_PROTECTED_ARTIFACTS_INTEGRITY.md
    ├── STAGE5B_4_A1_3_1_ERRATUM.md
    └── A1_3_1_EXECUTION_LOG.md
```

---

## Verification Commands

To run the complete unit and integration test suite (64 tests):
```bash
research_stages/.venv_phase20/bin/pytest research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_1_corrections/tests/ -v
```

To verify deliverables and the 10 protected baseline manifests:
```bash
research_stages/.venv_phase20/bin/python research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_3_1_corrections/scripts/verify_manifest.py
```
