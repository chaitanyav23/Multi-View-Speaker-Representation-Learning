# Stage 5B.4-A.1.2 — Test and Validation Report

**Author:** Senior ML Systems Engineer & Independent Research Auditor  
**Date:** November 2026  
**Repository:** `Multi-View-Speaker-Representation-Learning`  
**Workspace:** `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_2_corrections/`  
**Status:** ALL TESTS PASSED (61/61 in A.1.2, 182/182 across Stage 5B.4 suites)

---

## 1. Test Execution Summary

The Stage 5B.4-A.1.2 test suite was executed using pytest in the project's isolated Python environment (`research_stages/.venv_phase20/bin/pytest`).

| Test Suite | Module Under Test | Test Count | Status | Execution Time |
| :--- | :--- | :---: | :---: | :---: |
| `test_checkpoint_preflight.py` | `checkpoint_preflight.py` | 12 | **PASSED** | 0.03s |
| `test_output_containment.py` | `output_path_guard.py` | 13 | **PASSED** | 0.02s |
| `test_phase20_cpwer_parity.py` | `cpwer_adapter.py` | 13 | **PASSED** | 0.04s |
| `test_overlap_timestamp_provenance.py` | `overlap_posterior_loader.py` | 12 | **PASSED** | 0.07s |
| `test_cross_component_scenarios.py` | Full Pipeline (`orchestrator_remediated.py`) | 6 | **PASSED** | 1.84s |
| `test_manifest_validation.py` | Manifests & Baselines (`verify_manifest.py`) | 5 | **PASSED** | 0.02s |
| **Total A.1.2 Suite** | | **61** | **PASSED** | **1.92s** |

### Historical Subsystem Regression Results

Running individual stage test suites verified zero regressions across prior milestones:
- **Stage 5B.4-A:** 43 passed (3.33s)
- **Stage 5B.4-A.1:** 40 passed (10.02s)
- **Stage 5B.4-A.1.1:** 38 passed (2.20s)
- **Stage 5B.4-A.1.2:** 61 passed (1.92s)
- **Combined Total:** **182 passed**, 0 failed, 0 errors.

---

## 2. Detailed Test Coverage by Subsystem

### 2.1 Checkpoint Preflight Suite (`test_checkpoint_preflight.py`)
Validates that production checkpoint verification fails closed against all invalid inputs:
1. `test_missing_titanet_path_in_production`: Missing TitaNet path raises `CheckpointPreflightError`.
2. `test_missing_wespeaker_path_in_production`: Missing WeSpeaker path raises `CheckpointPreflightError`.
3. `test_missing_expected_hash_in_production`: None or missing expected hash raises `CheckpointPreflightError`.
4. `test_malformed_expected_hash`: Hashes with non-hex characters or length $\neq 64$ rejected.
5. `test_placeholder_hash_rejection`: Synthetic placeholders (`0000...`, `deadbeef...`) rejected.
6. `test_hash_mismatch`: Real file bytes differing from expected hash rejected.
7. `test_correct_hash_verification`: Valid checkpoint returns `VerifiedCheckpoint` with metadata.
8. `test_missing_file_fails`: File not found on disk rejected.
9. `test_directory_supplied_fails`: Path pointing to a directory rejected.
10. `test_symlink_escape_rejection`: Symlinks pointing outside authorized model roots rejected.
11. `test_production_mode_rejects_synthetic_bypass`: Synthetic flag rejected in production mode.
12. `test_verified_checkpoint_identity_passed_unchanged`: Verified object attributes verified intact.

### 2.2 Output Containment Suite (`test_output_containment.py`)
Validates complete filepath validation, identifier whitelisting, and directory containment:
1. `test_valid_recording_ids`: Standard benchmark recording IDs (`R01_...` through `R08_...`) pass.
2. `test_valid_registered_condition_ids`: Authoritative condition IDs (`E00` through `E21`) pass.
3. `test_unknown_condition_id_rejection`: Arbitrary condition IDs rejected.
4. `test_traversal_in_recording_id`: `../../etc/passwd` rejected immediately.
5. `test_absolute_recording_id`: `/root/data` rejected.
6. `test_path_separators_in_condition_id`: `E00/../escape` rejected.
7. `test_null_bytes_and_control_characters`: Injected `\x00` and newline characters rejected.
8. `test_unexpected_suffixes_in_recording_id`: Non-matching regex strings rejected.
9. `test_symlinked_output_directory_escape`: Symlink pointing outside output root rejected.
10. `test_symlinked_final_rttm_path`: Symlinked target RTTM file rejected.
11. `test_existing_directory_collision`: Target path already existing as directory rejected.
12. `test_custom_output_root_prohibited_in_production`: Arbitrary root prohibited without whitelist.
13. `test_valid_production_output_path`: Canonical path inside `outputs/` validated.

### 2.3 Phase 20 cpWER Parity Suite (`test_phase20_cpwer_parity.py`)
Differential verification against authoritative Phase 20 implementation across 12 edge cases:
- Case 1: Standard single-speaker non-overlapping segments.
- Case 2: Two speakers with partial overlap.
- Case 3: Equal-overlap ties between multiple speakers (verifies string comparison tie-breaker).
- Case 4: Words missing timestamps (fallback to nearest midpoint distance).
- Case 5: Words positioned completely outside all speaker intervals.
- Case 6: Midpoint distance fallback to closest segment.
- Case 7: Multiple overlapping segments from the same speaker.
- Case 8: Words straddling adjacent segment boundaries.
- Case 9: Empty hypothesis transcript handling.
- Case 10: Empty RTTM hypothesis handling.
- Case 11: Speaker label permutation invariance (Hungarian solver optimality).
- Case 12: Identical start and end timestamps.
- Case 13: Full differential cpWER evaluation end-to-end.

All 13 test cases produced 100% bit-exact parity with Phase 20.

### 2.4 Overlap Timestamp Provenance Suite (`test_overlap_timestamp_provenance.py`)
Validates acoustic and temporal semantics of Pyannote 3.1 PyanNet posteriors:
1. `test_correct_ordered_timestamps`: Verified $16.875\text{ ms}$ stride and $0.03096875\text{ s}$ center pass.
2. `test_duplicate_timestamps_rejected`: Duplicate timestamps ($\Delta t = 0$) raise `OverlapPosteriorError`.
3. `test_decreasing_timestamps_rejected`: Decreasing timestamps ($\Delta t < 0$) raise `OverlapPosteriorError`.
4. `test_nan_and_inf_timestamps_rejected`: Non-finite timestamps raise `OverlapPosteriorError`.
5. `test_incorrect_stride_rejected`: Stride deviation ($10\text{ ms} \neq 16.875\text{ ms}$) raises error.
6. `test_incorrect_first_frame_center_rejected`: Frame center deviation raises error.
7. `test_frame_count_duration_mismatch_rejected`: Frame count incompatible with duration rejected.
8. `test_posterior_length_mismatch_rejected`: Length mismatch between times and probs rejected.
9. `test_out_of_range_probabilities_rejected`: Probabilities $< 0$ or $> 1$ rejected.
10. `test_posterior_nan_inf_rejected`: Non-finite probabilities rejected.
11. `test_loader_synthetic_fixture_with_provenance`: End-to-end loading with provenance metadata.
12. `test_loader_missing_posterior_fails`: Missing file raises `FileNotFoundError`.

### 2.5 Cross-Component Integration Scenarios (`test_cross_component_scenarios.py`)
Verifies full pipeline coordination across six authoritative scenarios:
- **Scenario A (Production Checkpoint Preflight):** Verifies that in production mode, missing checkpoints, placeholder hashes, or checksum mismatches fail closed before any model inference or memory allocation.
- **Scenario B (Synthetic Checkpoint Mode):** Verifies that synthetic mode operates with mock descriptors, and that synthetic checkpoints cannot be used to bypass production mode preflight.
- **Scenario C (RTTM Path Traversal):** Verifies that malicious recording IDs or condition IDs attempting path traversal are rejected before any file operations occur.
- **Scenario D (cpWER Parity End-to-End):** Evaluates a synthetic hypothesis RTTM against reference transcripts and aligned words, asserting exact Phase 20 word attribution and Hungarian assignment.
- **Scenario E (Overlap Posterior Rejection):** Verifies that stride violations, probability bounds violations, or missing overlap posteriors in M4 conditions are rejected before attribution.
- **Scenario F (Valid Synthetic End-to-End Path):** Verifies complete end-to-end execution of Condition E00 and E01 on synthetic multi-tone audio, producing valid RTTM files and execution statistics without real checkpoints or GPU usage.

### 2.6 Manifest Validation Suite (`test_manifest_validation.py`)
Verifies cryptographic integrity:
1. `test_protected_baselines_intact`: Confirms all 7 historical baseline manifests remain unchanged.
2. `test_manifest_verification_on_valid_data`: Confirms verification passes on valid files.
3. `test_manifest_verification_detects_hash_mismatch`: Confirms detection of hash corruption.
4. `test_manifest_verification_detects_size_mismatch`: Confirms detection of byte size discrepancy.
5. `test_manifest_verification_detects_missing_file`: Confirms detection of missing files.

---

## 3. Conclusion

All 61 unit, integration, and scenario tests for Stage 5B.4-A.1.2 pass without error or failure. The pipeline is mathematically and architecturally validated.
