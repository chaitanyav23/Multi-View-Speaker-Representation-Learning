# Stage 5B.4-A.1.1 Test and Validation Report
## Comprehensive Non-Inference Test Execution, Regression Parity, and Scenario Coverage

**Author**: Lead Research Engineer / Code Auditor  
**Date**: November 6, 2026  
**Test Suite**: 121 Passing Non-Inference Tests  

---

## 1. Test Execution Summary

The Stage 5B.4 test framework now contains three unified test layers totaling 121 automated test cases. All tests execute in a non-inference environment without GPU dependencies or real audio loading.

### Test Execution Command
```bash
research_stages/.venv_phase20/bin/pytest \
  research_stages/stage5b/stage5b_4_fresh_diarization/tests/ \
  research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/tests/ \
  research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/a1_1_corrections/tests/ \
  -v
```

### Cumulative Results Breakdown

| Test Layer | Directory | Test Files | Total Tests | Passed | Failed | Execution Time |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Stage 5B.4-A** | `stage5b_4_fresh_diarization/tests/` | 10 | 43 | 43 | 0 | ~3.8s |
| **Stage 5B.4-A.1** | `a1_remediation/tests/` | 9 | 40 | 40 | 0 | ~6.2s |
| **Stage 5B.4-A.1.1** | `a1_1_corrections/tests/` | 7 | 38 | 38 | 0 | ~2.2s |
| **Total Regression** | Combined Full Suite | **26** | **121** | **121** | **0** | **12.18s** |

---

## 2. Detailed Remediation Suite Results (38 Tests)

### 2.1 Task 3: Centralized Path Resolution (`test_path_utils.py` — 4 Tests)
- `test_repo_root_resolution`: Verifies discovery of repo root via `.git` anchor. (**PASSED**)
- `test_path_resolution_no_duplication`: Confirms absolute paths contain no duplicate `research_stages/research_stages`. (**PASSED**)
- `test_resolution_from_arbitrary_working_directory`: Validates resolution from `/tmp` or detached CWD. (**PASSED**)
- `test_resolution_with_mock_git_tree`: Verifies upward traversal logic on synthetic nested trees. (**PASSED**)

### 2.2 Task 1: Audited cpWER Adapter (`test_cpwer_adapter.py` — 6 Tests)
- `test_rttm_parsing`: Validates segment extraction and duration calculation. (**PASSED**)
- `test_word_attribution_and_grouping`: Verifies maximum overlap word attribution and gap-based grouping. (**PASSED**)
- `test_cpwer_perfect_match`: Asserts cpWER = 0.0, zero errors on perfectly matching transcripts. (**PASSED**)
- `test_cpwer_speaker_permutation_invariance`: Asserts cpWER = 0.0 when hypothesis speaker IDs are swapped (proves Hungarian solver functions correctly). (**PASSED**)
- `test_cpwer_with_errors`: Validates insertion, deletion, and substitution error rates. (**PASSED**)
- `test_missing_files_fail_cleanly`: Confirms `FileNotFoundError` when words or transcript JSON is missing. (**PASSED**)

### 2.3 Task 2: Checkpoint Preflight Verification (`test_checkpoint_preflight.py` — 5 Tests)
- `test_checkpoint_preflight_success`: Verifies valid checkpoint file and valid 64-char SHA-256 digest. (**PASSED**)
- `test_checkpoint_preflight_rejects_truncated_placeholder`: Proves 32-char MD5 placeholders are rejected with `CheckpointPreflightError`. (**PASSED**)
- `test_checkpoint_preflight_missing_file`: Asserts explicit preflight failure on missing checkpoint file. (**PASSED**)
- `test_checkpoint_preflight_hash_mismatch`: Asserts failure on corrupted checkpoint bytes. (**PASSED**)
- `test_dual_checkpoint_preflight_selective_failure`: Demonstrates granular failure when only one encoder checkpoint fails. (**PASSED**)

### 2.4 Task 4: Overlap Posterior Input (`test_overlap_posterior_loader.py` — 6 Tests)
- `test_load_real_stage2_baseline_overlap_posteriors`: Verifies read-only loading of actual Stage 2 baseline posteriors. (**PASSED**)
- `test_missing_posterior_file`: Asserts `FileNotFoundError` on absent NPZ file. (**PASSED**)
- `test_missing_keys_in_npz`: Asserts `KeyError` on missing `p_overlap`. (**PASSED**)
- `test_dimension_mismatch`: Asserts `OverlapPosteriorError` on length mismatch between `frame_times` and `p_overlap`. (**PASSED**)
- `test_out_of_bounds_probabilities`: Asserts error on probabilities $< 0.0$ or $> 1.0$. (**PASSED**)
- `test_nan_probabilities`: Asserts error on NaN or Infinite values. (**PASSED**)

### 2.5 Task 5: Precise Cache Error Handling (`test_cache_precise_errors.py` — 4 Tests)
- `test_cache_hit_and_miss`: Distinguishes cold `MISS` from warm `HIT`. (**PASSED**)
- `test_cache_invalidation_on_checkpoint_change`: Confirms identity mismatch yields `INVALIDATION`. (**PASSED**)
- `test_cache_invalidation_on_corrupt_file`: Confirms corrupt NPZ yields `INVALIDATION`, not `MISS`. (**PASSED**)
- `test_cache_security_violation_raises_immediately`: Proves traversal in identifiers raises `CacheSecurityError` immediately. (**PASSED**)

### 2.6 Task 6: Output Path Guard (`test_output_containment.py` — 5 Tests)
- `test_valid_output_path_inside_stage5b4`: Allows valid path inside workspace. (**PASSED**)
- `test_valid_custom_test_root`: Allows explicitly designated test root. (**PASSED**)
- `test_rejection_of_prohibited_stages`: Rejects output paths targeting Stages 1, 2, 3, 5B.1, 5B.2. (**PASSED**)
- `test_rejection_of_directory_traversal`: Rejects `..` path traversal escapes. (**PASSED**)
- `test_rejection_of_symlink_escape`: Rejects symlinks pointing outside allowed root. (**PASSED**)

### 2.7 Cross-Component Scenarios A–H (`test_cross_component_scenarios.py` — 8 Tests)
- `test_scenario_a_clean_execution_m0`: Validates cold miss -> warm hit cycle for M0 baseline. (**PASSED**)
- `test_scenario_b_clean_execution_m4_and_cpwer`: Validates M4 overlap attribution and downstream cpWER scoring. (**PASSED**)
- `test_scenario_c_missing_overlap_posteriors`: Proves M4 fails explicitly if posteriors are missing; M0 succeeds. (**PASSED**)
- `test_scenario_d_corrupt_cache_invalidation`: Verifies corrupt cache entry is invalidated, re-extracted, and counted. (**PASSED**)
- `test_scenario_e_cache_security_violation`: Verifies security violation raises immediately with zero extraction. (**PASSED**)
- `test_scenario_f_checkpoint_hash_mismatch`: Verifies checkpoint mismatch halts execution in preflight. (**PASSED**)
- `test_scenario_g_partial_cache_reuse`: Verifies altered TitaNet hash yields 5 invalidations while WeSpeaker achieves 100% cache hits. (**PASSED**)
- `test_scenario_h_output_path_escape`: Verifies output path escape attempt raises `OutputSecurityError` with zero disk writes. (**PASSED**)

---

## 3. Regression Safety Summary

No existing test from Stage 5B.4-A or Stage 5B.4-A.1 was modified or broken. The 121 tests run completely deterministically in 12.18 seconds, establishing architectural integrity and backward compatibility.
