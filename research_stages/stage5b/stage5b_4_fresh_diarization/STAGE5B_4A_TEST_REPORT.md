# Stage 5B.4-A — Non-Inference Test Execution Report

**Execution Status**: All Tests Passed  
**Date**: 2026-11-06  
**Auditor**: Senior ML Research Engineer & Diarization Systems Auditor  
**Repository**: [Multi-View-Speaker-Representation-Learning](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning)  
**Deliverable Path**: `research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4A_TEST_REPORT.md`

---

## 1. Test Execution Overview

All test suites were executed using the repository's dedicated virtual environment:
`research_stages/.venv_phase20/bin/python -m pytest`

```
Command: research_stages/.venv_phase20/bin/python -m pytest research_stages/stage5b/stage5b_4_fresh_diarization/tests/ -v
Python Version: 3.10.20
PyTest Version: 9.0.3
Total Tests Run: 43
Passed: 43
Failed: 0
Skipped: 0
Wall-Clock Duration: 3.35s
```

---

## 2. Inventory of Executed Test Suites & Outcomes

| Test Module | Test Name | Result | Purpose |
| :--- | :--- | :--- | :--- |
| **`test_condition_registry.py`** | `test_registry_size` | PASSED | Verifies exact 6 conditions (`E00`–`E21`) |
| | `test_vad_configs` | PASSED | Validates C00, C50, C62 VAD parameters |
| | `test_overlap_configs` | PASSED | Validates M0 and M4-R2 parameters |
| | `test_get_condition_and_rejection` | PASSED | Tests retrieval and rejection of ambiguous `O02` ID |
| | `test_benchmark_recordings` | PASSED | Verifies 8 benchmark recordings (R01–R08) |
| | `test_validate_registry` | PASSED | Validates full registry invariants |
| **`test_vad_adapter.py`** | `test_canonical_interval_hash_determinism` | PASSED | Verifies interval hash determinism and change detection |
| | `test_binarize_hysteresis_logic` | PASSED | Tests hysteresis, min_on burst removal, min_off bridging |
| | `test_vad_adapter_mismatch_rejection` | PASSED | Rejects misaligned DFN/Mono posterior timestamps |
| **`test_subsegment_adapter.py`** | `test_scale_definitions` | PASSED | Verifies scales 0..4 window/shift definitions |
| | `test_short_interval_subsegmentation` | PASSED | Verifies handling of intervals shorter than window |
| | `test_sliding_window_with_trailing_remainder`| PASSED | Verifies NeMo 50ms trailing remainder clamp |
| | `test_generate_all_scales_coverage` | PASSED | Tests multi-scale coverage across all 5 scales |
| **`test_embedding_cache.py`** | `test_cache_hit_on_same_identity` | PASSED | Verifies atomic save and identical cache retrieval |
| | `test_cache_invalidation_on_altered_identity` | PASSED | Invalidation when VAD ID or hash changes |
| | `test_cache_invalidation_on_checkpoint_hash_change` | PASSED | Invalidation when model checkpoint hash changes |
| | `test_rejection_of_nan_and_inf` | PASSED | Rejects non-finite embedding matrices |
| | `test_rejection_of_wrong_embedding_dimension` | PASSED | Rejects dimension mismatches (e.g. 192 vs 256) |
| | `test_prohibited_stage3_cache_rejection` | PASSED | Verifies PermissionError on Stage 3 cache path |
| **`test_affinity_fusion.py`** | `test_cosine_affinity_matrix_properties` | PASSED | Verifies symmetry, diagonal=1.0, [-1, 1] range |
| | `test_offdiag_zscore_normalization` | PASSED | Verifies off-diagonal z-score mapping to [0, 1] |
| | `test_condition_j_equal_fusion` | PASSED | Verifies equal-weight fusion $0.5 A_{\text{zt}} + 0.5 A_{\text{zr}}$ |
| | `test_dimension_mismatch_rejection` | PASSED | Rejects shape mismatches between encoders |
| **`test_fresh_clustering.py`** | `test_stale_label_rejection_guard` | PASSED | Rejects frozen/ground-truth label keyword arguments |
| | `test_synthetic_nme_sc_speaker_separation` | PASSED | Verifies 2-speaker partitioning on block-diagonal affinity |
| | `test_cluster_fresh_affinity_interface` | PASSED | Tests fresh clustering interface output format |
| **`test_overlap_adapter.py`** | `test_extract_contiguous_turns` | PASSED | Verifies turn extraction without bridging None frames |
| | `test_m4_turn_alternation_success` | PASSED | Attributes secondary speaker on valid turn alternation |
| | `test_m4_same_speaker_adjacent_abstention` | PASSED | Abstains when same speaker appears before/after overlap |
| | `test_m4_short_turn_abstention` | PASSED | Abstains when adjacent turn < 0.20s |
| | `test_apply_overlap_m0_baseline` | PASSED | Verifies M0 preserves single-speaker segments |
| **`test_rttm_adapter.py`** | `test_subsegments_to_single_speaker_segments_merging` | PASSED | Merges contiguous same-speaker subsegments |
| | `test_rttm_file_roundtrip` | PASSED | Verifies NIST RTTM serialization and parsing |
| | `test_rasterize_segments_to_frame_spk` | PASSED | Tests frame-level discrete rasterization (10ms) |
| **`test_evaluation_adapter.py`** | `test_vad_union_metrics` | PASSED | Verifies Precision, Recall, F1 against reference union |
| | `test_speaker_count_error` | PASSED | Verifies signed and absolute speaker count error |
| | `test_format_cpwer_attribution_envelope_schema` | PASSED | Verifies canonical `{"segments": [...]}` schema |
| | `test_macro_and_pooled_summary` | PASSED | Tests macro vs duration-weighted pooled averaging |
| | `test_wilcoxon_and_bootstrap` | PASSED | Tests Wilcoxon signed-rank and bootstrap CI |
| **`test_cli_safety.py`** | `test_validate_config_mode` | PASSED | Verifies CLI `validate-config` exits code 0 |
| | `test_validate_cache_mode` | PASSED | Verifies CLI `validate-cache` exits code 0 |
| | `test_test_synthetic_mode` | PASSED | Verifies CLI `test-synthetic` exits code 0 |
| | `test_run_mode_without_authorization_flag_fails` | PASSED | Verifies CLI `run` fails without `--authorize-inference` |

---

## 3. Strict Boundary Verification

To ensure total compliance with Stage 5B.4-A execution constraints:

1. **Neural Checkpoint Loading**:
   - TitaNet checkpoint loaded: **NO (0 bytes loaded)**.
   - WeSpeaker checkpoint loaded: **NO (0 bytes loaded)**.
   - Pyannote PyanNet loaded: **NO (0 bytes loaded)**.
   - WhisperX model loaded: **NO (0 bytes loaded)**.
2. **Audio File Reads**:
   - Benchmark audio files read: **NO (0 audio files accessed)**.
3. **GPU Computation**:
   - CUDA device memory allocated during tests: **0 MiB**.
   - All tests executed exclusively on CPU using synthetic matrices and arrays.
4. **Benchmark Diarization / Metrics**:
   - Real RTTMs generated: **0**.
   - Real DER or cpWER evaluated: **0**.

---

## 4. Tests Deliberately Deferred to Separate Inference Authorization

The following integration checks require loading actual model checkpoints and real audio waveforms, and are therefore **deliberately deferred** until separate inference authorization is granted:
- `test_real_titanet_feature_extraction`: Extracting 192d embeddings from `R01_sara_sharif_dfn.wav`.
- `test_real_wespeaker_feature_extraction`: Extracting 256d embeddings from `R01_sara_sharif_mono.wav`.
- `test_real_nemo_cuda_spectral_clustering`: Executing `SpeakerClustering.forward_unit_infer` on CUDA using real affinity matrices.
- `test_real_whisperx_word_attribution`: Attributing WhisperX aligned words to real hypothesis RTTMs.
- `test_end_to_end_r01_pilot`: Evaluating `E00`, `E01`, and `E21` on recording R01.
