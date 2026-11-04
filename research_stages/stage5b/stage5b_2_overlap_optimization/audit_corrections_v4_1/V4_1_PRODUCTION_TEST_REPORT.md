# Stage 5B.2 V4.1: Production Test Suite Report

**Audit Phase**: Stage 5B.2 V4.1 Test Verification  
**Date**: November 4, 2026  
**Auditor**: Independent Forensic Research Engineer  
**Status**: 19/19 PASSED (100% Deterministic Pass Rate)

---

## 1. Test Suite Overview

A production test suite was developed and executed to validate the production implementation in `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/scripts/run_stage5b_2_corrected_v4_1.py`:
- **Test File**: `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/tests/test_stage5b_2_v4_1_production.py`
- **Environment**: Python 3.10.20, pytest 9.0.3, hydra-core 1.3.2
- **Execution Time**: 0.85 seconds

---

## 2. Test Execution Breakdown

### 2.1 Audit B: M2 Prototype-Purity Interval Union (13 Tests)
1. `test_clean_single_speaker_subsegment`: Validates standard single-speaker segment purity calculation. -> **PASSED**
2. `test_exact_duplicate_same_speaker_segments`: Validates that duplicate overlapping segments do not double-count duration. -> **PASSED**
3. `test_same_speaker_overlapping_segments`: Validates partial same-speaker overlaps are unified to net span. -> **PASSED**
4. `test_adjacent_same_speaker_intervals`: Validates abutted same-speaker intervals are properly merged. -> **PASSED**
5. `test_coverage_exactly_at_080`: Validates tolerance handling at exactly 80.0% coverage (`1.20 / 1.50`). -> **PASSED**
6. `test_coverage_just_below_080`: Validates strict rejection at 79.3% coverage (`1.19 / 1.50`). -> **PASSED**
7. `test_coverage_greater_than_080_with_competing_activity`: Validates 86.7% coverage with 13.3% competing activity. -> **PASSED**
8. `test_competing_speaker_with_02_seconds_activity`: Validates boundary at exactly 0.20s competing speech. -> **PASSED**
9. `test_competing_speaker_with_overlapping_duplicate_segments`: Validates that overlapping competing speech is unioned, preventing spurious rejection. -> **PASSED**
10. `test_multiple_speakers_with_equal_union_duration`: Validates deterministic behavior when multiple speakers share identical duration. -> **PASSED**
11. `test_input_order_permutation_invariance`: Validates that shuffling input segment list order produces identical prototype centroids. -> **PASSED**
12. `test_missing_embedding_cache`: Validates graceful error handling when embedding file is missing. -> **PASSED**
13. `test_prototype_dimensions_and_finiteness`: Validates finite L2-normalized embedding shape (512 for Wespeaker, 192 for Titanet). -> **PASSED**

### 2.2 Audit A: Deterministic Primary-Speaker Selection (2 Tests)
14. `test_majority_frame_count_priority`: Validates that speaker with strictly more frames in overlap segment is selected as primary. -> **PASSED**
15. `test_equal_count_alphabetical_tie_break`: Validates that when two speakers have identical frame counts in overlap, tie is broken strictly alphabetically (`"spk_0"` beats `"spk_1"`). -> **PASSED**

### 2.3 Regression Suite (4 Tests)
16. `test_hungarian_mapping`: Validates optimal bipartite Hungarian matching between predicted clusters and reference speakers. -> **PASSED**
17. `test_oracle_interval_union`: Validates union aggregation in oracle attribution logic. -> **PASSED**
18. `test_m3_m4_termination_at_speaker_change_and_silence`: Validates termination conditions for local context propagation methods. -> **PASSED**
19. `test_timestamp_frame_bin_support_boundaries`: Validates half-open interval binning and timestamp-to-frame alignment. -> **PASSED**

---

## 3. Pytest Summary Output
```
============================== 19 passed in 0.85s ==============================
```
Zero failures, zero warnings, zero flakes.
