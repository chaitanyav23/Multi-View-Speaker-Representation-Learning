# Stage 5B.2 Unit Test Report: Corrective Patch v2

## 1. Overview & Test Execution Summary

The unit test suite [`tests/test_stage5b_2_corrections.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v2/tests/test_stage5b_2_corrections.py) validates the six corrected modules.

- **Total Tests Executed**: 21
- **Passed**: 21
- **Failed**: 0
- **Execution Time**: 0.448 s
- **Python Environment**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python`

---

## 2. Test Coverage by Correction Dimension

### Correction A: M1 Temporal Context Attribution
1. `test_m1_equal_evidence_different_distances`:
   - *Scenario*: Two candidate alternate speakers (Spk_A and Spk_B) have identical activity duration (0.30s = 30 frames). Spk_A is at temporal distance 0.50s, whereas Spk_B is at distance 0.10s.
   - *Outcome*: PASS. The proximity tie-breaker deterministically selects Spk_B.
2. `test_m1_nearby_insufficient_evidence`:
   - *Scenario*: Spk_Near is very close (0.05s) but has only 0.10s of activity (< 0.20s threshold). Spk_Valid is 1.0s away but has 0.25s of activity.
   - *Outcome*: PASS. Spk_Near is rejected for insufficient evidence; Spk_Valid is selected.
3. `test_m1_distant_speaker_greater_activity`:
   - *Scenario*: Spk_Major has 0.50s evidence at 1.0s distance; Spk_Minor has 0.25s evidence at 0.20s distance.
   - *Outcome*: PASS. Evidence duration is the primary criterion, selecting Spk_Major.
4. `test_m1_no_valid_alternate`:
   - *Scenario*: Only primary speaker active in context window.
   - *Outcome*: PASS. Returns `None` with reason `insufficient_evidence`.
5. `test_m1_deterministic_tie`:
   - *Scenario*: Spk_A and Spk_B have identical evidence duration (0.30s) and identical rounded distance (0.20s).
   - *Outcome*: PASS. Lexicographical speaker ID ordering breaks the tie deterministically (Spk_A selected).

### Correction B: M3 & M4 Contiguous Local Turn Extraction
6. `test_m3_contiguous_alternate_turns`:
   - *Scenario*: Preceding turn Spk_A (0.40s) and subsequent turn Spk_B (0.40s) both flank overlap `[2.00, 2.50)`.
   - *Outcome*: PASS. M3 selects the nearest preceding turn Spk_A.
7. `test_m3_none_gap_terminates_turn`:
   - *Scenario*: Two 0.15s bursts of Spk_A are separated by a 0.10s silence (`None`) gap. Neither burst independently reaches the 0.20s threshold.
   - *Outcome*: PASS. The `None` gap terminates the turn run; runs are not bridged across silence. M3 abstains.
8. `test_m3_turn_outside_max_gap`:
   - *Scenario*: Alternate turn Spk_A occurs 6.5s before overlap start ($\Delta t_{\text{max}} = 5.0\text{ s}$).
   - *Outcome*: PASS. Context window bounds strictly enforced; returns `None`.
9. `test_m3_unresolved_primary`:
   - *Scenario*: Overlap interval has no speaker assigned in baseline single-label hypothesis.
   - *Outcome*: PASS. Primary speaker is marked unresolved; algorithm abstains without falling back to global nearest segment.
10. `test_m4_bidirectional_agreement_and_disagreement`:
    - *Scenario 1*: Preceding turn is Spk_A, subsequent turn is Spk_A $\implies$ PASS (Spk_A selected).
    - *Scenario 2*: Preceding turn is Spk_A, subsequent turn is Spk_B $\implies$ PASS (abstains on disagreement).

### Correction C: M2 Embedding Purity and Validity
11. `test_m2_embedding_purity_filtering`:
    - *Scenario*: Subsegment 0 has only 33.3% overlap intersection; Subsegment 1 has 66.7% overlap intersection.
    - *Outcome*: PASS. Subsegment 0 rejected ($< 40\%$ purity); Subsegment 1 retained and evaluated via cosine affinity.
12. `test_m2_insufficient_pure_embeddings_abstains`:
    - *Scenario*: All candidate embeddings have $< 40\%$ overlap intersection.
    - *Outcome*: PASS. Abstains with reason `insufficient_pure_embeddings` rather than manufacturing contaminated representations.
13. `test_m2_empty_prototypes_abstains`:
    - *Scenario*: No speaker prototypes available.
    - *Outcome*: PASS. Abstains cleanly.

### Correction D: Overlap Timestamp Alignment
14. `test_timestamp_single_positive_frame`:
    - *Scenario*: Single isolated positive frame (16.875ms bin duration).
    - *Outcome*: PASS. Pruned by `min_duration_on = 0.10s`.
15. `test_timestamp_consecutive_positive_frames`:
    - *Scenario*: 8 consecutive positive frames = $8 \times 0.016875 = 0.135\text{ s} \ge 0.10\text{ s}$.
    - *Outcome*: PASS. Exact physical duration $(e_i - s_i + 1) \times S$ reconstructed without receptive field dilation.
16. `test_timestamp_boundary_clipping`:
    - *Scenario*: All frames positive extending to recording start and end.
    - *Outcome*: PASS. Clamped strictly to $[0.0, T_{\text{rec}}]$.

### Correction E: Multi-Label Hungarian Speaker Mapping
17. `test_hungarian_multi_label_simultaneous_speakers`:
    - *Scenario*: Ground truth and hypothesis both contain simultaneous overlapping speakers in `[0.0, 1.0]`.
    - *Outcome*: PASS. Contingency matrix accumulates co-occurrences for all simultaneous pairs without overwriting.
18. `test_hungarian_input_order_invariance`:
    - *Scenario*: Permuting the order of overlapping segments in the input list.
    - *Outcome*: PASS. Resulting bijective speaker mapping is identical.
19. `test_hungarian_unequal_speaker_counts`:
    - *Scenario*: Ground truth has 1 speaker, hypothesis has 2 clusters.
    - *Outcome*: PASS. Rectangular Hungarian matching maps R1 to H1 and explicitly outputs H2 in `unmatched_hyps`.

### Correction F: Strict Metric Error Handling
20. `test_metric_empty_reference_raises_exception`:
    - *Scenario*: Reference annotation duration is 0.0s.
    - *Outcome*: PASS. Raises `ValueError` rather than silently outputting 0.0.
21. `test_metric_distinguishes_zero_from_none`:
    - *Scenario*: Perfect diarization hypothesis ($H = R$).
    - *Outcome*: PASS. Returns exact 0.0 for DER, Miss, FA, Confusion, and valid JER, with status `SUCCESS`.
