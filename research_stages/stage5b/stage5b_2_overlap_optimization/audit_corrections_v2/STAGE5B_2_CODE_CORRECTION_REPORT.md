# Stage 5B.2 Code Correction Report: Forensic Issue Resolution

## 1. Executive Summary

This report documents the forensic investigation and code-level resolution of six implementation issues identified in the Stage 5B.2 overlap optimization suite.

Source script:
[`research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections/scripts/run_stage5b_2_corrected.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections/scripts/run_stage5b_2_corrected.py)

Patched and verified script:
[`research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v2/scripts/run_stage5b_2_corrected_v2.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v2/scripts/run_stage5b_2_corrected_v2.py)

---

## 2. Issue-by-Issue Resolution

### Issue A: M1 Temporal Attribution Tie-Breaking
- **Defect**: In the original script, line 247 claimed to tie-break by temporal proximity, but only executed `valid_cands.sort(key=lambda x: x[1], reverse=True)`. Ties between speakers with equal evidence were broken by Python's arbitrary dictionary iteration order.
- **Correction**: Implemented multi-tier deterministic sorting:
  $$\text{Sort Key} = (-\text{evidence\_frames}, \; \text{round}(\text{min\_distance\_sec}, 6), \; \text{speaker\_id})$$
  1. Primary: valid alternate evidence duration in context window $[t_{\text{mid}} - 3.0, t_{\text{mid}} + 3.0]$ descending.
  2. Secondary: actual minimum temporal distance to overlap interval $[t_{\text{start}}, t_{\text{end}})$ ascending.
  3. Tertiary: deterministic lexicographical speaker ID ascending.
- **Verification**: Validated by unit tests `test_m1_equal_evidence_different_distances` and `test_m1_deterministic_tie`.

### Issue B: M3/M4 Contiguous Local Turn Extraction
- **Defect**:
  1. Turn searching was anchored to the overlap midpoint `mid_idx` rather than the overlap interval boundaries $[t_{\text{start}}, t_{\text{end}})$, causing searches to evaluate inside the overlap interval.
  2. The loop logic `if s is not None ... elif s == primary_spk:` ignored `None` frames when scanning, bridging across arbitrary silence intervals between distinct speaker turns.
  3. When no primary speaker was assigned at the midpoint, the code searched globally for the nearest segment in the recording.
- **Correction**:
  1. Bounded search relative to exact interval boundaries: preceding turns search backwards from $t_{\text{start}}$, subsequent turns search forwards from $t_{\text{end}}$.
  2. Enforced strict turn extraction via `extract_contiguous_turns`: any `None` frame or different speaker immediately terminates a contiguous turn run.
  3. Calculated temporal distance directly from the candidate turn boundary to the overlap boundary.
  4. If the overlap interval has no primary speaker assigned in the baseline hypothesis, it is marked as `unresolved_primary` and abstained, eliminating global file searches.
- **Verification**: Validated by unit tests `test_m3_none_gap_terminates_turn`, `test_m3_unresolved_primary`, and `test_m4_bidirectional_agreement_and_disagreement`.

### Issue C: M2 Prototype and Overlap-Specific Embedding Validity
- **Defect**: Every 1.5s subsegment embedding intersecting the overlap interval was included in the overlap representation via duration weighting, regardless of whether the overlap was 0.1s or 1.5s. In short overlaps, $>80\%$ of the embedding window was non-overlapping speech.
- **Correction**:
  1. Introduced explicit overlap purity filtering: a subsegment embedding is eligible only if its intersection with the overlap interval represents $\ge 40\%$ of the subsegment window duration ($f_{\text{ovl}} \ge 0.40$).
  2. Introduced prototype purity filtering: baseline subsegments used to construct speaker prototype centroids must contain $\ge 80\%$ duration of the assigned speaker with zero competing-speaker speech.
  3. If no cached embedding satisfies the purity threshold for a given overlap interval, the algorithm abstains (`insufficient_pure_embeddings`) rather than synthesizing an ungrounded representation.
- **Verification**: Validated by unit test `test_m2_embedding_purity_filtering`.

### Issue D: Overlap Timestamp Alignment & Receptive Field Decoupling
- **Defect**: Interval bounds were calculated as `start = s_i * step` and `end = e_i * step + win_dur`, expanding active regions by the full 61.9375 ms receptive field. The receptive field describes model input context, not the temporal support of the output label.
- **Correction**:
  1. Decoupled receptive field from output label temporal support.
  2. Each output frame $i$ is centered at `frame_times[i]` with bin width equal to the stride $S = 0.016875\text{ s}$ (270 audio samples).
  3. For contiguous active runs $[s_i, e_i]$, the temporal support is:
     $$t_{\text{start}} = \max(0.0, \; \text{frame\_times}[s_i] - \frac{S}{2})$$
     $$t_{\text{end}} = \min(T_{\text{rec}}, \; \text{frame\_times}[e_i] + \frac{S}{2})$$
  4. Yields exact physical duration $(e_i - s_i + 1) \times S$ without dilation.
- **Verification**: Validated by unit tests `test_timestamp_single_positive_frame` and `test_timestamp_consecutive_positive_frames`.

### Issue E: Multi-Label Hungarian Speaker Mapping
- **Defect**: The Hungarian cost matrix was constructed using 1D frame arrays `ref_frames[i] = spk` and `hyp_frames[i] = spk`. When multiple speakers were active simultaneously, later segments in the RTTM overwrote earlier segments, making the mapping order-dependent.
- **Correction**:
  1. Represented frame-level speaker activity as sets of strings: `ref_active[i]` and `hyp_active[i]`.
  2. Accumulated joint co-occurrences for all pairs $(r, h) \in \text{ref\_active}[i] \times \text{hyp\_active}[i]$.
  3. Solved the one-to-one optimal bipartite matching using `linear_sum_assignment` on the full multi-label co-occurrence matrix.
  4. Handled rectangular matrices ($K_{\text{ref}} \neq K_{\text{hyp}}$) and explicitly recorded unmatched reference and hypothesis speakers.
- **Verification**: Validated by unit tests `test_hungarian_multi_label_simultaneous_speakers` and `test_hungarian_input_order_invariance`.

### Issue F: Strict Metric Error Handling
- **Defect**: Exceptions in Jaccard Error Rate were silenced via `except Exception: jer = 0.0`, converting metric failures into apparently perfect scores. Detail keys in DER could default to zero.
- **Correction**:
  1. Removed all zero fallbacks. If JER fails, it returns `None` and records the explicit exception string in `jer_error`.
  2. If reference speech duration is zero or negative, raises `ValueError`.
  3. Missing required metric keys raise `KeyError`.
  4. Summary aggregations report `jer_valid_count` and `cpwer_valid_count` explicitly.
- **Verification**: Validated by unit tests `test_metric_empty_reference_raises_exception` and `test_metric_distinguishes_zero_from_none`.
