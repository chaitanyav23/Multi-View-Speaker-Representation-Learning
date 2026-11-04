# Stage 5B.2 V4.1: M2 Prototype-Purity Interval Union Audit Report

**Audit Phase**: Stage 5B.2 V4.1 Prototype Purity Audit  
**Date**: November 4, 2026  
**Auditor**: Independent Forensic Research Engineer  
**Status**: AUDITED & REPAIRED

---

## 1. Problem Definition: Naive Summation Flaw in M2 Purity

In Stage 5B.2, prototype extraction (Method M2: Dual-Encoder Prototype Nearest Neighbor) computes prototype embeddings from single-speaker regions by evaluating purity across candidate subsegments.

In previous versions (V2, V3, V4.0), `load_dual_encoder_prototypes_pure` calculated speaker coverage by naively summing intersecting interval lengths:
```python
# V4.0 FLAGGED IMPLEMENTATION:
for seg in baseline_segments:
    inter = max(0.0, min(s_end, seg['end']) - max(s_start, seg['start']))
    if inter > 0:
        spk_coverage[seg['speaker']] += inter
```

### Mathematical Vulnerability
If `baseline_segments` contained overlapping or duplicate intervals assigned to the same speaker within `[s_start, s_end]`, `spk_coverage[seg['speaker']]` would double-count the shared duration:
$$\text{spk\_coverage}[\text{spk}] = \sum_i |I_i \cap [s_\text{start}, s_\text{end}]| > \left| \bigcup_i (I_i \cap [s_\text{start}, s_\text{end}]) \right|$$

This could artificially inflate a speaker's perceived coverage beyond the actual subsegment duration `s_dur`, falsely elevating purity above the 0.80 threshold.

---

## 2. Corrective Implementation in V4.1

The function `load_dual_encoder_prototypes_pure` in `run_stage5b_2_corrected_v4_1.py` was replaced with an exact 1D interval union algorithm:

```python
# V4.1 MATHEMATICALLY EXACT IMPLEMENTATION:
# Group clipped intervals by speaker
spk_intervals: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
for seg in baseline_segments:
    inter_s = max(s_start, float(seg['start']))
    inter_e = min(s_end, float(seg['end']))
    if inter_e > inter_s:
        spk_intervals[str(seg['speaker'])].append((inter_s, inter_e))

# Compute exact 1D interval union duration per speaker
spk_coverage: Dict[str, float] = {}
for spk, intervals in spk_intervals.items():
    intervals.sort(key=lambda x: x[0])
    union_dur = 0.0
    cur_s, cur_e = intervals[0]
    for n_s, n_e in intervals[1:]:
        if n_s <= cur_e:
            cur_e = max(cur_e, n_e)
        else:
            union_dur += (cur_e - cur_s)
            cur_s, cur_e = n_s, n_e
    union_dur += (cur_e - cur_s)
    spk_coverage[spk] = union_dur
```

### Numerical Stability
To prevent machine-precision edge case rejections where `1.20 / 1.50 = 0.7999999999999999`, the purity check incorporates standard floating-point tolerance:
```python
purity = best_dur / s_dur
if (best_dur / s_dur) >= (min_prototype_purity - 1e-6) and competing_dur <= (max_competing_overlap + 1e-6):
    admitted = True
```

---

## 3. Verification

The repair was verified via 13 dedicated unit tests in `tests/test_stage5b_2_v4_1_production.py`:
- `test_exact_duplicate_same_speaker_segments`: PASS
- `test_same_speaker_overlapping_segments`: PASS
- `test_adjacent_same_speaker_intervals`: PASS
- `test_coverage_exactly_at_080`: PASS
- `test_coverage_just_below_080`: PASS
- `test_competing_speaker_with_overlapping_duplicate_segments`: PASS
- `test_input_order_permutation_invariance`: PASS

All tests pass deterministically.
