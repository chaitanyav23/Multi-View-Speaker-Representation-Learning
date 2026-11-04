# Stage 5B.2 V3 Oracle Mapping Verification Report

**Workspace:** `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v3/`  
**Production Function:** `attribute_secondary_oracle_multilabel` in [`scripts/run_stage5b_2_corrected_v3.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v3/scripts/run_stage5b_2_corrected_v3.py)  
**Test Suite:** [`tests/test_stage5b_2_production.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v3/tests/test_stage5b_2_production.py)  

---

## 1. Problem Identification: Order-Dependence in V2 Oracle Mapping

In Stage 5B.2 V2, the oracle attribution branches (M5B and M5C) contained the following logic:

```python
elif attribution_method == "M5B_predicted_ovl_oracle_attr":
    if gt_segs and hungarian_map and primary_spk is not None:
        active_gt = sorted(list(set(s["speaker"] for s in gt_segs if s["start"] <= mid <= s["end"])))
        active_mapped = sorted(list(set(hungarian_map[g] for g in active_gt if g in hungarian_map)))
        alts = [h for h in active_mapped if h != primary_spk]
        sec_spk = alts[0] if alts else None
```

### Forensic Defect Analysis:
1. **Arbitrary List Index Selection (`alts[0]`)**:
   When multiple alternate candidate speakers were active (e.g. in 3-speaker overlap, or when `primary_spk` was not among the active mapped speakers), `alts[0]` selected the first speaker purely according to alphabetical sorting of hypothesis speaker IDs (`sorted(...)`), rather than according to physical speech evidence or co-occurrence duration.
2. **Point-Sampling Defect (`s["start"] <= mid <= s["end"]`)**:
   Ground-truth activity was evaluated strictly at the single infinitesimal midpoint `mid = (st + en) / 2.0`. A reference speaker active for 80% of the overlap interval but pausing at `mid` was completely ignored.
3. **Single-Speaker False Alarm Contamination**:
   In predicted overlap intervals that actually contained only ONE ground-truth reference speaker (i.e. an overlap detector false alarm), if `primary_spk` differed from the mapped speaker, `alts[0]` assigned that mapped speaker as a *secondary* speaker alongside `primary_spk`, creating a spurious two-speaker overlap hypothesis on clean single-speaker audio.
4. **Order-Dependence**:
   If inputs were presented in non-sorted order or if speaker IDs were permuted, `alts[0]` could produce different selections.

---

## 2. Corrective Mathematical Architecture: `attribute_secondary_oracle_multilabel`

The production function `attribute_secondary_oracle_multilabel` replaces the arbitrary selection with a deterministic, multi-label co-occurrence mapping.

### Mathematical Formulation:
For an overlap interval $I = [t_{\text{start}}, t_{\text{end}})$ with duration $\Delta t_I = t_{\text{end}} - t_{\text{start}}$:

1. **Exact Interval Intersection**:
   For every reference segment $s = (t_s, t_e, r) \in \mathcal{S}_{\text{ref}}$:
   $$\Delta t(s, I) = \max\left(0, \min(t_{\text{end}}, t_e) - \max(t_{\text{start}}, t_s)\right)$$
   Total local reference duration for reference speaker $r$:
   $$D_{\text{local}}(r) = \sum_{s \in \mathcal{S}_{\text{ref}}, \text{spk}(s) = r} \Delta t(s, I)$$
   Active reference set: $\mathcal{R}_{\text{active}} = \{r : D_{\text{local}}(r) > 10^{-6}\text{s}\}$.

2. **Hungarian Hypothesis Mapping**:
   Using the global multi-label Hungarian mapping $\mathcal{M} : \mathcal{R} \to \mathcal{H}$:
   $$D_{\text{hyp}}(h) = \sum_{r \in \mathcal{R}_{\text{active}}, \mathcal{M}(r) = h} D_{\text{local}}(r)$$
   Mapped active hypothesis set: $\mathcal{H}_{\text{active}} = \{h : D_{\text{hyp}}(h) > 0\}$.

3. **Ground-Truth Speaker Count Guard**:
   If $|\mathcal{R}_{\text{active}}| \le 1$:
   The audio is physically single-speaker. The function explicitly returns:
   $$\text{speaker} = \text{None}, \quad \text{reason} = \text{"single\_reference\_speaker\_no\_overlap"}$$
   This eliminates spurious secondary attribution on overlap false alarms.

4. **Candidate Alternate Set**:
   If primary hypothesis speaker $h_{\text{prim}}$ is assigned:
   $$\mathcal{H}_{\text{cand}} = \mathcal{H}_{\text{active}} \setminus \{h_{\text{prim}}\}$$
   If $h_{\text{prim}}$ is unresolved (`None`):
   All mapped hypothesis speakers are ranked; the top speaker is designated primary, and the second top speaker is designated secondary.

5. **Deterministic Three-Tier Ranking**:
   Candidates $h \in \mathcal{H}_{\text{cand}}$ are sorted by the strict tuple key:
   $$\text{SortKey}(h) = \left( -\text{round}(D_{\text{hyp}}(h), 6), \; -\text{round}(C_{\text{global}}(h), 6), \; h \right)$$
   - **Primary Criterion**: Local overlap duration $D_{\text{hyp}}(h)$ within $[t_{\text{start}}, t_{\text{end}})$ (maximize).
   - **Secondary Criterion**: Global file-level co-occurrence $C_{\text{global}}(h) = -C_{\mathcal{M}^{-1}(h), h}$ from Hungarian contingency matrix (maximize).
   - **Tertiary Criterion**: Lexicographical hypothesis speaker ID string $h$ (deterministic tie-break).

6. **Two-Speaker Hypothesis Limit & Multi-Speaker Truncation Flag**:
   Standard RTTM evaluation pipelines accommodate at most 2 simultaneous tracks per interval.
   The top candidate $h_1$ is attributed as secondary speaker.
   Any remaining active candidates $h_2, h_3, \dots$ are recorded in `discarded_speakers`.
   If $|\mathcal{H}_{\text{cand}}| > 1$, the function sets:
   $$\text{is\_truncated\_multispeaker} = \text{True}$$
   This explicitly documents the two-speaker approximation used by the evaluation format.

---

## 3. Unit Test Verification Matrix

All test cases in [`tests/test_stage5b_2_production.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v3/tests/test_stage5b_2_production.py) directly import and execute `attribute_secondary_oracle_multilabel`:

| Test Name | Condition Tested | Result | Verification Proof |
| :--- | :--- | :---: | :--- |
| `test_oracle_order_invariance_reversed_and_shuffled` | Permuted segment orders: normal, reversed, random shuffle (seed 12345). | **PASSED** | Exact match on `speaker`, `reason`, and `discarded_speakers`. |
| `test_oracle_equal_cooccurrence_tie_break` | Equal local duration (0.50s) and equal co-occurrence between HYP_A and HYP_Z. | **PASSED** | Deterministically selects `HYP_A` via lexicographical tie-break; discards `HYP_Z`. |
| `test_oracle_three_speaker_overlap_truncation` | 3 simultaneous speakers in [1.8, 2.4]: REF_A (0.6s), REF_B (0.6s), REF_C (0.6s). | **PASSED** | Flags `is_truncated_multispeaker=True`, attributes top alternate, records 1 discarded speaker. |
| `test_oracle_missing_mapping_safety` | Reference speaker active in ground-truth but missing from Hungarian mapping. | **PASSED** | Safely returns `None` with reason `"no_alternate_mapped_speaker"` without KeyError. |
| `test_oracle_duplicate_labels_aggregation` | Multiple disjoint segments for same speaker in interval ([1.0, 1.4] and [1.6, 2.0]). | **PASSED** | Accurately aggregates total duration (0.80s) into single total $D_{\text{hyp}}$. |
| `test_empty_and_malformed_inputs` | Zero-length interval `(5.0, 5.0)` and empty ground-truth segment list. | **PASSED** | Safely returns `None` with reason `"empty_overlap_interval"`. |

---

## 4. Empirical Impact on Stage 5B.2 Oracle Configurations

Rerunning the oracle configurations (`O34-R2` and `O35-R2`) with `attribute_secondary_oracle_multilabel` produced the following metrics:

| Config ID | Description | V2 Macro DER | V3 Macro DER | V2 Pooled DER | V3 Pooled DER | Key Per-Recording Shift |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **O34-R2** | Predicted Overlap + Hungarian Oracle | 44.44% | **44.71%** | 47.13% | **47.55%** | Correctly rejects spurious secondary attribution on single-speaker false alarms in R03 (+0.52%) and R06 (+1.54%). |
| **O35-R2** | Full Oracle Overlap + Hungarian Oracle | 41.14% | **40.79%** | 42.53% | **42.05%** | **R02 DER drops from 50.50% to 47.73% (-2.77% absolute improvement)** by selecting the maximal-support interrupter during parliamentary exchanges. |

### Conclusion:
The V3 deterministic multi-label oracle mapping eliminates order-dependence, handles arbitrary speaker counts gracefully, flags multi-speaker truncation explicitly, and provides a mathematically sound upper bound for Stage 5B.2.
