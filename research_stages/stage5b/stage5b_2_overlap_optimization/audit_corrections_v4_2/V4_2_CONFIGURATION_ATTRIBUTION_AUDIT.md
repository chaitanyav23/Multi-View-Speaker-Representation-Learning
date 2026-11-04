# Stage 5B.2 V4.2: Configuration and Attribution Method Forensic Audit

**Audit Phase**: Stage 5B.2 V4.2 Configuration Audit  
**Date**: November 4, 2026  
**Auditor**: Independent Forensic Research Engineer  
**Status**: COMPLETE — ATTRIBUTION FORMALLY RECONCILED

---

## 1. Executive Summary

A forensic review of the source code and configuration dictionaries in Stage 5B.2 reveals that the experimental intervention `O42-R2` was misattributed in certain narrative sections of previous audit reports (including V4.1):
- `O42-R2` was erroneously described as using **Method M2** (Dual-Encoder Prototype Nearest Neighbor).
- **The Ground Truth Source Code**: `O42-R2` explicitly uses **Method M4** (`M4_conservative_abstention`), combined with C62 consensus VAD.
- **Method M2** is an independent embedding prototype nearest-neighbor attribution strategy evaluated under configuration `O31-R2` (and variants `O31-T-R2`, `O31-W-R2`, `O31-NF-R2`).

This report provides primary source code citations proving these definitions, disentangles the scientific conclusions of M4 and M2, and corrects the downstream attribution.

---

## 2. Source-Code Proof of Configuration Definitions

In [`run_stage5b_2_corrected_v4_1.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/scripts/run_stage5b_2_corrected_v4_1.py#L1026-L1060), configurations are formally defined in the production rerun matrix:

```python
# Lines 1026-1060 in run_stage5b_2_corrected_v4_1.py:
configurations = [
    # Baseline Single-Label (No Overlap)
    ("O00", "Baseline Single-Label (C00 VAD, No Overlap)", "C00", "none", 0.0, "M0_no_overlap"),
    ("O01", "Baseline Single-Label (C50 VAD, No Overlap)", "C50", "none", 0.0, "M0_no_overlap"),
    ("O02", "Baseline Single-Label (C62 VAD, No Overlap)", "C62", "none", 0.0, "M0_no_overlap"),

    # Attribution Strategies Comparison (C00, Mono, T=0.40)
    ("O30-R2", "Attribution: M1-R2 Proximity Tie-Break", "C00", "mono", 0.40, "M1_temporal_context"),
    ("O31-R2", "Attribution: M2-R2 Pure Prototype Concat (448d)", "C00", "mono", 0.40, "M2_prototype_concat"),
    ("O31-T-R2", "Attribution: M2-R2 Pure TitaNet (192d)", "C00", "mono", 0.40, "M2_prototype_titanet"),
    ("O31-W-R2", "Attribution: M2-R2 Pure WeSpeaker (256d)", "C00", "mono", 0.40, "M2_prototype_wespeaker"),
    ("O31-NF-R2", "Attribution: M2-R2 Pure Normalized Cosine Avg", "C00", "mono", 0.40, "M2_prototype_normfusion"),
    ("O32-R2", "Attribution: M3-R2 Contiguous Turn Alternation", "C00", "mono", 0.40, "M3_turn_alternation"),
    ("O33-R2", "Attribution: M4-R2 Conservative Contiguous Abstain", "C00", "mono", 0.40, "M4_conservative_abstention"),

    # Hungarian Multi-Label Oracle Diagnostics
    ("O34-R2", "Oracle Diag: M5B-R2 Pred Ovl + Hungarian Multi-Label Attr", "C00", "mono", 0.40, "M5B_predicted_ovl_oracle_attr"),
    ("O35-R2", "Oracle Diag: M5C-R2 Full Oracle Ovl + Hungarian Multi-Label Attr", "C00", "oracle_gt", 0.0, "M5C_full_oracle"),

    # Integrated Upstream Pipelines
    ("O40-R2", "Integrated: C50 Mono VAD + Mono Overlap (M1-R2)", "C50", "mono", 0.40, "M1_temporal_context"),
    ("O41-R2", "Integrated: C62 Consensus VAD + Mono Overlap (M1-R2)", "C62", "mono", 0.40, "M1_temporal_context"),
    ("O42-R2", "Integrated: C62 Consensus VAD + Conservative Overlap (M4-R2)", "C62", "mono", 0.40, "M4_conservative_abstention"),
]
```

### Detailed Parameter Breakdown:
- **`O02`**:
  - Backbone VAD: `C62` (Optimized Consensus VAD: DFN + Mono posterior consensus threshold $\theta=0.62$).
  - Overlap Stream: `none`.
  - Overlap Detection Threshold: `0.0`.
  - Attribution Method: `M0_no_overlap` (Single-label baseline; overlap frames remain single-label assigned to the primary speaker).
- **`O42-R2`**:
  - Backbone VAD: `C62` (Consensus VAD).
  - Overlap Stream: `mono` (Pyannote mono overlap detection posterior stream).
  - Overlap Detection Threshold: `0.40`.
  - Attribution Method: `M4_conservative_abstention` (**Method M4**, conservative contiguous turn alternation with abstention).

---

## 3. Algorithmic Distinction: M4 vs M2

### 3.1 Method M4 (`attribute_m3_m4_corrected`)
Implemented in [`run_stage5b_2_corrected_v4_1.py#L728-L800`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/scripts/run_stage5b_2_corrected_v4_1.py#L728-L800):
- **Mechanism**: Evaluates local conversational turn alternation immediately preceding and succeeding the overlap interval.
- **Rule**: If the speaker changes across the overlap boundary without intervening silence $> 5.0$s, M4 assigns the alternating speaker as the secondary speaker.
- **Conservative Abstention**: If the preceding and succeeding turns belong to the same primary speaker, or if no clear adjacent speaker turn exists, M4 **abstains** from secondary attribution rather than guessing.
- **Application**: Executed within `O33-R2` and the end-to-end integrated pipeline `O42-R2`.

### 3.2 Method M2 (`load_dual_encoder_prototypes_pure`)
Implemented in [`run_stage5b_2_corrected_v4_1.py#L320-L460`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4_1/scripts/run_stage5b_2_corrected_v4_1.py#L320-L460):
- **Mechanism**: Extracts L2-normalized embedding centroids (prototypes) for each speaker from pure single-speaker subsegments (purity $\ge 0.80$, competing speech $\le 0.20$s).
- **Attribution**: Overlap frames are extracted as 448-dimensional fused embeddings (or 192d TitaNet / 256d WeSpeaker) and assigned to the nearest non-primary prototype via cosine similarity.
- **Application**: Evaluated in `O31-R2`, `O31-T-R2`, `O31-W-R2`, `O31-NF-R2`.

---

## 4. Origin of Narrative Misattribution in Previous Reports

During Stage 5B.2 V4.1, the audit mandate contained two concurrent focal points:
1. Reconciling the benchmark numbers of the headline comparison (`O02` vs `O42-R2`).
2. Auditing the prototype-purity implementation of `load_dual_encoder_prototypes_pure` (Method M2).

Because both items were investigated concurrently, several narrative paragraphs in V4.1 inadvertently referred to `O42-R2` as "M2 prototype nearest neighbor" rather than "M4 conservative overlap attribution".

---

## 5. Corrected Scientific Conclusions

1. **Attribution of Benchmark Performance**:
   - The headline result ($\Delta = +0.52\%$ macro DER, $R04 = -8.09\%$ DER) belongs strictly to **`O42-R2` (C62 Consensus VAD + M4 Conservative Overlap Attribution)**, compared against **`O02` (C62 Consensus Baseline)**.
   - It is **not** an M2 result.
2. **Authoritative Phrasing for Freeze Addendum**:
   > *"Under the evaluated configuration, conservative M4 overlap attribution combined with C62 consensus VAD substantially improves diarization on the high-overlap R04 recording. However, the improvement does not generalize across the eight-recording benchmark, with degradation on R03, R05 and R06. The corpus-level difference is not statistically significant. These findings motivate selective overlap attribution rather than unconditional application."*
3. **Status of Method M2**:
   - The M2 prototype-purity interval-union repair and its dataset scan (0 flips across 1,840 subsegments) remain an independent implementation audit finding.
   - **Method M2 has not demonstrated downstream diarization improvement over baseline**, and must not be claimed to have done so.
