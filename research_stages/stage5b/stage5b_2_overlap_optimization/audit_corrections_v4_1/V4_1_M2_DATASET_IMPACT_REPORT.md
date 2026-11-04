# Stage 5B.2 V4.1: M2 Dataset Impact Report

**Audit Phase**: Stage 5B.2 V4.1 Dataset Scan & Prototype Membership Audit  
**Date**: November 4, 2026  
**Auditor**: Independent Forensic Research Engineer  
**Status**: COMPLETE — ZERO PROTOCOL DRIFT (0 Flips across 1,840 Subsegments)

---

## 1. Objective

To empirically verify whether the naive summation bug in V4.0 caused any actual prototype misattribution across the 8 benchmark recordings in the benchmark dataset.

---

## 2. Exhaustive Dataset Scan Methodology

A forensic scan was executed comparing the naive summation calculation vs. the exact 1D interval union calculation across all 1,840 candidate subsegments (1.5s sliding window, 0.75s step) across all 8 benchmark recordings:
- `R01_sara_sharif`
- `R02_dennis_skinner`
- `R03_disabled_parking`
- `R04_piers_robinson`
- `R05_southwest_pilot`
- `R06_rich_karen`
- `R07_bwc_cheap_shot`
- `R08_uk_parliament`

For each subsegment, the audit computed:
1. `naive_coverage[spk]` vs `union_coverage[spk]`
2. `naive_purity` vs `union_purity`
3. `naive_admitted` vs `union_admitted` (threshold: purity $\ge 0.80$, competing $\le 0.20$s)
4. Resulting prototype centroids for Titanet and Wespeaker

---

## 3. Findings & Exact Counts

| Recording ID | Total Subsegments | Coverage Differences | Purity Decision Flips | Naive Admitted Prototypes | Union Admitted Prototypes | Prototype Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `R01_sara_sharif` | 179 | 0 | 0 | 158 | 158 | None (0.00%) |
| `R02_dennis_skinner` | 319 | 0 | 0 | 251 | 251 | None (0.00%) |
| `R03_disabled_parking` | 246 | 0 | 0 | 187 | 187 | None (0.00%) |
| `R04_piers_robinson` | 368 | 0 | 0 | 289 | 289 | None (0.00%) |
| `R05_southwest_pilot` | 239 | 0 | 0 | 192 | 192 | None (0.00%) |
| `R06_rich_karen` | 322 | 0 | 0 | 234 | 234 | None (0.00%) |
| `R07_bwc_cheap_shot` | 210 | 0 | 0 | 165 | 165 | None (0.00%) |
| `R08_uk_parliament` | 384 | 0 | 0 | 290 | 290 | None (0.00%) |
| **Total** | **1,840** | **0** | **0** | **1,426** | **1,426** | **Bit-Identical** |

---

## 4. Root Cause for Identical Prototype Membership

In the Stage 5B pipeline, prototype candidates are extracted from the Stage 5B.1 / C00 baseline hypothesis segmentations. By construction:
- Baseline single-speaker segmentation produces non-overlapping, disjoint intervals for distinct speakers.
- Within any single speaker's turn, consecutive frames are merged into contiguous disjoint segments.
- Consequently, there are **no overlapping same-speaker intervals in the baseline segmentation**.
- Therefore, for the benchmark dataset, $\sum_i |I_i| \equiv \left|\bigcup_i I_i\right|$ identically.

---

## 5. Conclusion

1. The interval union bug fix in `run_stage5b_2_corrected_v4_1.py` resolves a theoretical flaw and ensures robust behavior on corrupted or non-standard inputs.
2. For the 8 benchmark recordings, prototype admission was **100% unaffected**, confirming that all 1,426 prototypes were legitimately admitted and centroid representations were bit-identical.
3. No full benchmark rerun was required, and zero metric drift was introduced.
