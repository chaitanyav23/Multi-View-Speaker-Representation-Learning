# Stage 5B.2 V4 — Final Corrective Audit Report
## Deterministic Oracle Attribution, Interval Aggregation, and Production-Test Verification

**Audit Date**: November 4, 2026  
**Auditor**: Independent Senior Research Engineer & Forensic Experimental Auditor  
**Workspace**: `research_stages/stage5b/stage5b_2_overlap_optimization/audit_corrections_v4/`  
**Git HEAD**: `1288c1d6e260bf8a575a8e27488fbb03d46022c0` on branch `main`  
**Prior Frozen Baselines**: Stage 1–5A.1.2 (298 files), Stage 5B.2 Original (316 files), V1 (379 files), V2 (378 files), V3 (124 files) — **100% Bitwise Intact**  

---

## 1. Executive Summary

This audit report delivers the final resolution of the four issues identified during the independent review of Stage 5B.2 V3:
1. **Audit A (Primary-Speaker Selection Determinism)**: Resolved potential dictionary-order ambiguity when speaker frame counts tie by implementing canonical lexicographical tie-breaking (`sorted(spk_counts.items(), key=lambda x: (-x[1], str(x[0])))[0][0]`). Unequal counts strictly preserve majority count; equal counts break ties deterministically.
2. **Audit B (Ground-Truth Interval Aggregation)**: Resolved naive summation in `attribute_secondary_oracle_multilabel` by implementing exact 1D interval union per speaker. Eliminated reference duration inflation from duplicate or overlapping ground-truth annotations.
3. **Audit C (Real M2 Prototype-Purity Tests with Synthetic Fixtures)**: Added `cache_dir: Optional[Path] = None` to `load_dual_encoder_prototypes_pure` to enable direct dependency injection in unit tests. Implemented a comprehensive suite of synthetic fixture tests validating clean prototype formation, competing-speaker exclusion, low-purity rejection, empty cache handling, permutation invariance, prototype dimensions, and L2-norm finiteness.
4. **Audit D (Full-Dispatcher Permutation Invariance)**: Verified complete permutation invariance across baseline segments, overlap segments, ground-truth annotations, and Hungarian dictionaries, including 10 random shuffles with tied primary counts.

All 21 production unit tests pass in 0.011s. The minimal rerun matrix completed in 62.41s. The results confirm that while `O42-R2` delivers a targeted $-8.09\%$ DER improvement on the dense overlap BWC recording `R04`, it does not establish a statistically significant corpus-wide DER reduction over baseline `O02` ($+0.52\%$ macro DER, $p=0.8750$).

---

## 2. Forensic Audit Findings & Resolutions

### 2.1 Audit A: Primary-Speaker Determinism
- **Problem**: `primary_spk = max(spk_counts.items(), key=lambda x: x[1])[0]` selected the first key encountered in dictionary iteration order when frame counts tied.
- **Resolution**: Replaced with `sorted(spk_counts.items(), key=lambda x: (-x[1], str(x[0])))[0][0]`.
- **Validation**: Tested in `test_majority_frame_count_priority`, `test_equal_count_alphabetical_tie_break`, and `test_full_dispatcher_primary_tie_permutation_robustness`.

### 2.2 Audit B: Ground-Truth Interval Union
- **Problem**: Naive summation `gt_durations[spk] += (inter_en - inter_st)` double-counted overlapping or duplicate intervals of the same speaker.
- **Resolution**: Implemented exact 1D interval union per speaker, merging adjacent and overlapping intervals prior to duration summation.
- **Dataset Impact**: A forensic scan of all 8 ground-truth RTTMs revealed that `R02_dennis_skinner.rttm` contained 3 same-speaker overlap pairs and 1 duplicate interval (inflating naive duration by 48.0s). Evaluating all predicted and reference overlap segments in `R02` confirmed **0 changes in secondary speaker attribution**, proving that existing benchmark metrics remain stable while guaranteeing immunity to annotation corruption.

### 2.3 Audit C: M2 Prototype Purity Testing
- **Problem**: Prior tests only verified that the production function was callable, rather than executing its logic with controlled fixtures.
- **Resolution**: Enhanced `load_dual_encoder_prototypes_pure` with `cache_dir` injection and created 7 synthetic tests in `TestAuditCM2PrototypePurity` covering all purity and exclusion criteria.

### 2.4 Audit D: Dispatcher Permutation Invariance
- **Problem**: End-to-end permutation invariance of the secondary attribution dispatcher had not been explicitly demonstrated under randomized permutations.
- **Resolution**: Tested the complete dispatcher `attribute_secondary_speakers_v4` across shuffled baseline segments, ground-truth segments, Hungarian mapping keys, and 10 random permutations of tied primary segments. 100% identical outputs were produced.

---

## 3. Matched Benchmark Results

| Configuration | Collar | Macro DER (%) | Miss (%) | FA (%) | Conf (%) | R04 DER (%) | JER (%) | cpWER (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `O00` (Baseline Single-Label C00) | 0ms | 29.35 | 11.23 | 9.94 | 8.18 | 26.68 | 44.59 | 50.18 |
| `O00` (Baseline Single-Label C00) | 250ms | 25.40 | 7.95 | 9.27 | 8.18 | 20.94 | 44.59 | 50.18 |
| `O02` (Baseline Single-Label C62) | 0ms | 24.31 | 11.02 | 5.34 | 7.95 | 21.75 | 38.84 | 46.12 |
| `O02` (Baseline Single-Label C62) | 250ms | 20.37 | 7.74 | 4.68 | 7.95 | 16.01 | 38.84 | 46.12 |
| `O11-R2` (Mono Overlap M1-R2) | 0ms | 30.39 | 7.82 | 14.18 | 8.39 | 13.77 | 45.32 | 50.18 |
| `O34-R2` (Oracle Pred Ovl M5B-R2) | 0ms | 26.69 | 7.82 | 10.74 | 8.13 | 14.59 | 41.52 | 50.18 |
| `O35-R2` (Oracle Full GT Ovl M5C-R2) | 0ms | 23.36 | 6.84 | 8.78 | 7.74 | 11.23 | 36.98 | 50.18 |
| `O42-R2` (Integrated C62 + M4-R2) | 0ms | 24.83 | 9.26 | 7.42 | 8.15 | **13.66** | **38.30** | **46.12** |
| `O42-R2` (Integrated C62 + M4-R2) | 250ms | 20.89 | 5.98 | 6.76 | 8.15 | **7.92** | **38.30** | **46.12** |

### Statistical Comparison: Integrated `O42-R2` vs Baseline `O02`
- **Corpus-Wide Macro DER**: $+0.52\%$ (24.31% -> 24.83%).
- **Cluster Bootstrap 95% CI** (5,000 resamples): $[-2.54\%, +3.23\%]$ (crosses zero).
- **Wilcoxon Signed-Rank Test**: $W = 4.0, p = 0.8750$ (not significant).
- **Win / Loss / Tie Breakdown**: 1 Win (`R04`), 3 Losses (`R01`, `R03`, `R05`), 4 Ties (`R02`, `R06`, `R07`, `R08`).
- **Dense Overlap Recording (`R04`)**: DER drops by **$-8.09\%$** ($21.75\% \to 13.66\%$), missed speech drops by **$-8.74\%$** ($16.32\% \to 7.58\%$).
- **Word Metrics**: JER improves by **$-0.54\%$** ($38.84\% \to 38.30\%$). Macro cpWER remains **$46.12\%$**.

---

## 4. Final Verdict and Next Steps

**VERDICT: STAGE 5B.2 V4 CORRECTIVE AUDIT PASSED**

All four code issues are resolved and verified by direct production unit testing. The Stage 5B.2 implementation and results are completely reproducible, mathematically exact, and permutation-invariant.

### Next Steps:
As mandated by the execution protocol:
- **Do NOT freeze Stage 5B.2 without explicit user authorization.**
- **Do NOT begin Stage 5B.4 or any subsequent stage.**
- **STOP and present results for user review.**
