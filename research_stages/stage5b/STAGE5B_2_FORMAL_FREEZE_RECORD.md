# Stage 5B.2 — Formal Freeze Record

**Project:** Multi-View Speaker Representation Learning  
**Stage:** 5B.2 — Multi-Speaker Overlap Optimization and Attribution  
**Frozen version:** V4.2.1  
**Status:** CLOSED AND FROZEN  
**Freeze authorization:** Explicitly authorized by the project owner.

---

## 1. Purpose

This document formally records the closure and freeze of Stage 5B.2.

Stage 5B.2 investigated overlap handling and speaker attribution under the existing diarization pipeline. The investigation progressed through V3, V4, V4.1, V4.2, and V4.2.1.

V4.2.1 is accepted as the final documented version of the investigation.

The freeze preserves the complete experimental and audit history. It does not imply that the final overlap configuration has demonstrated a statistically significant corpus-level improvement.

## 2. Frozen Scope

The freeze covers:

- V3: Corrected oracle multi-label mapping, temporal handling, and initial overlap-attribution evaluation.
- V4: Deterministic primary-speaker ordering, interval-union corrections, prototype-purity tests, and dispatcher permutation tests.
- V4.1: Independent correction and reconciliation of reported metrics, together with documentation of the execution-scope deviation.
- V4.2: Corrected configuration attribution, authoritative metric tables, and bootstrap analysis.
- V4.2.1: Documentation erratum, manifest reconciliation, and final freeze-readiness documentation.

All versions are retained as historical research records. V4.2.1 supersedes earlier versions as the final documented interpretation, but does not erase their history.

## 3. Authoritative Results

The final comparison is O02 versus O42-R2.

| Metric | O02 | O42-R2 | Difference |
|---|---:|---:|---:|
| Macro DER, c=0 | 44.90% | 45.42% | +0.5221 pp |
| Macro DER, c=0.25 | 42.62% | 42.97% | +0.3444 pp |
| Pooled DER, c=0 | 48.68% | 49.28% | +0.5980 pp |
| Macro JER | 61.27% | 60.73% | -0.54 pp |

The positive DER differences indicate worse performance for O42-R2.

### Per-recording behavior

| Recording | DER difference, O42-R2 − O02 |
|---|---:|
| R01 | Tie |
| R02 | Tie |
| R03 | +2.64 pp |
| R04 | -8.09 pp |
| R05 | +2.82 pp |
| R06 | +6.81 pp |
| R07 | Tie |
| R08 | Tie |

O42-R2 improved R04 substantially but degraded R03, R05, and R06.

There was one recording-level win, three losses, and four ties.

### Statistical analysis

The final bootstrap analysis used:

- 5,000 replicates.
- Random seed 42.
- Eight recording-level clusters.

The bootstrap confidence interval for the macro DER difference was:

**[-2.3501, +2.9050] percentage points.**

The paired Wilcoxon result was:

- W = 4.
- p = 0.8750.
- Effective non-tied sample size = 4.

The result does not establish a statistically significant corpus-level DER improvement.

## 4. Final Interpretation

The Stage 5B.2 investigation supports the following conclusions:

1. Overlap attribution changes can substantially improve individual recordings.
2. R04 benefits from the evaluated O42-R2 configuration.
3. R03, R05, and R06 regress under the same configuration.
4. The aggregate DER difference is not statistically significant.
5. Selective overlap attribution remains a possible future direction, not a validated solution.

**No claim of general corpus-level DER improvement is made.**

## 5. Methodological and Audit Corrections Preserved

The freeze retains the following corrections:

- Corrected oracle multi-label speaker mapping.
- Corrected Hungarian mapping tuple handling.
- Corrected timestamp-to-frame-bin handling.
- Deterministic primary-speaker ordering.
- Per-speaker interval-union handling.
- Prototype-purity test coverage.
- Dispatcher permutation tests.
- Correction of fabricated V4 report values in the subsequent audit.
- Reconciliation of authoritative metric CSVs.
- Correction of bootstrap difference and confidence-interval notation.
- Explicit documentation of the V4.1 protocol-scope deviation.

The V4.1 scope deviation remains part of the permanent record:

**48 downstream assignment/scoring evaluations were performed during V4.1 despite the conditional authorization not being triggered.**

This is documented transparently and must not be silently removed from the research history.

No model training, neural inference, embedding extraction, VAD inference, or ASR inference occurred during V4.2 or V4.2.1.

## 6. Integrity and Provenance

The following historical commits are preserved:

**Stage 5A.1.2 freeze:**

`ddf089de986f9e7cf026aa507324f3fccaf0cd8f`

**Stage 5B.2 V4.2.1 publication:**

`75a00bb356d062b35eb5ad61df20312325ac87b2`

Commit message:

`V4.2.1 Documentation Erratum & Manifest Reconciliation Report`

The V4.2.1 manifest contains six authoritative deliverables and was reported as successfully validated for:

- Valid JSON structure.
- Six manifest entries.
- No missing entries.
- No extra entries.
- No duplicate entries.
- Matching recorded file sizes.
- Matching SHA-256 hashes.

The four authoritative metric CSV hashes were reported identical across V3, V4, V4.1, and V4.2.1.

The final publication commit and push are preserved as historical provenance.

## 7. Freeze Boundaries

Following this freeze:

- Do not modify V3, V4, V4.1, V4.2, or V4.2.1 historical artifacts.
- Do not overwrite authoritative metric CSVs.
- Do not silently change the interpretation of the reported results.
- Do not present O42-R2 as a statistically established corpus-level improvement.
- Any future experiment must use a separate workspace and explicitly identify its dependencies on the frozen results.

Corrections to the frozen record, if ever necessary, must be added as a new, explicitly versioned audit or erratum. They must not rewrite the historical evidence.

## 8. Relationship to Stage 5B.4

Stage 5B.2 is closed.

Its results motivate, but do not authorize, a fresh-diarization experiment.

Stage 5B.4 is a separate proposed investigation. It will test whether improved VAD and overlap handling can improve speaker attribution when the system performs fresh diarization rather than projecting new masks onto frozen baseline speaker assignments.

Stage 5B.4 must not modify the frozen Stage 5B.2 artifacts.

## 9. Final Freeze Declaration

Stage 5B.2 is formally accepted as complete and frozen at V4.2.1.

The investigation is closed with a negative corpus-level result: the evaluated O42-R2 configuration did not establish a statistically significant DER improvement over O02.

The recording-specific R04 improvement is retained as a useful observation, while regressions in R03, R05, and R06 remain part of the final interpretation.

**Freeze status: CLOSED — V4.2.1.**
