# Stage 5B — Consolidated Results Report

**Project:** Multi-View Speaker Representation Learning  
**Scope:** Stage 5B.0, 5B.1, 5B.2 and 5B.3  
**Status:** Investigations completed; Stage 5B.2 frozen at V4.2.1; Stage 5B.4 proposed but not yet authorized for execution.

---

## 1. Executive Summary

Stage 5B investigates whether improvements to speaker-count estimation, voice activity detection (VAD), and overlap handling can improve speaker attribution in the existing multi-view speaker representation pipeline.

The investigation was divided into four independent tracks:

- **Stage 5B.0:** Reconnaissance and audit of the existing pipeline, annotations, VAD outputs, overlap, and speaker-count behavior.
- **Stage 5B.1:** VAD optimization under controlled configurations.
- **Stage 5B.2:** Multi-speaker overlap optimization and attribution.
- **Stage 5B.3:** Speaker-count estimation and clustering diagnostics.

The results do not establish a single configuration that reliably improves all downstream metrics across the eight recordings.

Three important conclusions emerge:

1. **VAD optimization can improve speech-region detection and projected diarization metrics**, but the measured downstream effect in Stage 5B.1 was evaluated under frozen baseline speaker assignments. It does not establish that a fresh diarization pipeline would achieve the same improvement.
2. **Overlap-handling changes can substantially help individual recordings**, especially R04, but the tested O42-R2 configuration did not produce a statistically established corpus-level DER improvement over O02.
3. **Speaker-count estimation remains difficult in recordings with weak representation separation or fleeting speakers.** Alternative clustering heuristics produced small aggregate changes, but none established a robust general solution.

The main unresolved question is therefore:

> Can improved VAD and overlap handling produce better speaker attribution when the system performs fresh diarization, rather than simply projecting new masks onto frozen baseline speaker assignments?

This is the central question proposed for Stage 5B.4.

---

## 2. Experimental Context and Evaluation Scope

### 2.1 Dataset

The investigation uses eight recordings:

- R01
- R02
- R03
- R04
- R05
- R06
- R07
- R08

Speaker identities are recording-local. Speaker labels from different recordings must not be treated as identities of the same person.

The dataset contains substantial variation in recording quality, speaker activity, overlap, and duration.

### 2.2 Evaluation metrics

The principal metrics used throughout the investigations include:

| Metric | Interpretation |
|---|---|
| DER | Diarization Error Rate |
| DER c=0 | DER with zero collar |
| DER c=0.25 | DER with a 250 ms collar |
| JER | Jaccard Error Rate |
| Miss | Reference speech missed by the system |
| False Alarm | Hypothesized speech outside reference speech |
| Confusion | Speech assigned to the wrong speaker |
| Speaker-count MAE | Mean absolute error in estimated speaker count |
| cpWER | Concatenated minimum-permutation Word Error Rate |
| VAD precision | Fraction of hypothesized speech that is reference speech |
| VAD recall | Fraction of reference speech detected |
| VAD F1 | Harmonic mean of VAD precision and recall |

Macro-averaged metrics treat recordings equally, while pooled metrics aggregate across the underlying duration or evaluation units. These must not be interchanged.

### 2.3 Statistical interpretation

The evaluation contains only eight recording-level clusters. Consequently:

- Per-recording improvements are useful diagnostic evidence.
- Small macro-average changes should not be interpreted as established general improvements.
- Bootstrap intervals and paired tests have limited power with this number of recording clusters.
- A result that is numerically better is not automatically statistically established.

---

## 3. Stage 5B.0 — Reconnaissance

### 3.1 Objective

Stage 5B.0 established the experimental baseline and audited the existing dataset and pipeline before optimization.

The investigation covered:

- Repository and pipeline dependencies.
- Dataset and annotation consistency.
- Existing VAD outputs.
- Overlap distribution.
- Speaker-count behavior.
- Experimental design for subsequent stages.

### 3.2 Annotation and overlap observations

The audit identified 1,362 annotation segments, of which 432 had corresponding audio.

Overlap was particularly concentrated in R02 and R04:

| Recording | Overlap duration | Share of recording overlap measure |
|---|---:|---:|
| R02 | 56 seconds | 34.57% |
| R04 | 95 seconds | 51.08% |

Together, R02 and R04 accounted for 96.79% of the reported overlap duration.

This motivated a separate overlap investigation rather than treating overlap as a minor uniform condition across all recordings.

### 3.3 Important correction

An earlier Stage 5B.0 oracle speaker-count claim for R07 suggested a large improvement from approximately 64.95% DER to 21.80%.

That claim was subsequently disproved during Stage 5B.3. It must not be used as evidence of an oracle speaker-count improvement.

The corrected Stage 5B.3 results show that forcing the true speaker count in R07 actually increased DER relative to the baseline configuration.

### 3.4 Stage 5B.0 conclusion

Stage 5B.0 was a reconnaissance stage, not a model-improvement experiment.

Its principal contribution was identifying:

- Strong recording-level variation.
- Concentrated overlap in R02 and R04.
- Speaker-count estimation as a potential failure point.
- The need to isolate VAD, overlap, and clustering investigations.

---

## 4. Stage 5B.1 — VAD Optimization

### 4.1 Objective

Stage 5B.1 investigated whether VAD configuration changes could improve speech-region detection and downstream evaluation.

The baseline configuration was:

- Configuration: C00
- Threshold: 0.481
- Minimum speech duration (`min_on`): 0.10 seconds
- Minimum silence duration (`min_off`): 0.15 seconds

Several configurations were compared, including threshold changes, hysteresis, minimum-duration constraints, gap handling, raw mono VAD, and consensus-based processing.

### 4.2 Original baseline and candidate results

The corrected baseline metrics were:

| Metric | C00 |
|---|---:|
| Macro DER, c=0 | 46.17% |
| Macro DER, c=0.25 | 39.65% |
| Miss | 24.42% |
| False Alarm | 13.33% |
| Confusion | 8.42% |
| JER | 62.20% |
| Frame F1 | 84.94% |

The two main candidates were C50 and C62.

| Metric | C00 | C50 | C62 |
|---|---:|---:|---:|
| VAD precision | — | 86.76% | 86.81% |
| VAD recall | — | 90.85% | 88.17% |
| VAD F1 | — | 88.34% | 87.03% |
| Projected DER, c=0 | 46.17% | 43.84% | 44.65% |
| Projected DER, c=0.25 | — | 36.99% | 38.03% |
| Miss | 16.39%* | 9.15% | 11.83% |
| False Alarm | 13.68%* | 15.94% | 15.33% |
| Confusion | — | 10.58% | 9.62% |
| JER | — | 61.11% | 61.18% |
| Macro cpWER | 37.12% | 37.59% | 37.12% |

\*Corrected union-based baseline VAD measurement.

The corrected miss and false-alarm measurements use the union of reference speech intervals for binary VAD evaluation.

### 4.3 Important methodological correction: overlap-aware VAD evaluation

The original VAD evaluation summed reference RTTM segment durations. This double-counted speech duration in overlapping regions.

The corrected evaluation uses the union of reference speech intervals.

The difference was material:

| Scope | Summed duration | Union duration | Inflation |
|---|---:|---:|---:|
| R02 | 284 s | 162 s | 75.3% |
| R04 | 284 s | 186 s | 52.7% |
| Corpus | 1,625 s | 1,400 s | 16.07% |

For binary VAD evaluation, the union is the appropriate reference speech denominator.

The corrected baseline miss rate is 16.39%, compared with the legacy value of 15.22%.

### 4.4 Critical interpretation: projected DER

The original Stage 5B.1 downstream evaluation did not rerun full diarization.

Instead, the evaluation:

1. Loaded frozen Condition J RTTM speaker assignments.
2. Projected the existing speaker labels onto 10 ms frames.
3. Filled unlabeled frames using the nearest known speaker label.
4. Applied the candidate VAD mask.
5. Generated an RTTM and evaluated it.

Therefore, the resulting metric must be called:

**VAD-mask-projected DER under frozen baseline speaker assignments.**

It is not equivalent to DER from a fresh diarization run.

Newly detected speech can inherit a nearby baseline speaker label. This propagation can influence the apparent benefit of the VAD configuration.

For C50, the amount of hypothesis speech receiving nearest-neighbor label propagation was substantial in some recordings:

| Recording | Propagated duration | Share of hypothesis speech |
|---|---:|---:|
| R01 | 16.87 s | 20.07% |
| R02 | 28.51 s | 17.87% |
| R05 | 24.71 s | 19.95% |

This is an important limitation when interpreting the projected DER improvement.

### 4.5 cpWER correction

The original cpWER evaluation had a serialization mismatch.

The metric implementation expected a dictionary containing a `segments` field, but the original serialized payload was a list.

The resulting exception was swallowed, leaving all 208 cpWER entries as NaN.

The corrective audit fixed the input structure and removed the unsafe fallback.

Corrected results:

- C00 macro cpWER: 37.12%.
- C50 macro cpWER: 37.59%.
- C62 macro cpWER: 37.12%.

These are unweighted macro averages across eight recordings.

### 4.6 Statistical results

C50 was numerically the strongest candidate on VAD F1, recall, and projected DER.

However, the paired statistical tests did not establish a significant improvement over C00:

- C50 vs C00: Wilcoxon p = 0.25.
- Paired t-test p = 0.1933.
- Bootstrap interval for the DER difference: approximately [-5.61, +0.15] percentage points.

C62 also did not establish a significant improvement.

C50 increases false alarms while reducing misses. C62 offers a more balanced alternative, with lower recall than C50 but a slightly lower false-alarm rate and better projected cpWER.

### 4.7 Stage 5B.1 conclusion

The VAD investigation identifies three configurations worth carrying forward:

- **C00:** Original baseline.
- **C50:** High-recall raw-mono candidate.
- **C62:** Consensus-based, more balanced candidate.

No final winner is established.

Most importantly, Stage 5B.1 demonstrates an improvement in VAD detection and projected metrics, not yet an established improvement in fresh end-to-end diarization.

---

## 5. Stage 5B.2 — Multi-Speaker Overlap Optimization and Attribution

### 5.1 Objective

Stage 5B.2 investigated whether changes to overlap handling and speaker attribution could improve diarization.

The investigation progressed through V3, V4, V4.1, V4.2, and V4.2.1.

The final accepted documented version is V4.2.1.

### 5.2 Methodological corrections

The development included several important corrections:

1. Corrected oracle multi-label speaker mapping.
2. Corrected Hungarian mapping tuple handling.
3. Corrected timestamp-to-frame-bin handling.
4. Added deterministic primary-speaker sorting.
5. Corrected interval-union handling for per-speaker oracle intervals.
6. Added prototype-purity tests for the M2 attribution prototype.
7. Added dispatcher permutation tests.

The V4 test suite included 21/21 passing dispatcher permutation tests.

A separate set of seven synthetic tests covered prototype-purity behavior, including clean centroids, competitor exclusion, low-purity rejection, empty caches, permutation invariance, embedding dimensions, and finite norms.

### 5.3 Same-speaker interval audit

The V4 audit found:

- Seven of eight RTTMs had no same-speaker overlaps or duplicates.
- R02 contained three same-speaker overlaps and one duplicate.
- Naively summing the R02 intervals inflated duration by 48 seconds.

The interval-union correction did not change downstream attribution rankings in the evaluated set.

### 5.4 Corrected authoritative results

The authoritative comparison is O02 versus O42-R2.

| Metric | O02 | O42-R2 | Difference |
|---|---:|---:|---:|
| Macro DER, c=0 | 44.90% | 45.42% | +0.52 pp |
| Macro DER, c=0.25 | 42.62% | 42.97% | +0.34 pp |
| Pooled DER, c=0 | 48.68% | 49.28% | +0.60 pp |
| Macro JER | 61.27% | 60.73% | -0.54 pp |

Positive DER differences indicate worse performance for O42-R2.

### 5.5 Per-recording analysis

| Recording | O02 DER | O42-R2 DER | Difference |
|---|---:|---:|---:|
| R01 | 31.19% | 31.19% | Tie |
| R02 | 64.67% | 64.67% | Tie |
| R03 | 34.29% | 36.92% | +2.64 pp |
| R04 | 39.35% | 31.26% | -8.09 pp |
| R05 | 60.38% | 63.20% | +2.82 pp |
| R06 | 71.07% | 77.87% | +6.81 pp |
| R07 | 41.98% | 41.98% | Tie |
| R08 | 16.28% | 16.28% | Tie |

The configuration improved R04 substantially but degraded R03, R05, and R06.

There were:

- 1 recording-level win.
- 3 recording-level losses.
- 4 ties.

### 5.6 Statistical analysis

The bootstrap analysis used:

- 5,000 replicates.
- Random seed 42.
- Eight recording-level clusters.

The macro DER difference was approximately +0.5221 percentage points.

The bootstrap confidence interval was:

**[-2.3501, +2.9050] percentage points.**

The paired Wilcoxon result was:

- W = 4
- p = 0.8750
- Effective non-tied sample size = 4

The corpus-level difference is not statistically significant.

### 5.7 Attribution interpretation

The M4 conservative contiguous-overlap attribution configuration helps R04 but harms R03, R05, and R06.

This suggests that overlap attribution may benefit from selective application rather than uniform application.

However, selective overlap attribution has not yet been established as a validated improvement.

The final V4.2.1 report corrects earlier documentation errors and preserves the distinction between observed metrics and statistical conclusions.

### 5.8 Scope and provenance

The V4.1 phase included 48 downstream assignment/scoring evaluations despite the conditional authorization not being triggered.

This was documented as a protocol-scope deviation.

No model training, neural inference, embedding extraction, VAD inference, or ASR inference occurred during V4.2 or V4.2.1.

The V4.2.1 documentation correction did not alter the authoritative metric CSV.

The subsequent Git publication was:

- Commit: `75a00bb356d062b35eb5ad61df20312325ac87b2`
- Commit message: `V4.2.1 Documentation Erratum & Manifest Reconciliation Report`
- Branch: `main`

### 5.9 Stage 5B.2 conclusion

Stage 5B.2 is accepted as complete and frozen at V4.2.1.

The investigation does not establish a corpus-level DER improvement for O42-R2 over O02.

The strongest result is recording-specific: R04 improved substantially, while R03, R05, and R06 regressed.

The investigation should be closed without claiming general overlap-handling superiority.

---

## 6. Stage 5B.3 — Speaker-Count Estimation

### 6.1 Objective

Stage 5B.3 investigated whether alternative speaker-count and clustering strategies could address failures such as the R07 single-speaker collapse.

The stage was conducted in an isolated workspace without retraining, embedding extraction, VAD inference, or ASR inference.

### 6.2 R07 representation diagnostics

NME-SC selected k=1 across the evaluated parameter range p=[1,108], despite the reference containing multiple speakers.

The representation diagnostics were:

| Quantity | Value |
|---|---:|
| S1 intra-speaker affinity | 0.5377 ± 0.19 |
| S2 intra-speaker affinity | 0.3422 ± 0.18 |
| S1–S2 inter-speaker affinity | 0.3292 ± 0.18 |
| S1/S2 inter-speaker margin | 0.0130 |
| Margin standard deviation | 0.1812 |

S3 had only two subsegments.

These results are consistent with weak separation between some speaker representations. They do not establish a definitive causal explanation for the R07 collapse.

### 6.3 R07 oracle speaker-count experiment

| Configuration | DER | Confusion | JER |
|---|---:|---:|---:|
| Baseline k=1 | 45.16% | 9.54% | 79.23% |
| Forced k=2 | 47.27% | 11.65% | 72.51% |
| True oracle k=3 | 71.72% | 36.10% | 82.54% |

The true oracle speaker count produced substantially worse DER than the baseline.

This is a key negative result.

Knowing the correct number of speakers does not guarantee correct speaker attribution or clustering.

### 6.4 R06 diagnostics

R06 had a reference speaker count of k=9.

The sweep over k=1 through k=10 produced:

- Minimum DER at k=6: 69.80%.
- Oracle k=9 DER: 76.46%.

Several fleeting speakers had only approximately 4–6 seconds of activity.

This indicates that short speaker duration can make speaker-count and clustering decisions difficult.

### 6.5 Methods evaluated

The investigation included:

- **M1:** Baseline NME-SC automatic speaker count.
- **M2:** Oracle speaker count.
- **M3:** Conversational floor enforcing k≥2.
- **M4:** Relative eigengap-based selection.
- **M5:** Silhouette-guided spectral clustering.
- **M6:** AHC with oracle speaker count.

### 6.6 Aggregate results

| Method | Macro DER c=0 | Macro DER c=0.25 | Confusion | JER | Count MAE | Collapse |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 46.1698% | 39.6444% | 8.4267% | 62.1951% | 1.25 | 1 |
| M2 | 55.2756% | 48.8491% | 17.5325% | — | 0 | 0 |
| M3 | 46.4331% | 39.9045% | 8.6901% | 61.3554% | 1.125 | 0 |
| M4 | 49.3577% | 42.2343% | 11.6147% | — | 1.875 | — |
| M5 | 46.0604% | 39.5285% | 8.3174% | 61.8440% | 1.375 | 0 |
| M6 | 47.8676% | 43.1467% | 10.1245% | — | 0 | — |

A dash indicates a value not included in the reported summary.

### 6.7 Interpretation

M5 produced a macro DER reduction of only 0.1094 percentage points relative to M1.

This difference was not statistically established.

M5 also had worse speaker-count MAE:

- M1: 1.25.
- M5: 1.375.

The R07 behavior illustrates the trade-off:

- M5 selected k=2.
- R07 DER increased from 45.1602% to 47.4252%.
- JER improved.

For R06:

- M5 selected k=6.
- DER improved from 72.5755% to 69.5728%.

Thus, the alternative method helps some recordings and harms others.

The absence of a single-speaker collapse does not mean speaker-count accuracy improved.

### 6.8 Implementation limitations

The audit identified several limitations in M5:

1. The candidate k range was hard-coded to 2–6.
2. It could not select more than six speakers.
3. R06 selected the upper candidate boundary.
4. Silhouette exceptions were swallowed and defaulted to k=2.
5. The silhouette calculation used eigenvector clustering but an affinity-derived distance.
6. The heuristic requires further validation.
7. Only three random trials were used.
8. The random seed was reset to 42.
9. Some raw diagnostic values were outside the nominal cosine range, likely because they represented aggregate multi-scale quantities.

Therefore, M5 should be treated as an exploratory heuristic, not a globally optimal speaker-count estimator.

### 6.9 Stage 5B.3 conclusion

Stage 5B.3 did not establish a robust speaker-count improvement.

It established that:

- R07 has weak representation separation under the examined diagnostics.
- Correct oracle speaker count can still worsen DER.
- R06 is difficult because of fleeting speakers and clustering behavior.
- M5 gives a very small aggregate DER improvement, without established statistical significance.
- Speaker-count accuracy and diarization quality are not interchangeable.

---

## 7. Cross-Stage Findings

### 7.1 VAD errors remain important

Under the fixed VAD baseline, miss and false-alarm errors account for a large portion of the measured DER.

The Stage 5B.3 summary attributed approximately 81.9% of macro DER to miss and false-alarm components.

This should not be interpreted as universally irreducible error. It indicates that speech-region detection is an important source of error in the evaluated pipeline.

### 7.2 Overlap is concentrated

R02 and R04 account for most of the audited overlap duration.

Uniform overlap-handling changes may not be appropriate because the effects vary considerably by recording.

### 7.3 Correct speaker count is not sufficient

The R07 oracle experiment is particularly important.

The true speaker count increased DER to 71.72%, compared with 45.16% for the baseline.

The challenge is not simply determining k. It also involves assigning frames and segments to the correct speakers.

### 7.4 Representation quality and clustering interact

The R07 affinity diagnostics suggest weak separation between at least some speaker representations.

This may contribute to clustering instability, but it is not proven to be the only cause.

### 7.5 Projected improvements must be distinguished from fresh diarization

Stage 5B.1 measured VAD-mask projection under frozen baseline speaker assignments.

Stage 5B.2 compared overlap-attribution configurations.

Stage 5B.3 compared speaker-count and clustering methods.

None of these results alone establishes the end-to-end effect of combining improved VAD and overlap handling with fresh diarization.

---

## 8. Evidence Classification

### 8.1 Supported findings

The following findings are supported by the completed investigations:

- R02 and R04 contain concentrated overlap.
- Union-based VAD evaluation materially changes the speech-duration denominator.
- C50 improves VAD recall and F1 relative to C00 in the evaluated recordings.
- C50 produces lower VAD-mask-projected DER than C00.
- C50 increases false alarms and has slightly worse macro cpWER.
- O42-R2 substantially improves R04 but regresses R03, R05, and R06.
- O42-R2 does not establish a statistically significant corpus-level DER improvement over O02.
- R07's true oracle speaker count does not improve DER.
- M5's aggregate DER improvement over M1 is small and not statistically established.
- Speaker-count accuracy and diarization accuracy are distinct evaluation outcomes.

### 8.2 Descriptive findings

These observations are useful but should not be presented as established general laws:

- R07's weak affinity separation may contribute to its speaker-count collapse.
- R06's fleeting speakers may contribute to clustering difficulty.
- Selective overlap attribution may be more suitable than uniform overlap attribution.
- C62 may provide a useful balance between recall and false alarms.

### 8.3 Negative findings

The following approaches did not establish the expected improvement:

- The earlier Stage 5B.0 R07 oracle-count improvement claim was disproved.
- Forced oracle speaker count did not improve R07 DER.
- M5 did not establish a meaningful aggregate improvement over M1.
- O42-R2 did not establish a corpus-level DER improvement over O02.
- C50's projected DER improvement does not establish a fresh diarization improvement.
- No tested gate or attribution heuristic should be described as a proven general solution.

### 8.4 Unresolved questions

The following questions remain open:

1. Does C50 improve fresh diarization, or does its apparent advantage depend on frozen speaker-label propagation?
2. Does C62 provide a better balance when speaker embeddings and clusters are recomputed?
3. Does improved VAD interact positively or negatively with overlap attribution?
4. Can O42-R2's R04 improvement be retained without degrading R03, R05, and R06?
5. Can selective overlap handling be designed using reliable overlap-specific evidence?
6. Can speaker-count estimation be improved without introducing false splits?
7. Do changes improve conversational attribution and cpWER, rather than only DER?

---

## 9. Stage 5B.4 — Proposed Next Investigation

The proposed research question is:

> Can improved VAD and overlap handling produce better speaker attribution when the system performs fresh diarization, rather than simply projecting new masks onto frozen baseline speaker assignments?

The proposed comparison is:

**VAD configurations:**
- C00
- C50
- C62

**Overlap configurations:**
- O02
- O42-R2

This creates six experimental conditions.

The key methodological requirement is that each condition must perform fresh downstream diarization using the corresponding VAD output. It must not reuse frozen baseline speaker assignments.

The detailed protocol is maintained separately in:

`STAGE5B_4_PROPOSED_EXPERIMENT_PROTOCOL.md`

Stage 5B.4 has not been executed or authorized for execution.

---

## 10. Reproducibility and Preservation

Stage 5B was conducted as a separate research track from the frozen Stage 5A.1.2 work.

The Stage 5A.1.2 freeze commit remains:

`ddf089de986f9e7cf026aa507324f3fccaf0cd8f`

The Stage 5B.2 V4.2.1 publication commit is:

`75a00bb356d062b35eb5ad61df20312325ac87b2`

The Stage 5B.2 investigation preserves the V3, V4, V4.1, V4.2, and V4.2.1 history.

The Stage 5B.1 corrective audit preserved the frozen Stage 1–5A.1.2 artifacts and the pre-existing Stage 5B.1 artifacts.

The Stage 5B.3 investigation was conducted in its own workspace.

No result in this report should be interpreted as permission to modify frozen historical artifacts.

---

## 11. Final Conclusion

Stage 5B has identified several useful directions for improving the multi-view speaker diarization pipeline, but it has not yet established a robust end-to-end improvement.

The most promising observations are:

- VAD configuration changes can reduce missed speech.
- Overlap attribution can substantially improve selected recordings.
- Speaker-count estimation remains a significant source of instability.
- Representation separation, VAD quality, overlap, and clustering interact.

However, the investigations also reveal important limitations:

- VAD projected DER is not fresh diarization DER.
- Improvements in individual recordings do not imply corpus-level improvement.
- Oracle speaker count does not guarantee better diarization.
- Small aggregate improvements must not be overstated.
- Several proposed heuristics remain exploratory.

The next scientifically meaningful step is therefore not another isolated metric optimization. It is a controlled experiment that combines the candidate VAD and overlap configurations while rerunning the downstream diarization process from fresh inputs.

**Stage 5B.2 is frozen at V4.2.1. Stage 5B.4 remains a proposed experiment pending protocol approval and explicit execution authorization.**
