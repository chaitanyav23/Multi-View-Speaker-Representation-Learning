# Stage 5B.4 — Proposed Experiment Protocol

**Project:** Multi-View Speaker Representation Learning  
**Stage:** 5B.4 — Fresh Diarization with Improved VAD and Overlap Handling  
**Status:** PROPOSED — PENDING PROTOCOL APPROVAL  
**Execution authorization:** Not yet granted.

---

## 1. Research Motivation

Stages 5B.1 and 5B.2 investigated VAD optimization and overlap attribution separately.

However, an important methodological limitation remains.

Stage 5B.1 evaluated candidate VAD configurations by projecting new VAD masks onto frozen baseline speaker assignments. This does not measure the behavior of a complete diarization pipeline when new speech regions are detected.

Stage 5B.2 evaluated overlap attribution changes, but did not establish a statistically significant corpus-level DER improvement.

Consequently, the combined end-to-end effect remains unknown.

Stage 5B.4 is designed to address this gap.

## 2. Primary Research Question

> Can improved VAD and overlap handling produce better speaker attribution when the system performs fresh diarization, rather than simply projecting new masks onto frozen baseline speaker assignments?

The experiment should determine whether VAD and overlap-handling improvements remain beneficial after fresh downstream diarization, speaker assignment, and clustering.

## 3. Hypotheses

### H1 — VAD effect

Changing the VAD configuration from C00 to C50 or C62 will change the speech regions supplied to the diarization pipeline.

The experiment will test whether those changes improve fresh diarization DER and speaker-attributed transcription metrics.

### H2 — Overlap-handling effect

Changing overlap handling from O02 to O42-R2 may improve overlap attribution.

The experiment will test whether this improvement generalizes beyond R04.

### H3 — Interaction effect

The effect of overlap handling may depend on the VAD configuration.

For example, an overlap-attribution configuration might behave differently when the VAD detects more short speech regions or additional overlapping speech.

The experiment will test this interaction rather than assuming that the independent gains of the two components are additive.

### H4 — End-to-end improvement

At least one combined configuration may improve fresh diarization relative to the C00 × O02 baseline.

This is the main hypothesis. It will not be considered supported solely because a projected DER improves.

---

## 4. Experimental Design

A factorial comparison will be performed using three VAD configurations and two overlap configurations.

### 4.1 VAD configurations

| ID | Configuration | Description |
|---|---|---|
| C00 | Baseline | Original VAD configuration |
| C50 | High-recall candidate | Raw-mono VAD candidate |
| C62 | Consensus candidate | Consensus-based configuration with hysteresis and gap handling |

The exact frozen configuration parameters and source artifacts must be loaded from the completed Stage 5B.1 workspace.

They must not be reconstructed from memory or silently modified.

### 4.2 Overlap configurations

| ID | Configuration | Description |
|---|---|---|
| O02 | Baseline overlap handling | Baseline overlap configuration |
| O42-R2 | Conservative overlap attribution | Final evaluated V4.2 configuration using the documented M4 attribution setup |

The exact implementation and parameters must be loaded from the frozen Stage 5B.2 V4.2.1 record.

No additional tuning is permitted within this experiment unless explicitly authorized through a separate protocol amendment.

### 4.3 Six-condition experiment matrix

| Condition | VAD | Overlap handling | Role |
|---|---|---|---|
| E00 | C00 | O02 | Primary baseline |
| E01 | C00 | O42-R2 | Isolated overlap effect |
| E10 | C50 | O02 | Isolated high-recall VAD effect |
| E11 | C50 | O42-R2 | Combined high-recall condition |
| E20 | C62 | O02 | Isolated consensus VAD effect |
| E21 | C62 | O42-R2 | Combined consensus condition |

All six conditions must be evaluated on the same eight recordings.

---

## 5. Critical Methodological Requirement: Fresh Diarization

This experiment must not repeat the frozen-label projection method used in the original Stage 5B.1 evaluation.

For every condition:

1. Load the original recording audio.
2. Apply the specified VAD configuration.
3. Generate the speech regions from that configuration.
4. Perform fresh downstream diarization using those regions.
5. Generate fresh speaker assignments and clusters.
6. Apply the specified overlap-handling configuration.
7. Produce the final RTTM.
8. Evaluate against the same reference annotations.
9. Generate the corresponding speaker-attributed transcription evaluation where supported.

The diarization stage must not reuse frozen baseline speaker labels to assign newly detected speech.

The baseline condition E00 must also be rerun through the same execution path. This controls for differences introduced by the new orchestration or evaluation code.

### 5.1 Embedding policy

Use the existing frozen encoder setup and the same embedding extraction configuration across all six conditions.

Do not introduce encoder fine-tuning, additional training, or new representation-learning objectives in Stage 5B.4.

If fresh embeddings are required because VAD changes the speech regions, extract them consistently for every relevant condition.

Any caching must be keyed by recording, segment boundaries, encoder, preprocessing configuration, and other inputs that affect the embedding.

A cache hit must not silently substitute embeddings generated for different speech regions.

---

## 6. Controlled Variables

The following should remain fixed:

- Eight-recording dataset.
- Audio preprocessing and normalization.
- Encoder checkpoints.
- Embedding extraction settings.
- Embedding dimensions.
- Clustering implementation, except for the explicitly specified overlap configuration.
- Speaker-label mapping procedure.
- Evaluation reference annotations.
- DER collars.
- Scoring implementation.
- Random seeds wherever stochastic operations are present.

The VAD and overlap configurations are the intended independent variables.

Any additional changes must be documented and approved before execution.

---

## 7. Evaluation Metrics

### 7.1 Primary metrics

**Primary endpoint 1: Macro DER, c=0.**

This measures diarization quality while treating all eight recordings equally.

**Primary endpoint 2: Macro cpWER.**

This measures speaker-attributed transcription quality across the recordings.

The corrected cpWER input schema from the Stage 5B.1 audit must be retained. Any exception must fail visibly rather than being converted to zero or silently omitted.

### 7.2 Secondary metrics

| Metric | Purpose |
|---|---|
| DER, c=0.25 | Collar-tolerant diarization comparison |
| Pooled DER | Duration-weighted corpus behavior |
| JER | Speaker-region overlap quality |
| Miss | Missed reference speech |
| False Alarm | Hypothesized non-reference speech |
| Confusion | Incorrect speaker attribution |
| Speaker-count MAE | Speaker-count estimation quality |
| VAD precision | Speech detection precision |
| VAD recall | Speech detection recall |
| VAD F1 | Overall VAD balance |
| cpWER per recording | Recording-level attribution behavior |

All macro and pooled metrics must be explicitly labeled.

### 7.3 Overlap-specific reporting

Because overlap is concentrated in R02 and R04, report:

- Overlap-region DER or equivalent supported overlap-specific scoring.
- Speaker attribution behavior in overlap regions.
- Per-recording results for R02 and R04.
- Any measurable degradation in non-overlap regions.

Do not infer overlap-specific improvement from overall DER alone.

---

## 8. Statistical Analysis Plan

The independent evaluation units are the eight recordings.

### 8.1 Paired comparisons

Compare each candidate against E00 using paired per-recording differences.

At minimum, report:

- Macro difference.
- Per-recording differences.
- Number of wins, losses, and ties.
- Paired Wilcoxon signed-rank test.
- Bootstrap confidence interval.

### 8.2 Bootstrap

Use the established project convention:

- 5,000 bootstrap replicates.
- Random seed 42.
- Resample recordings as clusters.
- Preserve pairing between compared configurations.

The bootstrap must operate on recording-level paired outcomes rather than treating individual frames or segments as independent recordings.

### 8.3 Multiple comparisons

There are five candidate conditions compared with E00.

The protocol should use a multiple-comparison correction for the planned primary comparisons.

A suitable choice is Holm correction across the five comparisons for each primary endpoint.

Report:

- Unadjusted p-values.
- Adjusted p-values.
- Confidence intervals.
- Effect sizes in percentage points.
- Per-recording outcomes.

With only eight recordings, statistical power remains limited. Non-significant results must not be interpreted as proof of equivalence.

### 8.4 Interaction analysis

Report the VAD × overlap-handling interaction descriptively.

For each VAD configuration, compare O42-R2 against O02.

Then compare the overlap effect across C00, C50, and C62.

Do not claim a reliable interaction based on a numerical pattern alone.

---

## 9. Execution Phases and Stop Conditions

### Phase A — Pre-execution audit

Before any GPU or inference execution:

1. Verify the Stage 5B.1 C00, C50, and C62 configurations.
2. Verify the Stage 5B.2 O02 and O42-R2 implementations.
3. Confirm the correct encoder checkpoints.
4. Confirm the fresh diarization path.
5. Confirm the evaluation code and cpWER schema.
6. Verify that no frozen artifacts will be modified.
7. Produce an execution manifest and hash record.

**Stop condition:** If any configuration cannot be reproduced from its frozen source, stop and request clarification.

### Phase B — Dry run

Run a minimal, non-destructive pipeline check.

Verify:

- Input and output paths.
- Segment timestamps.
- VAD output alignment.
- Embedding cache behavior.
- Fresh clustering behavior.
- RTTM validity.
- Metric serialization.
- Failure handling.

Do not begin the full experiment if the dry run fails.

### Phase C — Full six-condition execution

Execute all six conditions over the eight recordings.

The run should be deterministic wherever possible.

Record:

- Start and end times.
- Hardware and software environment.
- GPU utilization and peak memory.
- Configuration hashes.
- Checkpoint hashes.
- Input and output hashes.
- Failed runs and retries.
- Per-recording outputs.

Before starting, estimate GPU time and peak VRAM. Warn the project owner if execution is expected to be long-running so they can decide whether to use the GPU or schedule the run differently.

### Phase D — Evaluation

Evaluate all completed conditions using the same reference annotations and scoring code.

Do not modify reference annotations during evaluation.

Generate per-recording and aggregate metric tables.

### Phase E — Independent audit

Verify:

- All six conditions are present.
- All eight recordings are present per condition.
- No stale baseline speaker assignments were reused.
- No missing cpWER values were silently replaced.
- All RTTMs are valid.
- Metric calculations are reproducible.
- Frozen Stage 5A and Stage 5B historical artifacts remain unchanged.

### Phase F — Report and freeze decision

Produce:

- Experiment manifest.
- Execution log.
- Configuration comparison table.
- Per-recording metrics.
- Aggregate metrics.
- Statistical analysis.
- Integrity report.
- Final interpretation.

The stage must remain open until the independent audit is complete.

---

## 10. Isolation and Preservation Rules

Stage 5B.4 must use a new isolated workspace:

`research_stages/stage5b/stage5b_4_fresh_diarization/`

It must not:

- Modify Stage 5A.1.2 frozen artifacts.
- Modify Stage 5B.0 reconnaissance outputs.
- Modify Stage 5B.1 VAD optimization outputs.
- Modify Stage 5B.2 V4.2.1 frozen outputs.
- Modify Stage 5B.3 speaker-count investigation outputs.
- Rewrite historical reports or metric tables.
- Overwrite previous experiment outputs.

All outputs must be new, versioned artifacts.

---

## 11. Expected Outcomes and Interpretation

The experiment may produce several possible outcomes.

### Outcome A — Both VAD and overlap handling help

If one or more combined configurations improve both primary metrics with consistent per-recording behavior, this would support combining the two improvements.

Statistical evidence and recording-level consistency must still be considered.

### Outcome B — VAD helps, overlap handling does not

This would suggest that speech-region detection is the more useful optimization in the evaluated pipeline.

### Outcome C — Overlap handling helps only selected recordings

This would motivate a future selective overlap-attribution approach, but would not establish a general corpus-level improvement.

### Outcome D — Projected improvements disappear under fresh diarization

This would demonstrate that the earlier VAD-mask-projected DER improvements did not translate into equivalent end-to-end diarization gains.

### Outcome E — No configuration improves the baseline

This would be a valid negative result. It would indicate that the tested changes do not provide a reliable end-to-end improvement under the current pipeline.

---

## 12. Deliverables

The following artifacts are expected:

1. `STAGE5B_4_PROTOCOL.md`
2. `STAGE5B_4_PRE_EXECUTION_AUDIT.md`
3. `STAGE5B_4_EXECUTION_MANIFEST.json`
4. `STAGE5B_4_EXECUTION_LOG.md`
5. `STAGE5B_4_PER_RECORDING_METRICS.csv`
6. `STAGE5B_4_AGGREGATE_METRICS.csv`
7. `STAGE5B_4_STATISTICAL_ANALYSIS.md`
8. `STAGE5B_4_INDEPENDENT_AUDIT.md`
9. `STAGE5B_4_FINAL_REPORT.md`
10. `STAGE5B_4_HASH_VERIFICATION_REPORT.md`

The exact output names may be extended if required, but the above deliverables must remain identifiable.

---

## 13. Approval Boundary

This document defines a proposed experimental protocol.

**It does not authorize execution.**

Before running Stage 5B.4, the project owner must approve:

- The six-condition matrix.
- The fresh diarization execution path.
- The embedding extraction and caching policy.
- The primary and secondary metrics.
- The statistical analysis plan.
- The execution and resource plan.

No GPU inference, embedding extraction, VAD inference, ASR inference, or full experiment execution should begin until the protocol has been reviewed and explicitly approved.

---

## 14. Final Research Question

Stage 5B.4 is intended to answer one central question:

**Do the improvements identified independently in VAD optimization and overlap attribution translate into better speaker attribution when the complete diarization pipeline is rerun from fresh speech regions and fresh speaker assignments?**

This experiment is the necessary next step before claiming that the Stage 5B improvements translate into better end-to-end conversational intelligence.
