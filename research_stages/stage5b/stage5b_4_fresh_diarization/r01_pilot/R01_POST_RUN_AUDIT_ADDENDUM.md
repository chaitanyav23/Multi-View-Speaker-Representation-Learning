# Stage 5B.4 R01 Controlled Pilot — Post-Run Audit Addendum

**Document ID**: `R01_POST_RUN_AUDIT_ADDENDUM.md`  
**Date**: November 13, 2026  
**Auditor**: Senior ML Systems Engineer, Security Auditor & Research Reproducibility Specialist  
**Repository**: `https://github.com/chaitanyav23/Multi-View-Speaker-Representation-Learning`  
**Current HEAD**: `50940f5`  
**Parent / Pre-Run Commit**: `a7777ff`  
**Working-Tree Status**: Post-run reconciliation complete in the working tree; pending project-owner review and commit.  
**Scope**: Post-run reconciliation, artifact verification, and evidence freeze for the executed R01 controlled pilot (Conditions E20 vs E21).

---

## 1. Purpose and Scope

This document provides a post-run reconciliation and documentation record of the already-executed Stage 5B.4 R01 controlled pilot.

In strict adherence to governance boundaries:
- **No experimental reruns**: No neural inference, VAD inference, embedding extraction, affinity fusion, clustering, overlap attribution, RTTM generation, or evaluation was re-executed.
- **No GPU execution or model loading**: No checkpoints were reloaded and no GPU resources were engaged.
- **Strict artifact preservation**: All experimental artifacts in `outputs/`, `cache/`, and `production_integration/r01_pilot_outputs/` remain unaltered. All frozen configurations (C62, M0, M4) and historical Stage 1–5B.3 artifacts remain preserved.
- **Strict scope limitation**: The audit and pilot scope is strictly limited to recording `R01_sara_sharif`; no expansion to recordings R02–R08 is performed.
- **Non-destructive version control**: Git history is preserved as-is without rebase, reset, or history rewriting.

The purpose of this addendum is to reconcile documentation discrepancies identified in post-run review, independently verify cache and RTTM cryptographic integrity, classify test coverage, and formalize the final evidence reconciliation.

---

## 2. Frozen Experiment Identity

The authorized pilot evaluated exactly one controlled comparison on recording `R01_sara_sharif`:
- **Condition E20**: Single-label baseline combining **C62** Linear Consensus VAD and Method **M0** (no overlap attribution).
- **Condition E21**: Proposed integrated system combining **C62** Linear Consensus VAD and Method **M4** (conservative overlap attribution using provenance-verified **DFN** overlap posterior).

Both conditions share identical upstream processing:
- Audio sources: DFN audio (`90520b89...`) and 16k mono audio (`96f9af8d...`).
- VAD Backbone: Condition C62 Linear Consensus ($0.5 \cdot P_{\text{DFN}} + 0.5 \cdot P_{\text{mono}}$).
- Multi-scale embeddings: 5 temporal scales (1.5s, 1.25s, 1.0s, 0.75s, 0.5s) extracted from TitaNet-Large (`msdd._speaker_model.`, 192d) and WeSpeaker ResNet293-LM ONNX (`voxceleb_resnet293_LM`, 256d).
- Affinity Fusion: Condition J (Multiscale Equal Weight, $0.5 \cdot A_{\text{TitaNet}} + 0.5 \cdot A_{\text{WeSpeaker}}$).
- Clustering: NeMo `SpeakerClustering` (CUDA GPU, Max RP: 0.25, Sparse Vol: 30, Trials: 1).
- Overlap Posterior: Provenance-verified R01 DFN overlap posterior (`R01_sara_sharif_dfn_overlap.npz`, SHA-256: `4ffda87c...`, 5,983 frames).

---

## 3. Corrected Configuration Record (C62 Reconciliation)

### 3.1 Discrepancy Description
In `R01_PILOT_FINAL_REPORT.md` (Table 2, row "VAD Parameters"), the parameters were reported as:
`0.600 / 0.350 / 0.050s / 0.100s` (onset / offset / min_duration_on / min_duration_off).
However, the frozen Stage 5B.1 specification defines C62 as:
`onset=0.600`, `offset=0.350`, `min_duration_on=0.10s`, `min_duration_off=0.30s`.

### 3.2 Authoritative Reconciliation
1. **Frozen Protocol Definition**:
   In `research_stages/stage5b/stage5b_1_vad_optimization/STAGE5B_1_EXPERIMENT_PROTOCOL.md` (line 65), `STAGE5B_1_FINAL_REPORT.md` (line 78), and `STAGE5B_1_VAD_METRICS_PER_RECORDING.csv` (line 194), Condition C62 is formally defined with:
   - `onset = 0.600`
   - `offset = 0.350`
   - `min_duration_on = 0.10s`
   - `min_duration_off = 0.30s`
2. **Registry Configuration**:
   In `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/scripts/condition_registry.py` (lines 70–82):
   ```python
   VAD_C62 = VADCondition(
       condition_id="C62",
       audio_source="consensus",
       onset=0.600,
       offset=0.350,
       min_duration_on=0.10,
       min_duration_off=0.30,
       consensus_method="linear",
       consensus_weights=(0.5, 0.5),
       description="Robust consensus: linear combination of DFN and Mono posteriors",
   )
   ```
3. **Runtime Execution Verification**:
   In `vad_adapter_remediated.py` (lines 152–159, 282–289), the orchestrator passes `cfg.min_duration_on` (`0.10`) and `cfg.min_duration_off` (`0.30`) to `binarize_posteriors_audited_reference()`.
4. **Interval Hash Inspection and Discrepancy Resolution**:
   - Direct inspection of the `metadata_json` field in all 10 existing `.npz` cache files in `cache/C62/R01_sara_sharif/` reveals:
     `"vad_intervals_sha256": "edf308bd1f91d7c795d030ae8db41d1a679d1ee68481a148e7d58cc0b51f9c8a"`
   - All 10 cache files agree 100% on this exact 64-character hash.
   - Re-evaluating `canonical_interval_hash('R01_sara_sharif', 'C62', intervals)` using the authoritative C62 parameters `(0.10s, 0.30s)` yields exactly:
     `edf308bd1f91d7c795d030ae8db41d1a679d1ee68481a148e7d58cc0b51f9c8a` (16 speech intervals).
   - Re-evaluating with the typo parameters `(0.05s, 0.10s)` yields 20 speech intervals with hash prefix `5c3c8c49cfc45b48...`.
   - **Resolution of Hash Discrepancy**: The earlier recorded cache digest `edf308bd1f91d7c795d030ae8db41d1a679d1ee68481a148e7d58cc0b51f9c8a` is the authentic, physically verified hash present in all 10 cache files. The alternate tail `...1a3e6a2b271d5335198ecbc018072efd72e7960fc977e2ff` that appeared in the draft addendum was an unsupported transcription error during drafting and is hereby corrected.

**Conclusion**: The pilot execution strictly executed the authoritative frozen C62 parameters `(0.10s, 0.30s)`. The table entry `0.050s / 0.100s` in `R01_PILOT_FINAL_REPORT.md` was a typographic transcription error in the presentation markdown.

---

## 4. Corrected RTTM Accounting (Segment Count and Duration Reconciliation)

### 4.1 Discrepancy Description
- `R01_PILOT_FINAL_REPORT.md` (Table 3) reported `Hypothesis Segment Count: 9`, whereas the published RTTM files contain 19 `SPEAKER` lines.
- The draft addendum reported 76.315 seconds of summed duration and a speaker breakdown of 6, 9, 4 segments. An earlier inspection reported 19 records and 77.398 seconds.

### 4.2 Authoritative Reconciliation
Inspection of all six existing RTTM copies (`outputs/`, `production_integration/r01_pilot_outputs/`, `r01_pilot/`) confirms:
1. **Raw Acoustic RTTM Record Count**:
   All 6 RTTM files contain **exactly 19 `SPEAKER` records** across **3 unique speakers**:
   - `speaker_0`: 9 segments, summed duration = **47.014 s**
   - `speaker_1`: 4 segments, summed duration = **12.138 s**
   - `speaker_2`: 6 segments, summed duration = **18.246 s**
2. **Duration Accounting**:
   - **Sum of All Segment Durations**: Exactly **77.398 s** ($47.014 + 12.138 + 18.246 = 77.398\text{ s}$).
   - **Union-of-Time Duration**: Exactly **76.648 s**.
   - *Explanation of Sum vs Union Difference*: The 0.750s difference ($77.398 - 76.648 = 0.750\text{ s}$) arises because adjacent multiscale subsegments within contiguous speech turns share slight boundary overlap (e.g. between 14.122s and 18.627s). The union-of-time duration (76.648s) exactly matches the 16 VAD speech intervals (76.646s rounded to milliseconds).
   - *Cause of the 76.315s Discrepancy*: The draft addendum's figure of 76.315s (and 6, 9, 4 segment counts) was an inaccurate drafting transcription error. The earlier inspection of **77.398 seconds** is the true sum of segment durations verified across all six files.
3. **Hypothesis Word-Turn Grouping Count**:
   In `research_stages/stage5b/stage5b_4_fresh_diarization/a1_remediation/scripts/cpwer_adapter.py` (lines 185–188):
   ```python
   hyp_grouped_segments = _group_words_by_speaker(hyp_aligned_words)
   ...
   hyp_segment_count = len(hyp_grouped_segments)
   ```
   `cpwer_adapter.py` calculates word-level diarization error by assigning WhisperX timestamped words to RTTM intervals and grouping consecutive words attributed to the same speaker into conversational turns. For R01, there are **9 consecutive hypothesis word-turn groupings**.
4. **Reference Accounting**:
   - Reference RTTM (`research_stages/stage1_data_prep/annotations_rttm/R01_sara_sharif.rttm`): Exactly **7 acoustic `SPEAKER` records** (total 93.000s speech across 5 ground-truth speakers).
   - Reference word-turn groupings in cpWER: **7 reference word turns**.

**Conclusion**: The value `9` was the **Hypothesis Word-Turn Grouping Count** from cpWER, not the raw acoustic RTTM record count. The authoritative acoustic RTTM record count is **19**, across **3 unique speakers**, with summed duration **77.398 s** and union duration **76.648 s**.

---

## 5. Test Coverage Classification

The test suite consists of distinct validation categories that must not be conflated:

| Category | Description | Scope / Mocking Level | Execution / Test Files | Reported Executions |
|:---|:---|:---|:---|:---:|
| **Category A: Synthetic Unit & Interface Tests** | Unit tests validating interface contracts, schema validation, error handling, symlink rejection, and pipeline orchestration using synthetic numpy arrays, in-memory mocks, and fixture objects. | Synthetic mocks; no real R01 audio processed; no real neural inference. | `test_production_integration.py` (12)<br>`a1_3_2_2_rttm_atomic_publication/tests/` (67)<br>`a1_1_3_2_1_final_security_closure/tests/` (51)<br>`a1_1_3_2_final_corrections/tests/` (50)<br>`a1_1_3_1_corrections/tests/` (64)<br>`a1_1_3_corrections/tests/` (70) | **314 Executions Reported**<br>(**220 Unique Test Cases**) |
| **Category B: Static & Configuration Checks** | Structural verification of manifests, file hashes, configuration dictionaries, and immutable Stage 1–5B.3 baseline files. | Non-inferential filesystem inspections and hash checks. | `verify_manifest.py`<br>`condition_registry.py`<br>`overlap_provenance.py` validation passes | **All Passed** (Static / Config) |
| **Category C: Actual R01 Execution Evidence** | Real end-to-end execution evidence on actual `R01_sara_sharif` audio: real C62 linear consensus VAD binarization, real dual-encoder embedding extraction on RTX 4090, real multiscale affinity fusion, real CUDA GPU clustering, atomic RTTM publication, and Hungarian evaluation. | Real acoustic input, real neural inference, real evaluation adapters. | `outputs/E20/R01_sara_sharif.rttm`<br>`outputs/E21/R01_sara_sharif.rttm`<br>`cache/C62/R01_sara_sharif/*.npz` (10 caches)<br>`R01_PILOT_EXECUTION_LOG.md` | **Completed & Verified** (Real Pilot Execution) |

### Test Count Clarification
- **Reported Executions Across Suites**: $12 + 67 + 51 + 50 + 64 + 70 = 314$ cumulative test executions.
- **Unique Test Cases**: Static AST analysis of all test files across the 6 remediation workspaces reveals **220 unique `(file, function)` test definitions** (218 distinct test function names).
- **Regression Overlap**: 44 test functions were shared regression tests carried forward across successive remediation workspaces to prevent regressions while closing specific findings.
- **Independence from Evidence**: All 314 test executions are Category A synthetic/mocked tests, strictly separate from the Category C empirical pilot execution on real R01 audio.

---

## 6. Cache Metadata and Checkpoint Identity Verification

All 10 multi-scale embedding cache files located in `cache/C62/R01_sara_sharif/` were inspected and cryptographically verified:

### 6.1 Common Cache Metadata
- **Schema Version**: `5b4_a1_v1`
- **Recording ID**: `R01_sara_sharif`
- **Sample Rate**: `16000`
- **VAD Config ID**: `C62`
- **VAD Intervals Hash**: `edf308bd1f91d7c795d030ae8db41d1a679d1ee68481a148e7d58cc0b51f9c8a` (16 consensus intervals, verified in all 10 caches)

### 6.2 Per-Encoder Cache Breakdown

| Scale | Window / Shift | TitaNet Segments | WeSpeaker Segments | TitaNet Dimension | WeSpeaker Dimension |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scale 0** | 1.50s / 0.750s | 128 | 128 | 192 | 256 |
| **Scale 1** | 1.25s / 0.625s | 134 | 134 | 192 | 256 |
| **Scale 2** | 1.00s / 0.500s | 140 | 140 | 192 | 256 |
| **Scale 3** | 0.75s / 0.375s | 146 | 146 | 192 | 256 |
| **Scale 4** | 0.50s / 0.250s | 300 | 300 | 192 | 256 |

### 6.3 Encoder Checkpoint Provenance
1. **TitaNet-Large**:
   - Model source in cache: `dfn` (Audio input: `R01_sara_sharif_dfn.wav`, SHA-256: `90520b89...`)
   - Checkpoint recorded in cache: `450f6bd279a3f78348f6e883359f9b3b34687dbd15814fd0f10e446ca0f708f6`
   - Disk file: `/home/priyanka/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo`
   - Physical disk SHA-256: `450f6bd279a3f78348f6e883359f9b3b34687dbd15814fd0f10e446ca0f708f6` (**Exact Match**)
   - Extraction method: Submodule `msdd._speaker_model.` extracted to `EncDecSpeakerLabelModel` (192-dimensional embeddings).
2. **WeSpeaker ResNet293-LM**:
   - Model source in cache: `16k_mono` (Audio input: `R01_sara_sharif_16k_mono.wav`, SHA-256: `96f9af8d...`)
   - Checkpoint recorded in cache: `dbb1ccc7754caff552ebc46347a51aaee2669bb24efc740e665d1a1133d20e98`
   - Disk file: `/home/priyanka/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-resnet293-LM/snapshots/6e6bffe5bf3d772a1f143dc6dbfea58a0799ea83/voxceleb_resnet293_LM.onnx`
   - Physical disk SHA-256: `dbb1ccc7754caff552ebc46347a51aaee2669bb24efc740e665d1a1133d20e98` (**Exact Match**)
   - Extraction method: ONNX Runtime InferenceSession via `WeSpeakerONNXWrapper` (256-dimensional embeddings).

---

## 7. RTTM Integrity Verification Across All Six Copies

All six published RTTM files generated during the pilot were verified:

| Copy Index | Condition | Filesystem Path | Byte Size | Raw SHA-256 Digest | Normalized (LF) SHA-256 | Bit-Exact Match |
|:---:|:---:|:---|:---:|:---|:---|:---:|
| 1 | E20 | `research_stages/stage5b/stage5b_4_fresh_diarization/outputs/E20/R01_sara_sharif.rttm` | 1,311 | `109f49a6bd3da3503721c7f6ddc9b1c757985f316be0b5d95cb4995b47b9712a` | `109f49a6...` | Identical |
| 2 | E21 | `research_stages/stage5b/stage5b_4_fresh_diarization/outputs/E21/R01_sara_sharif.rttm` | 1,311 | `109f49a6bd3da3503721c7f6ddc9b1c757985f316be0b5d95cb4995b47b9712a` | `109f49a6...` | Identical |
| 3 | E20 | `research_stages/stage5b/stage5b_4_fresh_diarization/production_integration/r01_pilot_outputs/E20/R01_sara_sharif.rttm` | 1,311 | `109f49a6bd3da3503721c7f6ddc9b1c757985f316be0b5d95cb4995b47b9712a` | `109f49a6...` | Identical |
| 4 | E21 | `research_stages/stage5b/stage5b_4_fresh_diarization/production_integration/r01_pilot_outputs/E21/R01_sara_sharif.rttm` | 1,311 | `109f49a6bd3da3503721c7f6ddc9b1c757985f316be0b5d95cb4995b47b9712a` | `109f49a6...` | Identical |
| 5 | E20 | `research_stages/stage5b/stage5b_4_fresh_diarization/r01_pilot/E20/R01_sara_sharif.rttm` | 1,311 | `109f49a6bd3da3503721c7f6ddc9b1c757985f316be0b5d95cb4995b47b9712a` | `109f49a6...` | Identical |
| 6 | E21 | `research_stages/stage5b/stage5b_4_fresh_diarization/r01_pilot/E21/R01_sara_sharif.rttm` | 1,311 | `109f49a6bd3da3503721c7f6ddc9b1c757985f316be0b5d95cb4995b47b9712a` | `109f49a6...` | Identical |

**Canonical Location**:
The canonical output directory for Stage 5B.4 is:
`research_stages/stage5b/stage5b_4_fresh_diarization/outputs/{condition_id}/{recording_id}.rttm`.
The secondary copies in `production_integration/r01_pilot_outputs/` and `r01_pilot/` are bit-for-bit identical verification mirrors.

---

## 8. Pilot Results and Scientific Interpretation

### 8.1 Reconciled Pilot Result Table

| Evaluation Field | E20 (Baseline) | E21 (Intervention) | Parity / Difference ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Condition ID** | **E20** | **E21** | — |
| **VAD Backbone** | Condition C62 Linear Consensus | Condition C62 Linear Consensus | Identical |
| **Overlap Method** | **M0** (No Overlap Attribution) | **M4** (Conservative Abstention) | Independent Variable |
| **Overlap Posterior Stream** | None (N/A) | **DFN** (`R01_sara_sharif_dfn_overlap.npz`) | Verified Eligible |
| **DER (Collar = 0.0s, Strict)** | **32.91%** | **32.91%** | **+0.00%** |
| — Missed Speech Rate | 18.13% | 18.13% | +0.00% |
| — False Alarm Rate | 1.35% | 1.35% | +0.00% |
| — Speaker Confusion Rate | 13.42% | 13.42% | +0.00% |
| **DER (Collar = 0.25s, NIST)** | **31.77%** | **31.77%** | **+0.00%** |
| **Jaccard Error Rate (JER)** | **62.18%** | **62.18%** | **+0.00%** |
| **Estimated Speakers ($\hat{K}$)** | **3** | **3** | 0 |
| **True Speakers ($K$)** | 5 | 5 | — |
| **Speaker Count Error ($\hat{K} - K$)** | $-2$ | $-2$ | 0 |
| **Hungarian-Matched cpWER** | **52.30%** | **52.30%** | **+0.00%** |
| — Reference / Hypothesis Words | 283 / 266 | 283 / 266 | Identical |
| — Insertions / Deletions / Substitutions | 44 / 61 / 43 | 44 / 61 / 43 | Identical |
| **RTTM Segment Count (Raw Acoustic Records)** | **19** | **19** | **Reconciled (Bit-Exact)** |
| **RTTM Summed Duration** | **77.398 s** | **77.398 s** | **Reconciled (Bit-Exact)** |
| **RTTM Union-of-Time Duration** | **76.648 s** | **76.648 s** | **Reconciled (Bit-Exact)** |
| **Word-Turn Grouping Count (cpWER Word Turns)**| **9** | **9** | **Reconciled (Bit-Exact)** |
| **Output RTTM SHA-256 Digest** | `109f49a6...` | `109f49a6...` | **Bit-Exact Match** |

> [!NOTE] DER Component Rounding Qualification
> When summed individually from the displayed rounded components, $18.13\% (\text{Miss}) + 1.35\% (\text{FA}) + 13.42\% (\text{Confusion}) = 32.90\%$. The displayed total DER is $32.91\%$, differing by 0.01 percentage points. This 0.01% difference is an expected floating-point rounding artifact: total DER is computed directly from unrounded underlying error durations ($29.610\text{s} / 89.965\text{s} \approx 32.9128\% \rightarrow 32.91\%$) before rounding, whereas individual components round separately ($16.313\text{s} / 89.965\text{s} \approx 18.1326\% \rightarrow 18.13\%$, $1.218\text{s} / 89.965\text{s} \approx 1.3539\% \rightarrow 1.35\%$, $12.079\text{s} / 89.965\text{s} \approx 13.4263\% \rightarrow 13.42\%$).

### 8.2 Hungarian Speaker Alignment Details
Both conditions produced an identical Hungarian optimal bipartite mapping:
- Ground Truth `Speaker_1` $\rightarrow$ Unattributed
- Ground Truth `Speaker_2` $\rightarrow$ `speaker_0`
- Ground Truth `Speaker_3` $\rightarrow$ Unattributed
- Ground Truth `Speaker_4` $\rightarrow$ `speaker_1`
- Ground Truth `Speaker_5` $\rightarrow$ `speaker_2`

### 8.3 Rigorous Scientific Interpretation
1. **Absence of Ground-Truth Overlap on R01**:
   In the authoritative Stage 1 ground-truth annotations (`research_stages/stage1_data_prep/annotations_json/R01_sara_sharif.json`), all 7 reference segments have `has_gt_overlap: false`. Ground-truth overlapping speech duration is exactly **0.00 seconds**.
2. **DFN Overlap Posterior Distribution**:
   Analysis of all 5,983 frames of the PyAnNet DFN overlap posterior (`R01_sara_sharif_dfn_overlap.npz`) shows:
   - Maximum overlap posterior: **0.3804**
   - Mean overlap posterior: **0.0241**
   - Frames exceeding threshold $\tau = 0.40$: **0 / 5,983 (0.00%)**
   - Frames exceeding threshold $\tau = 0.50$: **0 / 5,983 (0.00%)**
3. **Validation of Method M4 Conservative Abstention**:
   Because the posterior evidence never crossed the calibrated threshold $\tau = 0.40$, Method M4's conservative abstention rule correctly refused to declare secondary speakers. 
   - It produced **0 secondary-speaker frames** and **0 false alarms**.
   - It preserved the single-label diarization hypothesis bit-for-bit.
4. **Methodological Significance & Generalizability Warning**:
   - This result is a **no-effect result on R01 because Method M4 did not activate**.
   - It demonstrates that M4 is safe against false-alarm inflation on recordings lacking multi-speaker overlap.
   - It is **NOT evidence that overlap attribution is ineffective**, nor is it evidence that M4 improves diarization performance corpus-wide. Overlap attribution efficacy can only be measured on recordings containing true overlapping speech (e.g., R02–R08).

---

## 9. Git and Artifact State

- **Current Git Branch**: `main`
- **Current HEAD Commit**: `50940f55763679d0def6c9283dc4b6077cf94d18`  
  *Commit Subject*: `Stage 5B.4 — Bounded Production Integration and Authorized R01 Pilot`
- **Parent / Pre-Run Commit**: `a7777ff`
- **Working Tree State**: Modifications present in working tree (`R01_PILOT_EXECUTION_REPORT.md`, `R01_PILOT_EXECUTION_LOG.md`) and untracked file (`R01_POST_RUN_AUDIT_ADDENDUM.md`).
- **Freeze Status**: Post-run reconciliation complete in the working tree; pending project-owner review and commit.
- **Manifest Integrity**: All 14 protected baseline manifests and all 29 production integration deliverables pass verification with 100% digest matches.

---

## 10. Limitations and Audit Verdict

### Documented Limitations
1. **Single Recording Scope**: The pilot evaluated exclusively recording `R01_sara_sharif`. Findings cannot be extrapolated to the full corpus without authorized evaluation on R02–R08.
2. **Zero Ground-Truth Overlap on Test Sample**: Because R01 contains 0.0s overlap, the sensitivity and precision of Method M4 during true simultaneous speech remains unmeasured by this pilot.
3. **Underclustering**: Both conditions underclustered the true 5 speakers into 3 clusters ($\hat{K}=3$, $K=5$, error $-2$), leaving `Speaker_1` and `Speaker_3` unattributed.
4. **WhisperX Mono Dependency**: The cpWER metric relies on WhisperX 16k mono word timestamps aligned with diarization hypotheses.

### Audit Determination

```
================================================================================
Stage 5B.4 R01 pilot: EXECUTED — POST-RUN RECONCILIATION COMPLETE, WITH DOCUMENTED LIMITATIONS.
Post-run reconciliation complete in the working tree; pending project-owner review and commit.
================================================================================
```

---

## 11. Final Evidence Reconciliation

The following compact status table synthesizes the resolution and evidence status across all reviewed points:

| Issue | Outcome | Evidence Status |
|:---|:---|:---|
| **C62 configuration** | Resolved | Verified `(0.10s, 0.30s)` across Stage 5B.1 protocol, registry, and runtime binarization code. Table entry `0.050s / 0.100s` resolved as markdown presentation typo. |
| **C62 interval hash** | Resolved | Verified exact hash `edf308bd1f91d7c795d030ae8db41d1a679d1ee68481a148e7d58cc0b51f9c8a` stored in all 10 cache files (`metadata_json`) and reproduced by canonical hashing. Draft addendum tail resolved as drafting transcription error. |
| **RTTM record count** | Resolved | Verified exactly 19 `SPEAKER` records in all 6 RTTM files across 3 speakers (`speaker_0`: 9, `speaker_1`: 4, `speaker_2`: 6). The value 9 was cpWER word-turn groupings. |
| **RTTM duration** | Resolved | Verified summed segment duration of **77.398 s** and union duration of **76.648 s** across all 6 files. Draft addendum figure (76.315s) resolved as drafting transcription error. |
| **Test count** | Resolved | Clarified 314 cumulative test executions reported across 6 remediation suites, representing 220 unique `(file, function)` test cases (with 44 regression tests carried forward). Category A synthetic tests kept distinct from Category C execution evidence. |
| **Six RTTM hashes** | Resolved | Verified 100% bit-exact parity across all 6 RTTM files on disk (SHA-256: `109f49a6bd3da3503721c7f6ddc9b1c757985f316be0b5d95cb4995b47b9712a`, size: 1,311 bytes). |
| **Git freeze** | Pending review | Post-run reconciliation complete in the working tree; pending project-owner review and commit. |
