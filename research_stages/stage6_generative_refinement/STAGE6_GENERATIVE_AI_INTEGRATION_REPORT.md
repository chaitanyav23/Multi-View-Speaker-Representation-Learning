# Stage 6 — Generative AI Integration Report
## DiarizationLM-Inspired Context-Aware Speaker Attribution Refinement

---

## 1. Executive Summary & Scope Compliance

Stage 6 introduces an isolated Generative AI post-processing module (`research_stages/stage6_generative_refinement/`) inspired by **DiarizationLM: Speaker Diarization Post-Processing with Large Language Models** (Wang et al., 2024, `arXiv:2401.03506`), adapted for multi-speaker Body-Worn Camera (BWC) recordings.

### Strict Scope & Non-Interference Guarantees
- **Upstream Diarization Pipeline Frozen**: Stage 1–5B.4 artifacts, audio files, VAD intervals, WhisperX ASR transcripts, TitaNet/WeSpeaker embeddings, affinity fusion matrices, clustering outputs, and RTTM files are accessed in **strictly read-only mode**.
- **Transcript & Timestamp Immutability**: Following Transcript-Preserving Speaker Transfer (TPST) principles, the post-processor never modifies, inserts, deletes, normalizes, or reorders ASR words or timestamps. Only validated speaker-ID labels from the recording's existing diarization speaker inventory (`available_speaker_ids`) may be transferred onto the original word sequence.
- **Reversibility**: Every refined word record preserves both `original_speaker_id` and `refined_speaker_id`, allowing 100% bit-exact recovery of the baseline speaker-attributed transcript.
- **Provider-Independent & Credential-Safe**: No API keys are hard-coded, and no external network call is assumed at import or default runtime.

---

## 2. System Architecture

The module is organized into five deterministic stages orchestrated by `Stage6RefinementPipeline`:

```
Frozen ASR Words JSON ──┐
                        ├──► [Step A: Word Orchestrator] ──► OrchestratedWord Sequence
Frozen Diarization RTTM ┘            │
                                     ▼
                       [Step B: Context-Window Builder] ──► Overlapping ContextWindow List
                                     │
                                     ▼
                       [Step C: LLM Refinement Interface] ──► Raw Structured JSON per Window
                                     │
                                     ▼
                       [Step D: Constrained Parser & TPST Reconciler] ──► RefinedWordOutput + Audit Telemetry
                                     │
                                     ▼
Frozen Reference JSON ──► [Step E: Evaluation & Integrity Framework] ──► WDER / cpWER / SHA-256 Comparison
```

### Step A — Word-Level Orchestration (`word_orchestrator.py`)
- Loads word-level ASR outputs (`R01_sara_sharif_aligned.json`) and diarization RTTM segments (`R01_sara_sharif.rttm`).
- Attributes each word using exact Phase 20 `max_overlap_speaker` parity:
  1. **Direct Temporal Overlap (`DIRECT_OVERLAP`)**: Selects the RTTM speaker with maximum positive overlap $\max(0, \min(w_{\text{end}}, s_{\text{end}}) - \max(w_{\text{start}}, s_{\text{start}}))$.
  2. **Multi-Speaker Overlap (`MULTI_SPEAKER_OVERLAP`)**: Explicitly flags words where $\ge 2$ distinct RTTM speakers have positive temporal overlap, recording all `overlapping_speakers` and their overlap durations in `overlap_durations_by_speaker`.
  3. **Midpoint Tolerance Fallback (`MIDPOINT_TOLERANCE_FALLBACK`)**: When direct overlap is zero, falls back to the nearest RTTM segment boundary within `tolerance = 0.5s` of the word midpoint.
  4. **Unassigned (`UNASSIGNED_UNK`) / Missing Timestamps (`MISSING_TIMESTAMPS`)**: Preserves `"UNK"` explicitly without fabricating speaker assignments.

### Step B — Context-Window Builder (`context_window_builder.py`)
- Partitions long transcripts into bounded core windows (`window_size_words = 60` by default) with configurable left/right word overlap buffers (`context_overlap_words = 15`) and surrounding conversational turn summaries (`surrounding_turns_radius = 4`).
- Each word index in the transcript belongs to the `core_word_indices` of **exactly one** window, guaranteeing deterministic precedence when reconciling overlapping windows.

### Step C — Provider-Independent LLM Interface (`llm_interface.py`)
- Constructs `LLMRefinementRequest` payloads containing `available_speaker_ids`, `overlap_info`, `surrounding_conversational_turns`, and `window_words`.
- Provides five configurable `BaseLLMProvider` implementations:
  1. `OfflineAbstainingLLMProvider` (`offline_abstaining`): Safe default offline provider that abstains on all words, preserving 100% of baseline assignments.
  2. `OfflineContextualRuleDiagnosticProvider` (`offline_contextual_diagnostic`): Deterministic non-neural diagnostic analyzer that inspects isolated 1-word speaker islands and `"UNK"` boundary words.
  3. `MockScriptedLLMProvider` (`mock_scripted`): Deterministic scripted provider for unit tests and controlled conversational Q&A simulations.
  4. `OpenAICompatibleLLMProvider` (`openai_compatible`): Standard urllib-based client for OpenAI-compatible chat completion endpoints (`OPENAI_API_KEY`).
  5. `GeminiLLMProvider` (`gemini`): Standard urllib-based client for Google Gemini `generateContent` endpoints (`GEMINI_API_KEY`).

### Step D — Constrained Parser & TPST-Inspired Reconciliation (`tpst_reconciler.py`)
- Enforces strict JSON syntax and schema validation (`window_id` and `decisions` list).
- Validates every decision against the original `OrchestratedWord` sequence and rejects:
  - Malformed JSON (`REJECTED_MALFORMED_JSON`) or schema violations (`REJECTED_SCHEMA_VIOLATION`)
  - Extra words (`REJECTED_EXTRA_WORDS`) or missing words (`REJECTED_MISSING_WORD`)
  - Reordered or mutated words (`REJECTED_REORDERED_OR_MUTATED_WORDS`)
  - Hallucinated or out-of-inventory speaker IDs (`REJECTED_INVALID_SPEAKER_ID`)
  - Low-confidence proposals (`REJECTED_LOW_CONFIDENCE`)
- **Why Hungarian Matching Alone Does Not Guarantee Word Alignment**:
  Hungarian (Kuhn-Munkres) bipartite matching solves an unconstrained permutation assignment problem that **does not enforce sequential monotonicity** ($p_1 < p_2 \iff q_1 < q_2$). In conversational BWC transcripts containing frequent repeated function words (`"you"`, `"the"`, `"yes"`, `"bag"`), unconstrained bipartite matching can cross-align identical tokens across different turns or silently mask word reorderings. Therefore, `TPSTReconciler` enforces **strict monotonic index and lexical verification** for word-sequence alignment, whereas Hungarian matching is used exclusively in Step E for permutation-invariant speaker-ID mapping between predicted cluster IDs (`speaker_0..k`) and reference speaker labels (`Speaker_1..m`).

---

## 3. Data Schemas (`schemas.py`)

| Schema Class | Purpose | Key Fields |
|---|---|---|
| `OrchestratedWord` | Input word attributed to baseline RTTM | `word_index`, `word`, `start`, `end`, `asr_confidence`, `original_speaker_id`, `attribution_status`, `is_overlapping_speech`, `overlapping_speakers`, `overlap_durations_by_speaker` |
| `ConversationalTurnSummary` | Grouped contiguous speaker turn | `turn_index`, `speaker_id`, `start`, `end`, `text`, `word_count`, `has_overlap` |
| `ContextWindow` | Sliding conversational window | `window_id`, `core_word_indices`, `context_word_indices`, `words`, `surrounding_turns`, `available_speaker_ids`, `overlap_info` |
| `LLMRefinementRequest` | Provider-independent prompt bundle | `window_id`, `available_speaker_ids`, `words_payload`, `surrounding_turns_payload`, `overlap_info`, `system_prompt`, `user_prompt` |
| `RefinedWordOutput` | Reconciled word output with full provenance | `word_index`, `word`, `start`, `end`, `original_speaker_id`, `proposed_speaker_id`, `refined_speaker_id`, `abstained`, `accepted`, `confidence`, `reconciliation_reason`, `reasoning`, `window_id` |

---

## 4. Evaluation Framework & Mathematical Definitions (`evaluation.py`)

Stage 6 computes two complementary definitions of **Word Diarization Error Rate (WDER)** alongside **Concatenated Minimum-Permutation Word Error Rate (`cpWER`)**:

1. **Timestamp-Aligned WDER (`compute_timestamp_wder`)**:
   Each ASR word with valid timestamps is matched to the ground-truth reference segment covering its temporal interval (or midpoint within `0.5s`). Let $M^*$ be the optimal 1-to-1 Hungarian speaker mapping maximizing word-level speaker agreement:
   $$\text{WDER}_{\text{assigned}} = \frac{N_{\text{assigned}} - N_{\text{correct}}(M^*)}{N_{\text{assigned}}}, \qquad \text{WDER}_{\text{all\_ref}} = \frac{N_{\text{all\_ref}} - N_{\text{correct}}(M^*)}{N_{\text{all\_ref}}}$$
   where $N_{\text{assigned}}$ counts ASR words in reference speech assigned a non-`"UNK"` speaker (exact parity with Phase 20 `1 - speaker_attribution_accuracy`), and $N_{\text{all\_ref}}$ also penalizes `"UNK"` words falling inside reference speech.

2. **Lexical-Aligned WDER (`compute_lexical_wder`, Shafey et al. / DiarizationLM definition)**:
   Aligns the chronological hypothesis word sequence against the chronological reference word sequence via minimum-edit-distance Levenshtein alignment. Over the $C + S$ aligned word pairs (lexical matches $C$ plus lexical substitutions $S$), using the optimal speaker mapping $M^*$:
   $$\text{WDER}_{\text{lexical}} = \frac{S_{\text{IS}} + C_{\text{IS}}}{S + C}$$
   where $C_{\text{IS}}$ and $S_{\text{IS}}$ denote lexically correct and substituted words whose mapped hypothesis speaker disagrees with the reference speaker.

3. **Concatenated Minimum-Permutation WER (`cpWER`)**:
   Uses `phase20_standalone/scripts/cpwer.py` (`cpwer_stats`) to group adjacent same-speaker words into turns (max gap `2.0s`), concatenate turns per speaker, and solve the minimum-WER bipartite speaker assignment against reference speaker transcripts.

---

## 5. Files Created

All Stage 6 files reside exclusively inside `research_stages/stage6_generative_refinement/`:

1. `research_stages/stage6_generative_refinement/__init__.py` — Package exports.
2. `research_stages/stage6_generative_refinement/schemas.py` — Dataclasses and enums (`AttributionStatus`, `ConfidenceStatus`, `ReconciliationReason`, `OrchestratedWord`, `ConversationalTurnSummary`, `ContextWindow`, `LLMRefinementRequest`, `RefinedWordOutput`).
3. `research_stages/stage6_generative_refinement/word_orchestrator.py` — Step A word-level orchestration and overlap attribution.
4. `research_stages/stage6_generative_refinement/context_window_builder.py` — Step B bounded sliding conversational window builder.
5. `research_stages/stage6_generative_refinement/llm_interface.py` — Step C provider-independent LLM interface and offline/live providers.
6. `research_stages/stage6_generative_refinement/tpst_reconciler.py` — Step D constrained JSON parser and TPST reconciler.
7. `research_stages/stage6_generative_refinement/evaluation.py` — Step E WDER (`timestamp` and `lexical`), `cpWER`, telemetry, and SHA-256 integrity checks.
8. `research_stages/stage6_generative_refinement/pipeline.py` — End-to-end `Stage6RefinementPipeline` orchestrator.
9. `research_stages/stage6_generative_refinement/run_stage6_r01_pilot.py` — Read-only R01 integration runner for E20 and E21.
10. `research_stages/stage6_generative_refinement/tests/test_stage6_generative_refinement.py` — 13 synthetic and R01 integration tests.
11. `research_stages/stage6_generative_refinement/outputs/` — Generated R01 evaluation summary and per-condition/mode JSON artifacts.

---

## 6. Synthetic Test Results

Executed via `research_stages/.venv_phase20/bin/pytest research_stages/stage6_generative_refinement/tests/test_stage6_generative_refinement.py -v`:

| Test ID | Test Case | Result |
|---|---|---|
| `test_01` | Correct speaker reassignment on synthetic Q&A turn boundary (improves WDER & cpWER to `0.0`) | **PASSED** |
| `test_02` | Invalid / hallucinated speaker IDs (`speaker_99`, `Officer_Smith`) rejected (`REJECTED_INVALID_SPEAKER_ID`) | **PASSED** |
| `test_03` | Missing words in LLM output rejected (`REJECTED_MISSING_WORD`) while preserving original words | **PASSED** |
| `test_04` | Extra words in LLM output rejected (`REJECTED_EXTRA_WORDS`) | **PASSED** |
| `test_05` | Reordered words in LLM output rejected (`REJECTED_REORDERED_OR_MUTATED_WORDS`) | **PASSED** |
| `test_06` | Multi-speaker overlapping speech explicitly tracked (`MULTI_SPEAKER_OVERLAP` + overlap durations) | **PASSED** |
| `test_07` | Missing speaker assignments (`UNK`) and missing timestamps (`None`) handled safely | **PASSED** |
| `test_08` | LLM abstention (`"abstain": true`) preserves 100% of baseline assignments (`ABSTAINED_BY_LLM`) | **PASSED** |
| `test_09` | Malformed JSON and schema violations cleanly rejected without crashing | **PASSED** |
| `test_10` | Original word text, timestamps, and recoverable baseline transcript verified via SHA-256 | **PASSED** |
| `test_11` | Sliding window core-word precedence verified across overlapping context buffers | **PASSED** |
| `test_12` | External providers (`OpenAICompatibleLLMProvider`, `GeminiLLMProvider`) fail cleanly without API keys | **PASSED** |
| `test_13` | Read-only R01 pilot integration (`E20` & `E21`) and input hash immutability verified | **PASSED** |

**Total**: **13 / 13 passed** (`1.98s`).

---

## 7. R01 Pilot Integration Results (`R01_sara_sharif`)

The pipeline was executed in strictly read-only mode on the frozen R01 inputs:
- **ASR Words**: `research_stages/stage2_baseline/whisperx/R01_sara_sharif_aligned.json` (`266` words, SHA-256 `2dab0db97539fc00865c90c81e9f8fcacf011ce7c3ee2bd59beea10966511add`)
- **Diarization RTTMs**: `outputs/E20/R01_sara_sharif.rttm` and `outputs/E21/R01_sara_sharif.rttm` (`19` segments, `3` speakers: `speaker_0`, `speaker_1`, `speaker_2`, SHA-256 `109f49a6bd3da3503721c7f6ddc9b1c757985f316be0b5d95cb4995b47b9712a`)
- **Reference Transcript**: `research_stages/stage1_data_prep/annotations_json/R01_sara_sharif.json` (`7` segments, `5` reference speakers: `Speaker_1..Speaker_5`, SHA-256 `20a8300a780d9fefab9f9e05774776ea55b58674f744e1b8059d18932e1f9ce1`)

### Quantitative Comparison (E20 and E21)

Because `E20` (M0) and `E21` (M4-DFN) produce identical RTTM segments on `R01_sara_sharif` (as established in the Stage 5B.4 R01 pilot reconciliation), both conditions yield identical word-level orchestration and Stage 6 evaluation metrics:

| Condition | Execution Mode | Provider Type | Proposed | Accepted | Rejected | Abstained | $\text{WDER}_{\text{ts,assigned}}$ | $\text{WDER}_{\text{ts,all\_ref}}$ | $\text{WDER}_{\text{lexical}}$ | $\text{cpWER}$ | Word/Timestamp SHA-256 Match |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **E20 / E21** | **Baseline (Stage 5B.4)** | — | — | — | — | — | `20.08%` (`53/264`) | `20.08%` (`53/264`) | `19.38%` (`50/258`) | `52.30%` (`148/283`) | Verified |
| **E20 / E21** | `offline_abstaining` | Offline Default | `0` | `0` | `0` | `266` | `20.08%` (`0.00 pp`) | `20.08%` (`0.00 pp`) | `19.38%` (`0.00 pp`) | `52.30%` (`0.00 pp`) | `b5e13093...` / `9c7ba033...` |
| **E20 / E21** | `offline_contextual_diagnostic` | Offline Rule Diagnostic | `0` | `0` | `0` | `266` | `20.08%` (`0.00 pp`) | `20.08%` (`0.00 pp`) | `19.38%` (`0.00 pp`) | `52.30%` (`0.00 pp`) | `b5e13093...` / `9c7ba033...` |
| **E20 / E21** | `simulated_conversational_qa` | Scripted Simulation (`mock_scripted`) | `3` | `3` | `0` | `263` | **`19.70%` (`-0.38 pp`)** | **`19.70%` (`-0.38 pp`)** | **`18.99%` (`-0.39 pp`)** | `52.30%` (`0.00 pp`, `S: 43→41`) | `b5e13093...` / `9c7ba033...` |

### Key Structural Findings on R01
1. **Bit-Exact Baseline Preservation in Default Offline Modes**:
   Both `offline_abstaining` and `offline_contextual_diagnostic` abstain on all 266 words (`offline_contextual_diagnostic` abstains on word 43 `"Yes."` because it lies at a 3-way speaker transition `speaker_2 -> speaker_1 -> speaker_0` rather than a symmetric `A -> B -> A` island).
2. **Targeted Boundary Refinement (`simulated_conversational_qa`)**:
   In the scripted conversational simulation, reassigning word 43 (`"Yes."` at `14.492–14.833s`) from `speaker_1` to `speaker_0` (which maps to reference `Speaker_2` covering `3.00–16.00s`) and assigning the initial unassigned words 0–1 (`"Yes sir."` at `2.052–2.377s`) from `"UNK"` to `speaker_1` reduces timestamp-aligned WDER from `20.08%` to `19.70%` (`-0.38 pp`) and lexical WDER from `19.38%` to `18.99%` (`-0.39 pp`), while reducing `cpWER` substitution errors from `43` to `41`.
3. **Why Cluster-Count Under-Estimation Bounds Post-Processing Gains on R01**:
   Ground-truth R01 contains **5 reference speakers** (`Speaker_1..Speaker_5`), whereas upstream acoustic clustering (`E20`/`E21`) produced **3 speaker clusters** (`speaker_0`, `speaker_1`, `speaker_2`). Because Stage 6 is strictly prohibited from inventing new speaker IDs outside `available_speaker_ids`, turns spoken by unmapped minor speakers (such as `Speaker_3`'s 5-word response `"I don't have any bag."` at `17.98–18.65s`) cannot be assigned to a dedicated cluster without introducing a new speaker ID.

---

## 8. Limitations & Next Steps

1. **Cluster Inventory Bound**: Because TPST reconciliation restricts `proposed_speaker_id` to `available_speaker_ids` from the acoustic RTTM, Stage 6 can correct boundary attribution errors and speaker swaps among discovered speakers, but cannot create new speaker clusters when upstream clustering merges minor speakers.
2. **Running Live Instruction-Following LLMs**:
   To evaluate a live external LLM (e.g., `gpt-4o-mini` or `gemini-2.5-flash`) on R01 once API credentials are provided in the environment:
   ```bash
   export OPENAI_API_KEY="<your-key>"
   research_stages/.venv_phase20/bin/python -m research_stages.stage6_generative_refinement.run_stage6_r01_pilot \
       --live-provider openai_compatible --live-model gpt-4o-mini
   ```
   All responses from live providers pass through the exact same deterministic `TPSTReconciler` guardrails.
