# ConvoFuse — Ground-Truth-Isolated Generative Refinement & TPST Report

**Module**: [`convofuse/generative_refiner.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/convofuse/generative_refiner.py)  
**Model**: `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (`Q4_K_M` GGUF via Local Ollama on CUDA)  

---

## 1. Strict Ground-Truth Isolation Architecture (Part 4.1 & 4.5)

`ConvoFuseGenerativeRefiner` enforces strict architectural isolation from reference annotations:

1. **Signature & Runtime Guard**: `ConvoFuseGenerativeRefiner._enforce_no_ground_truth_kwargs` inspects all keyword arguments passed to `refine_words()` and raises a `TypeError("Ground-truth isolation violation...")` if any reference parameter (`reference_transcript`, `reference_segments`, `reference_rttm`, `ground_truth`, `gt_segments`, `gt_labels`, `reference_path`, `reference_json_path`, `evaluation_metrics`) is supplied.
2. **Module-Level Separation**: Neither `convofuse/generative_refiner.py` nor `convofuse/audio_diarization_adapter.py` imports or reads `annotations_json/` or reference RTTMs. Reference annotations are loaded exclusively inside `convofuse/evaluator.py` (`evaluate_post_inference`) at Stage 13 (`EVALUATION`), strictly after Stage 11 (`GENERATIVE_REFINEMENT`) and Stage 12 (`TRANSCRIPT_RECONSTRUCTION`) have completed.

---

## 2. DiarizationLM-Gemma Prompt Construction & Local Ollama Execution (Part 4.2–4.3)

- **Prompt Template**: Uses the official `DiarizationLM-Gemma-4-E4B-v1` turn format:
  ```text
  <|turn>user
  <speaker:1> ... <speaker:2> ... --> <turn|>
  <|turn>model
  ```
- **Deterministic Decoding**: `OllamaDiarizationLMClient` queries `http://localhost:11434/api/generate` with `model="diarizationlm-gemma"`, `temperature=0.0`, `top_k=1`, `seed=42`, `num_predict=2048`, and `stop=["<turn|>", "[eod]"]`.
- **Measured Telemetry (`SegmentRefinementTelemetry`)**: Records actual `prompt_eval_count` (`prompt_tokens`), `eval_count` (`completion_tokens`), `done`, `done_reason`, `runtime_sec`, `provider_version` (`ollama-0.32.6`), `prompt_text`, `raw_completion`, and `cleaned_completion` for every segment.

---

## 3. Strict TPST Reconciliation & Rejection Rules (Part 4.4)

`extract_text_and_spk_strict` and `reconcile_segment` replace the permissive fallback behavior of the upstream `diarizationlm` utility (`skip_meaningless_speaker=True`) with strict deterministic validation:

| Validation Rule | Condition | Decision | Reason Code |
| :--- | :--- | :--- | :--- |
| **Truncated Completion** | `not done` or `done_reason in ("length", "max_tokens")` | `REJECTED_PROPOSAL` | `REJECTED_TRUNCATED_COMPLETION` |
| **Incomplete Alignment** | Hypothesis word omitted by LLM completion (`aligned_tag is None`) | `REJECTED_PROPOSAL` | `REJECTED_INCOMPLETE_ALIGNMENT` |
| **UNK Preservation** | `baseline_speaker_id == "UNK"` and LLM proposes `<speaker:1..K>` | `REJECTED_PROPOSAL` | `REJECTED_UNKNOWN_SPEAKER_MAPPING` |
| **Missing Initial Tag** | Word appears before any `<speaker:...>` tag in completion | `REJECTED_PROPOSAL` | `REJECTED_MISSING_TAG` |
| **Malformed Speaker Tag** | Unclosed or non-integer tag (e.g., `<speaker:abc>`) | `REJECTED_PROPOSAL` | `REJECTED_MALFORMED_TAG` |
| **Out-of-Inventory Tag** | Integer tag `<speaker:M>` where $M \notin \{1, \dots, K\}$ | `REJECTED_PROPOSAL` | `REJECTED_OUT_OF_INVENTORY_TAG` |
| **Simultaneous Overlap Guard** | Proposed speaker change on word with `is_overlapping_speech == True` | `REJECTED_PROPOSAL` | `REJECTED_UNSUPPORTED_OVERLAP_CORRECTION` |
| **Locality Gate (LG-TPST)** | Contiguous speaker-flip span $> 5$ words | `ABSTAINED` | `ABSTAINED_LOCALITY_GATE` |
| **Baseline Confirmed** | Validated tag matches `baseline_speaker_id` | `UNCHANGED` | `UNCHANGED_BASELINE_CONFIRMED` |
| **Validated Local Transfer** | Validated tag in `1..K`, span $\le 5$ words, non-overlapping | `ACCEPTED_CORRECTION` | `ACCEPTED_VALIDATED_TRANSFER` |

`ConvoFuseGenerativeRefiner.verify_invariants` asserts after every run that word count, `word_index`, `word` text, `start`/`end` timestamps, `baseline_speaker_id`, `UNK` invariance, and `overlap_metadata` are 100% unchanged.
