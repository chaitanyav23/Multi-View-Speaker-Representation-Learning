# ConvoFuse — Canonical Data Contracts & State Lifecycle Specification

**Module**: [`convofuse/contracts.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/convofuse/contracts.py)  

---

## 1. Separation of the Three Distinct Fusion & Conditioning Operations (Part 2.1)

ConvoFuse enforces a strict architectural separation across three distinct operations that must never be conflated:

1. **Task-Specific Dual-Path Audio Conditioning (`TaskSpecificAudioPaths`)**:
   - Routes `dfn_conditioned_wav` (16 kHz mono DeepFilterNet3-enhanced WAV) to `TitaNet-Large` (192-D).
   - Routes `canonical_16k_mono_wav` (16 kHz mono unenhanced WAV) to `WeSpeaker ResNet293-LM` (256-D) and raw-audio overlap posterior detection.
   - Routes `normalized_asr_wav` (16 kHz mono EBU R128 loudness-normalized WAV) to `Faster-Whisper Large-v3` ASR and `Wav2Vec2` CTC forced alignment.
2. **Dual-Encoder Speaker Affinity Fusion**:
   - Computes independent cosine similarity matrices $S_{\text{TitaNet}}$ and $S_{\text{WeSpeaker}}$ over 1.5s / 0.75s subsegments.
   - Applies off-diagonal row-wise Z-score normalization and combines them via Equal-Weight (`0.5 / 0.5`) or Reliability-Gated (`[0.35, 0.65]`) fusion prior to NME-SC spectral clustering.
3. **Temporal Word-to-Speaker Association (`build_baseline_convofuse_words`)**:
   - Associates Wav2Vec2-aligned ASR words (`[start, end]`) with NME-SC diarization RTTM intervals using `max_overlap_speaker` and a `0.5s` midpoint collar (`orchestrate_words`), preserving unassigned words as `UNK` and attaching frame-level overlap metadata.

---

## 2. Execution Mode Contract (`ConvoFuseExecutionMode`)

| Mode | Enum Value | Description |
| :--- | :--- | :--- |
| **Full Processing Mode** | `FULL_PROCESSING` | Accepts raw uploaded audio/video media, extracts canonical 16 kHz mono + EBU R128 normalized audio, runs ASR/alignment/diarization, executes word-speaker association, runs `DiarizationLM-Gemma` TPST refinement, and reconstructs the complete transcript. |
| **Frozen-Input Replay Mode** | `FROZEN_INPUT_REPLAY` | Reuses SHA-256-verified frozen Stage 1 audio, Stage 2 aligned ASR words, Stage 5 dual-encoder predicted RTTMs, and Stage 5B.2 overlap posteriors; executes live word-speaker association, live ground-truth-isolated `DiarizationLM-Gemma` inference, TPST reconciliation, transcript reconstruction, and separate post-inference evaluation. |

---

## 3. Canonical 15-State Job Lifecycle (`ConvoFuseJobStage`)

Every ConvoFuse execution records an ordered audit trail (`stage_history`) across the 15 canonical lifecycle stages:

1. `UPLOADED`
2. `QUEUED`
3. `AUDIO_EXTRACTION`
4. `VAD`
5. `ASR`
6. `WORD_ALIGNMENT`
7. `SPEAKER_REPRESENTATION`
8. `AFFINITY_FUSION`
9. `CLUSTERING`
10. `WORD_SPEAKER_ASSIGNMENT`
11. `GENERATIVE_REFINEMENT`
12. `TRANSCRIPT_RECONSTRUCTION`
13. `EVALUATION`
14. `COMPLETED`
15. `FAILED`

---

## 4. Deterministic Speaker Identity Contract (`SpeakerIdentityMapper`)

Implemented in [`SpeakerIdentityMapper`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/convofuse/contracts.py#L297-L388):

1. **RTTM to Canonical Mapping**: Upstream RTTM labels (`speaker_0`, `speaker_1`, `speaker_2`, ...) are natural-sorted and mapped deterministically to `SPEAKER_00`, `SPEAKER_01`, `SPEAKER_02`, ...
2. **Known Speaker Prompt Tagging**: Canonical speakers `SPEAKER_00..SPEAKER_{K-1}` are mapped 1-to-1 to 1-indexed DiarizationLM tags `<speaker:1>..<speaker:K>`.
3. **`UNK` Invariance**: Words without a valid diarization assignment have `baseline_speaker_id = "UNK"`. `UNK` is excluded from `canonical_inventory` (`1..K`) and is tagged `<speaker:0>` in prompt segments. Any LLM proposal to map an `UNK` word to `<speaker:1..K>` or to map a known word to `<speaker:0>` is rejected with `REJECTED_UNKNOWN_SPEAKER_MAPPING`.
4. **Strict Tag Validation (`resolve_dlm_tag`)**:
   - Missing tag before initial `<speaker:...>` token $\rightarrow$ `REJECTED_MISSING_TAG`
   - Malformed tag (e.g., `<speaker:abc>`, `<speaker:>`) $\rightarrow$ `REJECTED_MALFORMED_TAG`
   - Tag `0` or `UNK` $\rightarrow$ `REJECTED_UNKNOWN_SPEAKER_MAPPING`
   - Integer tag outside `1..K` $\rightarrow$ `REJECTED_OUT_OF_INVENTORY_TAG`

---

## 5. Word, Turn, and Speaker Summary Data Contracts

- **`ConvoFuseWord`**: Immutable `word_index`, `word`, `start`, `end`, `baseline_speaker_id`, `refined_speaker_id`, `asr_confidence`, `calibrated_confidence`, `alignment_score`, `alignment_status`, `overlap_metadata` (`OverlapMetadata`), `provenance` (`WordProvenance`), `refinement_decision` (`UNCHANGED`, `ACCEPTED_CORRECTION`, `REJECTED_PROPOSAL`, `ABSTAINED`), `reconciliation_reason` (`RejectionOrAbstentionReason`), `proposed_speaker_id`, and `reasoning_note`.
- **`ConvoFuseTurn`**: `turn_id`, `turn_index`, `speaker_id`, `start`, `end`, `duration`, `text`, `word_indices`, `word_count`, `avg_confidence`, `has_overlapping_speech`, `overlapping_speakers`, refinement decision counts, `formatted_header` (`[HH:MM:SS.mmm – HH:MM:SS.mmm] SPEAKER_XX`), and `formatted_line`.
- **`ConvoFuseSpeakerSummary`**: `speaker_id`, `total_speaking_duration_sec`, `turn_count`, `word_count`, `first_appearance_sec`, `last_appearance_sec`, `mean_word_confidence`, `accepted_corrections_into_speaker`, and `accepted_corrections_out_of_speaker`.
