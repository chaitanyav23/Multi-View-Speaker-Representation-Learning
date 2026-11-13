# ConvoFuse — Conversational Transcript Reconstruction & Multi-Format Export Specification

**Module**: [`convofuse/transcript_reconstructor.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/convofuse/transcript_reconstructor.py)  

---

## 1. Chronological Turn Reconstruction (`reconstruct_turns`)

ConvoFuse reconstructs human-readable chronological speaker turns from the reconciled `ConvoFuseWord` sequence:

1. **Speaker Boundary Preservation**: Consecutive words are merged into a `ConvoFuseTurn` only if they share the same `refined_speaker_id` (or `baseline_speaker_id` for baseline turn comparison) and the inter-word silence gap is $\le 1.5\text{s}$. Words are never merged across speaker changes.
2. **Timestamp Formatting**: `format_hhmmss_mmm` formats start and end timestamps to millisecond precision (`HH:MM:SS.mmm`), producing headers such as:
   ```text
   [00:00:02.987 – 00:00:09.329] SPEAKER_02: We are, yes. Are you okay just to swap over, just yourself come with me and if you go with my colleague over there, is that alright?
   [00:00:09.409 – 00:00:09.850] SPEAKER_01: Alright.
   ```
3. **Overlap & Refinement Metadata per Turn**: Each `ConvoFuseTurn` tracks `has_overlapping_speech`, `overlapping_speakers`, `accepted_refinement_count`, `rejected_refinement_count`, `abstained_refinement_count`, and `unchanged_word_count`.

---

## 2. Speaker Summary Index (`build_speaker_summaries`)

For every speaker in the recording (`SPEAKER_00`, `SPEAKER_01`, ..., plus `UNK` if unassigned words exist), `build_speaker_summaries` computes:
- `total_speaking_duration_sec`
- `turn_count`
- `word_count`
- `first_appearance_sec` and `last_appearance_sec`
- `mean_word_confidence`
- `accepted_corrections_into_speaker` and `accepted_corrections_out_of_speaker`

No demographic, role, or identity descriptions are fabricated.

---

## 3. Seven Mandatory Output Artifacts (`export_convofuse_artifacts`)

Every ConvoFuse execution exports 7 structured files per recording:

1. `<rec_id>__transcript.txt` — Complete human-readable chronological conversational transcript with header metadata and speaker summary index.
2. `<rec_id>__words.json` — Word-level JSON with `baseline_speaker_id`, `refined_speaker_id`, `overlap_metadata`, `provenance`, `refinement_decision`, and `reconciliation_reason`.
3. `<rec_id>__turns.json` — Chronological turn-level JSON (`turns` and `baseline_turns`).
4. `<rec_id>__speakers.json` — Speaker summary index and RTTM-to-canonical speaker mapping.
5. `<rec_id>__comparison.json` — Baseline-versus-refined word/turn diff and segment-level LLM telemetry.
6. `<rec_id>__processing_manifest.json` — Complete 15-stage execution audit trail, audio stream routing, and SHA-256 verification hashes.
7. `<rec_id>__evaluation_report.json` — Post-inference evaluation metrics (Timestamp WDER, Lexical WDER, cpWER, correction precision, and cryptographic word/timestamp integrity hashes).
