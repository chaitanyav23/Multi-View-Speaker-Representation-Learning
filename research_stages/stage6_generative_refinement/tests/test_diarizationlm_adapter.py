"""Unit and integration tests for DiarizationLM-Gemma adapter and TPST transfer.

Tests:
1. Multiple speakers mapping and reverse-mapping
2. Repeated words handling in Levenshtein alignment
3. Overlapping turns and turn-boundary corrections
4. Ambiguous alignments (insertions/deletions in LLM completion)
5. Out-of-inventory speaker rejection
6. Strict word and timestamp invariance
7. Prompt segmentation boundary completeness
"""

from __future__ import annotations

import unittest
from research_stages.stage6_generative_refinement.diarizationlm_adapter import (
    PromptOptions,
    extract_text_and_spk,
    truncate_suffix_and_tailing_text,
    transcript_preserving_speaker_transfer,
    apply_locality_gated_speaker_transfer,
    DiarizationLMGemmaAdapter,
)
from research_stages.stage6_generative_refinement.schemas import (
    AttributionStatus,
    ConfidenceStatus,
    OrchestratedWord,
    ReconciliationReason,
)


class TestDiarizationLMAdapter(unittest.TestCase):

  def setUp(self) -> None:
    self.adapter = DiarizationLMGemmaAdapter(
        max_chars_per_segment=4000,
        use_locality_gating=True,
        max_local_span_words=5,
    )

  def test_01_truncate_suffix_and_tailing_text(self) -> None:
    raw = "<speaker:1> Hello world [eod] extra hallucinated text"
    cleaned = truncate_suffix_and_tailing_text(raw, " [eod]")
    self.assertEqual(cleaned, "<speaker:1> Hello world")

    raw2 = "<speaker:1> Hello <turn|> trailing stuff"
    cleaned2 = truncate_suffix_and_tailing_text(raw2, "<turn|>")
    self.assertEqual(cleaned2, "<speaker:1> Hello ")

  def test_02_multiple_speakers_extraction_and_mapping(self) -> None:
    comp = "<speaker:1> Welcome everyone. <speaker:2> Thanks. <speaker:3> Hello! <speaker:4> Let's begin."
    text, spk = extract_text_and_spk(comp)
    self.assertEqual(text, "Welcome everyone. Thanks. Hello! Let's begin.")
    self.assertEqual(spk, "1 1 2 3 4 4")

  def test_03_repeated_words_alignment(self) -> None:
    """Tests that repeated words ('you', 'know', 'that') are aligned monotonically."""
    hyp_text = "You know that you know that I know"
    hyp_spk = "1 1 1 1 1 1 1 1"
    llm_text = "You know that you know that I know"
    llm_spk = "1 1 1 2 2 2 2 2"

    transferred = transcript_preserving_speaker_transfer(
        src_text=llm_text,
        src_spk=llm_spk,
        tgt_text=hyp_text,
        tgt_spk=hyp_spk,
    )
    self.assertEqual(transferred, "1 1 1 2 2 2 2 2")

  def test_04_overlapping_and_shifted_turn_boundary(self) -> None:
    """Tests turn-boundary correction on single-word affirmation."""
    hyp_text = "Are you ready? Yes. Okay let's go."
    hyp_spk = "1 1 1 1 2 2 2"
    llm_text = "Are you ready? Yes. Okay let's go."
    llm_spk = "1 1 1 2 2 2 2"

    gated = apply_locality_gated_speaker_transfer(
        hyp_spk=hyp_spk.split(),
        llm_spk=llm_spk.split(),
        max_local_span_words=5,
    )
    # The single word 'Yes.' (span_len=1 <= 5) should accept the correction to speaker 2
    self.assertEqual(gated, ["1", "1", "1", "2", "2", "2", "2"])

  def test_05_ambiguous_llm_insertions_and_deletions(self) -> None:
    """Verifies TPST preserves target word count even if LLM inserts or deletes words."""
    hyp_text = "Which one is your bag?"
    hyp_spk = "1 1 1 1 1"
    # LLM dropped 'one' and inserted 'please'
    llm_text = "Which is your bag please?"
    llm_spk = "2 2 2 2 2"

    transferred = transcript_preserving_speaker_transfer(
        src_text=llm_text,
        src_spk=llm_spk,
        tgt_text=hyp_text,
        tgt_spk=hyp_spk,
    )
    # Output MUST match target length exactly (5 words)
    self.assertEqual(len(transferred.split()), len(hyp_text.split()))

  def test_06_word_and_timestamp_immutability(self) -> None:
    """Verifies that reconciliation never mutates word strings or timestamps."""
    words = [
        OrchestratedWord(
            word_index=0,
            word="Which",
            start=16.92,
            end=17.06,
            asr_confidence=0.95,
            original_speaker_id="speaker_0",
            attribution_status=AttributionStatus.SINGLE_SPEAKER_OVERLAP.value,
        ),
        OrchestratedWord(
            word_index=1,
            word="bag?",
            start=17.48,
            end=17.70,
            asr_confidence=0.95,
            original_speaker_id="speaker_0",
            attribution_status=AttributionStatus.SINGLE_SPEAKER_OVERLAP.value,
        ),
    ]
    segs = self.adapter.build_prompt_segments(
        words, available_speaker_ids=["speaker_0", "speaker_1"]
    )
    self.assertEqual(len(segs), 1)

    llm_completion = "<speaker:2> Which bag? [eod]"
    refined, telemetry = self.adapter.reconcile_segment(
        segs[0], llm_completion
    )
    self.assertEqual(len(refined), 2)
    self.assertEqual(refined[0].word, "Which")
    self.assertEqual(refined[0].start, 16.92)
    self.assertEqual(refined[0].end, 17.06)
    self.assertEqual(refined[1].word, "bag?")
    self.assertEqual(refined[1].start, 17.48)
    self.assertEqual(refined[1].end, 17.70)
    # Speaker 2 in prompt maps to speaker_1
    self.assertEqual(refined[0].refined_speaker_id, "speaker_1")

  def test_07_prompt_segmentation_no_word_loss(self) -> None:
    """Verifies prompt segmentation chunks large word sequence without losing a single word."""
    words = [
        OrchestratedWord(
            word_index=i,
            word=f"word_{i}",
            start=float(i),
            end=float(i + 0.5),
            asr_confidence=0.95,
            original_speaker_id="speaker_0" if i % 2 == 0 else "speaker_1",
            attribution_status=AttributionStatus.SINGLE_SPEAKER_OVERLAP.value,
        )
        for i in range(150)
    ]
    # Small character limit to force multiple segments
    adapter = DiarizationLMGemmaAdapter(max_chars_per_segment=300)
    segs = adapter.build_prompt_segments(
        words, available_speaker_ids=["speaker_0", "speaker_1"]
    )
    self.assertGreater(len(segs), 1)

    total_segmented_words = sum(len(s.words) for s in segs)
    self.assertEqual(total_segmented_words, 150)

    # Check contiguous sequence
    for i, seg in enumerate(segs):
      if i > 0:
        self.assertEqual(
            seg.start_word_index, segs[i - 1].end_word_index + 1
        )


if __name__ == "__main__":
  unittest.main()
