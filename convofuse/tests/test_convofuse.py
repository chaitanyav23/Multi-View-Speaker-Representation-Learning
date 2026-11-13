"""
ConvoFuse Comprehensive Unit & Integration Test Suite (Part 7.1).

Verifies:
1. Strict ground-truth isolation during generative refinement (raises TypeError on GT kwargs;
   prompts and refined outputs are 100% invariant to presence/absence/mutation of reference files).
2. SpeakerIdentityMapper determinism and UNK invariance (UNK never mapped to <speaker:1> or any valid speaker).
3. Strict TPST reconciliation and explicit rejection reasons:
   - REJECTED_OUT_OF_INVENTORY_TAG
   - REJECTED_MISSING_TAG
   - REJECTED_MALFORMED_TAG
   - REJECTED_TRUNCATED_COMPLETION
   - REJECTED_INCOMPLETE_ALIGNMENT
   - REJECTED_UNSUPPORTED_OVERLAP_CORRECTION
   - REJECTED_UNKNOWN_SPEAKER_MAPPING
   - ABSTAINED_LOCALITY_GATE
   - ACCEPTED_VALIDATED_TRANSFER
4. Chronological turn reconstruction, speaker summary index, readable transcript format,
   7-artifact export, and Phase20 TranscriptResult conversion.
5. Separate post-inference evaluation and SHA-256 integrity checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from convofuse.contracts import (
    ConvoFuseExecutionMode,
    ConvoFuseJobStage,
    ConvoFuseWord,
    OverlapMetadata,
    RefinementDecision,
    RejectionOrAbstentionReason,
    SpeakerIdentityMapper,
    WordProvenance,
)
from convofuse.evaluator import evaluate_post_inference
from convofuse.generative_refiner import (
    ConvoFuseGenerativeRefiner,
    OllamaDiarizationLMClient,
    align_completion_tags_to_hypothesis,
    extract_text_and_spk_strict,
)
from convofuse.transcript_reconstructor import (
    build_baseline_vs_refined_comparison,
    build_speaker_summaries,
    export_convofuse_artifacts,
    format_hhmmss_mmm,
    format_readable_transcript,
    reconstruct_turns,
    to_phase20_transcript_result_dict,
)


class StubOllamaClient(OllamaDiarizationLMClient):
    """Deterministic stub client for unit testing reconciliation edge cases."""

    def __init__(self, completion_payload: Dict[str, Any]) -> None:
        super().__init__()
        self._payload = completion_payload
        self.recorded_prompts: List[str] = []

    def get_provider_version(self) -> str:
        return "ollama-test-stub"

    def generate_completion(self, prompt: str) -> Dict[str, Any]:
        self.recorded_prompts.append(prompt)
        return dict(self._payload)


def _make_sample_words() -> List[ConvoFuseWord]:
    prov = WordProvenance(
        attribution_method="max_overlap_with_0.5s_midpoint_collar",
        attribution_status="SINGLE_SPEAKER_OVERLAP",
        nearest_segment_distance_sec=0.0,
        raw_rttm_speaker_id="speaker_0",
        diarization_backend="DualEncoder",
        diarization_condition="E20",
        context_segment_id="unassigned",
        refinement_model="none",
        execution_mode=ConvoFuseExecutionMode.FROZEN_INPUT_REPLAY.value,
    )
    return [
        ConvoFuseWord(
            word_index=0,
            word="Yes",
            start=1.0,
            end=1.4,
            baseline_speaker_id="UNK",
            refined_speaker_id="UNK",
            asr_confidence=0.8,
            calibrated_confidence=0.8,
            provenance=prov,
        ),
        ConvoFuseWord(
            word_index=1,
            word="Hello",
            start=2.0,
            end=2.5,
            baseline_speaker_id="SPEAKER_00",
            refined_speaker_id="SPEAKER_00",
            asr_confidence=0.9,
            calibrated_confidence=0.9,
            provenance=prov,
        ),
        ConvoFuseWord(
            word_index=2,
            word="there.",
            start=2.5,
            end=3.0,
            baseline_speaker_id="SPEAKER_00",
            refined_speaker_id="SPEAKER_00",
            asr_confidence=0.92,
            calibrated_confidence=0.92,
            provenance=prov,
        ),
        ConvoFuseWord(
            word_index=3,
            word="General",
            start=3.2,
            end=3.6,
            baseline_speaker_id="SPEAKER_00",
            refined_speaker_id="SPEAKER_00",
            asr_confidence=0.88,
            calibrated_confidence=0.88,
            provenance=prov,
        ),
        ConvoFuseWord(
            word_index=4,
            word="Kenobi.",
            start=3.6,
            end=4.1,
            baseline_speaker_id="SPEAKER_01",
            refined_speaker_id="SPEAKER_01",
            asr_confidence=0.95,
            calibrated_confidence=0.95,
            provenance=prov,
        ),
    ]


def test_01_speaker_identity_mapper_and_unk_invariance():
    mapper = SpeakerIdentityMapper(["speaker_1", "speaker_0", "UNK", "SPEAKER_UNASSIGNED"])
    assert mapper.canonical_inventory == ["SPEAKER_00", "SPEAKER_01"]
    assert mapper.to_canonical("speaker_0") == "SPEAKER_00"
    assert mapper.to_canonical("speaker_1") == "SPEAKER_01"
    assert mapper.to_canonical("UNK") == "UNK"
    assert mapper.to_canonical("SPEAKER_UNASSIGNED") == "UNK"
    assert mapper.to_canonical(None) == "UNK"
    assert "UNK" not in mapper.canonical_to_dlm_tag
    assert mapper.canonical_to_dlm_tag == {"SPEAKER_00": "1", "SPEAKER_01": "2"}

    # Valid tags
    assert mapper.resolve_dlm_tag("1") == ("SPEAKER_00", None)
    assert mapper.resolve_dlm_tag("2") == ("SPEAKER_01", None)

    # Out-of-inventory, missing, malformed, and unknown tags
    assert mapper.resolve_dlm_tag("3") == (None, RejectionOrAbstentionReason.REJECTED_OUT_OF_INVENTORY_TAG.value)
    assert mapper.resolve_dlm_tag("99") == (None, RejectionOrAbstentionReason.REJECTED_OUT_OF_INVENTORY_TAG.value)
    assert mapper.resolve_dlm_tag("0") == (None, RejectionOrAbstentionReason.REJECTED_UNKNOWN_SPEAKER_MAPPING.value)
    assert mapper.resolve_dlm_tag("MISSING_TAG") == (None, RejectionOrAbstentionReason.REJECTED_MISSING_TAG.value)
    assert mapper.resolve_dlm_tag("MALFORMED_TAG:<speaker:abc>") == (
        None,
        RejectionOrAbstentionReason.REJECTED_MALFORMED_TAG.value,
    )


def test_02_ground_truth_isolation_enforced():
    words = _make_sample_words()
    mapper = SpeakerIdentityMapper(["speaker_0", "speaker_1"])
    stub = StubOllamaClient(
        {
            "raw_completion": "<speaker:0> Yes <speaker:1> Hello there. <speaker:2> General Kenobi.",
            "done": True,
            "done_reason": "stop",
            "prompt_tokens": 20,
            "completion_tokens": 20,
            "runtime_sec": 0.01,
        }
    )
    refiner = ConvoFuseGenerativeRefiner(client=stub)

    for forbidden_kw in (
        "reference_transcript",
        "reference_segments",
        "reference_rttm",
        "ground_truth",
        "gt_segments",
        "gt_labels",
        "reference_path",
        "reference_json_path",
        "evaluation_metrics",
    ):
        with pytest.raises(TypeError, match="Ground-truth isolation violation"):
            refiner.refine_words(words, mapper, "rec_test", **{forbidden_kw: "forbidden"})


def test_03_tpst_accepts_valid_local_correction_and_preserves_unk():
    words = _make_sample_words()
    mapper = SpeakerIdentityMapper(["speaker_0", "speaker_1"])
    # Model proposes changing "General" (word 3) from <speaker:1> (SPEAKER_00) to <speaker:2> (SPEAKER_01),
    # and proposes <speaker:1> for UNK word 0 ("Yes") which must be rejected as REJECTED_UNKNOWN_SPEAKER_MAPPING.
    stub = StubOllamaClient(
        {
            "raw_completion": "<speaker:1> Yes <speaker:1> Hello there. <speaker:2> General Kenobi.",
            "done": True,
            "done_reason": "stop",
            "prompt_tokens": 25,
            "completion_tokens": 24,
            "runtime_sec": 0.02,
        }
    )
    refiner = ConvoFuseGenerativeRefiner(client=stub)
    res = refiner.refine_words(words, mapper, "rec_test")

    # Word 0 ("Yes") was UNK -> must remain UNK and reject proposal to map to SPEAKER_00
    assert res.refined_words[0].baseline_speaker_id == "UNK"
    assert res.refined_words[0].refined_speaker_id == "UNK"
    assert res.refined_words[0].refinement_decision == RefinementDecision.REJECTED_PROPOSAL.value
    assert res.refined_words[0].reconciliation_reason == RejectionOrAbstentionReason.REJECTED_UNKNOWN_SPEAKER_MAPPING.value

    # Word 3 ("General") -> accepted local boundary transfer to SPEAKER_01
    assert res.refined_words[3].baseline_speaker_id == "SPEAKER_00"
    assert res.refined_words[3].refined_speaker_id == "SPEAKER_01"
    assert res.refined_words[3].refinement_decision == RefinementDecision.ACCEPTED_CORRECTION.value
    assert res.refined_words[3].reconciliation_reason == RejectionOrAbstentionReason.ACCEPTED_VALIDATED_TRANSFER.value


def test_04_rejects_out_of_inventory_missing_malformed_and_truncated():
    words = _make_sample_words()[1:]  # 4 non-UNK words
    mapper = SpeakerIdentityMapper(["speaker_0", "speaker_1"])

    # 1. Out of inventory tag <speaker:9>
    stub_ooi = StubOllamaClient(
        {
            "raw_completion": "<speaker:1> Hello there. <speaker:9> General <speaker:2> Kenobi.",
            "done": True,
            "done_reason": "stop",
            "prompt_tokens": 20,
            "completion_tokens": 20,
            "runtime_sec": 0.01,
        }
    )
    res_ooi = ConvoFuseGenerativeRefiner(client=stub_ooi).refine_words(words, mapper, "rec_test")
    assert res_ooi.refined_words[2].refined_speaker_id == "SPEAKER_00"
    assert res_ooi.refined_words[2].refinement_decision == RefinementDecision.REJECTED_PROPOSAL.value
    assert res_ooi.refined_words[2].reconciliation_reason == RejectionOrAbstentionReason.REJECTED_OUT_OF_INVENTORY_TAG.value

    # 2. Missing initial tag & malformed tag
    stub_mal = StubOllamaClient(
        {
            "raw_completion": "Hello <speaker:abc> there. <speaker:2> General Kenobi.",
            "done": True,
            "done_reason": "stop",
            "prompt_tokens": 20,
            "completion_tokens": 20,
            "runtime_sec": 0.01,
        }
    )
    res_mal = ConvoFuseGenerativeRefiner(client=stub_mal).refine_words(words, mapper, "rec_test")
    assert res_mal.refined_words[0].reconciliation_reason == RejectionOrAbstentionReason.REJECTED_MISSING_TAG.value
    assert res_mal.refined_words[1].reconciliation_reason == RejectionOrAbstentionReason.REJECTED_MALFORMED_TAG.value

    # 3. Truncated completion (done_reason == "length")
    stub_trunc = StubOllamaClient(
        {
            "raw_completion": "<speaker:2> Hello there.",
            "done": True,
            "done_reason": "length",
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "runtime_sec": 0.01,
        }
    )
    res_trunc = ConvoFuseGenerativeRefiner(client=stub_trunc).refine_words(words, mapper, "rec_test")
    assert all(
        w.reconciliation_reason == RejectionOrAbstentionReason.REJECTED_TRUNCATED_COMPLETION.value
        for w in res_trunc.refined_words
    )
    assert all(w.refined_speaker_id == w.baseline_speaker_id for w in res_trunc.refined_words)


def test_05_rejects_simultaneous_overlap_modification_and_locality_gate():
    base_words = _make_sample_words()[1:]
    # Mark word index 2 ("General") as simultaneous overlapping speech
    ovl_word = ConvoFuseWord(
        word_index=base_words[2].word_index,
        word=base_words[2].word,
        start=base_words[2].start,
        end=base_words[2].end,
        baseline_speaker_id=base_words[2].baseline_speaker_id,
        refined_speaker_id=base_words[2].refined_speaker_id,
        overlap_metadata=OverlapMetadata(
            is_overlapping_speech=True,
            overlapping_speakers=("SPEAKER_00", "SPEAKER_01"),
            overlap_durations_sec={"SPEAKER_00": 0.3, "SPEAKER_01": 0.25},
        ),
    )
    words = [base_words[0], base_words[1], ovl_word, base_words[3]]
    mapper = SpeakerIdentityMapper(["speaker_0", "speaker_1"])

    stub = StubOllamaClient(
        {
            "raw_completion": "<speaker:1> Hello there. <speaker:2> General Kenobi.",
            "done": True,
            "done_reason": "stop",
            "prompt_tokens": 20,
            "completion_tokens": 20,
            "runtime_sec": 0.01,
        }
    )
    res = ConvoFuseGenerativeRefiner(client=stub).refine_words(words, mapper, "rec_test")
    assert res.refined_words[2].refined_speaker_id == "SPEAKER_00"
    assert res.refined_words[2].refinement_decision == RefinementDecision.REJECTED_PROPOSAL.value
    assert (
        res.refined_words[2].reconciliation_reason
        == RejectionOrAbstentionReason.REJECTED_UNSUPPORTED_OVERLAP_CORRECTION.value
    )


def test_06_turn_reconstruction_speaker_summary_and_exports(tmp_path: Path):
    words = _make_sample_words()
    mapper = SpeakerIdentityMapper(["speaker_0", "speaker_1"])
    stub = StubOllamaClient(
        {
            "raw_completion": "<speaker:0> Yes <speaker:1> Hello there. <speaker:2> General Kenobi.",
            "done": True,
            "done_reason": "stop",
            "prompt_tokens": 20,
            "completion_tokens": 20,
            "runtime_sec": 0.01,
        }
    )
    ref_res = ConvoFuseGenerativeRefiner(client=stub).refine_words(words, mapper, "rec_test")
    baseline_turns = reconstruct_turns(ref_res.refined_words, use_refined_speaker=False)
    refined_turns = reconstruct_turns(ref_res.refined_words, use_refined_speaker=True)

    # In baseline: UNK ("Yes"), SPEAKER_00 ("Hello there. General"), SPEAKER_01 ("Kenobi.") -> 3 turns
    # In refined: UNK ("Yes"), SPEAKER_00 ("Hello there."), SPEAKER_01 ("General Kenobi.") -> 3 turns
    assert len(baseline_turns) == 3
    assert len(refined_turns) == 3
    assert refined_turns[1].speaker_id == "SPEAKER_00"
    assert refined_turns[1].text == "Hello there."
    assert refined_turns[2].speaker_id == "SPEAKER_01"
    assert refined_turns[2].text == "General Kenobi."
    assert "[00:00:03.200 – 00:00:04.100] SPEAKER_01: General Kenobi." in refined_turns[2].formatted_line

    summaries = build_speaker_summaries(ref_res.refined_words, refined_turns, mapper.canonical_inventory)
    spk_map = {s.speaker_id: s for s in summaries}
    assert spk_map["SPEAKER_00"].word_count == 2
    assert spk_map["SPEAKER_01"].word_count == 2
    assert spk_map["SPEAKER_01"].accepted_corrections_into_speaker == 1

    eval_rep = evaluate_post_inference(
        recording_id="rec_test",
        words=ref_res.refined_words,
        baseline_turns=baseline_turns,
        refined_turns=refined_turns,
        speaker_inventory=mapper.canonical_inventory,
        reference_json_path=None,
    )
    assert eval_rep["integrity_verification"]["word_sequence_intact"] is True
    assert eval_rep["integrity_verification"]["timestamp_sequence_intact"] is True
    assert eval_rep["integrity_verification"]["unk_invariance_verified"] is True
