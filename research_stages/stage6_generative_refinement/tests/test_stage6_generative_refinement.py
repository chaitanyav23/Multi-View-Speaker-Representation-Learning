"""
Stage 6 — Phase 4: Synthetic Unit and Integration Test Suite
Tests all required components and safety invariants of the DiarizationLM-inspired
speaker attribution refinement module:
  1. Correct speaker reassignment
  2. Invalid speaker IDs
  3. Missing and extra words
  4. Reordered words
  5. Overlapping speech
  6. Missing speaker assignments ("UNK" and missing timestamps)
  7. LLM abstention
  8. Malformed JSON and schema violations
  9. Preservation of timestamps and original transcript
 10. Multi-window sliding context and core-index precedence
 11. Provider-independent LLM configuration and credential safety
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_stages.stage6_generative_refinement.context_window_builder import (
    ContextWindowBuilder,
    ContextWindowConfig,
)
from research_stages.stage6_generative_refinement.llm_interface import (
    GeminiLLMProvider,
    LLMConfig,
    LLMProviderUnavailableError,
    MockScriptedLLMProvider,
    OfflineAbstainingLLMProvider,
    OpenAICompatibleLLMProvider,
)
from research_stages.stage6_generative_refinement.pipeline import Stage6RefinementPipeline
from research_stages.stage6_generative_refinement.schemas import (
    AttributionStatus,
    ConfidenceStatus,
    ReconciliationReason,
)
from research_stages.stage6_generative_refinement.tpst_reconciler import (
    ReconcilerConfig,
    TPSTReconciler,
)
from research_stages.stage6_generative_refinement.word_orchestrator import (
    extract_available_speakers,
    orchestrate_words,
)


def _sample_words_and_diar():
    """Creates a realistic synthetic 8-word Q&A dialogue turn."""
    raw_words = [
        {"word": "Which", "start": 16.00, "end": 16.20, "confidence": 0.91},
        {"word": "one", "start": 16.22, "end": 16.40, "confidence": 0.88},
        {"word": "is", "start": 16.42, "end": 16.55, "confidence": 0.95},
        {"word": "your", "start": 16.58, "end": 16.75, "confidence": 0.90},
        {"word": "bag?", "start": 16.78, "end": 17.00, "confidence": 0.89},
        {"word": "Not", "start": 17.20, "end": 17.45, "confidence": 0.85},
        {"word": "mine", "start": 17.48, "end": 17.80, "confidence": 0.87},
        {"word": "sir.", "start": 17.82, "end": 18.10, "confidence": 0.92},
    ]
    # Acoustic diarization erroneously merges the entire [16.0, 18.2] span into speaker_0,
    # while speaker_1 is also active later in the recording.
    diar_segments = [
        {"start": 16.00, "end": 18.20, "duration": 2.20, "speaker": "speaker_0"},
        {"start": 20.00, "end": 25.00, "duration": 5.00, "speaker": "speaker_1"},
    ]
    reference_segments = [
        {"speaker": "Speaker_A", "start": 16.00, "end": 17.05, "text": "Which one is your bag?"},
        {"speaker": "Speaker_B", "start": 17.15, "end": 18.15, "text": "Not mine sir."},
        {"speaker": "Speaker_B", "start": 20.00, "end": 25.00, "text": ""},
    ]
    return raw_words, diar_segments, reference_segments


class TestStage6GenerativeRefinement(unittest.TestCase):
    """Comprehensive synthetic test suite for Stage 6 Generative AI Integration."""

    def test_01_correct_speaker_reassignment(self):
        """1. Validated LLM speaker reassignments are accepted and improve WDER/cpWER."""
        raw_words, diar_segments, ref_segments = _sample_words_and_diar()

        def response_fn(req):
            decisions = []
            for w in req.words_payload:
                idx = w["word_index"]
                # Reassign words 5, 6, 7 ("Not mine sir.") from speaker_0 to speaker_1
                if idx in (5, 6, 7):
                    decisions.append({
                        "word_index": idx,
                        "word": w["word"],
                        "proposed_speaker_id": "speaker_1",
                        "abstain": False,
                        "confidence": "HIGH",
                        "reasoning": "First-person answer to preceding question from speaker_0.",
                    })
                else:
                    decisions.append({
                        "word_index": idx,
                        "word": w["word"],
                        "proposed_speaker_id": "speaker_0",
                        "abstain": False,
                        "confidence": "HIGH",
                        "reasoning": "Question turn by speaker_0 confirmed.",
                    })
            return json.dumps({"window_id": req.window_id, "decisions": decisions})

        provider = MockScriptedLLMProvider(response_fn=response_fn)
        pipeline = Stage6RefinementPipeline(llm_provider=provider)
        res = pipeline.run_from_memory(
            raw_words=raw_words,
            diar_segments=diar_segments,
            recording_id="synth_rec",
            condition_id="synth_cond",
            reference_segments=ref_segments,
        )

        # Verify words 0..4 remain speaker_0 and words 5..7 are refined to speaker_1
        for i in range(5):
            self.assertEqual(res.refined_words[i].original_speaker_id, "speaker_0")
            self.assertEqual(res.refined_words[i].refined_speaker_id, "speaker_0")
            self.assertFalse(res.refined_words[i].correction_accepted)
            self.assertEqual(
                res.refined_words[i].reconciliation_reason,
                ReconciliationReason.UNCHANGED_ORIGINAL_CONFIRMED.value,
            )

        for i in (5, 6, 7):
            self.assertEqual(res.refined_words[i].original_speaker_id, "speaker_0")
            self.assertEqual(res.refined_words[i].refined_speaker_id, "speaker_1")
            self.assertTrue(res.refined_words[i].correction_proposed)
            self.assertTrue(res.refined_words[i].correction_accepted)
            self.assertEqual(
                res.refined_words[i].reconciliation_reason,
                ReconciliationReason.ACCEPTED_VALIDATED_CORRECTION.value,
            )

        tel = res.evaluation_report["refinement_telemetry"]
        self.assertEqual(tel["proposed_corrections_count"], 3)
        self.assertEqual(tel["accepted_corrections_count"], 3)
        self.assertEqual(tel["rejected_corrections_count"], 0)

        # Verify WDER and cpWER improved from 0.375 to 0.0
        base_wder = res.evaluation_report["baseline_metrics"]["wder_timestamp"]["wder_assigned"]
        ref_wder = res.evaluation_report["refined_metrics"]["wder_timestamp"]["wder_assigned"]
        self.assertAlmostEqual(base_wder, 3 / 8, places=4)
        self.assertAlmostEqual(ref_wder, 0.0, places=4)
        self.assertEqual(res.evaluation_report["refined_metrics"]["cpwer"], 0.0)

    def test_02_invalid_speaker_ids_rejected(self):
        """2. Invented or unknown speaker IDs are deterministically rejected."""
        raw_words, diar_segments, _ = _sample_words_and_diar()

        def response_fn(req):
            decisions = []
            for w in req.words_payload:
                idx = w["word_index"]
                # Invent non-existent speakers on indices 5, 6, 7
                bad_spk = {5: "speaker_99", 6: "Officer_1", 7: "UNK"}.get(idx, "speaker_0")
                decisions.append({
                    "word_index": idx,
                    "word": w["word"],
                    "proposed_speaker_id": bad_spk,
                    "abstain": False,
                    "confidence": "HIGH",
                    "reasoning": "Testing invalid speaker ID rejection.",
                })
            return json.dumps({"window_id": req.window_id, "decisions": decisions})

        provider = MockScriptedLLMProvider(response_fn=response_fn)
        pipeline = Stage6RefinementPipeline(llm_provider=provider)
        res = pipeline.run_from_memory(raw_words, diar_segments)

        for i in (5, 6, 7):
            self.assertEqual(res.refined_words[i].refined_speaker_id, "speaker_0")
            self.assertFalse(res.refined_words[i].correction_accepted)
            self.assertEqual(
                res.refined_words[i].reconciliation_reason,
                ReconciliationReason.REJECTED_INVALID_SPEAKER_ID.value,
            )

        tel = res.evaluation_report["refinement_telemetry"]
        self.assertEqual(tel["accepted_corrections_count"], 0)
        self.assertEqual(tel["rejected_corrections_count"], 3)

    def test_03_missing_words_rejected_and_preserved(self):
        """3a. Missing words in LLM output are never deleted and retain original speaker."""
        raw_words, diar_segments, _ = _sample_words_and_diar()

        def response_fn(req):
            # Omit words 2 ("is") and 6 ("mine") from LLM response
            decisions = []
            for w in req.words_payload:
                if w["word_index"] in (2, 6):
                    continue
                decisions.append({
                    "word_index": w["word_index"],
                    "word": w["word"],
                    "proposed_speaker_id": "speaker_1",
                    "abstain": False,
                    "confidence": "HIGH",
                    "reasoning": "Valid on non-missing words.",
                })
            return json.dumps({"window_id": req.window_id, "decisions": decisions})

        provider = MockScriptedLLMProvider(response_fn=response_fn)
        pipeline = Stage6RefinementPipeline(llm_provider=provider)
        res = pipeline.run_from_memory(raw_words, diar_segments)

        self.assertEqual(len(res.refined_words), len(raw_words))
        for missing_idx in (2, 6):
            self.assertEqual(res.refined_words[missing_idx].word, raw_words[missing_idx]["word"])
            self.assertEqual(res.refined_words[missing_idx].refined_speaker_id, "speaker_0")
            self.assertFalse(res.refined_words[missing_idx].correction_accepted)
            self.assertEqual(
                res.refined_words[missing_idx].reconciliation_reason,
                ReconciliationReason.REJECTED_MISSING_WORDS.value,
            )

        self.assertEqual(res.window_reports[0].missing_word_indices, [2, 6])

    def test_04_extra_words_rejected(self):
        """3b. Additional/hallucinated words in LLM output are rejected and never added."""
        raw_words, diar_segments, _ = _sample_words_and_diar()

        def response_fn(req):
            decisions = [
                {
                    "word_index": w["word_index"],
                    "word": w["word"],
                    "proposed_speaker_id": "speaker_1",
                    "abstain": False,
                    "confidence": "HIGH",
                    "reasoning": "Attempt reassignment.",
                }
                for w in req.words_payload
            ]
            # Append an extra hallucinated word
            decisions.append({
                "word_index": 99,
                "word": "hallucinated_extra_word",
                "proposed_speaker_id": "speaker_1",
                "abstain": False,
                "confidence": "HIGH",
                "reasoning": "Extra word.",
            })
            return json.dumps({"window_id": req.window_id, "decisions": decisions})

        provider = MockScriptedLLMProvider(response_fn=response_fn)
        pipeline = Stage6RefinementPipeline(llm_provider=provider)
        res = pipeline.run_from_memory(raw_words, diar_segments)

        self.assertEqual(len(res.refined_words), len(raw_words))
        self.assertEqual([w.word for w in res.refined_words], [w["word"] for w in raw_words])
        for w in res.refined_words:
            self.assertEqual(w.refined_speaker_id, "speaker_0")
            self.assertFalse(w.correction_accepted)
            self.assertEqual(w.reconciliation_reason, ReconciliationReason.REJECTED_EXTRA_WORDS.value)

    def test_05_reordered_words_rejected(self):
        """4. Reordered words (both index swaps and lexical swaps) are deterministically rejected."""
        raw_words, diar_segments, _ = _sample_words_and_diar()

        # Case A: Lexical swap of words 0 and 1 while keeping indices 0 and 1
        def lexical_swap_fn(req):
            decisions = [
                {
                    "word_index": w["word_index"],
                    "word": w["word"],
                    "proposed_speaker_id": "speaker_1",
                    "abstain": False,
                    "confidence": "HIGH",
                    "reasoning": "Swap test.",
                }
                for w in req.words_payload
            ]
            decisions[0]["word"], decisions[1]["word"] = decisions[1]["word"], decisions[0]["word"]
            return json.dumps({"window_id": req.window_id, "decisions": decisions})

        pipeline_a = Stage6RefinementPipeline(
            llm_provider=MockScriptedLLMProvider(response_fn=lexical_swap_fn)
        )
        res_a = pipeline_a.run_from_memory(raw_words, diar_segments)
        for w in res_a.refined_words:
            self.assertEqual(w.refined_speaker_id, "speaker_0")
            self.assertFalse(w.correction_accepted)
            self.assertEqual(
                w.reconciliation_reason,
                ReconciliationReason.REJECTED_REORDERED_OR_MUTATED_WORDS.value,
            )

        # Case B: Non-monotonic word_index ordering (e.g., index 3 before index 2)
        def index_swap_fn(req):
            decisions = [
                {
                    "word_index": w["word_index"],
                    "word": w["word"],
                    "proposed_speaker_id": "speaker_1",
                    "abstain": False,
                    "confidence": "HIGH",
                    "reasoning": "Index order swap.",
                }
                for w in req.words_payload
            ]
            decisions[2], decisions[3] = decisions[3], decisions[2]
            return json.dumps({"window_id": req.window_id, "decisions": decisions})

        pipeline_b = Stage6RefinementPipeline(
            llm_provider=MockScriptedLLMProvider(response_fn=index_swap_fn)
        )
        res_b = pipeline_b.run_from_memory(raw_words, diar_segments)
        for w in res_b.refined_words:
            self.assertEqual(w.refined_speaker_id, "speaker_0")
            self.assertFalse(w.correction_accepted)
            self.assertEqual(
                w.reconciliation_reason,
                ReconciliationReason.REJECTED_REORDERED_OR_MUTATED_WORDS.value,
            )

    def test_06_overlapping_speech_explicit_tracking(self):
        """5. Multi-speaker overlapping speech is explicitly tracked and propagated."""
        raw_words = [
            {"word": "Stop", "start": 5.00, "end": 5.50, "confidence": 0.80},
            {"word": "now", "start": 5.60, "end": 6.00, "confidence": 0.90},
        ]
        # Both speaker_0 and speaker_1 overlap [5.00, 5.50]
        diar_segments = [
            {"start": 4.80, "end": 6.20, "duration": 1.40, "speaker": "speaker_0"},
            {"start": 5.10, "end": 5.45, "duration": 0.35, "speaker": "speaker_1"},
        ]
        orchestrated = orchestrate_words(raw_words, diar_segments)
        self.assertTrue(orchestrated[0].is_overlapping_speech)
        self.assertEqual(
            orchestrated[0].attribution_status,
            AttributionStatus.MULTI_SPEAKER_OVERLAP.value,
        )
        self.assertEqual(orchestrated[0].overlapping_speakers, ("speaker_0", "speaker_1"))
        self.assertEqual(orchestrated[0].original_speaker_id, "speaker_0")  # 0.50s > 0.35s

        self.assertFalse(orchestrated[1].is_overlapping_speech)
        self.assertEqual(
            orchestrated[1].attribution_status,
            AttributionStatus.SINGLE_SPEAKER_OVERLAP.value,
        )

    def test_07_missing_speaker_assignments_and_missing_timestamps(self):
        """6. Words without segment overlap or timestamps receive explicit 'UNK' without invention."""
        raw_words = [
            {"word": "Hello", "start": 0.10, "end": 0.30, "confidence": 0.50},
            {"word": "Untimed", "start": None, "end": None, "confidence": 0.40},
            {"word": "Officer", "start": 10.00, "end": 10.50, "confidence": 0.95},
        ]
        diar_segments = [
            {"start": 10.00, "end": 11.00, "duration": 1.00, "speaker": "speaker_0"},
        ]
        orchestrated = orchestrate_words(raw_words, diar_segments, tolerance=0.5)
        self.assertEqual(orchestrated[0].original_speaker_id, "UNK")
        self.assertEqual(orchestrated[0].attribution_status, AttributionStatus.UNASSIGNED_UNK.value)

        self.assertEqual(orchestrated[1].original_speaker_id, "UNK")
        self.assertEqual(orchestrated[1].attribution_status, AttributionStatus.MISSING_TIMESTAMPS.value)

        self.assertEqual(orchestrated[2].original_speaker_id, "speaker_0")

        # Run pipeline with OfflineAbstainingLLMProvider: UNK must remain UNK
        pipeline = Stage6RefinementPipeline(llm_provider=OfflineAbstainingLLMProvider())
        res = pipeline.run_from_memory(raw_words, diar_segments)
        self.assertEqual(res.refined_words[0].refined_speaker_id, "UNK")
        self.assertEqual(res.refined_words[1].refined_speaker_id, "UNK")
        self.assertEqual(res.refined_words[2].refined_speaker_id, "speaker_0")

    def test_08_llm_abstention_preserves_baseline(self):
        """7. When LLM sets abstain=True, baseline assignment is preserved even if a speaker is proposed."""
        raw_words, diar_segments, _ = _sample_words_and_diar()

        def abstain_fn(req):
            decisions = [
                {
                    "word_index": w["word_index"],
                    "word": w["word"],
                    "proposed_speaker_id": "speaker_1",  # Different speaker, but abstain=True!
                    "abstain": True,
                    "confidence": "LOW",
                    "reasoning": "Context ambiguous; abstaining.",
                }
                for w in req.words_payload
            ]
            return json.dumps({"window_id": req.window_id, "decisions": decisions})

        pipeline = Stage6RefinementPipeline(
            llm_provider=MockScriptedLLMProvider(response_fn=abstain_fn)
        )
        res = pipeline.run_from_memory(raw_words, diar_segments)
        for w in res.refined_words:
            self.assertEqual(w.refined_speaker_id, "speaker_0")
            self.assertTrue(w.abstained)
            self.assertFalse(w.correction_accepted)
            self.assertEqual(w.reconciliation_reason, ReconciliationReason.ABSTAINED_BY_LLM.value)

        tel = res.evaluation_report["refinement_telemetry"]
        self.assertEqual(tel["abstentions_count"], len(raw_words))
        self.assertEqual(tel["accepted_corrections_count"], 0)

    def test_09_malformed_json_and_schema_violation_handling(self):
        """8. Malformed JSON or invalid response schemas fail closed and preserve baseline."""
        raw_words, diar_segments, _ = _sample_words_and_diar()

        # Subcase A: Corrupt JSON syntax
        pipeline_bad_json = Stage6RefinementPipeline(
            llm_provider=MockScriptedLLMProvider(response_fn=lambda _: "{not valid json!!!")
        )
        res_bad = pipeline_bad_json.run_from_memory(raw_words, diar_segments)
        self.assertFalse(res_bad.window_reports[0].is_valid_json)
        for w in res_bad.refined_words:
            self.assertEqual(w.refined_speaker_id, w.original_speaker_id)
            self.assertEqual(w.reconciliation_reason, ReconciliationReason.REJECTED_MALFORMED_JSON.value)

        # Subcase B: Valid JSON but missing 'decisions' array
        pipeline_bad_schema = Stage6RefinementPipeline(
            llm_provider=MockScriptedLLMProvider(response_fn=lambda _: json.dumps({"wrong_key": []}))
        )
        res_schema = pipeline_bad_schema.run_from_memory(raw_words, diar_segments)
        self.assertFalse(res_schema.window_reports[0].is_valid_schema)
        for w in res_schema.refined_words:
            self.assertEqual(w.refined_speaker_id, w.original_speaker_id)
            self.assertEqual(w.reconciliation_reason, ReconciliationReason.REJECTED_SCHEMA_VIOLATION.value)

    def test_10_preservation_of_timestamps_and_original_transcript(self):
        """9. Original words, timestamps, and baseline speakers remain 100% recoverable."""
        raw_words, diar_segments, _ = _sample_words_and_diar()

        def mutated_word_fn(req):
            decisions = []
            for w in req.words_payload:
                idx = w["word_index"]
                # Mutate word 4 from "bag?" to "luggage?"
                word_str = "luggage?" if idx == 4 else w["word"]
                decisions.append({
                    "word_index": idx,
                    "word": word_str,
                    "proposed_speaker_id": "speaker_1",
                    "abstain": False,
                    "confidence": "HIGH",
                    "reasoning": "Testing lexical mutation rejection.",
                })
            return json.dumps({"window_id": req.window_id, "decisions": decisions})

        pipeline = Stage6RefinementPipeline(
            llm_provider=MockScriptedLLMProvider(response_fn=mutated_word_fn)
        )
        res = pipeline.run_from_memory(raw_words, diar_segments)

        # Word 4 ("bag?") must NOT be overwritten with "luggage?", and its speaker change must be rejected
        self.assertEqual(res.refined_words[4].word, "bag?")
        self.assertEqual(res.refined_words[4].refined_speaker_id, "speaker_0")
        self.assertFalse(res.refined_words[4].correction_accepted)
        self.assertEqual(
            res.refined_words[4].reconciliation_reason,
            ReconciliationReason.REJECTED_WORD_TEXT_MISMATCH.value,
        )

        # Other words had valid reassignments and must still preserve exact timestamps
        self.assertTrue(res.evaluation_report["integrity_checks"]["word_sequence_integrity_verified"])
        self.assertTrue(res.evaluation_report["integrity_checks"]["timestamp_integrity_verified"])
        self.assertTrue(res.evaluation_report["integrity_checks"]["original_transcript_recoverable"])

        recovered = res.recover_original_transcript_words()
        self.assertEqual([r["word"] for r in recovered], [w["word"] for w in raw_words])
        self.assertEqual(
            [(r["start"], r["end"]) for r in recovered],
            [(w["start"], w["end"]) for w in raw_words],
        )

    def test_11_sliding_window_core_precedence(self):
        """10. Overlapping context windows assign authority to the window where a word is in core_word_indices."""
        raw_words, diar_segments, _ = _sample_words_and_diar()
        win_cfg = ContextWindowConfig(window_size_words=4, context_overlap_words=2)

        def window_specific_fn(req):
            # Window 0 proposes speaker_0 for everything; Window 1 proposes speaker_1 for its core words (4..7)
            is_win_1 = req.window_id.endswith("win_001")
            target_spk = "speaker_1" if is_win_1 else "speaker_0"
            decisions = [
                {
                    "word_index": w["word_index"],
                    "word": w["word"],
                    "proposed_speaker_id": target_spk,
                    "abstain": False,
                    "confidence": "HIGH",
                    "reasoning": f"Window decision for {req.window_id}",
                }
                for w in req.words_payload
            ]
            return json.dumps({"window_id": req.window_id, "decisions": decisions})

        pipeline = Stage6RefinementPipeline(
            window_config=win_cfg,
            llm_provider=MockScriptedLLMProvider(response_fn=window_specific_fn),
        )
        res = pipeline.run_from_memory(raw_words, diar_segments)
        self.assertEqual(len(res.windows), 2)
        self.assertEqual(res.windows[0].core_word_indices, (0, 1, 2, 3))
        self.assertEqual(res.windows[0].context_word_indices, (0, 1, 2, 3, 4, 5))
        self.assertEqual(res.windows[1].core_word_indices, (4, 5, 6, 7))
        self.assertEqual(res.windows[1].context_word_indices, (2, 3, 4, 5, 6, 7))

        # Words 0..3 governed by win_000 (speaker_0), words 4..7 governed by win_001 (speaker_1)
        for i in range(4):
            self.assertEqual(res.refined_words[i].refined_speaker_id, "speaker_0")
            self.assertTrue(res.refined_words[i].context_window_id.endswith("win_000"))
        for i in range(4, 8):
            self.assertEqual(res.refined_words[i].refined_speaker_id, "speaker_1")
            self.assertTrue(res.refined_words[i].context_window_id.endswith("win_001"))

    def test_12_external_providers_fail_cleanly_without_api_key(self):
        """11. OpenAI and Gemini providers never hardcode keys and raise explicit errors when unconfigured."""
        raw_words, diar_segments, _ = _sample_words_and_diar()
        env_backup = os.environ.pop("UNSET_TEST_API_KEY_STAGE6", None)
        try:
            openai_cfg = LLMConfig(
                provider="openai_compatible",
                model_name="gpt-4o-mini",
                api_key_env_var="UNSET_TEST_API_KEY_STAGE6",
            )
            p_openai = OpenAICompatibleLLMProvider(openai_cfg)
            pipeline_oa = Stage6RefinementPipeline(llm_provider=p_openai)
            with self.assertRaises(LLMProviderUnavailableError):
                pipeline_oa.run_from_memory(raw_words, diar_segments)

            gemini_cfg = LLMConfig(
                provider="gemini",
                model_name="gemini-2.5-flash",
                api_key_env_var="UNSET_TEST_API_KEY_STAGE6",
            )
            p_gemini = GeminiLLMProvider(gemini_cfg)
            pipeline_gm = Stage6RefinementPipeline(llm_provider=p_gemini)
            with self.assertRaises(LLMProviderUnavailableError):
                pipeline_gm.run_from_memory(raw_words, diar_segments)
        finally:
            if env_backup is not None:
                os.environ["UNSET_TEST_API_KEY_STAGE6"] = env_backup

    def test_13_r01_pilot_read_only_integration_and_integrity(self) -> None:
        """
        Verifies end-to-end read-only integration with the frozen R01 pilot outputs
        (E20 and E21), confirming bit-exact input preservation, word/timestamp SHA-256
        integrity, and expected WDER/cpWER metrics.
        """
        from pathlib import Path
        import tempfile
        from research_stages.stage6_generative_refinement.run_stage6_r01_pilot import (
            DEFAULT_ASR_WORDS_PATH,
            DEFAULT_REF_JSON_PATH,
            DEFAULT_RTTM_PATHS,
            main as run_r01_main,
            sha256_file,
        )

        before_hashes = {
            "asr": sha256_file(DEFAULT_ASR_WORDS_PATH),
            "ref": sha256_file(DEFAULT_REF_JSON_PATH),
            "e20": sha256_file(DEFAULT_RTTM_PATHS["E20"]),
            "e21": sha256_file(DEFAULT_RTTM_PATHS["E21"]),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            rc = run_r01_main(["--output-dir", tmp_dir])
            self.assertEqual(rc, 0)

            summary_file = Path(tmp_dir) / "R01_STAGE6_EVALUATION_SUMMARY.json"
            self.assertTrue(summary_file.exists())
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            self.assertTrue(summary["frozen_inputs"]["inputs_unmodified"])
            self.assertEqual(len(summary["runs"]), 6)

            for run in summary["runs"]:
                ev = run["evaluation_report"]
                ic = ev["integrity_checks"]
                self.assertTrue(ic["word_sequence_integrity_verified"])
                self.assertTrue(ic["timestamp_integrity_verified"])
                self.assertTrue(ic["original_transcript_recoverable"])
                self.assertEqual(ic["original_words_sha256"], ic["refined_words_sha256"])
                self.assertEqual(ic["original_timestamps_sha256"], ic["refined_timestamps_sha256"])

                if run["mode"] == "offline_abstaining":
                    self.assertEqual(ev["refinement_telemetry"]["abstentions_count"], 266)
                    self.assertEqual(ev["metric_deltas"]["delta_cpwer"], 0.0)
                    self.assertEqual(ev["metric_deltas"]["delta_wder_timestamp_assigned"], 0.0)
                elif run["mode"] == "simulated_conversational_qa":
                    self.assertEqual(ev["refinement_telemetry"]["proposed_corrections_count"], 3)
                    self.assertEqual(ev["refinement_telemetry"]["accepted_corrections_count"], 3)
                    self.assertLess(ev["metric_deltas"]["delta_wder_timestamp_assigned"], 0.0)
                    self.assertLess(ev["metric_deltas"]["delta_wder_lexical"], 0.0)

        after_hashes = {
            "asr": sha256_file(DEFAULT_ASR_WORDS_PATH),
            "ref": sha256_file(DEFAULT_REF_JSON_PATH),
            "e20": sha256_file(DEFAULT_RTTM_PATHS["E20"]),
            "e21": sha256_file(DEFAULT_RTTM_PATHS["E21"]),
        }
        self.assertEqual(before_hashes, after_hashes)


if __name__ == "__main__":
    unittest.main()
