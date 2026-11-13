"""Stage 6 R01 Integration Runner: DiarizationLM-Inspired Post-Processing on R01 Pilot Outputs.

This script connects the Stage 6 Generative AI post-processing pipeline to the frozen
R01 pilot outputs (E20 / M0 and E21 / M4-DFN) and reference annotations in strictly
read-only mode, writing isolated Stage 6 evaluation and artifact files under
`research_stages/stage6_generative_refinement/outputs/`.

Supported execution modes:
1. `offline_abstaining` (default safe offline baseline-preserving mode):
   Uses `OfflineAbstainingLLMProvider` to verify end-to-end orchestration, context
   windowing, TPST reconciliation, and bit-exact preservation of the Stage 5B.4
   baseline diarization outputs when no external LLM API key is configured.
2. `offline_contextual_diagnostic`:
   Uses `OfflineContextualRuleDiagnosticProvider` (a deterministic non-neural rule
   diagnostic provider) to inspect isolated 1-word speaker islands and `UNK`
   boundary words.
3. `simulated_conversational_qa`:
   Uses `MockScriptedLLMProvider` with a scripted conversational Q&A response for R01's
   multi-party dialogue exchange (`[14.49s - 29.02s]`) to demonstrate and verify how
   context-aware conversational turn reassignment is reconciled and scored by the
   TPST reconciler and WDER/cpWER evaluator.
4. `openai_compatible` / `gemini`:
   Live external LLM API execution (requires user-configured environment variable
   `OPENAI_API_KEY` or `GEMINI_API_KEY`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from research_stages.stage6_generative_refinement.context_window_builder import (
    ContextWindowConfig,
)
from research_stages.stage6_generative_refinement.llm_interface import (
    BaseLLMProvider,
    LLMConfig,
    MockScriptedLLMProvider,
    create_llm_provider,
)
from research_stages.stage6_generative_refinement.pipeline import (
    Stage6PipelineResult,
    Stage6RefinementPipeline,
)
from research_stages.stage6_generative_refinement.schemas import (
    ConfidenceStatus,
    LLMRefinementRequest,
)
from research_stages.stage6_generative_refinement.tpst_reconciler import (
    ReconcilerConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ASR_WORDS_PATH = (
    REPO_ROOT / "research_stages/stage2_baseline/whisperx/R01_sara_sharif_aligned.json"
)
DEFAULT_REF_JSON_PATH = (
    REPO_ROOT / "research_stages/stage1_data_prep/annotations_json/R01_sara_sharif.json"
)
DEFAULT_RTTM_PATHS: Dict[str, Path] = {
    "E20": REPO_ROOT
    / "research_stages/stage5b/stage5b_4_fresh_diarization/outputs/E20/R01_sara_sharif.rttm",
    "E21": REPO_ROOT
    / "research_stages/stage5b/stage5b_4_fresh_diarization/outputs/E21/R01_sara_sharif.rttm",
}
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "research_stages/stage6_generative_refinement/outputs"
)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in read-only mode."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_r01_simulated_conversational_qa_provider() -> MockScriptedLLMProvider:
    """Construct a scripted conversational Q&A provider for R01 demonstration.

    In R01 (`R01_sara_sharif`), the acoustic diarization baseline (3 clusters:
    `speaker_0`, `speaker_1`, `speaker_2`) exhibits two targeted boundary phenomena
    that can be resolved within the existing 3-cluster inventory:
    1. Words 0-1 (`"Yes"`, `"sir."` at `2.052-2.377s`) have no RTTM segment within
       0.5s (`"UNK"`), and lexically match the compliant subject (`speaker_1`).
    2. Word 43 (`"Yes."` at `14.492-14.833s`) is attributed to `speaker_1` as an
       isolated 1-word island during the officer's procedural exchange (`[3.00-16.00s]`
       reference `Speaker_2`, mapped to `speaker_0`), before the bag-inquiry turn.
    3. For the bag-inquiry self-dialogue (`[16.92-19.53s]`: `"Which one is your bag?
       I don't have any bag. You have no bag?"`), ground truth involves a third
       unmapped speaker (`Speaker_3`, 5 words total) not present in the 3-cluster
       RTTM inventory (`speaker_0`, `speaker_1`, `speaker_2`); because the LLM is
       strictly forbidden from inventing new speaker IDs, it abstains on those words.

    Note: This provider is explicitly labeled as a deterministic scripted simulation of
    an instruction-following LLM analyzing R01's conversational structure, used to
    demonstrate TPST reconciliation and evaluation on R01 without requiring external
    network calls or claiming live LLM inference.
    """

    def _response_fn(request: LLMRefinementRequest) -> str:
        decisions: List[Dict[str, Any]] = []
        for w in request.words_payload:
            idx = int(w["word_index"])
            word_str = str(w["word"])
            orig_spk = str(w["current_speaker_id"])

            if idx in (0, 1) and orig_spk == "UNK":
                decisions.append(
                    {
                        "word_index": idx,
                        "word": word_str,
                        "proposed_speaker_id": "speaker_1",
                        "abstain": False,
                        "confidence": ConfidenceStatus.MEDIUM.value,
                        "reasoning": (
                            "Initial unassigned 'Yes sir.' lexically matches compliant "
                            "subject response pattern attributed to speaker_1."
                        ),
                    }
                )
            elif idx == 43:
                decisions.append(
                    {
                        "word_index": idx,
                        "word": word_str,
                        "proposed_speaker_id": "speaker_0",
                        "abstain": False,
                        "confidence": ConfidenceStatus.MEDIUM.value,
                        "reasoning": (
                            "Isolated single-word affirmative 'Yes.' at 14.49s occurs "
                            "within the officer procedural turn mapped to speaker_0."
                        ),
                    }
                )
            else:
                decisions.append(
                    {
                        "word_index": idx,
                        "word": word_str,
                        "proposed_speaker_id": orig_spk,
                        "abstain": True,
                        "confidence": ConfidenceStatus.UNCERTAIN.value,
                        "reasoning": "Abstain: preserve acoustic diarization baseline.",
                    }
                )
        return json.dumps(
            {
                "window_id": request.window_id,
                "decisions": decisions,
            }
        )

    return MockScriptedLLMProvider(
        response_fn=_response_fn,
        config=LLMConfig(
            provider="mock_scripted",
            model_name="r01-scripted-conversational-qa-simulation-v1",
        ),
    )


def run_single_condition(
    condition_name: str,
    mode_name: str,
    asr_words_path: Path,
    rttm_path: Path,
    ref_json_path: Path,
    provider: BaseLLMProvider,
    window_config: ContextWindowConfig,
    reconciler_config: ReconcilerConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the Stage 6 pipeline for one condition (E20 or E21) and one provider mode."""
    pipeline = Stage6RefinementPipeline(
        window_config=window_config,
        reconciler_config=reconciler_config,
        llm_provider=provider,
    )
    out_path = (
        output_dir / f"R01_sara_sharif__{condition_name}__{mode_name}.json"
    ).resolve()
    result: Stage6PipelineResult = pipeline.run_from_files(
        words_json_path=asr_words_path,
        diarization_rttm_path=rttm_path,
        recording_id="R01_sara_sharif",
        condition_id=f"{condition_name}__{mode_name}",
        reference_transcript_json_path=ref_json_path,
        output_json_path=out_path,
    )
    try:
        rel_or_abs = str(out_path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        rel_or_abs = str(out_path.resolve())
    return {
        "condition": condition_name,
        "mode": mode_name,
        "provider_name": provider.config.provider,
        "model_name": provider.config.model_name,
        "output_artifact": rel_or_abs,
        "output_sha256": sha256_file(out_path),
        "evaluation_report": result.evaluation_report,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage 6 DiarizationLM-inspired post-processing on R01 pilot outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Stage 6 R01 output JSON artifacts.",
    )
    parser.add_argument(
        "--live-provider",
        type=str,
        default=None,
        choices=["openai_compatible", "gemini"],
        help="Optional live external LLM provider to run in addition to offline modes.",
    )
    parser.add_argument(
        "--live-model",
        type=str,
        default=None,
        help=(
            "Optional model identifier when --live-provider is specified. "
            "Defaults to GEMINI_MODEL for Gemini or gpt-4o-mini for OpenAI-compatible."
        ),
    )
    args = parser.parse_args(argv)

    raw_output_dir = Path(args.output_dir)
    output_dir: Path = (
        (REPO_ROOT / raw_output_dir).resolve()
        if not raw_output_dir.is_absolute()
        else raw_output_dir.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Record input hashes BEFORE execution to verify read-only integrity
    input_hashes_before = {
        "asr_words": sha256_file(DEFAULT_ASR_WORDS_PATH),
        "ref_json": sha256_file(DEFAULT_REF_JSON_PATH),
        "rttm_E20": sha256_file(DEFAULT_RTTM_PATHS["E20"]),
        "rttm_E21": sha256_file(DEFAULT_RTTM_PATHS["E21"]),
    }

    window_config = ContextWindowConfig(
        window_size_words=60,
        context_overlap_words=15,
        surrounding_turns_radius=4,
        turn_grouping_max_gap_sec=2.0,
    )
    reconciler_config = ReconcilerConfig(
        strict_word_text_match=True,
        reject_window_on_extra_words=True,
        reject_window_on_reordering=True,
        min_confidence_to_accept=ConfidenceStatus.LOW.value,
        allow_reassigning_unk=True,
    )

    runs_summary: List[Dict[str, Any]] = []

    for cond_name, rttm_path in DEFAULT_RTTM_PATHS.items():
        # Mode 1: Default safe offline abstaining mode
        abstaining_provider = create_llm_provider(
            LLMConfig(
                provider="offline_abstaining",
                model_name="offline-abstaining-v1",
            )
        )
        runs_summary.append(
            run_single_condition(
                condition_name=cond_name,
                mode_name="offline_abstaining",
                asr_words_path=DEFAULT_ASR_WORDS_PATH,
                rttm_path=rttm_path,
                ref_json_path=DEFAULT_REF_JSON_PATH,
                provider=abstaining_provider,
                window_config=window_config,
                reconciler_config=reconciler_config,
                output_dir=output_dir,
            )
        )

        # Mode 2: Offline deterministic contextual rule diagnostic provider
        diagnostic_provider = create_llm_provider(
            LLMConfig(
                provider="offline_contextual_diagnostic",
                model_name="deterministic-bwc-dialogue-analyzer-v1",
            )
        )
        runs_summary.append(
            run_single_condition(
                condition_name=cond_name,
                mode_name="offline_contextual_diagnostic",
                asr_words_path=DEFAULT_ASR_WORDS_PATH,
                rttm_path=rttm_path,
                ref_json_path=DEFAULT_REF_JSON_PATH,
                provider=diagnostic_provider,
                window_config=window_config,
                reconciler_config=reconciler_config,
                output_dir=output_dir,
            )
        )

        # Mode 3: Scripted conversational Q&A demonstration on R01
        sim_provider = build_r01_simulated_conversational_qa_provider()
        runs_summary.append(
            run_single_condition(
                condition_name=cond_name,
                mode_name="simulated_conversational_qa",
                asr_words_path=DEFAULT_ASR_WORDS_PATH,
                rttm_path=rttm_path,
                ref_json_path=DEFAULT_REF_JSON_PATH,
                provider=sim_provider,
                window_config=window_config,
                reconciler_config=reconciler_config,
                output_dir=output_dir,
            )
        )

        # Optional Mode 4: Live external provider.
        # Initial live experiment is restricted to R01 E20 only.
        if args.live_provider is not None and cond_name == "E20":

            # Resolve model identifier based on the selected provider.
            if args.live_model:
                live_model = args.live_model

            elif args.live_provider == "gemini":
                live_model = os.getenv(
                    "GEMINI_MODEL",
                    "gemini-3.5-flash-lite",
                )

            else:
                live_model = "gpt-4o-mini"

            # Explicitly select the appropriate API-key environment variable.
            if args.live_provider == "gemini":
                api_key_env_var = "GEMINI_API_KEY"
            else:
                api_key_env_var = "OPENAI_API_KEY"

            live_config = LLMConfig(
                provider=args.live_provider,
                model_name=live_model,
                api_key_env_var=api_key_env_var,
            )

            live_provider = create_llm_provider(live_config)

            runs_summary.append(
                run_single_condition(
                    condition_name=cond_name,
                    mode_name=f"live_{args.live_provider}",
                    asr_words_path=DEFAULT_ASR_WORDS_PATH,
                    rttm_path=rttm_path,
                    ref_json_path=DEFAULT_REF_JSON_PATH,
                    provider=live_provider,
                    window_config=window_config,
                    reconciler_config=reconciler_config,
                    output_dir=output_dir,
                )
            )

    # Record input hashes AFTER execution to prove read-only immutability
    input_hashes_after = {
        "asr_words": sha256_file(DEFAULT_ASR_WORDS_PATH),
        "ref_json": sha256_file(DEFAULT_REF_JSON_PATH),
        "rttm_E20": sha256_file(DEFAULT_RTTM_PATHS["E20"]),
        "rttm_E21": sha256_file(DEFAULT_RTTM_PATHS["E21"]),
    }
    assert input_hashes_before == input_hashes_after, (
        "CRITICAL: Frozen R01 input hashes changed during Stage 6 execution!"
    )

    summary_payload = {
        "stage": "Stage 6 — Generative AI Integration (DiarizationLM-Inspired Post-Processing)",
        "recording_id": "R01_sara_sharif",
        "frozen_inputs": {
            "asr_words_path": str(DEFAULT_ASR_WORDS_PATH.relative_to(REPO_ROOT)),
            "reference_json_path": str(DEFAULT_REF_JSON_PATH.relative_to(REPO_ROOT)),
            "rttm_E20_path": str(DEFAULT_RTTM_PATHS["E20"].relative_to(REPO_ROOT)),
            "rttm_E21_path": str(DEFAULT_RTTM_PATHS["E21"].relative_to(REPO_ROOT)),
            "sha256_before": input_hashes_before,
            "sha256_after": input_hashes_after,
            "inputs_unmodified": input_hashes_before == input_hashes_after,
        },
        "window_config": asdict(window_config),
        "reconciler_config": asdict(reconciler_config),
        "runs": runs_summary,
    }

    summary_path = output_dir / "R01_STAGE6_EVALUATION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Stage 6 R01 evaluation complete. Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
