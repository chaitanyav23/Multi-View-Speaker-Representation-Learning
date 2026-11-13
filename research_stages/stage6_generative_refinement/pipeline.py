"""
Stage 6 — End-to-End Generative AI Speaker Attribution Refinement Pipeline
Connects:
  A. Word-level orchestration (ASR words + Diarization RTTM -> OrchestratedWord list)
  B. Context-window builder (Sliding conversational windows + overlap metadata)
  C. Provider-independent LLM refinement
  D. Constrained parser & TPST-inspired reconciliation
  E. Baseline vs. Refined evaluation and integrity verification
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .context_window_builder import ContextWindowBuilder, ContextWindowConfig
from .evaluation import evaluate_refinement_comparison
from .llm_interface import (
    BaseLLMProvider,
    LLMConfig,
    build_llm_refinement_request,
    create_llm_provider,
)
from .schemas import (
    ContextWindow,
    OrchestratedWord,
    RefinedWordOutput,
)
from .tpst_reconciler import (
    ReconcilerConfig,
    TPSTReconciler,
    WindowReconciliationReport,
)
from .word_orchestrator import (
    extract_available_speakers,
    load_asr_words,
    orchestrate_words,
    parse_rttm_file,
)


@dataclass
class Stage6PipelineResult:
    """Complete result bundle from running the Stage 6 refinement pipeline on a recording."""
    recording_id: str
    condition_id: str
    available_speaker_ids: List[str]
    orchestrated_words: List[OrchestratedWord]
    windows: List[ContextWindow]
    refined_words: List[RefinedWordOutput]
    window_reports: List[WindowReconciliationReport]
    evaluation_report: Dict[str, Any]
    llm_provider_name: str
    llm_model_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "condition_id": self.condition_id,
            "llm_provider": self.llm_provider_name,
            "llm_model": self.llm_model_name,
            "available_speaker_ids": list(self.available_speaker_ids),
            "evaluation_summary": self.evaluation_report,
            "window_reconciliation_reports": [r.to_dict() for r in self.window_reports],
            "words": [w.to_dict() for w in self.refined_words],
        }

    def recover_original_transcript_words(self) -> List[Dict[str, Any]]:
        """Recovers the original unrefined ASR word-level transcript and baseline speakers."""
        return [
            {
                "word_index": w.word_index,
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "confidence": w.asr_confidence,
                "speaker": w.original_speaker_id,
            }
            for w in self.refined_words
        ]


class Stage6RefinementPipeline:
    """
    Isolated post-processing orchestrator for DiarizationLM-inspired speaker
    attribution refinement. Never modifies input files or upstream artifacts.
    """

    def __init__(
        self,
        window_config: ContextWindowConfig | None = None,
        reconciler_config: ReconcilerConfig | None = None,
        llm_provider: BaseLLMProvider | None = None,
        llm_config: LLMConfig | None = None,
        overlap_tolerance_sec: float = 0.5,
    ) -> None:
        self.window_builder = ContextWindowBuilder(window_config)
        self.reconciler = TPSTReconciler(reconciler_config)
        if llm_provider is not None:
            self.llm_provider = llm_provider
        else:
            self.llm_provider = create_llm_provider(llm_config or LLMConfig())
        self.overlap_tolerance_sec = overlap_tolerance_sec

    def run_from_memory(
        self,
        raw_words: Sequence[Dict[str, Any]],
        diar_segments: Sequence[Dict[str, Any]],
        recording_id: str = "unknown_recording",
        condition_id: str = "default",
        reference_segments: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Stage6PipelineResult:
        """
        Executes the full Stage 6 refinement pipeline on in-memory word and RTTM lists.
        """
        # Step A: Word-level orchestration
        available_speakers = extract_available_speakers(diar_segments)
        orchestrated = orchestrate_words(
            raw_words=raw_words,
            diar_segments=diar_segments,
            tolerance=self.overlap_tolerance_sec,
        )

        # Step B: Context-window construction
        windows = self.window_builder.build_windows(
            words=orchestrated,
            available_speaker_ids=available_speakers,
            recording_id=recording_id,
            condition_id=condition_id,
        )

        # Step C: LLM refinement per window
        raw_outputs_by_window: Dict[str, str] = {}
        for win in windows:
            req = build_llm_refinement_request(win)
            raw_outputs_by_window[win.window_id] = self.llm_provider.generate_refinement_json(req)

        # Step D: Constrained parsing & TPST reconciliation
        refined_words, window_reports = self.reconciler.reconcile_all_windows(
            original_words=orchestrated,
            windows=windows,
            raw_outputs_by_window=raw_outputs_by_window,
        )

        # Step E: Evaluation & comparison
        eval_report = evaluate_refinement_comparison(
            original_words=orchestrated,
            refined_words=refined_words,
            reference_segments=reference_segments,
        )

        return Stage6PipelineResult(
            recording_id=recording_id,
            condition_id=condition_id,
            available_speaker_ids=list(available_speakers),
            orchestrated_words=orchestrated,
            windows=windows,
            refined_words=refined_words,
            window_reports=window_reports,
            evaluation_report=eval_report,
            llm_provider_name=self.llm_provider.config.provider,
            llm_model_name=self.llm_provider.config.model_name,
        )

    def run_from_files(
        self,
        words_json_path: Path,
        diarization_rttm_path: Path,
        recording_id: str,
        condition_id: str,
        reference_transcript_json_path: Optional[Path] = None,
        output_json_path: Optional[Path] = None,
    ) -> Stage6PipelineResult:
        """
        Loads frozen ASR words, diarization RTTM, and optional reference transcript
        read-only from disk, runs the refinement pipeline, and optionally writes the
        isolated Stage 6 output JSON.
        """
        raw_words = load_asr_words(words_json_path)
        diar_segments = parse_rttm_file(diarization_rttm_path)

        ref_segments: Optional[List[Dict[str, Any]]] = None
        if reference_transcript_json_path is not None:
            ref_p = Path(reference_transcript_json_path)
            if not ref_p.exists():
                raise FileNotFoundError(f"Reference transcript JSON not found: {ref_p}")
            with ref_p.open("r", encoding="utf-8") as f:
                ref_data = json.load(f)
            if isinstance(ref_data, dict):
                ref_segments = list(ref_data.get("segments", []))
            elif isinstance(ref_data, list):
                ref_segments = list(ref_data)

        result = self.run_from_memory(
            raw_words=raw_words,
            diar_segments=diar_segments,
            recording_id=recording_id,
            condition_id=condition_id,
            reference_segments=ref_segments,
        )

        if output_json_path is not None:
            out_p = Path(output_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        return result
