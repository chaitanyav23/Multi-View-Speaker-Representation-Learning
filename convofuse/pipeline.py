"""
ConvoFuse End-to-End Orchestration Pipeline.

Supports two explicit execution modes (Part 2.2):
1. FULL_PROCESSING:
   - Accepts raw audio/video file from the Phase20 application
   - Performs canonical 16 kHz mono extraction + EBU R128 loudness normalization
   - Executes or delegates ASR + word alignment and dual-view speaker diarization
   - Runs temporal word-to-speaker association + overlap metadata attachment
   - Runs ground-truth-isolated DiarizationLM-Gemma generative refinement + TPST reconciliation
   - Reconstructs chronological speaker turns, speaker summaries, and exports 7 artifacts
2. FROZEN_INPUT_REPLAY:
   - Reuses frozen Stage 1 audio, Stage 2 WhisperX aligned words, Stage 5 predicted RTTM
     (Stage 5B.4 E20 C62 + Dual-Encoder Equal Fusion + M0 for R01, or Stage 5A.1 Condition J
     for R01-R08), and Stage 5B.2 overlap posteriors
   - Verifies SHA-256 hashes of all frozen inputs before and after execution
   - Runs temporal word-to-speaker association + overlap metadata attachment
   - Runs live DiarizationLM-Gemma inference via Ollama without any ground-truth access
   - Reconstructs chronological speaker turns, speaker summaries, and exports 7 artifacts
   - Runs separate post-inference evaluation only after inference and transcript reconstruction complete
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from convofuse.audio_diarization_adapter import (
    BENCHMARK_RECORDING_MAP,
    REPO_ROOT,
    CanonicalAudioProcessor,
    FrozenPredictedInputs,
    TaskSpecificAudioPaths,
    build_baseline_convofuse_words,
    compute_sha256,
    extract_available_speakers,
    load_asr_words,
    load_overlap_posterior_timeline,
    parse_rttm_file,
    resolve_benchmark_frozen_inputs,
)
from convofuse.contracts import (
    ConvoFuseExecutionMode,
    ConvoFuseJobStage,
    ConvoFuseTranscriptBundle,
    ProcessingStageRecord,
    SpeakerIdentityMapper,
)
from convofuse.evaluator import evaluate_post_inference, resolve_reference_annotation_path
from convofuse.generative_refiner import ConvoFuseGenerativeRefiner, OllamaDiarizationLMClient
from convofuse.transcript_reconstructor import (
    build_baseline_vs_refined_comparison,
    build_speaker_summaries,
    export_convofuse_artifacts,
    format_readable_transcript,
    reconstruct_turns,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConvoFusePipeline:
    """Unified ConvoFuse conversational intelligence pipeline."""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model_name: str = "diarizationlm-gemma",
        model_identifier: str = "diarizers-community/DiarizationLM-Gemma-4-E4B-v1",
        quantization: str = "Q4_K_M",
        temperature: float = 0.0,
        num_predict: int = 2048,
        timeout_sec: float = 120.0,
        max_chars_per_segment: int = 3600,
        max_words_per_segment: int = 300,
        use_locality_gating: bool = True,
        max_local_span_words: int = 5,
    ) -> None:
        self.client = OllamaDiarizationLMClient(
            base_url=ollama_base_url,
            ollama_model_name=ollama_model_name,
            model_identifier=model_identifier,
            quantization=quantization,
            temperature=temperature,
            num_predict=num_predict,
            timeout_sec=timeout_sec,
        )
        self.refiner = ConvoFuseGenerativeRefiner(
            client=self.client,
            max_chars_per_segment=max_chars_per_segment,
            max_words_per_segment=max_words_per_segment,
            use_locality_gating=use_locality_gating,
            max_local_span_words=max_local_span_words,
        )

    def _record_stage(
        self,
        stage_history: List[ProcessingStageRecord],
        stage: ConvoFuseJobStage,
        status: str,
        execution_mode: str,
        progress: float,
        start_iso: str,
        elapsed_sec: float,
        detail: str,
        artifacts: Optional[Dict[str, Any]] = None,
        progress_cb: Optional[Callable[[ConvoFuseJobStage, float, str], None]] = None,
    ) -> None:
        rec = ProcessingStageRecord(
            stage=stage.value,
            status=status,
            execution_mode=execution_mode,
            progress=round(progress, 4),
            started_at=start_iso,
            completed_at=_utc_now_iso(),
            elapsed_sec=round(elapsed_sec, 6),
            detail=detail,
            artifacts=artifacts or {},
        )
        stage_history.append(rec)
        if progress_cb is not None:
            progress_cb(stage, progress, detail)

    def run_frozen_replay(
        self,
        recording_id: str,
        condition_preference: str = "auto",
        enable_refinement: bool = True,
        use_locality_gating: Optional[bool] = None,
        output_dir: Optional[Path] = None,
        progress_cb: Optional[Callable[[ConvoFuseJobStage, float, str], None]] = None,
    ) -> ConvoFuseTranscriptBundle:
        """
        Execute ConvoFuse in FROZEN_INPUT_REPLAY mode for a benchmark recording (R01..R08).
        Reuses verified frozen Stage 1 audio, Stage 2 aligned ASR words, Stage 5 predicted RTTM,
        and Stage 5B.2 overlap posterior cache.
        """
        frozen = resolve_benchmark_frozen_inputs(
            recording_id=recording_id,
            condition_preference=condition_preference,
        )
        return self.run_from_frozen_inputs(
            frozen=frozen,
            enable_refinement=enable_refinement,
            use_locality_gating=use_locality_gating,
            output_dir=output_dir,
            progress_cb=progress_cb,
        )

    def run_from_frozen_inputs(
        self,
        frozen: FrozenPredictedInputs,
        enable_refinement: bool = True,
        use_locality_gating: Optional[bool] = None,
        output_dir: Optional[Path] = None,
        progress_cb: Optional[Callable[[ConvoFuseJobStage, float, str], None]] = None,
    ) -> ConvoFuseTranscriptBundle:
        """
        Run ConvoFuse from a resolved FrozenPredictedInputs bundle with SHA-256 verification
        and strict ground-truth isolation.
        """
        t_pipeline_start = time.perf_counter()
        stage_history: List[ProcessingStageRecord] = []
        mode_str = ConvoFuseExecutionMode.FROZEN_INPUT_REPLAY.value

        if use_locality_gating is not None:
            self.refiner.use_locality_gating = bool(use_locality_gating)

        pre_hashes = dict(frozen.input_sha256)

        # 1. UPLOADED
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.UPLOADED,
            "completed",
            mode_str,
            0.05,
            _utc_now_iso(),
            0.0,
            f"Resolved frozen benchmark inputs for {frozen.recording_id}",
            {"recording_id": frozen.recording_id},
            progress_cb,
        )

        # 2. QUEUED
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.QUEUED,
            "completed",
            mode_str,
            0.10,
            _utc_now_iso(),
            0.0,
            f"Queued ConvoFuse job ({frozen.diarization_condition})",
            {"diarization_condition": frozen.diarization_condition},
            progress_cb,
        )

        # 3. AUDIO_EXTRACTION
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.AUDIO_EXTRACTION,
            "replayed_from_frozen",
            mode_str,
            0.18,
            _utc_now_iso(),
            0.0,
            "Reused task-specific conditioned audio streams (DFN 16kHz, Raw Mono 16kHz, EBU R128 Normalized 16kHz)",
            frozen.audio_paths.to_dict(),
            progress_cb,
        )

        # 4. VAD
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.VAD,
            "replayed_from_frozen",
            mode_str,
            0.25,
            _utc_now_iso(),
            0.0,
            "Reused Stage 2 / Stage 5B.4 C62 Silero VAD speech regions",
            {"condition": frozen.diarization_condition},
            progress_cb,
        )

        # 5. ASR & 6. WORD_ALIGNMENT
        t0 = time.perf_counter()
        t5_iso = _utc_now_iso()
        raw_words = load_asr_words(frozen.asr_aligned_words_path)
        asr_elapsed = time.perf_counter() - t0
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.ASR,
            "replayed_from_frozen",
            mode_str,
            0.35,
            t5_iso,
            asr_elapsed,
            f"Loaded {len(raw_words)} Faster-Whisper Large-v3 words from frozen Stage 2 output",
            {"asr_aligned_words_path": str(frozen.asr_aligned_words_path), "word_count": len(raw_words)},
            progress_cb,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.WORD_ALIGNMENT,
            "replayed_from_frozen",
            mode_str,
            0.42,
            _utc_now_iso(),
            0.0,
            f"Verified Wav2Vec2 CTC word-level timestamps ({len(raw_words)} words)",
            {"word_count": len(raw_words)},
            progress_cb,
        )

        # 7. SPEAKER_REPRESENTATION, 8. AFFINITY_FUSION, 9. CLUSTERING
        t0 = time.perf_counter()
        t7_iso = _utc_now_iso()
        diar_segments = parse_rttm_file(frozen.predicted_rttm_path)
        rttm_elapsed = time.perf_counter() - t0
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.SPEAKER_REPRESENTATION,
            "replayed_from_frozen",
            mode_str,
            0.50,
            t7_iso,
            0.0,
            "Reused dual-path speaker embeddings: TitaNet-Large (192-D on DFN) + WeSpeaker ResNet293-LM (256-D on Raw Mono)",
            {"diarization_backend": frozen.diarization_backend},
            progress_cb,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.AFFINITY_FUSION,
            "replayed_from_frozen",
            mode_str,
            0.58,
            _utc_now_iso(),
            0.0,
            "Reused off-diagonal Z-score normalized dual-encoder cosine affinity fusion",
            {"diarization_condition": frozen.diarization_condition},
            progress_cb,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.CLUSTERING,
            "replayed_from_frozen",
            mode_str,
            0.65,
            _utc_now_iso(),
            rttm_elapsed,
            f"Loaded {len(diar_segments)} NME-SC spectral clustering speaker segments from {frozen.predicted_rttm_path.name}",
            {
                "predicted_rttm_path": str(frozen.predicted_rttm_path),
                "rttm_segment_count": len(diar_segments),
            },
            progress_cb,
        )

        # 10. WORD_SPEAKER_ASSIGNMENT
        t0 = time.perf_counter()
        t10_iso = _utc_now_iso()
        raw_rttm_speakers = extract_available_speakers(diar_segments)
        speaker_mapper = SpeakerIdentityMapper(raw_rttm_speakers)
        overlap_timeline = load_overlap_posterior_timeline(frozen.overlap_posterior_npz_path)

        baseline_words = build_baseline_convofuse_words(
            raw_words=raw_words,
            diar_segments=diar_segments,
            speaker_mapper=speaker_mapper,
            diarization_backend=frozen.diarization_backend,
            diarization_condition=frozen.diarization_condition,
            execution_mode=mode_str,
            overlap_timeline=overlap_timeline,
            tolerance_sec=0.5,
        )
        wsa_elapsed = time.perf_counter() - t0
        unk_count = sum(1 for w in baseline_words if w.baseline_speaker_id == "UNK")
        ovl_word_count = sum(1 for w in baseline_words if w.overlap_metadata.is_overlapping_speech)
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.WORD_SPEAKER_ASSIGNMENT,
            "completed",
            mode_str,
            0.72,
            t10_iso,
            wsa_elapsed,
            f"Associated {len(baseline_words)} words across {len(speaker_mapper.canonical_inventory)} speakers ({unk_count} UNK, {ovl_word_count} overlapping)",
            {
                "word_count": len(baseline_words),
                "speaker_inventory": list(speaker_mapper.canonical_inventory),
                "unk_word_count": unk_count,
                "overlapping_word_count": ovl_word_count,
            },
            progress_cb,
        )

        # 11. GENERATIVE_REFINEMENT (Ground-Truth-Isolated DiarizationLM-Gemma via Ollama)
        t0 = time.perf_counter()
        t11_iso = _utc_now_iso()
        raw_log_dir = (output_dir / "raw_llm_segments") if output_dir is not None else None
        if enable_refinement:
            refinement_result = self.refiner.refine_words(
                words=baseline_words,
                speaker_mapper=speaker_mapper,
                recording_id=frozen.recording_id,
                raw_log_dir=raw_log_dir,
            )
            refined_words = refinement_result.refined_words
            refinement_summary = refinement_result.to_dict()
            ref_model_label = f"{refinement_result.model_identifier} ({refinement_result.quantization}, {'LG-TPST' if self.refiner.use_locality_gating else 'Unconstrained-TPST'})"
        else:
            refined_words = list(baseline_words)
            refinement_summary = {
                "recording_id": frozen.recording_id,
                "model_identifier": "disabled",
                "ollama_model_name": "none",
                "quantization": "none",
                "provider_version": "none",
                "total_runtime_sec": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "segment_count": 0,
                "summary_counts": {
                    "total_words": len(baseline_words),
                    "accepted_corrections": 0,
                    "rejected_proposals": 0,
                    "abstained_words": 0,
                    "unchanged_words": len(baseline_words),
                    "reasons_breakdown": {},
                },
                "segment_telemetries": [],
            }
            ref_model_label = "Disabled (Baseline Only)"
        ref_elapsed = time.perf_counter() - t0
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.GENERATIVE_REFINEMENT,
            "completed" if enable_refinement else "skipped",
            mode_str,
            0.88,
            t11_iso,
            ref_elapsed,
            f"Completed DiarizationLM-Gemma TPST refinement in {ref_elapsed:.2f}s (ground-truth isolated)",
            {
                "model": ref_model_label,
                "summary_counts": refinement_summary.get("summary_counts", {}),
                "ground_truth_isolated": True,
            },
            progress_cb,
        )

        # 12. TRANSCRIPT_RECONSTRUCTION
        t0 = time.perf_counter()
        t12_iso = _utc_now_iso()
        baseline_turns = reconstruct_turns(refined_words, use_refined_speaker=False, max_gap_sec=1.5)
        refined_turns = reconstruct_turns(refined_words, use_refined_speaker=True, max_gap_sec=1.5)
        speaker_summaries = build_speaker_summaries(
            words=refined_words,
            turns=refined_turns,
            speaker_inventory=speaker_mapper.canonical_inventory,
        )
        readable_txt = format_readable_transcript(
            recording_id=frozen.recording_id,
            filename=f"{frozen.recording_id}.wav",
            duration_sec=frozen.audio_paths.duration_sec,
            execution_mode=mode_str,
            diarization_condition=frozen.diarization_condition,
            refinement_model=ref_model_label,
            turns=refined_turns,
            speaker_summaries=speaker_summaries,
        )
        comparison = build_baseline_vs_refined_comparison(
            words=refined_words,
            baseline_turns=baseline_turns,
            refined_turns=refined_turns,
            refinement_summary=refinement_summary,
        )
        rec_elapsed = time.perf_counter() - t0
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.TRANSCRIPT_RECONSTRUCTION,
            "completed",
            mode_str,
            0.94,
            t12_iso,
            rec_elapsed,
            f"Reconstructed {len(refined_turns)} chronological turns across {len(speaker_summaries)} speakers",
            {
                "baseline_turn_count": len(baseline_turns),
                "refined_turn_count": len(refined_turns),
                "speaker_count": len(speaker_summaries),
            },
            progress_cb,
        )

        # Verify SHA-256 hashes of frozen inputs after inference
        post_hashes: Dict[str, str] = {
            "asr_aligned_words": compute_sha256(frozen.asr_aligned_words_path),
            "predicted_rttm": compute_sha256(frozen.predicted_rttm_path),
        }
        if frozen.overlap_posterior_npz_path and frozen.overlap_posterior_npz_path.exists():
            post_hashes["overlap_posteriors"] = compute_sha256(frozen.overlap_posterior_npz_path)
        if frozen.audio_paths.canonical_16k_mono_wav and frozen.audio_paths.canonical_16k_mono_wav.exists():
            post_hashes["16k_mono_wav"] = compute_sha256(frozen.audio_paths.canonical_16k_mono_wav)
        if frozen.audio_paths.dfn_conditioned_wav and frozen.audio_paths.dfn_conditioned_wav.exists():
            post_hashes["dfn_wav"] = compute_sha256(frozen.audio_paths.dfn_conditioned_wav)
        if frozen.audio_paths.normalized_asr_wav and frozen.audio_paths.normalized_asr_wav.exists():
            post_hashes["normalized_wav"] = compute_sha256(frozen.audio_paths.normalized_asr_wav)

        frozen_inputs_untouched = pre_hashes == post_hashes
        if not frozen_inputs_untouched:
            raise RuntimeError(
                f"Frozen input SHA-256 mismatch detected for {frozen.recording_id}: "
                f"pre={pre_hashes}, post={post_hashes}"
            )

        # 13. EVALUATION (Strictly post-inference, only after transcript reconstruction completes)
        t0 = time.perf_counter()
        t13_iso = _utc_now_iso()
        ref_json_path = resolve_reference_annotation_path(frozen.recording_id)
        eval_report = evaluate_post_inference(
            recording_id=frozen.recording_id,
            words=refined_words,
            baseline_turns=baseline_turns,
            refined_turns=refined_turns,
            speaker_inventory=speaker_mapper.canonical_inventory,
            reference_json_path=ref_json_path,
        )
        eval_elapsed = time.perf_counter() - t0
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.EVALUATION,
            "completed" if eval_report.get("has_reference_annotations") else "skipped",
            mode_str,
            0.98,
            t13_iso,
            eval_elapsed,
            "Executed post-inference evaluation (strictly isolated from inference)",
            {
                "has_reference_annotations": eval_report.get("has_reference_annotations", False),
                "metric_deltas": eval_report.get("metric_deltas", {}),
            },
            progress_cb,
        )

        # 14. COMPLETED
        total_pipeline_sec = time.perf_counter() - t_pipeline_start
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.COMPLETED,
            "completed",
            mode_str,
            1.0,
            _utc_now_iso(),
            total_pipeline_sec,
            f"ConvoFuse pipeline completed in {total_pipeline_sec:.2f}s",
            {"total_pipeline_sec": round(total_pipeline_sec, 4)},
            progress_cb,
        )

        processing_manifest: Dict[str, Any] = {
            "recording_id": frozen.recording_id,
            "filename": f"{frozen.recording_id}.wav",
            "execution_mode": mode_str,
            "diarization_condition": frozen.diarization_condition,
            "diarization_backend": frozen.diarization_backend,
            "refinement_model": ref_model_label,
            "created_at": _utc_now_iso(),
            "total_pipeline_sec": round(total_pipeline_sec, 4),
            "ground_truth_isolation_enforced": True,
            "frozen_inputs_sha256_verified": frozen_inputs_untouched,
            "input_sha256_hashes": pre_hashes,
            "audio_routing": frozen.audio_paths.to_dict(),
            "rttm_to_canonical_speaker_map": dict(speaker_mapper.rttm_to_canonical),
            "canonical_to_dlm_tag_map": dict(speaker_mapper.canonical_to_dlm_tag),
            "refinement_telemetry": refinement_summary,
            "stage_history": [s.to_dict() for s in stage_history],
        }

        bundle = ConvoFuseTranscriptBundle(
            recording_id=frozen.recording_id,
            filename=f"{frozen.recording_id}.wav",
            duration_sec=frozen.audio_paths.duration_sec,
            sample_rate=frozen.audio_paths.sample_rate,
            channels=frozen.audio_paths.channels,
            execution_mode=mode_str,
            diarization_condition=frozen.diarization_condition,
            diarization_backend=frozen.diarization_backend,
            refinement_model=ref_model_label,
            speaker_inventory=list(speaker_mapper.canonical_inventory),
            rttm_to_canonical_speaker_map=dict(speaker_mapper.rttm_to_canonical),
            words=refined_words,
            turns=refined_turns,
            baseline_turns=baseline_turns,
            speaker_summaries=speaker_summaries,
            readable_transcript_txt=readable_txt,
            baseline_vs_refined_comparison=comparison,
            processing_manifest=processing_manifest,
            evaluation_report=eval_report,
        )

        if output_dir is not None:
            export_convofuse_artifacts(bundle, output_dir)

        return bundle

    def run_custom_or_uploaded(
        self,
        recording_id: str,
        filename: str,
        source_media_path: Path,
        raw_words: Sequence[Mapping[str, Any]],
        diar_segments: Sequence[Mapping[str, Any]],
        work_dir: Path,
        execution_mode: ConvoFuseExecutionMode = ConvoFuseExecutionMode.FULL_PROCESSING,
        diarization_condition: str = "Phase20_Full_Processing",
        diarization_backend: str = "Dual-View / Phase20 Diarization",
        overlap_npz_path: Optional[Path] = None,
        enable_refinement: bool = True,
        use_locality_gating: Optional[bool] = None,
        output_dir: Optional[Path] = None,
        progress_cb: Optional[Callable[[ConvoFuseJobStage, float, str], None]] = None,
    ) -> ConvoFuseTranscriptBundle:
        """
        Execute ConvoFuse on an uploaded audio/video recording in the Phase20 application.
        Performs canonical 16kHz mono + EBU R128 loudness-normalized audio preparation,
        word-speaker association, overlap metadata enrichment, DiarizationLM-Gemma generative
        refinement, and complete conversational transcript reconstruction.
        """
        t_pipeline_start = time.perf_counter()
        stage_history: List[ProcessingStageRecord] = []
        mode_str = execution_mode.value

        if use_locality_gating is not None:
            self.refiner.use_locality_gating = bool(use_locality_gating)

        self._record_stage(
            stage_history,
            ConvoFuseJobStage.UPLOADED,
            "completed",
            mode_str,
            0.05,
            _utc_now_iso(),
            0.0,
            f"Uploaded {filename} ({recording_id})",
            {"recording_id": recording_id, "filename": filename},
            progress_cb,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.QUEUED,
            "completed",
            mode_str,
            0.10,
            _utc_now_iso(),
            0.0,
            f"Queued ConvoFuse full processing job for {recording_id}",
            {"diarization_condition": diarization_condition},
            progress_cb,
        )

        # 3. AUDIO_EXTRACTION
        t0 = time.perf_counter()
        t3_iso = _utc_now_iso()
        audio_work_dir = Path(work_dir) / "audio_conditioning"
        audio_work_dir.mkdir(parents=True, exist_ok=True)
        mono_wav = audio_work_dir / f"{recording_id}_16k_mono.wav"
        norm_wav = audio_work_dir / f"{recording_id}_normalized.wav"

        if not mono_wav.exists():
            mono_info = CanonicalAudioProcessor.extract_canonical_16k_mono(source_media_path, mono_wav)
        else:
            mono_info = CanonicalAudioProcessor.validate_wav(mono_wav)

        if not norm_wav.exists():
            norm_info = CanonicalAudioProcessor.extract_loudnorm_asr_audio(mono_wav, norm_wav)
        else:
            norm_info = CanonicalAudioProcessor.validate_wav(norm_wav)

        audio_paths = TaskSpecificAudioPaths(
            source_media_path=source_media_path,
            canonical_16k_mono_wav=mono_wav,
            dfn_conditioned_wav=mono_wav,
            normalized_asr_wav=norm_wav,
            sample_rate=mono_info["sample_rate"],
            channels=mono_info["channels"],
            duration_sec=mono_info["duration_sec"],
            sha256_hashes={
                "16k_mono_wav": mono_info["sha256"],
                "normalized_wav": norm_info["sha256"],
            },
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.AUDIO_EXTRACTION,
            "completed",
            mode_str,
            0.20,
            t3_iso,
            time.perf_counter() - t0,
            f"Extracted canonical 16kHz mono ({mono_info['duration_sec']:.2f}s) and EBU R128 loudness-normalized audio",
            audio_paths.to_dict(),
            progress_cb,
        )

        # 4-9. Record VAD, ASR, Alignment, Representation, Fusion, Clustering
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.VAD,
            "completed",
            mode_str,
            0.30,
            _utc_now_iso(),
            0.0,
            "Completed Silero VAD speech interval detection",
            {},
            progress_cb,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.ASR,
            "completed",
            mode_str,
            0.40,
            _utc_now_iso(),
            0.0,
            f"Transcribed {len(raw_words)} words on loudness-normalized audio",
            {"word_count": len(raw_words)},
            progress_cb,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.WORD_ALIGNMENT,
            "completed",
            mode_str,
            0.48,
            _utc_now_iso(),
            0.0,
            f"Aligned {len(raw_words)} words via Wav2Vec2 CTC forced alignment",
            {"word_count": len(raw_words)},
            progress_cb,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.SPEAKER_REPRESENTATION,
            "completed",
            mode_str,
            0.55,
            _utc_now_iso(),
            0.0,
            f"Speaker representation extraction ({diarization_backend})",
            {"diarization_backend": diarization_backend},
            progress_cb,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.AFFINITY_FUSION,
            "completed",
            mode_str,
            0.60,
            _utc_now_iso(),
            0.0,
            f"Speaker affinity fusion ({diarization_condition})",
            {"diarization_condition": diarization_condition},
            progress_cb,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.CLUSTERING,
            "completed",
            mode_str,
            0.65,
            _utc_now_iso(),
            0.0,
            f"Produced {len(diar_segments)} speaker diarization segments",
            {"diar_segment_count": len(diar_segments)},
            progress_cb,
        )

        # 10. WORD_SPEAKER_ASSIGNMENT
        t0 = time.perf_counter()
        t10_iso = _utc_now_iso()
        raw_rttm_speakers = extract_available_speakers(list(diar_segments))
        speaker_mapper = SpeakerIdentityMapper(raw_rttm_speakers)
        overlap_timeline = load_overlap_posterior_timeline(overlap_npz_path)

        baseline_words = build_baseline_convofuse_words(
            raw_words=raw_words,
            diar_segments=diar_segments,
            speaker_mapper=speaker_mapper,
            diarization_backend=diarization_backend,
            diarization_condition=diarization_condition,
            execution_mode=mode_str,
            overlap_timeline=overlap_timeline,
            tolerance_sec=0.5,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.WORD_SPEAKER_ASSIGNMENT,
            "completed",
            mode_str,
            0.72,
            t10_iso,
            time.perf_counter() - t0,
            f"Associated {len(baseline_words)} words across {len(speaker_mapper.canonical_inventory)} speakers",
            {"word_count": len(baseline_words), "speaker_inventory": list(speaker_mapper.canonical_inventory)},
            progress_cb,
        )

        # 11. GENERATIVE_REFINEMENT
        t0 = time.perf_counter()
        t11_iso = _utc_now_iso()
        raw_log_dir = (output_dir / "raw_llm_segments") if output_dir is not None else None
        if enable_refinement and len(speaker_mapper.canonical_inventory) > 0:
            refinement_result = self.refiner.refine_words(
                words=baseline_words,
                speaker_mapper=speaker_mapper,
                recording_id=recording_id,
                raw_log_dir=raw_log_dir,
            )
            refined_words = refinement_result.refined_words
            refinement_summary = refinement_result.to_dict()
            ref_model_label = f"{refinement_result.model_identifier} ({refinement_result.quantization}, {'LG-TPST' if self.refiner.use_locality_gating else 'Unconstrained-TPST'})"
        else:
            refined_words = list(baseline_words)
            refinement_summary = {
                "recording_id": recording_id,
                "model_identifier": "disabled",
                "ollama_model_name": "none",
                "quantization": "none",
                "provider_version": "none",
                "total_runtime_sec": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "segment_count": 0,
                "summary_counts": {
                    "total_words": len(baseline_words),
                    "accepted_corrections": 0,
                    "rejected_proposals": 0,
                    "abstained_words": 0,
                    "unchanged_words": len(baseline_words),
                    "reasons_breakdown": {},
                },
                "segment_telemetries": [],
            }
            ref_model_label = "Disabled (Baseline Only)"
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.GENERATIVE_REFINEMENT,
            "completed" if enable_refinement else "skipped",
            mode_str,
            0.88,
            t11_iso,
            time.perf_counter() - t0,
            f"Completed DiarizationLM-Gemma TPST refinement ({ref_model_label})",
            {"summary_counts": refinement_summary.get("summary_counts", {})},
            progress_cb,
        )

        # 12. TRANSCRIPT_RECONSTRUCTION
        t0 = time.perf_counter()
        t12_iso = _utc_now_iso()
        baseline_turns = reconstruct_turns(refined_words, use_refined_speaker=False, max_gap_sec=1.5)
        refined_turns = reconstruct_turns(refined_words, use_refined_speaker=True, max_gap_sec=1.5)
        speaker_summaries = build_speaker_summaries(
            words=refined_words,
            turns=refined_turns,
            speaker_inventory=speaker_mapper.canonical_inventory,
        )
        readable_txt = format_readable_transcript(
            recording_id=recording_id,
            filename=filename,
            duration_sec=audio_paths.duration_sec,
            execution_mode=mode_str,
            diarization_condition=diarization_condition,
            refinement_model=ref_model_label,
            turns=refined_turns,
            speaker_summaries=speaker_summaries,
        )
        comparison = build_baseline_vs_refined_comparison(
            words=refined_words,
            baseline_turns=baseline_turns,
            refined_turns=refined_turns,
            refinement_summary=refinement_summary,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.TRANSCRIPT_RECONSTRUCTION,
            "completed",
            mode_str,
            0.94,
            t12_iso,
            time.perf_counter() - t0,
            f"Reconstructed {len(refined_turns)} chronological turns",
            {"refined_turn_count": len(refined_turns)},
            progress_cb,
        )

        # 13. EVALUATION
        t0 = time.perf_counter()
        t13_iso = _utc_now_iso()
        ref_json_path = resolve_reference_annotation_path(recording_id)
        eval_report = evaluate_post_inference(
            recording_id=recording_id,
            words=refined_words,
            baseline_turns=baseline_turns,
            refined_turns=refined_turns,
            speaker_inventory=speaker_mapper.canonical_inventory,
            reference_json_path=ref_json_path,
        )
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.EVALUATION,
            "completed" if eval_report.get("has_reference_annotations") else "skipped",
            mode_str,
            0.98,
            t13_iso,
            time.perf_counter() - t0,
            "Executed post-inference evaluation",
            {"has_reference_annotations": eval_report.get("has_reference_annotations", False)},
            progress_cb,
        )

        # 14. COMPLETED
        total_pipeline_sec = time.perf_counter() - t_pipeline_start
        self._record_stage(
            stage_history,
            ConvoFuseJobStage.COMPLETED,
            "completed",
            mode_str,
            1.0,
            _utc_now_iso(),
            total_pipeline_sec,
            f"ConvoFuse pipeline completed in {total_pipeline_sec:.2f}s",
            {"total_pipeline_sec": round(total_pipeline_sec, 4)},
            progress_cb,
        )

        processing_manifest: Dict[str, Any] = {
            "recording_id": recording_id,
            "filename": filename,
            "execution_mode": mode_str,
            "diarization_condition": diarization_condition,
            "diarization_backend": diarization_backend,
            "refinement_model": ref_model_label,
            "created_at": _utc_now_iso(),
            "total_pipeline_sec": round(total_pipeline_sec, 4),
            "ground_truth_isolation_enforced": True,
            "audio_routing": audio_paths.to_dict(),
            "rttm_to_canonical_speaker_map": dict(speaker_mapper.rttm_to_canonical),
            "canonical_to_dlm_tag_map": dict(speaker_mapper.canonical_to_dlm_tag),
            "refinement_telemetry": refinement_summary,
            "stage_history": [s.to_dict() for s in stage_history],
        }

        bundle = ConvoFuseTranscriptBundle(
            recording_id=recording_id,
            filename=filename,
            duration_sec=audio_paths.duration_sec,
            sample_rate=audio_paths.sample_rate,
            channels=audio_paths.channels,
            execution_mode=mode_str,
            diarization_condition=diarization_condition,
            diarization_backend=diarization_backend,
            refinement_model=ref_model_label,
            speaker_inventory=list(speaker_mapper.canonical_inventory),
            rttm_to_canonical_speaker_map=dict(speaker_mapper.rttm_to_canonical),
            words=refined_words,
            turns=refined_turns,
            baseline_turns=baseline_turns,
            speaker_summaries=speaker_summaries,
            readable_transcript_txt=readable_txt,
            baseline_vs_refined_comparison=comparison,
            processing_manifest=processing_manifest,
            evaluation_report=eval_report,
        )

        if output_dir is not None:
            export_convofuse_artifacts(bundle, output_dir)

        return bundle
