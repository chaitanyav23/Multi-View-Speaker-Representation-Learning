"""
ConvoFuse Application Service — Integrates the Research-Validated ConvoFuse Pipeline
into the Phase20 FastAPI Backend and SQLite Persistence Layer (Part 6).

Supports:
1. FROZEN_INPUT_REPLAY mode (replaying frozen Stage 1 audio, Stage 2 aligned words,
   Stage 5 dual-encoder RTTMs, and Stage 5B.2 overlap posteriors through live
   ground-truth-isolated DiarizationLM-Gemma refinement and TPST reconciliation).
2. FULL_PROCESSING mode (processing uploaded audio/video recordings through canonical
   dual-path audio conditioning, Phase20 ASR/alignment/diarization, temporal word-speaker
   association, DiarizationLM-Gemma refinement, and complete transcript reconstruction).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional

from backend.config import settings
from backend.database.models import (
    JobRecord,
    JobRepository,
    RecordingRecord,
    RecordingRepository,
    TranscriptRepository,
)
from backend.services.job_service import JobService
from backend.services.transcription_service import TranscriptionService

# Resolve Repository B root relative to app/backend/services/convofuse_service.py (parents[3])
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_REPO_ROOT = Path(os.getenv("CONVOFUSE_REPO_ROOT", str(DEFAULT_REPO_ROOT))).resolve()
if str(RESEARCH_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_REPO_ROOT))

from convofuse.audio_diarization_adapter import (  # noqa: E402
    BENCHMARK_RECORDING_MAP,
    resolve_benchmark_frozen_inputs,
)
from convofuse.contracts import (  # noqa: E402
    ConvoFuseExecutionMode,
    ConvoFuseJobStage,
    ConvoFuseTranscriptBundle,
)
from convofuse.pipeline import ConvoFusePipeline  # noqa: E402
from convofuse.transcript_reconstructor import (  # noqa: E402
    to_phase20_transcript_result_dict,
)


class ConvoFuseService:
    """Application-level orchestration service for ConvoFuse jobs and artifacts."""

    _pipeline: Optional[ConvoFusePipeline] = None

    @classmethod
    def get_pipeline(cls) -> ConvoFusePipeline:
        if cls._pipeline is None:
            cls._pipeline = ConvoFusePipeline()
        return cls._pipeline

    @classmethod
    def set_pipeline(cls, pipeline: ConvoFusePipeline) -> None:
        cls._pipeline = pipeline

    @classmethod
    def get_convofuse_dir(cls, recording_id: str) -> Path:
        canonical_id = BENCHMARK_RECORDING_MAP.get(recording_id, recording_id)
        d = settings.results_dir / "convofuse" / canonical_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def list_benchmark_recordings(cls) -> List[Dict[str, str]]:
        return [
            {"short_id": "R01", "recording_id": "R01_sara_sharif", "default_condition": "E20 (Stage 5B.4 Fresh Dual-Encoder M0)"},
            {"short_id": "R02", "recording_id": "R02_dennis_skinner", "default_condition": "Condition J (Stage 5A.1 Dual-Encoder Equal Fusion)"},
            {"short_id": "R03", "recording_id": "R03_disabled_parking", "default_condition": "Condition J (Stage 5A.1 Dual-Encoder Equal Fusion)"},
            {"short_id": "R04", "recording_id": "R04_piers_robinson", "default_condition": "Condition J (Stage 5A.1 Dual-Encoder Equal Fusion)"},
            {"short_id": "R05", "recording_id": "R05_southwest_pilot", "default_condition": "Condition J (Stage 5A.1 Dual-Encoder Equal Fusion)"},
            {"short_id": "R06", "recording_id": "R06_rich_karen", "default_condition": "Condition J (Stage 5A.1 Dual-Encoder Equal Fusion)"},
            {"short_id": "R07", "recording_id": "R07_bwc_cheap_shot", "default_condition": "Condition J (Stage 5A.1 Dual-Encoder Equal Fusion)"},
            {"short_id": "R08", "recording_id": "R08_uk_parliament", "default_condition": "Condition J (Stage 5A.1 Dual-Encoder Equal Fusion)"},
        ]

    @classmethod
    def execute_convofuse(
        cls,
        recording_id: str,
        execution_mode: str = ConvoFuseExecutionMode.FROZEN_INPUT_REPLAY.value,
        condition_preference: str = "auto",
        enable_refinement: bool = True,
        use_locality_gating: bool = True,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> ConvoFuseTranscriptBundle:
        """
        Execute ConvoFuse synchronously (or inside a background JobService worker) and
        persist both the 7 ConvoFuse artifacts and the Phase20-compatible TranscriptResult.
        """
        pipeline = cls.get_pipeline()

        def _stage_cb(stage: ConvoFuseJobStage, prog: float, detail: str) -> None:
            if progress_cb is not None:
                progress_cb(prog, f"{stage.value}: {detail}")

        if execution_mode == ConvoFuseExecutionMode.FROZEN_INPUT_REPLAY.value or recording_id in BENCHMARK_RECORDING_MAP:
            canonical_id = BENCHMARK_RECORDING_MAP.get(recording_id, recording_id)
            frozen = resolve_benchmark_frozen_inputs(
                recording_id=canonical_id,
                condition_preference=condition_preference,
            )
            out_dir = cls.get_convofuse_dir(canonical_id)

            # Ensure a RecordingRecord exists in SQLite so Phase20 UI & media endpoints work seamlessly
            existing_rec = RecordingRepository.get(canonical_id)
            if existing_rec is None and frozen.audio_paths.canonical_16k_mono_wav:
                wav_p = frozen.audio_paths.canonical_16k_mono_wav
                now_iso = datetime.now(timezone.utc).isoformat()
                RecordingRepository.create(
                    RecordingRecord(
                        recording_id=canonical_id,
                        filename=f"{canonical_id}.wav",
                        filepath=str(wav_p),
                        duration=frozen.audio_paths.duration_sec,
                        size_bytes=wav_p.stat().st_size if wav_p.exists() else 0,
                        sha256=frozen.input_sha256.get("16k_mono_wav", canonical_id),
                        created_at=now_iso,
                    )
                )

            bundle = pipeline.run_from_frozen_inputs(
                frozen=frozen,
                enable_refinement=enable_refinement,
                use_locality_gating=use_locality_gating,
                output_dir=out_dir,
                progress_cb=_stage_cb,
            )
            target_id = canonical_id
        else:
            rec = RecordingRepository.get(recording_id)
            if not rec:
                raise ValueError(f"Recording '{recording_id}' not found.")

            # Run or retrieve Phase20 transcription to obtain ASR words and diarization segments
            p20_res = TranscriptionService.get_transcript(recording_id)
            if p20_res is None:
                p20_res = TranscriptionService.execute_transcription(
                    recording_id=recording_id,
                    progress_cb=progress_cb,
                )

            raw_words = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                    "asr_confidence": w.asr_confidence if w.asr_confidence is not None else w.raw_confidence,
                    "alignment_score": w.alignment_score,
                    "alignment_status": w.alignment_status,
                }
                for w in p20_res.words
            ]
            diar_segments = [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "duration": round(max(0.0, seg.end - seg.start), 3),
                    "speaker": seg.speaker,
                }
                for seg in p20_res.segments
            ]
            out_dir = cls.get_convofuse_dir(recording_id)
            work_dir = settings.processed_dir / "convofuse" / recording_id

            bundle = pipeline.run_custom_or_uploaded(
                recording_id=recording_id,
                filename=rec.filename,
                source_media_path=Path(rec.filepath),
                raw_words=raw_words,
                diar_segments=diar_segments,
                work_dir=work_dir,
                execution_mode=ConvoFuseExecutionMode.FULL_PROCESSING,
                diarization_condition="Phase20_Full_Processing",
                diarization_backend=str(p20_res.summary.get("diarization_model") or "Phase20-Diarization"),
                enable_refinement=enable_refinement,
                use_locality_gating=use_locality_gating,
                output_dir=out_dir,
                progress_cb=_stage_cb,
            )
            target_id = recording_id

        # Persist complete bundle JSON and Phase20-compatible TranscriptResult
        out_dir = cls.get_convofuse_dir(target_id)
        bundle_json_path = out_dir / f"{target_id}__bundle.json"
        bundle_json_path.write_text(
            json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        p20_dict = to_phase20_transcript_result_dict(bundle)
        now_iso = datetime.now(timezone.utc).isoformat()
        TranscriptRepository.save(target_id, p20_dict, now_iso)
        p20_json_path = settings.results_dir / f"{target_id}.json"
        p20_json_path.write_text(json.dumps(p20_dict, indent=2, ensure_ascii=False), encoding="utf-8")

        return bundle

    @classmethod
    def submit_convofuse_job(
        cls,
        recording_id: str,
        execution_mode: str = ConvoFuseExecutionMode.FROZEN_INPUT_REPLAY.value,
        condition_preference: str = "auto",
        enable_refinement: bool = True,
        use_locality_gating: bool = True,
        sync: bool = False,
    ) -> JobRecord:
        """Create and dispatch a ConvoFuse job (asynchronously via JobService or synchronously if sync=True)."""
        canonical_id = (
            BENCHMARK_RECORDING_MAP.get(recording_id, recording_id)
            if execution_mode == ConvoFuseExecutionMode.FROZEN_INPUT_REPLAY.value or recording_id in BENCHMARK_RECORDING_MAP
            else recording_id
        )
        job = JobService.create_job(canonical_id)

        if sync:
            now_iso = datetime.now(timezone.utc).isoformat()
            JobRepository.update_status(job.job_id, "processing", 0.05, "UPLOADED", now_iso)

            def _sync_cb(prog: float, stage_str: str) -> None:
                upd_iso = datetime.now(timezone.utc).isoformat()
                JobRepository.update_status(job.job_id, "processing", prog, stage_str, upd_iso)

            try:
                cls.execute_convofuse(
                    recording_id=canonical_id,
                    execution_mode=execution_mode,
                    condition_preference=condition_preference,
                    enable_refinement=enable_refinement,
                    use_locality_gating=use_locality_gating,
                    progress_cb=_sync_cb,
                )
                fin_iso = datetime.now(timezone.utc).isoformat()
                JobRepository.update_status(job.job_id, "completed", 1.0, "COMPLETED", fin_iso)
            except Exception as exc:
                err_iso = datetime.now(timezone.utc).isoformat()
                JobRepository.update_status(job.job_id, "failed", 0.0, "FAILED", err_iso, error=str(exc))
                raise
            return JobService.get_job(job.job_id) or job

        def _task(j_id: str, prog_cb: Callable[[float, str], None]) -> None:
            cls.execute_convofuse(
                recording_id=canonical_id,
                execution_mode=execution_mode,
                condition_preference=condition_preference,
                enable_refinement=enable_refinement,
                use_locality_gating=use_locality_gating,
                progress_cb=prog_cb,
            )

        JobService.submit_task(job.job_id, _task)
        return job

    @classmethod
    def _get_pregenerated_benchmark_dir(cls, canonical_id: str) -> Optional[Path]:
        cand = RESEARCH_REPO_ROOT / "convofuse_outputs" / "r01_r08_benchmark" / canonical_id
        if cand.exists() and cand.is_dir():
            return cand
        return None

    @classmethod
    def get_artifact_json(cls, recording_id: str, suffix: str) -> Optional[Dict[str, Any]]:
        canonical_id = BENCHMARK_RECORDING_MAP.get(recording_id, recording_id)
        out_dir = cls.get_convofuse_dir(canonical_id)
        target = out_dir / f"{canonical_id}__{suffix}.json"
        if target.exists():
            return json.loads(target.read_text(encoding="utf-8"))
        # Fallback 1: check if embedded in TranscriptRepository
        t_data = TranscriptRepository.get(canonical_id)
        if t_data and isinstance(t_data.get("summary"), dict):
            cf = t_data["summary"].get("convofuse")
            if isinstance(cf, dict):
                if suffix == "processing_manifest":
                    return cf.get("processing_manifest")
                if suffix == "evaluation_report":
                    return cf.get("evaluation_report")
                if suffix == "comparison":
                    return cf.get("baseline_vs_refined_comparison")
                if suffix == "words":
                    return {"recording_id": canonical_id, "words": cf.get("words", [])}
                if suffix == "turns":
                    return {
                        "recording_id": canonical_id,
                        "turns": cf.get("turns", []),
                        "baseline_turns": cf.get("baseline_turns", []),
                    }
                if suffix == "speakers":
                    return {
                        "recording_id": canonical_id,
                        "speaker_inventory": cf.get("speaker_inventory", []),
                        "speaker_summaries": cf.get("speaker_summaries", []),
                    }
                if suffix == "bundle":
                    return cf
        # Fallback 2: read-only fallback to Repository B's convofuse_outputs/r01_r08_benchmark/<canonical_id>
        pre_dir = cls._get_pregenerated_benchmark_dir(canonical_id)
        if pre_dir is not None:
            pre_target = pre_dir / f"{canonical_id}__{suffix}.json"
            if pre_target.exists():
                return json.loads(pre_target.read_text(encoding="utf-8"))
            if suffix == "bundle":
                manifest_p = pre_dir / f"{canonical_id}__processing_manifest.json"
                words_p = pre_dir / f"{canonical_id}__words.json"
                turns_p = pre_dir / f"{canonical_id}__turns.json"
                spks_p = pre_dir / f"{canonical_id}__speakers.json"
                comp_p = pre_dir / f"{canonical_id}__comparison.json"
                eval_p = pre_dir / f"{canonical_id}__evaluation_report.json"
                txt_p = pre_dir / f"{canonical_id}__transcript.txt"
                if all(p.exists() for p in (manifest_p, words_p, turns_p, spks_p, comp_p, eval_p, txt_p)):
                    manifest_d = json.loads(manifest_p.read_text(encoding="utf-8"))
                    words_d = json.loads(words_p.read_text(encoding="utf-8"))
                    turns_d = json.loads(turns_p.read_text(encoding="utf-8"))
                    spks_d = json.loads(spks_p.read_text(encoding="utf-8"))
                    comp_d = json.loads(comp_p.read_text(encoding="utf-8"))
                    eval_d = json.loads(eval_p.read_text(encoding="utf-8"))
                    txt_s = txt_p.read_text(encoding="utf-8")
                    words_list = words_d.get("words", [])
                    turns_list = turns_d.get("turns", [])
                    return {
                        "recording_id": canonical_id,
                        "filename": manifest_d.get("filename", f"{canonical_id}.wav"),
                        "duration": manifest_d.get("duration", 0.0),
                        "execution_mode": manifest_d.get("execution_mode", "FROZEN_INPUT_REPLAY"),
                        "speaker_inventory": spks_d.get("speaker_inventory", []),
                        "word_count": len(words_list),
                        "turn_count": len(turns_list),
                        "words": words_list,
                        "turns": turns_list,
                        "baseline_turns": turns_d.get("baseline_turns", []),
                        "speaker_summaries": spks_d.get("speaker_summaries", []),
                        "processing_manifest": manifest_d,
                        "baseline_vs_refined_comparison": comp_d,
                        "evaluation_report": eval_d,
                        "readable_transcript_txt": txt_s,
                    }
        return None

    @classmethod
    def get_readable_txt(cls, recording_id: str) -> Optional[str]:
        canonical_id = BENCHMARK_RECORDING_MAP.get(recording_id, recording_id)
        out_dir = cls.get_convofuse_dir(canonical_id)
        txt_path = out_dir / f"{canonical_id}__transcript.txt"
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8")
        t_data = TranscriptRepository.get(canonical_id)
        if t_data and isinstance(t_data.get("summary"), dict):
            cf = t_data["summary"].get("convofuse")
            if isinstance(cf, dict):
                return cf.get("readable_transcript_txt")
        pre_dir = cls._get_pregenerated_benchmark_dir(canonical_id)
        if pre_dir is not None:
            pre_txt = pre_dir / f"{canonical_id}__transcript.txt"
            if pre_txt.exists():
                return pre_txt.read_text(encoding="utf-8")
        return None
