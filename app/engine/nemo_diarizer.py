"""NVIDIA NeMo MSDD / TitaNet Diarization Adapter.

Integrates the isolated 'cs698b_nemo' subprocess worker into the production
application pipeline, managing VAD oracle manifest generation, subprocess
execution, error capture, and RTTM parsing.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("engine.nemo_diarizer")

DEFAULT_NEMO_PYTHON = Path("/home/priyanka/anaconda3/envs/cs698b_nemo/bin/python")
DEFAULT_CHECKPOINT = Path(
    "/home/priyanka/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo"
)
WORKER_SCRIPT = Path(__file__).resolve().parent / "diarization_worker.py"


class NemoMsddDiarizer:
    """Adapter for running isolated NeMo MSDD/TitaNet speaker diarization."""

    def __init__(
        self,
        nemo_python: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        timeout_seconds: int = 300,
        max_speakers: int = 10,
        device: str = "cuda",
    ) -> None:
        self.nemo_python = nemo_python or DEFAULT_NEMO_PYTHON
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.timeout_seconds = timeout_seconds
        self.max_speakers = max_speakers
        self.device = device

    def is_available(self) -> Tuple[bool, Optional[str]]:
        """Verify whether NeMo executable, worker script, and checkpoint exist."""
        if not self.nemo_python.exists():
            return False, f"NeMo Python executable not found: {self.nemo_python}"
        if not WORKER_SCRIPT.exists():
            return False, f"Diarization worker script not found: {WORKER_SCRIPT}"
        if not self.checkpoint_path.exists():
            return False, f"NeMo MSDD checkpoint not found: {self.checkpoint_path}"
        return True, None

    @staticmethod
    def generate_oracle_vad_rttm(
        vad_intervals: List[Dict[str, float]],
        output_rttm: Path,
        file_id: str,
    ) -> Path:
        """
        Generate a speech-only oracle VAD RTTM for NeMo.
        Uses a dummy speaker label 'speaker' as required by Phase 20 specification.
        """
        output_rttm.parent.mkdir(parents=True, exist_ok=True)
        with open(output_rttm, "w", encoding="utf-8") as f:
            for seg in vad_intervals:
                start = float(seg["start"])
                end = float(seg["end"])
                dur = max(0.0, end - start)
                if dur > 0.001:
                    f.write(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> speaker <NA> <NA>\n")
        return output_rttm

    @staticmethod
    def parse_rttm(rttm_path: Path) -> List[Dict[str, Any]]:
        """Parse standard NIST RTTM file into a list of segment dictionaries."""
        segments: List[Dict[str, Any]] = []
        if not rttm_path.exists():
            return segments

        with open(rttm_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    dur = float(parts[4])
                    if dur > 0.0:
                        segments.append({
                            "start": round(start, 3),
                            "end": round(start + dur, 3),
                            "speaker": parts[7],
                        })
        return segments

    def diarize(
        self,
        audio_path: Path,
        vad_intervals: Optional[List[Dict[str, float]]] = None,
        vad_rttm: Optional[Path] = None,
        out_dir: Optional[Path] = None,
    ) -> Tuple[List[Dict[str, Any]], str, Optional[str], Dict[str, Any]]:
        """
        Execute isolated NeMo MSDD diarization subprocess.
        Returns:
            (segments, diar_status, diar_blocker, stats)
        """
        available, reason = self.is_available()
        if not available:
            logger.warning("NeMo diarizer unavailable: %s", reason)
            return [], "DIARIZATION_UNAVAILABLE", reason, {}

        if not audio_path.exists():
            return [], "DIARIZATION_FAILED", f"Audio file not found: {audio_path}", {}

        file_id = audio_path.stem

        # Short-circuit if empty VAD intervals provided
        if vad_intervals is not None and len(vad_intervals) == 0:
            logger.info("Empty VAD intervals provided for '%s'; short-circuiting without worker invocation.", file_id)
            stats = {
                "speaker_count": 0,
                "speakers": [],
                "turn_count": 0,
                "total_speech_seconds": 0.0,
                "inference_time_seconds": 0.0,
            }
            return [], "NO_SPEECH", "No speech intervals detected by VAD.", stats

        # Short-circuit if empty VAD RTTM provided
        if vad_rttm is not None and vad_rttm.exists():
            existing_turns = self.parse_rttm(vad_rttm)
            if len(existing_turns) == 0:
                logger.info("Empty VAD RTTM provided for '%s'; short-circuiting without worker invocation.", file_id)
                stats = {
                    "speaker_count": 0,
                    "speakers": [],
                    "turn_count": 0,
                    "total_speech_seconds": 0.0,
                    "inference_time_seconds": 0.0,
                }
                return [], "NO_SPEECH", "VAD RTTM contains 0 speech intervals.", stats

        temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
        unique_work_dir: Path
        if out_dir is None:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"nemo_diar_{file_id}_")
            unique_work_dir = Path(temp_dir_obj.name).resolve()
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            import time
            unique_name = f"run_{file_id}_{os.getpid()}_{int(time.time() * 1000)}"
            unique_work_dir = (out_dir / unique_name).resolve()
            unique_work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Prepare speech-only VAD RTTM
            if vad_rttm is not None and vad_rttm.exists():
                active_vad_rttm = vad_rttm
            elif vad_intervals is not None:
                active_vad_rttm = unique_work_dir / f"{file_id}_oracle_vad.rttm"
                self.generate_oracle_vad_rttm(vad_intervals, active_vad_rttm, file_id)
            else:
                return [], "DIARIZATION_FAILED", "Neither vad_intervals nor vad_rttm provided.", {}

            cmd = [
                str(self.nemo_python.resolve()),
                str(WORKER_SCRIPT.resolve()),
                "--audio",
                str(audio_path.resolve()),
                "--vad-rttm",
                str(active_vad_rttm.resolve()),
                "--out-dir",
                str(unique_work_dir),
                "--checkpoint",
                str(self.checkpoint_path.resolve()),
                "--max-speakers",
                str(self.max_speakers),
                "--device",
                self.device,
            ]

            logger.info("Spawning NeMo diarization worker in isolated workspace: %s", unique_work_dir)
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
            )

            if proc.returncode != 0:
                err_msg = proc.stderr.strip() or proc.stdout.strip() or "Unknown worker crash"
                logger.error("NeMo diarization worker failed (exit code %d): %s", proc.returncode, err_msg)
                return [], "DIARIZATION_FAILED", f"Worker exited with code {proc.returncode}: {err_msg[:400]}", {}

            # Strict worker JSON protocol validation
            json_lines = []
            for line in proc.stdout.splitlines():
                line_str = line.strip()
                if line_str.startswith("{") and line_str.endswith("}"):
                    json_lines.append(line_str)

            if not json_lines:
                logger.error("Worker exited with code 0 but emitted no JSON payload.")
                return [], "DIARIZATION_FAILED", "Worker completed but emitted no structured JSON response.", {}

            try:
                worker_meta = json.loads(json_lines[-1])
            except Exception as exc:
                logger.error("Worker emitted malformed JSON: %s", exc)
                return [], "DIARIZATION_FAILED", f"Worker emitted malformed JSON response: {exc}", {}

            if not isinstance(worker_meta, dict):
                return [], "DIARIZATION_FAILED", "Worker JSON response is not a valid dictionary.", {}

            worker_status = worker_meta.get("status")
            VALID_STATUSES = {"SUCCESS", "NO_SPEECH", "FAILED"}
            if worker_status not in VALID_STATUSES:
                logger.error("Unrecognized worker status: %s", worker_status)
                return [], "DIARIZATION_FAILED", f"Unrecognized worker status: '{worker_status}'.", worker_meta

            if worker_status == "FAILED":
                err = worker_meta.get("error", "Unspecified worker failure")
                logger.error("Worker reported FAILED status: %s", err)
                return [], "DIARIZATION_FAILED", f"Worker reported failure: {err}", worker_meta

            if worker_status == "NO_SPEECH":
                logger.info("Worker reported NO_SPEECH for audio '%s'", file_id)
                stats = {
                    "speaker_count": 0,
                    "speakers": [],
                    "turn_count": 0,
                    "total_speech_seconds": 0.0,
                    "inference_time_seconds": worker_meta.get("inference_time_seconds", 0.0),
                }
                return [], "NO_SPEECH", worker_meta.get("reason", "No speech intervals detected by VAD."), stats

            # For SUCCESS status, validate RTTM path existence and workspace provenance
            raw_rttm_path = worker_meta.get("rttm_path")
            if not raw_rttm_path or not isinstance(raw_rttm_path, str):
                return [], "DIARIZATION_FAILED", "Worker reported SUCCESS but omitted 'rttm_path'.", worker_meta

            rttm_path = Path(raw_rttm_path).resolve()

            # Workspace provenance check: rttm_path must reside within unique_work_dir
            try:
                rttm_path.relative_to(unique_work_dir)
            except ValueError:
                logger.error("Provenance violation: RTTM path '%s' is outside invocation workspace '%s'", rttm_path, unique_work_dir)
                return [], "DIARIZATION_FAILED", f"Worker returned RTTM outside invocation workspace: {rttm_path}", worker_meta

            if not rttm_path.is_file():
                return [], "DIARIZATION_FAILED", f"Worker output RTTM does not exist as a regular file: {rttm_path}", worker_meta

            try:
                segments = self.parse_rttm(rttm_path)
            except Exception as exc:
                return [], "DIARIZATION_FAILED", f"Failed to parse worker output RTTM: {exc}", worker_meta

            if not segments:
                logger.info("Worker RTTM parsed to 0 segments; returning NO_SPEECH.")
                stats = {
                    "speaker_count": 0,
                    "speakers": [],
                    "turn_count": 0,
                    "total_speech_seconds": 0.0,
                    "inference_time_seconds": worker_meta.get("inference_time_seconds", 0.0),
                }
                return [], "NO_SPEECH", "Worker produced empty RTTM with zero speaker segments.", stats

            # Validate metadata consistency
            speaker_count = worker_meta.get("speaker_count")
            if speaker_count is None or not isinstance(speaker_count, int):
                speaker_count = len({s["speaker"] for s in segments})

            speakers = worker_meta.get("speakers")
            if not isinstance(speakers, list):
                speakers = sorted(list({s["speaker"] for s in segments}))

            stats = {
                "speaker_count": speaker_count,
                "speakers": speakers,
                "turn_count": len(segments),
                "rttm_path": str(rttm_path),
                "inference_time_seconds": worker_meta.get("inference_time_seconds", 0.0),
            }

            return segments, "DIARIZATION_AVAILABLE", None, stats

        except subprocess.TimeoutExpired:
            logger.error("NeMo diarization worker timed out after %ds", self.timeout_seconds)
            return [], "DIARIZATION_FAILED", f"Execution timed out after {self.timeout_seconds}s", {}
        except Exception as exc:
            logger.error("Unexpected error during NeMo diarization: %s", exc, exc_info=True)
            return [], "DIARIZATION_FAILED", str(exc), {}
        finally:
            if temp_dir_obj is not None:
                try:
                    temp_dir_obj.cleanup()
                except Exception:
                    pass
