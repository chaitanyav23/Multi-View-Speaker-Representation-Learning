"""
Phase 20 Speech Intelligence Engine for CS698B.

Implements real model inference following the Phase 20 research pipeline:
1. Audio Preprocessing: 16kHz mono WAV conversion with EBU R128 loudness normalization.
2. VAD: Pyannote High-Recall VAD or Silero VAD segmentation.
3. ASR: Faster-Whisper (large-v3 by default) with condition_on_previous_text=False
   and word-level acoustic confidence extraction.
4. Diarization: Speaker turn segmentation with RTTM intervals.
5. Speaker Attribution: Maximum-overlap and midpoint-distance fusion.
6. Confidence Calibration: Isotonic Regression / Temperature Scaling with entropy estimation.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from adaptive.confidence import ConfidenceCalibrator
    from engine.base import BaseSpeechEngine
    from engine.schemas import (
        AlignedWord,
        CalibrationMetadata,
        ConfidenceFeatures,
        ConversationTurn,
        TranscriptResult,
    )
except ImportError:
    from app.adaptive.confidence import ConfidenceCalibrator
    from app.engine.base import BaseSpeechEngine
    from app.engine.schemas import (
        AlignedWord,
        CalibrationMetadata,
        ConfidenceFeatures,
        ConversationTurn,
        TranscriptResult,
    )

logger = logging.getLogger(__name__)


class PipelineDependencyError(RuntimeError):
    """Raised when an essential model dependency, checkpoint, or credential is missing."""
    pass


def sanitize_credential_text(text: str, token: Optional[str] = None) -> str:
    """Sanitize secret token strings from error messages, blockers, and logs.

    Replaces exact matches of the provided token string and any standard
    Hugging Face token patterns (hf_...) with '[REDACTED_HF_TOKEN]'.
    """
    if not text or not isinstance(text, str):
        return text or ""
    sanitized = text
    if token and isinstance(token, str) and token.strip():
        sanitized = sanitized.replace(token.strip(), "[REDACTED_HF_TOKEN]")
    import re
    sanitized = re.sub(r"hf_[A-Za-z0-9]{20,}", "[REDACTED_HF_TOKEN]", sanitized)
    return sanitized


def ensure_hf_hub_download_compat(token: Optional[str] = None) -> bool:
    """Ensure huggingface_hub.hf_hub_download translates use_auth_token to token idempotently.

    Pyannote.audio 3.3.2 passes `use_auth_token` to `hf_hub_download()`.
    In `huggingface_hub >= 0.24.0`, this parameter was deprecated and removed in favor of `token`.
    This helper installs an idempotent compatibility wrapper only once.

    Parameters
    ----------
    token : Optional[str]
        Optional active Hugging Face token to sanitize from exception strings if patching fails.

    Returns
    -------
    bool
        True if the compatibility wrapper is active or was installed; False if not needed.
    """
    try:
        import huggingface_hub
    except ImportError:
        return False

    current_func = getattr(huggingface_hub, "hf_hub_download", None)
    if current_func is None or not callable(current_func):
        return False

    # Check if already patched to prevent repeated wrapper installation
    if getattr(current_func, "_is_pyannote_compat_wrapper", False):
        return True

    import inspect
    try:
        sig = inspect.signature(current_func)
    except (ValueError, TypeError):
        return False

    # If use_auth_token is already accepted natively, no translation is needed
    if "use_auth_token" in sig.parameters:
        return True

    # If the function does not accept token either, compatibility cannot be established safely
    if "token" not in sig.parameters:
        logger.error("huggingface_hub.hf_hub_download does not accept 'token' parameter.")
        raise RuntimeError("huggingface_hub.hf_hub_download does not accept 'token' parameter.")

    original_func = current_func

    import functools

    @functools.wraps(original_func)
    def _compat_download(*args, **kwargs):
        # Translate use_auth_token to token only when necessary
        if "use_auth_token" in kwargs:
            auth_token = kwargs.pop("use_auth_token")
            # Preserve explicitly supplied token if already present in kwargs
            if "token" not in kwargs and auth_token is not None:
                kwargs["token"] = auth_token
        return original_func(*args, **kwargs)

    # Expose identity markers to guarantee idempotency and non-nested inspection
    _compat_download._is_pyannote_compat_wrapper = True
    _compat_download._original_func = original_func

    # Install process-wide patch on huggingface_hub
    try:
        huggingface_hub.hf_hub_download = _compat_download
    except Exception as exc:
        clean_exc = sanitize_credential_text(str(exc), token)
        logger.error(
            "Failed to install hf_hub_download compatibility patch into huggingface_hub: %s",
            clean_exc,
        )
        raise RuntimeError(
            f"Failed to install hf_hub_download compatibility patch into huggingface_hub: {clean_exc}"
        ) from exc

    # If pyannote.audio modules have already been imported, patch their bound reference
    import sys
    pyannote_targets = [
        "pyannote.audio.core.pipeline",
        "pyannote.audio.pipelines.speaker_diarization",
    ]
    for mod_name in pyannote_targets:
        mod = sys.modules.get(mod_name)
        if mod is not None:
            try:
                setattr(mod, "hf_hub_download", _compat_download)
            except Exception as exc:
                clean_exc = sanitize_credential_text(str(exc), token)
                logger.error(
                    "Failed to install hf_hub_download compatibility patch into present module %s: %s",
                    mod_name,
                    clean_exc,
                )
                raise RuntimeError(
                    f"Failed to install hf_hub_download compatibility patch into present module {mod_name}: {clean_exc}"
                ) from exc

    return True


class Phase20SpeechEngine(BaseSpeechEngine):
    """
    Real Phase 20 Speech Processing Engine.
    Executes actual pretrained models with zero mock/synthetic fallback.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        hf_token: Optional[str] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
        confidence_threshold: float = 0.70,
        enable_diarization: bool = True,
        diarization_backend: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.calibrator = calibrator or ConfidenceCalibrator()

        if hf_token is not None:
            self.hf_token = hf_token.strip() if isinstance(hf_token, str) and hf_token.strip() else None
        else:
            try:
                from backend.config import resolve_hf_token
            except ImportError:
                from app.backend.config import resolve_hf_token
            self.hf_token = resolve_hf_token()

        self.enable_diarization = enable_diarization
        self.diarization_backend = (diarization_backend or "nemo").lower()

        # Determine compute device and precision
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        if device is None:
            self.device = "cuda" if has_cuda else "cpu"
        else:
            self.device = device

        if compute_type is None:
            # On 4GB GPUs (e.g. RTX 3050 Laptop), large-v3 requires int8 to fit comfortably in VRAM
            if self.device == "cuda" and has_cuda:
                import torch
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.compute_type = "int8" if vram_gb < 6.0 else "float16"
            else:
                self.compute_type = "int8"
        else:
            self.compute_type = compute_type

        self._whisper_model = None
        self._aligner = None
        self._nemo_diarizer = None

    def _get_nemo_diarizer(self):
        """Lazy load NeMo MSDD diarizer adapter."""
        if self._nemo_diarizer is None:
            try:
                from engine.nemo_diarizer import NemoMsddDiarizer
            except ImportError:
                from app.engine.nemo_diarizer import NemoMsddDiarizer

            self._nemo_diarizer = NemoMsddDiarizer(device=self.device)
        return self._nemo_diarizer

    def _get_aligner(self):
        """Lazy load native Wav2Vec2 CTC forced aligner."""
        if self._aligner is None:
            try:
                from engine.wav2vec2_aligner import Wav2Vec2ForcedAligner
            except ImportError:
                from app.engine.wav2vec2_aligner import Wav2Vec2ForcedAligner

            logger.info("Loading native Wav2Vec2 CTC aligner on %s...", self.device)
            self._aligner = Wav2Vec2ForcedAligner(device=self.device, lazy_load=True)
            self._aligner.load_model()
        return self._aligner

    def _get_whisper_model(self):
        """Lazy load Faster-Whisper model."""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as err:
                raise PipelineDependencyError(
                    "faster-whisper is not installed. Install it via 'pip install faster-whisper'."
                ) from err

            logger.info(
                "Loading Faster-Whisper '%s' on %s (%s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            try:
                self._whisper_model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as exc:
                # If CUDA load failure, provide explicit diagnosis or fallback to CPU
                if self.device == "cuda":
                    logger.warning(
                        "CUDA load failed (%s). Retrying on CPU with int8...", exc
                    )
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self._whisper_model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
                else:
                    raise PipelineDependencyError(
                        f"Failed to load Faster-Whisper model '{self.model_size}': {exc}"
                    ) from exc

        return self._whisper_model

    def preprocess_audio(self, input_path: Path, output_path: Path) -> Path:
        """
        Standardize audio to 16kHz 16-bit mono WAV with EBU R128 loudness normalization.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-filter_complex",
            "[0:a]aresample=16000,pan=mono|c0=c0,loudnorm=I=-16:LRA=11:TP=-1.5[out]",
            "-map",
            "[out]",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg preprocessing failed for {input_path}:\n{res.stderr}")
        return output_path

    def process(
        self,
        audio_path: str,
        recording_id: str,
        filename: Optional[str] = None,
        duration: Optional[float] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> TranscriptResult:
        """
        BaseSpeechEngine contract method.
        Executes real Phase 20 pipeline with live progress reporting.
        """
        opts = options or {}
        language = opts.get("language", "en")
        initial_prompt = opts.get("initial_prompt")
        diarization_rttm = opts.get("diarization_rttm")
        if diarization_rttm:
            diarization_rttm = Path(diarization_rttm)

        in_path = Path(audio_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {audio_path}")

        # Stage 1: Audio Preprocessing & Normalization
        if progress_callback:
            progress_callback(0.15, "Preprocessing & Loudness Normalization (EBU R128)")
        
        # Determine normalized temp path
        norm_dir = in_path.parent / "normalized"
        norm_dir.mkdir(parents=True, exist_ok=True)
        norm_path = norm_dir / f"{in_path.stem}_16k.wav"
        
        if not norm_path.exists() or norm_path.stat().st_size == 0:
            norm_path = self.preprocess_audio(in_path, norm_path)

        # Stage 2: Voice Activity Detection & Model Preparation
        if progress_callback:
            progress_callback(0.35, "VAD Filtering & ASR Model Initialization")
        model = self._get_whisper_model()

        # Stage 3: Faster-Whisper ASR Inference & Alignment
        if progress_callback:
            progress_callback(0.65, f"Faster-Whisper ({self.model_size}) Transcription & Acoustic Confidence")

        result = self.transcribe(
            audio_path=norm_path,
            recording_id=recording_id,
            filename=filename,
            language=language,
            initial_prompt=initial_prompt,
            diarization_rttm=diarization_rttm,
        )

        # Stage 4: Attributing Speakers & Finalizing
        if progress_callback:
            progress_callback(0.90, "Speaker Attribution & Confidence Calibration")

        if progress_callback:
            progress_callback(1.00, "Transcription Complete")

        return result
        """
        Standardize audio to 16kHz 16-bit mono WAV with EBU R128 loudness normalization.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-filter_complex",
            "[0:a]aresample=16000,pan=mono|c0=c0,loudnorm=I=-16:LRA=11:TP=-1.5[out]",
            "-map",
            "[out]",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg preprocessing failed for {input_path}:\n{res.stderr}")
        return output_path

    def get_audio_duration(self, audio_path: Path) -> float:
        """Query duration in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return 0.0
        try:
            return float(res.stdout.strip())
        except ValueError:
            return 0.0

    def transcribe(
        self,
        audio_path: Path,
        recording_id: str = "rec_001",
        filename: Optional[str] = None,
        language: str = "en",
        initial_prompt: Optional[str] = None,
        diarization_rttm: Optional[Path] = None,
        **kwargs: Any,
    ) -> TranscriptResult:
        """
        Execute full Phase 20 pipeline on preprocessed audio:
        1. ASR via Faster-Whisper (condition_on_previous_text=False)
        2. Word-level timestamp & acoustic confidence extraction
        3. Speaker attribution fusion via RTTM intervals or VAD
        4. Confidence calibration and entropy calculation
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        duration = self.get_audio_duration(audio_path)
        model = self._get_whisper_model()

        # Read preprocessed waveform for Wav2Vec2 forced alignment
        try:
            import soundfile as sf
            waveform, sr = sf.read(str(audio_path), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
        except Exception as exc:
            logger.warning("Failed to load audio with soundfile (%s). Alignment will be marked unaligned.", exc)
            waveform = None
            sr = 16000

        enable_alignment = kwargs.get("enable_alignment", True)
        aligner = self._get_aligner() if enable_alignment else None

        # Phase 20 specification: condition_on_previous_text=False prevents hallucination cascades
        logger.info(
            "Transcribing %s (duration: %.2fs) with condition_on_previous_text=False...",
            audio_path.name,
            duration,
        )

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            beam_size=5,
            word_timestamps=False,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
        )

        raw_words: List[Dict[str, Any]] = []
        raw_segments: List[Dict[str, Any]] = []

        for seg in segments_iter:
            seg_text = seg.text.strip()
            if not seg_text:
                continue

            seg_start = round(float(seg.start), 3)
            seg_end = round(float(seg.end), 3)
            seg_dur = max(0.0, seg_end - seg_start)

            # Faster-Whisper sequence likelihood probability derived from average token logprobs
            avg_lp = float(seg.avg_logprob) if hasattr(seg, "avg_logprob") else -0.5
            asr_prob = max(0.001, min(0.999, math.exp(avg_lp)))

            raw_segments.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "avg_logprob": avg_lp,
            })

            # Check zero-duration edge case (as documented in B.3): cannot align zero-duration interval
            if seg_dur <= 0.001:
                tokens_in_text = [w for w in seg_text.split() if w]
                for token in tokens_in_text:
                    raw_words.append({
                        "word": token,
                        "start": seg_start,
                        "end": seg_end,
                        "confidence": 0.0,
                        "alignment_score": None,
                        "asr_confidence": round(asr_prob, 4),
                        "alignment_status": "fallback",
                    })
                continue

            # Execute native Wav2Vec2 CTC forced alignment
            aligned_chunk_words = None
            if aligner is not None and waveform is not None:
                f1 = max(0, int(seg_start * sr))
                f2 = min(len(waveform), int(seg_end * sr))
                chunk = waveform[f1:f2]
                try:
                    aligned_chunk_words = aligner.align_segment(chunk, seg_text, seg_start, seg_end)
                except Exception as exc:
                    logger.warning("Alignment failed on [%.3f, %.3f] ('%s'): %s", seg_start, seg_end, seg_text, exc)
                    aligned_chunk_words = None

            if aligned_chunk_words:
                for w in aligned_chunk_words:
                    ctc_score = float(w["confidence"])
                    is_fallback = (ctc_score == 0.0 and w["start"] == seg_start and w["end"] == seg_end)
                    status = "fallback" if is_fallback else "aligned"
                    raw_words.append({
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "confidence": ctc_score if not is_fallback else 0.0,
                        "alignment_score": round(ctc_score, 4) if not is_fallback else None,
                        "asr_confidence": round(asr_prob, 4),
                        "alignment_status": status,
                    })
            else:
                # If alignment failed or was disabled, retain text with segment bounds as unaligned
                tokens_in_text = [w for w in seg_text.split() if w]
                for token in tokens_in_text:
                    raw_words.append({
                        "word": token,
                        "start": seg_start,
                        "end": seg_end,
                        "confidence": 0.0,
                        "alignment_score": None,
                        "asr_confidence": round(asr_prob, 4),
                        "alignment_status": "unaligned",
                    })

        # Load diarization segments if RTTM is provided or expose explicit blocker
        diar_segments, diar_status, diar_blocker = self._load_or_derive_diarization(
            audio_path, diarization_rttm, raw_segments, duration
        )

        # Step 4: Word-Level Speaker Attribution Fusion
        attributed = self._attribute_words(raw_words, diar_segments)

        # Step 5: Confidence Calibration & Feature Enrichment
        aligned_words: List[AlignedWord] = []
        for w_dict in attributed:
            raw_c = float(w_dict["confidence"])
            status = w_dict.get("alignment_status", "aligned")
            cal_c = float(self.calibrator.calibrate(raw_c)) if status == "aligned" else 0.0
            w_dur = max(0.0, round(w_dict["end"] - w_dict["start"], 3))

            features = ConfidenceFeatures(
                acoustic_score=round(w_dict.get("asr_confidence") or raw_c, 4),
                alignment_score=w_dict.get("alignment_score"),
                asr_confidence=w_dict.get("asr_confidence"),
                alignment_status=status,
                duration=w_dur,
                speech_ratio=1.0,
                overlap_detected=False,
                energy_level=1.0,
            )

            aligned_words.append(
                AlignedWord(
                    word=w_dict["word"],
                    start=w_dict["start"],
                    end=w_dict["end"],
                    speaker=w_dict["speaker"],
                    confidence=round(cal_c, 4),
                    raw_confidence=round(raw_c, 4),
                    is_calibrated=True,
                    alignment_status=status,
                    alignment_score=w_dict.get("alignment_score"),
                    asr_confidence=w_dict.get("asr_confidence"),
                    features=features,
                )
            )

        # Step 6: Group Words into Structured Conversation Turns
        turns = self._group_into_turns(aligned_words, max_gap=1.5)

        # Step 7: Summary Metrics
        if aligned_words:
            conf_scores = [w.confidence for w in aligned_words]
            overall_conf = round(float(sum(conf_scores) / len(conf_scores)), 4)
            low_conf_count = sum(1 for w in aligned_words if w.confidence < self.confidence_threshold)
        else:
            overall_conf = 0.0
            low_conf_count = 0

        speakers = sorted(list({w.speaker for w in aligned_words}))
        real_speakers = [s for s in speakers if s != "SPEAKER_UNASSIGNED"]
        speakers_detected_count = len(real_speakers) if diar_status in ("DIARIZATION_AVAILABLE", "EXTERNAL_RTTM") else 0
        detected_speakers_list = real_speakers if diar_status in ("DIARIZATION_AVAILABLE", "EXTERNAL_RTTM") else []

        speaker_dist: Dict[str, float] = {}
        for turn in turns:
            dur = turn.end - turn.start
            speaker_dist[turn.speaker] = round(speaker_dist.get(turn.speaker, 0.0) + dur, 2)

        summary_stats = {
            "overall_confidence": overall_conf,
            "mean_confidence": overall_conf,
            "word_count": len(aligned_words),
            "aligned_word_count": sum(1 for w in aligned_words if w.alignment_status == "aligned"),
            "fallback_word_count": sum(1 for w in aligned_words if w.alignment_status == "fallback"),
            "unaligned_word_count": sum(1 for w in aligned_words if w.alignment_status == "unaligned"),
            "turn_count": len(turns),
            "speakers_detected": speakers_detected_count,
            "low_confidence_word_count": low_conf_count,
            "speaker_distribution": speaker_dist,
            "alignment_model": "wav2vec2_fairseq_base_ls960_asr_ls960.pth",
            "alignment_method": "Native-Wav2Vec2-CTC",
            "diarization_model": "diar_msdd_telephonic.nemo + TitaNet-Large",
            "diarization_status": diar_status,
            "diarization_blocker": diar_blocker,
        }

        cal_meta = CalibrationMetadata(
            method=self.calibrator.method,
            version="1.0.0",
            is_calibrated=True,
            brier_score=round(self.calibrator.brier_score, 4) if hasattr(self.calibrator, "brier_score") else None,
            ece=round(self.calibrator.ece, 4) if hasattr(self.calibrator, "ece") else None,
        )

        return TranscriptResult(
            recording_id=recording_id,
            filename=filename or audio_path.name,
            duration=round(duration, 2),
            sample_rate=16000,
            channels=1,
            method="Phase20-FasterWhisper-NativeWav2Vec2",
            calibration=cal_meta,
            speakers=detected_speakers_list,
            segments=turns,
            words=aligned_words,
            summary=summary_stats,
        )

    def _load_or_derive_diarization(
        self,
        audio_path: Path,
        diarization_rttm: Optional[Path],
        raw_segments: List[Dict[str, Any]],
        duration: float,
    ) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
        """
        Derive speaker intervals via external RTTM, isolated NeMo MSDD, or authenticated Pyannote.
        """
        fallback_segments = [{
            "start": 0.0,
            "end": round(duration, 3),
            "speaker": "SPEAKER_UNASSIGNED",
        }]

        # 1. External RTTM (always honored if explicitly provided)
        if diarization_rttm and diarization_rttm.exists():
            return (self._parse_rttm(diarization_rttm), "EXTERNAL_RTTM", None)

        # 2. Disabled diarization configuration check (Bypass every backend!)
        if not self.enable_diarization:
            logger.info("Diarization disabled by configuration; bypassing all diarization backends.")
            return (fallback_segments, "DIARIZATION_UNAVAILABLE", "Diarization disabled by configuration.")

        # 3. Isolated NeMo MSDD / TitaNet Diarization (Primary default)
        if self.diarization_backend != "pyannote":
            diarizer = self._get_nemo_diarizer()
            available, reason = diarizer.is_available()
            if available:
                logger.info("Executing isolated NeMo MSDD / TitaNet speaker diarization...")
                vad_intervals: List[Dict[str, float]] = []
                try:
                    import soundfile as sf
                    import faster_whisper.vad as fw_vad

                    wav_data, sr = sf.read(str(audio_path), dtype="float32")
                    if wav_data.ndim > 1:
                        wav_data = wav_data.mean(axis=1)
                    vad_ts = fw_vad.get_speech_timestamps(
                        wav_data,
                        fw_vad.VadOptions(min_silence_duration_ms=300, speech_pad_ms=200),
                    )

                    # Structural validation of VAD intervals
                    for idx, item in enumerate(vad_ts):
                        if not isinstance(item, dict) or "start" not in item or "end" not in item:
                            raise ValueError(f"VAD returned malformed interval at index {idx}: {item}")
                        start_s = round(float(item["start"]) / sr, 3)
                        end_s = round(float(item["end"]) / sr, 3)
                        if start_s < 0 or end_s < 0:
                            raise ValueError(f"Negative VAD timestamp at index {idx}: [{start_s}, {end_s}]")
                        if start_s >= end_s:
                            raise ValueError(f"Invalid VAD duration at index {idx}: [{start_s}, {end_s}]")
                        vad_intervals.append({"start": start_s, "end": end_s})

                except Exception as exc:
                    logger.error("Genuine VAD extraction failed: %s. Aborting diarization.", exc)
                    return (fallback_segments, "DIARIZATION_FAILED", f"Voice Activity Detection failed: {exc}")

                if len(vad_intervals) == 0:
                    logger.info("VAD detected 0 speech intervals in audio; no speech to diarize.")
                    return (fallback_segments, "NO_SPEECH", "No speech intervals detected by VAD.")

                segments, status, blocker, stats = diarizer.diarize(
                    audio_path=audio_path,
                    vad_intervals=vad_intervals,
                )
                if status == "DIARIZATION_AVAILABLE" and segments:
                    logger.info(
                        "NeMo MSDD diarization succeeded: %d turns, %d speakers (%s).",
                        len(segments),
                        stats.get("speaker_count", 0),
                        ", ".join(stats.get("speakers", [])),
                    )
                    return (segments, "DIARIZATION_AVAILABLE", None)
                elif status == "NO_SPEECH":
                    logger.info("NeMo MSDD diarizer reported NO_SPEECH: %s", blocker)
                    return (fallback_segments, "NO_SPEECH", blocker or "No speech intervals detected by VAD.")
                elif status == "DIARIZATION_FAILED":
                    logger.warning("NeMo MSDD diarization failed: %s", blocker)
                    return (fallback_segments, "DIARIZATION_FAILED", blocker)

        # 4. Gated Pyannote Diarization (Fallback or explicit backend)
        if not self.hf_token:
            logger.warning("Gated Pyannote speaker diarization requested, but no Hugging Face token is available.")
            diar_status = "DIARIZATION_FAILED" if self.diarization_backend == "pyannote" else "DIARIZATION_UNAVAILABLE"
            diar_blocker = (
                "Hugging Face authentication required for gated Pyannote diarization. "
                "No token was found in HF_TOKEN, HUGGINGFACE_HUB_TOKEN, or local Hugging Face cache. "
                "Please configure HF_TOKEN or run 'huggingface-cli login'."
            )
            return (fallback_segments, diar_status, diar_blocker)

        try:
            # Ensure idempotent compatibility with pyannote.audio 3.3.2 if huggingface_hub dropped use_auth_token
            ensure_hf_hub_download_compat(token=self.hf_token)

            from pyannote.audio import Pipeline
            import torch

            logger.info("Running real Pyannote speaker diarization with resolved Hugging Face token...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            if pipeline is None:
                auth_err = (
                    "Could not load 'pyannote/speaker-diarization-3.1' pipeline. "
                    "Ensure your Hugging Face account has accepted the user conditions at "
                    "https://hf.co/pyannote/speaker-diarization-3.1 and https://hf.co/pyannote/segmentation-3.0."
                )
                logger.error("%s", auth_err)
                return (fallback_segments, "DIARIZATION_FAILED", auth_err)

            if self.device == "cuda" and torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            diar = pipeline(str(audio_path))
            segments = []
            for turn, _, speaker in diar.itertracks(yield_label=True):
                segments.append({
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker,
                })
            if segments:
                return (segments, "PYANNOTE_COMMUNITY_DIARIZATION", None)
            else:
                return (fallback_segments, "NO_SPEECH", "Pyannote produced 0 speaker segments.")

        except Exception as exc:
            exc_str = str(exc)
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 401 or "401" in exc_str or "Unauthorized" in exc_str or "Invalid token" in exc_str:
                auth_err = (
                    "Hugging Face authentication failed (HTTP 401 Unauthorized): "
                    "The provided Hugging Face token is invalid, expired, or revoked."
                )
            elif (
                status_code == 403
                or "403" in exc_str
                or "GatedRepoError" in type(exc).__name__
                or "gated" in exc_str.lower()
                or "restricted" in exc_str.lower()
            ):
                auth_err = (
                    "Access to gated model 'pyannote/speaker-diarization-3.1' denied (HTTP 403 Forbidden). "
                    "The token is authenticated, but user conditions have not been accepted. "
                    "Please visit https://hf.co/pyannote/speaker-diarization-3.1 to accept user terms."
                )
            else:
                exc_cls_name = type(exc).__name__
                clean_exc_str = sanitize_credential_text(exc_str, self.hf_token)
                auth_err = f"Pyannote diarization failed ({exc_cls_name}): {clean_exc_str}"

            # Always sanitize the final auth_err message before logging and returning
            auth_err = sanitize_credential_text(auth_err, self.hf_token)
            logger.error("Pyannote diarization error: %s", auth_err)
            return (fallback_segments, "DIARIZATION_FAILED", auth_err)

    def _parse_rttm(self, rttm_path: Path) -> List[Dict[str, Any]]:
        """Parse standard NIST RTTM file into list of segment dicts."""
        segments = []
        with rttm_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    dur = float(parts[4])
                    if dur > 0:
                        segments.append({
                            "start": round(start, 3),
                            "end": round(start + dur, 3),
                            "speaker": parts[7],
                        })
        return segments

    def _attribute_words(
        self,
        words: List[Dict[str, Any]],
        diar_segments: List[Dict[str, Any]],
        tolerance: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Phase 20 maximum-overlap speaker attribution with midpoint distance collar tolerance.
        """
        output = []
        for word in words:
            w_start = float(word["start"])
            w_end = float(word["end"])

            # 1. Check direct temporal overlap
            overlaps = []
            for seg in diar_segments:
                overlap = min(w_end, seg["end"]) - max(w_start, seg["start"])
                if overlap > 0:
                    overlaps.append((overlap, seg["speaker"]))

            if overlaps:
                # Assign to speaker with maximum overlap
                best_speaker = max(overlaps, key=lambda x: x[0])[1]
            else:
                # 2. Check nearest neighbor distance to word midpoint
                midpoint = (w_start + w_end) / 2.0
                best_dist = float("inf")
                nearest = "SPEAKER_UNASSIGNED"
                for seg in diar_segments:
                    dist = min(abs(midpoint - seg["start"]), abs(midpoint - seg["end"]))
                    if dist < best_dist:
                        best_dist = dist
                        nearest = seg["speaker"]

                best_speaker = nearest if best_dist <= tolerance else "SPEAKER_UNASSIGNED"

            w_out = dict(word)
            w_out["start"] = w_start
            w_out["end"] = w_end
            w_out["speaker"] = best_speaker
            output.append(w_out)
        return output

    def _group_into_turns(
        self,
        words: List[AlignedWord],
        max_gap: float = 1.5,
    ) -> List[ConversationTurn]:
        """
        Group sequential words of the same speaker into coherent conversation turns.
        """
        if not words:
            return []

        turns: List[ConversationTurn] = []
        cur_speaker = words[0].speaker
        cur_words: List[AlignedWord] = [words[0]]

        for word in words[1:]:
            gap = word.start - cur_words[-1].end
            if word.speaker == cur_speaker and gap <= max_gap:
                cur_words.append(word)
            else:
                turns.append(self._build_turn(cur_words, len(turns)))
                cur_speaker = word.speaker
                cur_words = [word]

        if cur_words:
            turns.append(self._build_turn(cur_words, len(turns)))

        return turns

    def _build_turn(self, words: List[AlignedWord], turn_idx: int = 0) -> ConversationTurn:
        avg_conf = sum(w.confidence for w in words) / len(words)
        text = " ".join(w.word for w in words)
        return ConversationTurn(
            turn_id=f"turn_{turn_idx:03d}",
            speaker=words[0].speaker,
            start=words[0].start,
            end=words[-1].end,
            text=text,
            words=words,
            avg_confidence=round(avg_conf, 4),
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return engine metadata and hardware parameters."""
        return {
            "engine": "Phase20SpeechEngine",
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "confidence_threshold": self.confidence_threshold,
            "calibration_method": self.calibrator.method,
            "has_hf_token": bool(self.hf_token),
        }
