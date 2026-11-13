"""
ConvoFuse — Audio Ingestion, Dual-Path Conditioning, Dual-Encoder Diarization,
and Temporal Word-to-Speaker Association Adapter (Part 3).

Wraps the authoritative Stage 1–5B.4 research implementations and Phase20 audio/ASR
components without modifying frozen Stage 5 source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np

from .contracts import (
    ConvoFuseExecutionMode,
    ConvoFuseJobStage,
    ConvoFuseWord,
    OverlapMetadata,
    RefinementDecision,
    RejectionOrAbstentionReason,
    SpeakerIdentityMapper,
    WordProvenance,
)
from research_stages.stage6_generative_refinement.word_orchestrator import (
    extract_available_speakers,
    load_asr_words,
    orchestrate_words,
    parse_rttm_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_RECORDING_MAP: Dict[str, str] = {
    "R01": "R01_sara_sharif",
    "R01_sara_sharif": "R01_sara_sharif",
    "R02": "R02_dennis_skinner",
    "R02_dennis_skinner": "R02_dennis_skinner",
    "R03": "R03_disabled_parking",
    "R03_disabled_parking": "R03_disabled_parking",
    "R04": "R04_piers_robinson",
    "R04_piers_robinson": "R04_piers_robinson",
    "R05": "R05_southwest_pilot",
    "R05_southwest_pilot": "R05_southwest_pilot",
    "R06": "R06_rich_karen",
    "R06_rich_karen": "R06_rich_karen",
    "R07": "R07_bwc_cheap_shot",
    "R07_bwc_cheap_shot": "R07_bwc_cheap_shot",
    "R08": "R08_uk_parliament",
    "R08_uk_parliament": "R08_uk_parliament",
}


def compute_sha256(path: Path) -> str:
    """Compute read-only SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class TaskSpecificAudioPaths:
    """
    Task-specific dual-path audio conditioning contract (Part 1.3.B & Part 3.2):
    - `canonical_16k_mono_wav`: Unenhanced 16 kHz mono PCM WAV (used for WeSpeaker ResNet293-LM 256-D
      and raw-audio overlap detection).
    - `dfn_conditioned_wav`: DeepFilterNet3-enhanced 16 kHz mono PCM WAV (used for TitaNet-Large 192-D).
    - `normalized_asr_wav`: EBU R128 loudness-normalized 16 kHz mono PCM WAV (used for Faster-Whisper
      Large-v3 ASR and Wav2Vec2 CTC word alignment).
    """
    source_media_path: Optional[Path]
    canonical_16k_mono_wav: Optional[Path]
    dfn_conditioned_wav: Optional[Path]
    normalized_asr_wav: Optional[Path]
    sample_rate: int = 16000
    channels: int = 1
    duration_sec: float = 0.0
    sha256_hashes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_media_path": str(self.source_media_path) if self.source_media_path else None,
            "titanet_dfn_path": str(self.dfn_conditioned_wav) if self.dfn_conditioned_wav else None,
            "wespeaker_raw_mono_path": str(self.canonical_16k_mono_wav) if self.canonical_16k_mono_wav else None,
            "asr_normalized_path": str(self.normalized_asr_wav) if self.normalized_asr_wav else None,
            "overlap_raw_mono_path": str(self.canonical_16k_mono_wav) if self.canonical_16k_mono_wav else None,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_sec": self.duration_sec,
            "sha256_hashes": dict(self.sha256_hashes),
        }


@dataclass(frozen=True)
class FrozenPredictedInputs:
    """
    Validated frozen predicted inputs for replay/integration mode (Part 6.5 & Part 7.2–7.3).
    Strictly excludes reference/ground-truth annotations to enforce ground-truth isolation.
    """
    recording_id: str
    diarization_condition: str
    diarization_backend: str
    asr_aligned_words_path: Path
    predicted_rttm_path: Path
    overlap_posterior_npz_path: Optional[Path]
    audio_paths: TaskSpecificAudioPaths
    input_sha256: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "diarization_condition": self.diarization_condition,
            "diarization_backend": self.diarization_backend,
            "asr_aligned_words_path": str(self.asr_aligned_words_path),
            "predicted_rttm_path": str(self.predicted_rttm_path),
            "overlap_posterior_npz_path": str(self.overlap_posterior_npz_path) if self.overlap_posterior_npz_path else None,
            "audio_paths": self.audio_paths.to_dict(),
            "input_sha256": dict(self.input_sha256),
        }


class CanonicalAudioProcessor:
    """
    Extracts and validates canonical 16 kHz mono audio and task-specific conditioned paths (Part 3.1–3.2).
    """

    @staticmethod
    def extract_canonical_16k_mono(input_path: Path, output_wav_path: Path) -> Dict[str, Any]:
        """Extract canonical 16 kHz 16-bit mono PCM WAV with temporal origin preservation."""
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vn",
            "-af", "aresample=async=1:first_pts=0",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_wav_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Canonical FFmpeg extraction failed for {input_path}: {res.stderr}")
        return CanonicalAudioProcessor.validate_wav(output_wav_path)

    @staticmethod
    def extract_loudnorm_asr_audio(input_path: Path, output_wav_path: Path) -> Dict[str, Any]:
        """Extract EBU R128 loudness-normalized 16 kHz mono WAV for ASR & Wav2Vec2 alignment."""
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vn",
            "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_wav_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"EBU R128 loudness normalization failed for {input_path}: {res.stderr}")
        return CanonicalAudioProcessor.validate_wav(output_wav_path)

    @staticmethod
    def validate_wav(wav_path: Path) -> Dict[str, Any]:
        """Validate sample rate, channels, duration, finite samples, and SHA-256 digest."""
        p = Path(wav_path)
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(f"Audio file missing or empty: {p}")
        try:
            import soundfile as sf
            data, sr = sf.read(str(p), dtype="float32")
            channels = 1 if data.ndim == 1 else data.shape[1]
            n_samples = len(data)
            duration = round(float(n_samples) / float(sr), 4) if sr > 0 else 0.0
            has_nan_or_inf = bool(not np.all(np.isfinite(data)))
            if has_nan_or_inf:
                raise ValueError(f"Audio waveform contains NaN or Inf values: {p}")
        except ImportError:
            import wave
            with wave.open(str(p), "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                n_samples = wf.getnframes()
                duration = round(float(n_samples) / float(sr), 4) if sr > 0 else 0.0
        return {
            "path": str(p),
            "sample_rate": sr,
            "channels": channels,
            "samples": n_samples,
            "duration_sec": duration,
            "sha256": compute_sha256(p),
        }


def resolve_benchmark_frozen_inputs(
    recording_id: str,
    condition_preference: str = "auto",
    repo_root: Path = REPO_ROOT,
) -> FrozenPredictedInputs:
    """
    Resolves a benchmark recording ID (e.g., 'R01' or 'R01_sara_sharif') to its validated
    frozen Stage 1 audio paths, Stage 2 WhisperX aligned words, Stage 5 predicted RTTM,
    and Stage 5B.2 overlap posteriors.
    Never loads reference/ground-truth annotations.
    """
    canonical_id = BENCHMARK_RECORDING_MAP.get(recording_id, recording_id)
    audio_dir = repo_root / "research_stages" / "stage1_data_prep" / "audio" / canonical_id
    if not audio_dir.exists():
        audio_dir = repo_root / "research_stages" / "stage1_data_prep" / "audio"
    mono_wav = audio_dir / f"{canonical_id}_16k_mono.wav"
    dfn_wav = audio_dir / f"{canonical_id}_dfn.wav"
    norm_wav = audio_dir / f"{canonical_id}_normalized.wav"

    asr_words_path = (
        repo_root / "research_stages" / "stage2_baseline" / "whisperx" / f"{canonical_id}_aligned.json"
    )
    if not asr_words_path.exists():
        raise FileNotFoundError(f"Frozen ASR aligned words not found for {canonical_id}: {asr_words_path}")

    # Determine authoritative predicted RTTM:
    # - For R01, Stage 5B.4 E20 (fresh C62 + TitaNet-DFN + WeSpeaker-Mono + Equal Affinity Fusion + NME-SC M0)
    #   is available if condition_preference in ("auto", "E20").
    # - For R01-R08 corpus-wide consistency, Stage 5A.1 Condition J (TitaNet-DFN + WeSpeaker-Mono + Normalized
    #   Equal-Weight Affinity Fusion + NME-SC) is available for all 8 recordings.
    e20_rttm = (
        repo_root
        / "research_stages"
        / "stage5b"
        / "stage5b_4_fresh_diarization"
        / "outputs"
        / "E20"
        / f"{canonical_id}.rttm"
    )
    cond_j_rttm = (
        repo_root
        / "research_stages"
        / "stage5a_1"
        / "predictions"
        / "rttms"
        / "condition_J_norm_equal_weight_fusion"
        / f"{canonical_id}_dfn.rttm"
    )
    cond_k_rttm = (
        repo_root
        / "research_stages"
        / "stage5a_1"
        / "predictions"
        / "rttms"
        / "condition_K_norm_reliability_gate"
        / f"{canonical_id}_dfn.rttm"
    )

    if condition_preference == "E20" or (condition_preference == "auto" and e20_rttm.exists()):
        rttm_path = e20_rttm
        cond_label = "Stage5B.4_E20_C62_DualEncoder_EqualFusion_M0"
        backend_label = "TitaNet-Large(DFN,192D) + WeSpeaker-ResNet293-LM(Mono,256D) -> ZScore-Equal-Affinity-Fusion -> NME-SC (C62 VAD, M0)"
    elif condition_preference == "Condition_K" and cond_k_rttm.exists():
        rttm_path = cond_k_rttm
        cond_label = "Stage5A.1_Condition_K_DualEncoder_ReliabilityGate"
        backend_label = "TitaNet-Large(DFN,192D) + WeSpeaker-ResNet293-LM(Mono,256D) -> LORO-Clipped-Reliability-Gate -> NME-SC"
    elif cond_j_rttm.exists():
        rttm_path = cond_j_rttm
        cond_label = "Stage5A.1_Condition_J_DualEncoder_EqualFusion"
        backend_label = "TitaNet-Large(DFN,192D) + WeSpeaker-ResNet293-LM(Mono,256D) -> ZScore-Equal-Affinity-Fusion -> NME-SC"
    else:
        raise FileNotFoundError(
            f"No predicted Stage 5 RTTM found for {canonical_id} (checked {e20_rttm} and {cond_j_rttm})"
        )

    # Overlap posterior cache (raw mono stream preferred per Part 1.3.B & Part 3.2)
    ovl_mono = (
        repo_root
        / "research_stages"
        / "stage5b"
        / "stage5b_2_overlap_optimization"
        / "overlap_cache"
        / f"{canonical_id}_mono_overlap.npz"
    )
    ovl_dfn = (
        repo_root
        / "research_stages"
        / "stage5b"
        / "stage5b_2_overlap_optimization"
        / "overlap_cache"
        / f"{canonical_id}_dfn_overlap.npz"
    )
    ovl_path = ovl_mono if ovl_mono.exists() else (ovl_dfn if ovl_dfn.exists() else None)

    audio_hashes: Dict[str, str] = {}
    duration_sec = 0.0
    if mono_wav.exists():
        info = CanonicalAudioProcessor.validate_wav(mono_wav)
        audio_hashes["16k_mono_wav"] = info["sha256"]
        duration_sec = info["duration_sec"]
    if dfn_wav.exists():
        audio_hashes["dfn_wav"] = compute_sha256(dfn_wav)
    if norm_wav.exists():
        audio_hashes["normalized_wav"] = compute_sha256(norm_wav)

    audio_paths = TaskSpecificAudioPaths(
        source_media_path=mono_wav if mono_wav.exists() else None,
        canonical_16k_mono_wav=mono_wav if mono_wav.exists() else None,
        dfn_conditioned_wav=dfn_wav if dfn_wav.exists() else None,
        normalized_asr_wav=norm_wav if norm_wav.exists() else None,
        sample_rate=16000,
        channels=1,
        duration_sec=duration_sec,
        sha256_hashes=audio_hashes,
    )

    input_sha256 = {
        "asr_aligned_words": compute_sha256(asr_words_path),
        "predicted_rttm": compute_sha256(rttm_path),
    }
    if ovl_path and ovl_path.exists():
        input_sha256["overlap_posteriors"] = compute_sha256(ovl_path)
    input_sha256.update(audio_hashes)

    return FrozenPredictedInputs(
        recording_id=canonical_id,
        diarization_condition=cond_label,
        diarization_backend=backend_label,
        asr_aligned_words_path=asr_words_path,
        predicted_rttm_path=rttm_path,
        overlap_posterior_npz_path=ovl_path,
        audio_paths=audio_paths,
        input_sha256=input_sha256,
    )


def load_overlap_posterior_timeline(
    overlap_npz_path: Optional[Path],
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Loads frame-level overlap probabilities from a Stage 5B.2 overlap cache `.npz` file.
    Returns (frame_times_sec, overlap_probs, source_name) or None.
    """
    if overlap_npz_path is None or not Path(overlap_npz_path).exists():
        return None
    try:
        with np.load(str(overlap_npz_path), allow_pickle=False) as data:
            keys = list(data.files)
            prob_key = next(
                (k for k in ("overlap_prob", "p_overlap", "probs", "posteriors") if k in keys),
                keys[0] if keys else None,
            )
            if prob_key is None:
                return None
            probs = np.asarray(data[prob_key], dtype=np.float32).reshape(-1)
            if "frame_times" in keys:
                times = np.asarray(data["frame_times"], dtype=np.float32).reshape(-1)
            elif "timestamps" in keys:
                times = np.asarray(data["timestamps"], dtype=np.float32).reshape(-1)
            else:
                step = float(data["step"]) if "step" in keys else 0.016875
                start_offset = float(data["start"]) if "start" in keys else 0.0
                times = start_offset + np.arange(len(probs), dtype=np.float32) * step
            return times, probs, Path(overlap_npz_path).name
    except Exception:
        return None


def build_baseline_convofuse_words(
    raw_words: Sequence[Mapping[str, Any]],
    diar_segments: Sequence[Mapping[str, Any]],
    speaker_mapper: SpeakerIdentityMapper,
    diarization_backend: str,
    diarization_condition: str,
    execution_mode: str,
    overlap_timeline: Optional[Tuple[np.ndarray, np.ndarray, str]] = None,
    tolerance_sec: float = 0.5,
) -> List[ConvoFuseWord]:
    """
    Associates aligned ASR words with diarization RTTM intervals (Part 3.6 & Part 3.7):
    - Uses validated `orchestrate_words` (`max_overlap_speaker` + `0.5s` midpoint tolerance).
    - Maps RTTM speaker IDs deterministically to `SPEAKER_00`, `SPEAKER_01`, ..., while
      preserving `UNK` for unassigned words.
    - Enriches each word with multi-speaker RTTM overlap metadata and frame-level overlap
      posterior statistics without forcing simultaneous speakers into false single-speaker claims.
    """
    orchestrated = orchestrate_words(
        raw_words=list(raw_words),
        diar_segments=list(diar_segments),
        tolerance=tolerance_sec,
    )

    convofuse_words: List[ConvoFuseWord] = []
    for idx, ow in enumerate(orchestrated):
        raw_dict = raw_words[idx] if idx < len(raw_words) else {}
        canonical_spk = speaker_mapper.to_canonical(ow.original_speaker_id)
        canonical_ovl_spks = tuple(
            sorted(
                {
                    speaker_mapper.to_canonical(s)
                    for s in ow.overlapping_speakers
                    if speaker_mapper.to_canonical(s) != "UNK"
                }
            )
        )
        canonical_ovl_durs = {
            speaker_mapper.to_canonical(k): float(v)
            for k, v in ow.overlap_durations.items()
            if speaker_mapper.to_canonical(k) != "UNK"
        }

        frame_mean: Optional[float] = None
        frame_max: Optional[float] = None
        ovl_source: Optional[str] = None
        if overlap_timeline is not None and ow.start is not None and ow.end is not None:
            times, probs, src_name = overlap_timeline
            ovl_source = src_name
            mask = (times >= float(ow.start)) & (times <= float(ow.end))
            if np.any(mask):
                sub_p = probs[mask]
                frame_mean = round(float(np.mean(sub_p)), 4)
                frame_max = round(float(np.max(sub_p)), 4)
            elif len(times) > 0:
                mid = 0.5 * (float(ow.start) + float(ow.end))
                nearest_idx = int(np.argmin(np.abs(times - mid)))
                val = round(float(probs[nearest_idx]), 4)
                frame_mean = val
                frame_max = val

        is_multi_spk = bool(ow.is_overlapping_speech or len(canonical_ovl_spks) > 1)
        high_posterior_ovl = bool(frame_max is not None and frame_max >= 0.40)

        limitation_note: Optional[str] = None
        if is_multi_spk:
            limitation_note = (
                "Multiple simultaneous speakers overlap this word interval; single-speaker-per-word "
                "schema assigns dominant overlap speaker while preserving secondary speakers in metadata."
            )
        elif high_posterior_ovl:
            limitation_note = (
                "Acoustic overlap posterior >= 0.40 detected on raw audio; single-speaker clustering "
                "attribution retained per Stage 5B.2 V4.2.1 conservative policy."
            )

        ovl_meta = OverlapMetadata(
            is_overlapping_speech=is_multi_spk,
            overlapping_speakers=canonical_ovl_spks,
            overlap_durations_sec=canonical_ovl_durs,
            frame_overlap_prob_mean=frame_mean,
            frame_overlap_prob_max=frame_max,
            overlap_posterior_Dragging_source=ovl_source,
            simultaneous_attribution_supported=False,
            limitation_note=limitation_note,
        )

        prov = WordProvenance(
            attribution_method="max_overlap_with_0.5s_midpoint_collar",
            attribution_status=ow.attribution_status,
            nearest_segment_distance_sec=ow.nearest_segment_distance,
            raw_rttm_speaker_id=ow.original_speaker_id,
            diarization_backend=diarization_backend,
            diarization_condition=diarization_condition,
            context_segment_id="unassigned",
            refinement_model="none",
            execution_mode=execution_mode,
        )

        cal_conf = raw_dict.get("confidence", ow.asr_confidence)
        align_score = raw_dict.get("alignment_score", ow.asr_confidence)
        align_status = str(raw_dict.get("alignment_status", "aligned"))

        convofuse_words.append(
            ConvoFuseWord(
                word_index=ow.word_index,
                word=ow.word,
                start=ow.start,
                end=ow.end,
                baseline_speaker_id=canonical_spk,
                refined_speaker_id=canonical_spk,
                asr_confidence=ow.asr_confidence,
                calibrated_confidence=float(cal_conf) if cal_conf is not None else ow.asr_confidence,
                alignment_score=float(align_score) if align_score is not None else None,
                alignment_status=align_status,
                overlap_metadata=ovl_meta,
                provenance=prov,
                refinement_decision=RefinementDecision.UNCHANGED.value,
                reconciliation_reason=RejectionOrAbstentionReason.UNCHANGED_BASELINE_CONFIRMED.value,
                proposed_speaker_id=canonical_spk,
                reasoning_note="Baseline acoustic speaker assignment prior to generative refinement.",
            )
        )

    return convofuse_words
