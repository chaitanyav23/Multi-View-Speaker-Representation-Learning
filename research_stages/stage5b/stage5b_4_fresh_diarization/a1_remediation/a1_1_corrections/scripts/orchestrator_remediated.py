#!/usr/bin/env python3
"""
Stage 5B.4-A.1.1 Remediation: Fully Integrated Fresh Diarization Orchestrator
Coordinates VAD interval generation, multiscale subsegmentation, dual-encoder caching,
Condition J affinity fusion, fresh spectral clustering, overlap attribution, output writing,
and dual-collar DER + cpWER evaluation.

Remediates all six A.1 issues:
1. Production cpWER evaluation with {"segments": [...]} envelope
2. Cryptographic checkpoint verification preflight on disk
3. Centralized non-duplicating repository root resolution
4. M4 overlap posteriors loaded and validated; M0 remains independent
5. Precise cache error handling (miss vs invalidation vs security violation)
6. Output RTTM path containment guard against traversal and symlink escape
"""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from path_utils import (
    get_repo_root,
    get_stage5b_4_dir,
    get_stage5b_4_cache_dir,
    get_stage2_posteriors_dir,
)
from condition_registry import CONDITIONS, BENCHMARK_RECORDINGS, get_condition, validate_registry
from vad_adapter_remediated import VADAdapterRemediated, canonical_interval_hash
from subsegment_generator_remediated import generate_all_scales, SCALES
from cache_manager_remediated import (
    IsolatedEmbeddingCacheManagerRemediated,
    CacheIdentity,
    CacheSecurityError,
    CacheValidationError,
    CACHE_SCHEMA_VERSION,
)
from feature_extraction_interface import (
    extract_titanet_embedding,
    extract_wespeaker_embedding,
    TITANET_EMBEDDING_DIM,
    WESPEAKER_EMBEDDING_DIM,
    MockTitaNetModel,
    MockWeSpeakerModel,
)
from affinity_fusion_remediated import (
    compute_multiscale_cosine_affinity,
    compute_condition_j_fusion,
)
from clustering_backends import (
    BaseClusteringBackend,
    SyntheticSpectralClusteringBackend,
    NeMoProductionClusteringBackend,
    BackendUnavailableError,
)
from overlap_adapter_remediated import apply_overlap_attribution
from overlap_posterior_loader import OverlapPosteriorLoader, OverlapPosteriorError
from rttm_adapter_remediated import (
    subsegments_to_single_speaker_segments,
    write_rttm_file,
)
from output_path_guard import validate_output_path, OutputSecurityError
from evaluation_adapters_remediated import (
    BaseEvaluationAdapter,
    SyntheticEvaluationAdapter,
    ProductionEvaluationAdapterRemediated,
)
from checkpoint_preflight import (
    verify_encoder_checkpoint_preflight,
    CheckpointPreflightError,
)


@dataclass
class PipelineExecutionStats:
    condition_id: str
    recording_id: str
    vad_id: str
    vad_interval_count: int
    vad_intervals_sha256: str
    subsegment_counts: Dict[int, int]
    titanet_cache_hits: int = 0
    titanet_cache_misses: int = 0
    titanet_cache_invalidations: int = 0
    wespeaker_cache_hits: int = 0
    wespeaker_cache_misses: int = 0
    wespeaker_cache_invalidations: int = 0
    extracted_embeddings: int = 0
    clustering_backend: str = "synthetic"
    estimated_k: int = 0
    overlap_method: str = "M0_no_overlap"
    output_rttm_path: Optional[str] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FreshDiarizationOrchestratorRemediated:
    """
    Remediated orchestrator for Stage 5B.4 fresh diarization.
    Enforces checkpoint verification preflight, precise cache tracking,
    containment guards, and full evaluation integration.
    """

    def __init__(
        self,
        cache_manager: IsolatedEmbeddingCacheManagerRemediated,
        vad_adapter: VADAdapterRemediated,
        clustering_backend: BaseClusteringBackend,
        evaluation_adapter: BaseEvaluationAdapter,
        overlap_loader: Optional[OverlapPosteriorLoader] = None,
        titanet_model: Optional[Any] = None,
        wespeaker_model: Optional[Any] = None,
        titanet_checkpoint_path: Optional[Path] = None,
        titanet_checkpoint_sha256: Optional[str] = None,
        wespeaker_checkpoint_path: Optional[Path] = None,
        wespeaker_checkpoint_sha256: Optional[str] = None,
        skip_checkpoint_preflight: bool = False,
        allowed_output_roots: Optional[List[Path]] = None,
    ):
        self.cache_mgr = cache_manager
        self.vad_adapter = vad_adapter
        self.clustering_backend = clustering_backend
        self.evaluation_adapter = evaluation_adapter
        self.overlap_loader = overlap_loader or OverlapPosteriorLoader()
        self.titanet_model = titanet_model
        self.wespeaker_model = wespeaker_model
        self.allowed_output_roots = allowed_output_roots or [get_stage5b_4_dir()]

        # Checkpoint preflight verification
        self.titanet_ckpt_path = Path(titanet_checkpoint_path) if titanet_checkpoint_path else None
        self.titanet_ckpt_hash = titanet_checkpoint_sha256
        self.wespeaker_ckpt_path = Path(wespeaker_checkpoint_path) if wespeaker_checkpoint_path else None
        self.wespeaker_ckpt_hash = wespeaker_checkpoint_sha256

        if not skip_checkpoint_preflight:
            self._run_preflight_checks()

    def _run_preflight_checks(self) -> None:
        """Validates encoder checkpoints on disk using SHA-256 before running."""
        if self.titanet_ckpt_path is not None:
            if not self.titanet_ckpt_hash:
                raise CheckpointPreflightError("titanet_checkpoint_sha256 required when path is provided.")
            res_tita = verify_encoder_checkpoint_preflight(
                "titanet_large", self.titanet_ckpt_path, self.titanet_ckpt_hash
            )
            self.titanet_ckpt_hash = res_tita.computed_sha256

        if self.wespeaker_ckpt_path is not None:
            if not self.wespeaker_ckpt_hash:
                raise CheckpointPreflightError("wespeaker_checkpoint_sha256 required when path is provided.")
            res_res = verify_encoder_checkpoint_preflight(
                "wespeaker_resnet293", self.wespeaker_ckpt_path, self.wespeaker_ckpt_hash
            )
            self.wespeaker_ckpt_hash = res_res.computed_sha256

    def process_recording(
        self,
        condition_id: str,
        recording_id: str,
        dfn_waveform: np.ndarray,
        mono_waveform: np.ndarray,
        posteriors_override: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None,
        sample_rate: int = 16000,
        output_dir: Optional[Path] = None,
        reference_rttm_path: Optional[Path] = None,
        aligned_words_json: Optional[Path] = None,
        reference_transcript_json: Optional[Path] = None,
    ) -> PipelineExecutionStats:
        """
        Executes the full fresh diarization pipeline for one recording and condition.
        """
        cond = get_condition(condition_id)
        vad_id = cond.vad_config.vad_id
        overlap_method = cond.overlap_config.method_id

        # 1. VAD Interval Generation & Overlap Posterior Loading
        if posteriors_override is not None:
            # Synthetic / test mode
            f_times, p_speech, p_ovl = posteriors_override
            from vad_adapter_remediated import binarize_posteriors_audited_reference
            vad_intervals = binarize_posteriors_audited_reference(
                times=f_times,
                probs=p_speech,
                onset=cond.vad_config.onset,
                offset=cond.vad_config.offset,
                min_duration_on=cond.vad_config.min_duration_on,
                min_duration_off=cond.vad_config.min_duration_off,
            )
            vad_hash = canonical_interval_hash(recording_id, vad_id, vad_intervals)

            # In override mode: M4 requires p_ovl to be provided
            if overlap_method == "M4_conservative_abstention" and p_ovl is None:
                raise OverlapPosteriorError(
                    f"Condition {condition_id} requires M4 overlap posteriors, but p_overlap is None."
                )
        else:
            vad_intervals, vad_hash = self.vad_adapter.generate_vad_intervals(recording_id, vad_id)
            f_times, _ = self.vad_adapter.load_posteriors(recording_id, "16k_mono")

            # M4 Overlap posterior integration
            if overlap_method == "M4_conservative_abstention":
                # Must load and validate real overlap posteriors from Stage 2 baseline
                f_times_ovl, p_ovl = self.overlap_loader.load_overlap_posteriors(recording_id)
                # Verify timestamp parity
                if len(f_times) != len(f_times_ovl) or not np.allclose(f_times, f_times_ovl, atol=1e-4):
                    raise OverlapPosteriorError(
                        f"Overlap frame times misaligned with VAD frame times for recording {recording_id}"
                    )
            else:
                # M0 single-speaker baseline does not require or use p_overlap
                p_ovl = None

        # 2. Dynamic Multiscale Subsegmentation
        all_scales = generate_all_scales(vad_intervals)
        base_subsegs = all_scales[4]  # Base scale 4 (0.50s win / 0.25s shift)

        sub_counts = {k: len(all_scales[k]) for k in range(5)}
        stats = PipelineExecutionStats(
            condition_id=condition_id,
            recording_id=recording_id,
            vad_id=vad_id,
            vad_interval_count=len(vad_intervals),
            vad_intervals_sha256=vad_hash,
            subsegment_counts=sub_counts,
            clustering_backend=type(self.clustering_backend).__name__,
            overlap_method=overlap_method,
        )

        if len(base_subsegs) == 0:
            # Degenerate case: no speech detected in recording
            return stats

        # 3. Dual-Encoder Embedding Extraction with Precise Cache Error Handling
        tita_embs_by_scale: Dict[int, torch.Tensor] = {}
        res_embs_by_scale: Dict[int, torch.Tensor] = {}

        tita_hash = self.titanet_ckpt_hash or "450f6bd279a3f783687dbd15814fd0f100000000000000000000000000000000"
        wespeaker_hash = self.wespeaker_ckpt_hash or "e7b8447ea84345e21af34a3985a4382600000000000000000000000000000000"

        for scale_k in range(5):
            subsegs_k = all_scales[scale_k]
            win_sec, shift_sec, _ = SCALES[scale_k][1], SCALES[scale_k][2], SCALES[scale_k][3]

            # --- TitaNet-Large (DFN Audio, 192d) ---
            id_tita = CacheIdentity(
                schema_version=CACHE_SCHEMA_VERSION,
                recording_id=recording_id,
                vad_config_id=vad_id,
                vad_intervals_sha256=vad_hash,
                encoder_name="titanet_large",
                encoder_checkpoint_sha256=tita_hash,
                audio_source_id="dfn",
                audio_preprocessing_id="raw_16k",
                scale_index=scale_k,
                window_sec=win_sec,
                shift_sec=shift_sec,
                sample_rate=sample_rate,
                frontend_config_version="nemo_msdd_v1",
            )

            res_load, status = self.cache_mgr.load_or_status(
                id_tita, expected_dim=TITANET_EMBEDDING_DIM, expected_subsegments=subsegs_k
            )

            if status == "HIT":
                stats.titanet_cache_hits += 1
                t_embs, t_norms = res_load
            else:
                if status == "INVALIDATION":
                    stats.titanet_cache_invalidations += 1
                else:
                    stats.titanet_cache_misses += 1

                if self.titanet_model is None:
                    raise BackendUnavailableError("TitaNet extraction model not provided for cache miss/invalidation.")

                extracted_list = []
                norms_list = []
                for sub in subsegs_k:
                    st_i = max(0, int(round(sub["start"] * sample_rate)))
                    en_i = min(len(dfn_waveform), int(round(sub["end"] * sample_rate)))
                    slice_wav = dfn_waveform[st_i:en_i]
                    res_t = extract_titanet_embedding(slice_wav, sample_rate, self.titanet_model)
                    extracted_list.append(res_t.embedding)
                    norms_list.append(res_t.l2_norm)
                    stats.extracted_embeddings += 1

                t_embs = np.array(extracted_list, dtype=np.float32)
                t_norms = np.array(norms_list, dtype=np.float32)
                starts = np.array([s["start"] for s in subsegs_k], dtype=np.float64)
                ends = np.array([s["end"] for s in subsegs_k], dtype=np.float64)
                self.cache_mgr.save_embeddings(id_tita, t_embs, t_norms, starts, ends)

            tita_embs_by_scale[scale_k] = torch.from_numpy(t_embs)

            # --- WeSpeaker ResNet293-LM (Mono Audio, 256d) ---
            id_res = CacheIdentity(
                schema_version=CACHE_SCHEMA_VERSION,
                recording_id=recording_id,
                vad_config_id=vad_id,
                vad_intervals_sha256=vad_hash,
                encoder_name="wespeaker_resnet293",
                encoder_checkpoint_sha256=wespeaker_hash,
                audio_source_id="16k_mono",
                audio_preprocessing_id="raw_16k",
                scale_index=scale_k,
                window_sec=win_sec,
                shift_sec=shift_sec,
                sample_rate=sample_rate,
                frontend_config_version="wespeaker_fbank80_v1",
            )

            res_load, status = self.cache_mgr.load_or_status(
                id_res, expected_dim=WESPEAKER_EMBEDDING_DIM, expected_subsegments=subsegs_k
            )

            if status == "HIT":
                stats.wespeaker_cache_hits += 1
                r_embs, r_norms = res_load
            else:
                if status == "INVALIDATION":
                    stats.wespeaker_cache_invalidations += 1
                else:
                    stats.wespeaker_cache_misses += 1

                if self.wespeaker_model is None:
                    raise BackendUnavailableError("WeSpeaker extraction model not provided for cache miss/invalidation.")

                extracted_list = []
                norms_list = []
                for sub in subsegs_k:
                    st_i = max(0, int(round(sub["start"] * sample_rate)))
                    en_i = min(len(mono_waveform), int(round(sub["end"] * sample_rate)))
                    slice_wav = mono_waveform[st_i:en_i]
                    res_w = extract_wespeaker_embedding(slice_wav, sample_rate, self.wespeaker_model)
                    extracted_list.append(res_w.embedding)
                    norms_list.append(res_w.l2_norm)
                    stats.extracted_embeddings += 1

                r_embs = np.array(extracted_list, dtype=np.float32)
                r_norms = np.array(norms_list, dtype=np.float32)
                starts = np.array([s["start"] for s in subsegs_k], dtype=np.float64)
                ends = np.array([s["end"] for s in subsegs_k], dtype=np.float64)
                self.cache_mgr.save_embeddings(id_res, r_embs, r_norms, starts, ends)

            res_embs_by_scale[scale_k] = torch.from_numpy(r_embs)

        # 4. Multiscale Cosine Affinity Matrix Construction
        A_tita = compute_multiscale_cosine_affinity(tita_embs_by_scale, all_scales)
        A_res = compute_multiscale_cosine_affinity(res_embs_by_scale, all_scales)

        # 5. Condition J Equal-Weight Fusion
        A_eq, _, _ = compute_condition_j_fusion(A_tita, A_res)
        # Guarantee floating-point numerical symmetry
        A_eq = 0.5 * (A_eq + A_eq.t())
        A_eq.fill_diagonal_(1.0)

        # 6. Fresh Clustering Backend Execution
        spk_labels, k_est = self.clustering_backend.cluster(
            affinity_matrix=A_eq,
            base_subsegments=base_subsegs,
            max_num_speakers=10,
        )
        stats.estimated_k = k_est

        # 7. Convert Clustering Output to Single-Speaker RTTM Segments
        single_spk_segs = subsegments_to_single_speaker_segments(base_subsegs, spk_labels)

        # 8. Overlap Attribution Handling
        final_segs, _ = apply_overlap_attribution(
            single_speaker_segments=single_spk_segs,
            p_overlap=p_ovl,
            frame_times=f_times,
            method_id=overlap_method,
            tau_overlap=cond.overlap_config.tau_overlap,
            min_overlap_dur=cond.overlap_config.min_overlap_dur,
            max_gap_sec=cond.overlap_config.max_gap_sec,
            min_turn_dur_sec=cond.overlap_config.min_turn_dur_sec,
        )

        # 9. Output RTTM Writing with Strict Containment Enforcement
        if output_dir is not None:
            validated_output_dir = validate_output_path(output_dir, allowed_roots=self.allowed_output_roots)
            rttm_path = validated_output_dir / f"{recording_id}_{condition_id}.rttm"
            write_rttm_file(final_segs, recording_id, rttm_path)
            stats.output_rttm_path = str(rttm_path)

            # 10. Evaluation if reference RTTM provided
            if reference_rttm_path is not None:
                eval_metrics = self.evaluation_adapter.evaluate(
                    hypothesis_rttm=rttm_path,
                    reference_rttm=reference_rttm_path,
                    aligned_words_json=aligned_words_json,
                    reference_transcript_json=reference_transcript_json,
                    k_hat=k_est,
                )
                stats.evaluation_metrics = eval_metrics

        return stats
