#!/usr/bin/env python3
"""
Stage 5B.4-A.1.2 Remediation: Fully Hardened Fresh Diarization Orchestrator
Coordinates VAD interval generation, multiscale subsegmentation, dual-encoder caching,
Condition J affinity fusion, fresh spectral clustering, overlap attribution, output writing,
and dual-collar DER + Phase 20 cpWER evaluation.

Remediates all four A.1.2 issues:
1. Mandatory production checkpoint verification (no bypass, no placeholder hashes)
2. Final RTTM filename and identifier containment validation
3. Differential cpWER word-attribution parity with Phase 20
4. Strict overlap timestamp validation and provenance verification
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
    get_stage5b_4_output_dir,
    get_stage2_posteriors_dir,
)

# Reference condition registry from A.1
A1_SCRIPTS = get_stage5b_4_dir() / "a1_remediation" / "scripts"
if str(A1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A1_SCRIPTS))

from condition_registry import CONDITIONS, BENCHMARK_RECORDINGS, get_condition, validate_registry
from vad_adapter_remediated import VADAdapterRemediated, canonical_interval_hash
from subsegment_generator_remediated import generate_all_scales, SCALES
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
from rttm_adapter_remediated import (
    subsegments_to_single_speaker_segments,
    write_rttm_file,
)
from feature_extraction_interface import (
    extract_titanet_embedding,
    extract_wespeaker_embedding,
    TITANET_EMBEDDING_DIM,
    WESPEAKER_EMBEDDING_DIM,
    MockTitaNetModel,
    MockWeSpeakerModel,
)

# A.1.2 remediated modules
from checkpoint_preflight import (
    ExecutionMode,
    VerifiedCheckpoint,
    run_production_checkpoint_preflight,
    verify_encoder_checkpoint_strict,
    CheckpointPreflightError,
)
from output_path_guard import (
    validate_final_rttm_path,
    validate_recording_id,
    validate_condition_id,
    OutputSecurityError,
)
from cpwer_adapter import evaluate_cpwer
from overlap_posterior_loader import (
    OverlapPosteriorLoaderStrict,
    OverlapPosteriorError,
    validate_frame_timestamps_strict,
    validate_posterior_values_strict,
)
from cache_manager_remediated import (
    IsolatedEmbeddingCacheManagerRemediated,
    CacheIdentity,
    CacheSecurityError,
    CacheValidationError,
    CACHE_SCHEMA_VERSION,
)
from evaluation_adapters_remediated import (
    BaseEvaluationAdapter,
    SyntheticEvaluationAdapter,
    ProductionEvaluationAdapterRemediated,
)


@dataclass
class PipelineExecutionStats:
    condition_id: str
    recording_id: str
    vad_id: str
    vad_interval_count: int
    vad_intervals_sha256: str
    subsegment_counts: Dict[int, int]
    execution_mode: str = "production"
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
    provenance_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FreshDiarizationOrchestratorRemediated:
    """
    Hardened Fresh Diarization Orchestrator for Stage 5B.4-A.1.2.
    Enforces mandatory preflight checkpoint verification in production mode,
    full filename and identifier containment, Phase 20 parity, and strict overlap provenance.
    """

    def __init__(
        self,
        cache_manager: IsolatedEmbeddingCacheManagerRemediated,
        vad_adapter: VADAdapterRemediated,
        clustering_backend: BaseClusteringBackend,
        evaluation_adapter: BaseEvaluationAdapter,
        overlap_loader: Optional[OverlapPosteriorLoaderStrict] = None,
        titanet_model: Optional[Any] = None,
        wespeaker_model: Optional[Any] = None,
        titanet_checkpoint_path: Optional[Path] = None,
        titanet_checkpoint_sha256: Optional[str] = None,
        wespeaker_checkpoint_path: Optional[Path] = None,
        wespeaker_checkpoint_sha256: Optional[str] = None,
        execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
        custom_permitted_checkpoint_roots: Optional[List[Path]] = None,
        custom_allowed_output_roots: Optional[List[Path]] = None,
    ):
        self.execution_mode = execution_mode
        self.cache_mgr = cache_manager
        self.vad_adapter = vad_adapter
        self.clustering_backend = clustering_backend
        self.evaluation_adapter = evaluation_adapter
        self.overlap_loader = overlap_loader or OverlapPosteriorLoaderStrict(execution_mode=execution_mode)
        self.titanet_model = titanet_model
        self.wespeaker_model = wespeaker_model
        self.custom_permitted_checkpoint_roots = custom_permitted_checkpoint_roots
        self.custom_allowed_output_roots = custom_allowed_output_roots

        # Checkpoint paths and expected hashes
        self.titanet_ckpt_path = Path(titanet_checkpoint_path) if titanet_checkpoint_path else None
        self.titanet_ckpt_hash = titanet_checkpoint_sha256
        self.wespeaker_ckpt_path = Path(wespeaker_checkpoint_path) if wespeaker_checkpoint_path else None
        self.wespeaker_ckpt_hash = wespeaker_checkpoint_sha256

        self.verified_titanet: Optional[VerifiedCheckpoint] = None
        self.verified_wespeaker: Optional[VerifiedCheckpoint] = None

        # Mandatory Preflight Verification
        self._run_preflight_checks()

    def _run_preflight_checks(self) -> None:
        """
        Executes preflight verification.
        In PRODUCTION mode: both checkpoints are mandatory, verified on disk with 64-char hex SHA-256.
        In SYNTHETIC mode: verifies test checkpoints if supplied or uses mock descriptors.
        """
        if self.execution_mode == ExecutionMode.PRODUCTION:
            v_tita, v_wes = run_production_checkpoint_preflight(
                titanet_path=self.titanet_ckpt_path,
                titanet_expected_sha256=self.titanet_ckpt_hash,
                wespeaker_path=self.wespeaker_ckpt_path,
                wespeaker_expected_sha256=self.wespeaker_ckpt_hash,
                custom_permitted_roots=self.custom_permitted_checkpoint_roots,
            )
            self.verified_titanet = v_tita
            self.verified_wespeaker = v_wes
        elif self.execution_mode == ExecutionMode.SYNTHETIC:
            if self.titanet_ckpt_path is not None and self.titanet_ckpt_hash is not None:
                self.verified_titanet = verify_encoder_checkpoint_strict(
                    encoder_name="titanet_large",
                    checkpoint_path=self.titanet_ckpt_path,
                    expected_sha256=self.titanet_ckpt_hash,
                    execution_mode=ExecutionMode.SYNTHETIC,
                    custom_permitted_roots=self.custom_permitted_checkpoint_roots,
                )
            else:
                self.verified_titanet = VerifiedCheckpoint(
                    encoder_name="titanet_large",
                    verified_path=Path("/mock/titanet.nemo"),
                    computed_sha256="synth_titanet_placeholder_hash",
                    file_size_bytes=0,
                    is_synthetic=True,
                )

            if self.wespeaker_ckpt_path is not None and self.wespeaker_ckpt_hash is not None:
                self.verified_wespeaker = verify_encoder_checkpoint_strict(
                    encoder_name="wespeaker_resnet293",
                    checkpoint_path=self.wespeaker_ckpt_path,
                    expected_sha256=self.wespeaker_ckpt_hash,
                    execution_mode=ExecutionMode.SYNTHETIC,
                    custom_permitted_roots=self.custom_permitted_checkpoint_roots,
                )
            else:
                self.verified_wespeaker = VerifiedCheckpoint(
                    encoder_name="wespeaker_resnet293",
                    verified_path=Path("/mock/wespeaker.onnx"),
                    computed_sha256="synth_wespeaker_placeholder_hash",
                    file_size_bytes=0,
                    is_synthetic=True,
                )
        else:
            raise CheckpointPreflightError(f"Unsupported execution mode: {self.execution_mode}")

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
        expected_duration: Optional[float] = None,
    ) -> PipelineExecutionStats:
        """
        Executes the full fresh diarization pipeline for one recording and condition.
        """
        # 1. Identifier validation
        cond_id = validate_condition_id(condition_id)
        rec_id = validate_recording_id(recording_id, execution_mode=self.execution_mode)

        cond = get_condition(cond_id)
        vad_id = cond.vad_config.vad_id
        overlap_method = cond.overlap_config.method_id

        # 2. VAD Interval Generation & Overlap Posterior Loading with Strict Provenance
        provenance_verified = False
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
            vad_hash = canonical_interval_hash(rec_id, vad_id, vad_intervals)

            # In override mode: M4 requires p_ovl to be provided and valid
            if overlap_method == "M4_conservative_abstention":
                if p_ovl is None:
                    raise OverlapPosteriorError(
                        f"Condition {cond_id} requires M4 overlap posteriors, but p_overlap is None."
                    )
                # Validate synthetic posterior arrays
                validate_frame_timestamps_strict(f_times, expected_duration=expected_duration)
                p_ovl = validate_posterior_values_strict(p_ovl, len(f_times))
                provenance_verified = True
        else:
            vad_intervals, vad_hash = self.vad_adapter.generate_vad_intervals(rec_id, vad_id)
            f_times, _ = self.vad_adapter.load_posteriors(rec_id, "16k_mono")

            # M4 Overlap posterior integration with strict provenance
            if overlap_method == "M4_conservative_abstention":
                f_times_ovl, p_ovl, prov = self.overlap_loader.load_and_validate(
                    rec_id, expected_duration=expected_duration
                )
                # Verify exact timestamp alignment
                if len(f_times) != len(f_times_ovl) or not np.allclose(f_times, f_times_ovl, atol=1e-4):
                    raise OverlapPosteriorError(
                        f"Overlap frame times misaligned with VAD frame times for recording {rec_id}"
                    )
                provenance_verified = prov.is_verified
            else:
                p_ovl = None
                provenance_verified = True

        # 3. Dynamic Multiscale Subsegmentation
        all_scales = generate_all_scales(vad_intervals)
        base_subsegs = all_scales[4]  # Base scale 4 (0.50s win / 0.25s shift)

        sub_counts = {k: len(all_scales[k]) for k in range(5)}
        stats = PipelineExecutionStats(
            condition_id=cond_id,
            recording_id=rec_id,
            vad_id=vad_id,
            vad_interval_count=len(vad_intervals),
            vad_intervals_sha256=vad_hash,
            subsegment_counts=sub_counts,
            execution_mode=self.execution_mode.value,
            clustering_backend=type(self.clustering_backend).__name__,
            overlap_method=overlap_method,
            provenance_verified=provenance_verified,
        )

        if len(base_subsegs) == 0:
            return stats

        # 4. Dual-Encoder Embedding Extraction with Precise Cache Error Handling
        tita_embs_by_scale: Dict[int, torch.Tensor] = {}
        res_embs_by_scale: Dict[int, torch.Tensor] = {}

        tita_hash = self.verified_titanet.computed_sha256
        wespeaker_hash = self.verified_wespeaker.computed_sha256

        for scale_k in range(5):
            subsegs_k = all_scales[scale_k]
            win_sec, shift_sec, _ = SCALES[scale_k][1], SCALES[scale_k][2], SCALES[scale_k][3]

            # --- TitaNet-Large (DFN Audio, 192d) ---
            id_tita = CacheIdentity(
                schema_version=CACHE_SCHEMA_VERSION,
                recording_id=rec_id,
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
                recording_id=rec_id,
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

        # 5. Multiscale Cosine Affinity Matrix Construction
        A_tita = compute_multiscale_cosine_affinity(tita_embs_by_scale, all_scales)
        A_res = compute_multiscale_cosine_affinity(res_embs_by_scale, all_scales)

        # 6. Condition J Equal-Weight Fusion
        A_eq, _, _ = compute_condition_j_fusion(A_tita, A_res)
        # Numerical floating point symmetry guarantee
        A_eq = 0.5 * (A_eq + A_eq.t())
        A_eq.fill_diagonal_(1.0)

        # 7. Fresh Clustering Backend Execution
        spk_labels, k_est = self.clustering_backend.cluster(
            affinity_matrix=A_eq,
            base_subsegments=base_subsegs,
            max_num_speakers=10,
        )
        stats.estimated_k = k_est

        # 8. Convert Clustering Output to Single-Speaker RTTM Segments
        single_spk_segs = subsegments_to_single_speaker_segments(base_subsegs, spk_labels)

        # 9. Overlap Attribution Handling
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

        # 10. Output RTTM Writing with Final Path Containment Enforcement
        if output_dir is not None:
            validated_final_rttm = validate_final_rttm_path(
                output_dir=output_dir,
                recording_id=rec_id,
                condition_id=cond_id,
                execution_mode=self.execution_mode,
                custom_allowed_roots=self.custom_allowed_output_roots,
            )
            write_rttm_file(final_segs, rec_id, validated_final_rttm)
            stats.output_rttm_path = str(validated_final_rttm)

            # 11. Evaluation if reference RTTM provided
            if reference_rttm_path is not None:
                eval_metrics = self.evaluation_adapter.evaluate(
                    hypothesis_rttm=validated_final_rttm,
                    reference_rttm=reference_rttm_path,
                    aligned_words_json=aligned_words_json,
                    reference_transcript_json=reference_transcript_json,
                    k_hat=k_est,
                )
                stats.evaluation_metrics = eval_metrics

        return stats
