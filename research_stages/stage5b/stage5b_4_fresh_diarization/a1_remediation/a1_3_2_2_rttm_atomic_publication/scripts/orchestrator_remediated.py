#!/usr/bin/env python3
"""
Stage 5B.4-A.1.3.1 Remediation: Fully Hardened Fresh Diarization Orchestrator
Coordinates VAD interval generation, multiscale subsegmentation, dual-encoder caching,
Condition J affinity fusion, fresh spectral clustering, overlap attribution, output writing,
and dual-collar DER + Phase 20 cpWER evaluation.

Remediates all A.1.3.1 findings:
1. Production checkpoint roots cannot be expanded by caller arguments.
2. Direct byte-level cryptographic verification of source audio and PyanNet checkpoint.
3. Trusted specification validation for generation configuration and posterior semantics.
4. Official WeSpeaker ResNet293 model construction and TOCTOU in-memory byte loading.
5. RTTM destination symlink replacement race protection via os.O_NOFOLLOW.
6. Formalized M0 vs M4 execution eligibility and overlap provenance state separation.
"""

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
    get_stage5b_4_outputs_dir,
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
from feature_extraction_interface import (
    extract_titanet_embedding,
    extract_wespeaker_embedding,
    TITANET_EMBEDDING_DIM,
    WESPEAKER_EMBEDDING_DIM,
    MockTitaNetModel,
    MockWeSpeakerModel,
)

# A.1.3.1 remediated modules
from checkpoint_preflight import (
    ExecutionMode,
    VerifiedCheckpoint,
    run_production_checkpoint_preflight,
    verify_encoder_checkpoint_strict,
    create_synthetic_checkpoint_fixture,
    CheckpointPreflightError,
)
from model_factory import (
    VerifiedModelHandle,
    ControlledModelFactory,
    validate_model_handle_strict,
    ModelFactoryError,
)
from output_path_guard import (
    FinalRTTMPath,
    validate_final_rttm_path,
    validate_recording_id,
    validate_condition_id,
    revalidate_and_write_rttm,
    OutputSecurityError,
)
from cpwer_adapter import evaluate_cpwer
from overlap_provenance import (
    OverlapProvenanceError,
    StructuralValidationState,
    ProvenanceValidationState,
    OverallValidationState,
    ValidatedPosteriorContainer,
    OverlapProvenanceVerifier,
)
from overlap_posterior_loader import (
    OverlapPosteriorLoaderStrict,
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
    """
    Execution statistics and multi-state provenance record.
    Remediates Finding 6 by formalizing distinct fields for:
    - m0_execution_eligible: Whether M0 is authorized to run (does not consume overlap posteriors).
    - m4_execution_eligible: Whether M4 is authorized to run (requires verified overlap provenance).
    - overlap_provenance_verified: Whether overlap posterior provenance was verified.
    - overlap_production_eligible: Whether overlap posterior artifact is production eligible.
    """
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
    # Structural validity
    structurally_valid: bool = False
    # Finding 6 formalized eligibility fields
    m0_execution_eligible: bool = False
    m4_execution_eligible: bool = False
    overlap_provenance_verified: bool = False
    overlap_production_eligible: bool = False
    # Backwards compatibility aliases
    provenance_verified: bool = False
    production_eligible: bool = False

    def __post_init__(self):
        # Synchronize backward compatibility aliases
        if not self.provenance_verified and self.overlap_provenance_verified:
            self.provenance_verified = self.overlap_provenance_verified
        if not self.production_eligible and self.overlap_production_eligible:
            self.production_eligible = self.overlap_production_eligible

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FreshDiarizationOrchestratorRemediated:
    """
    Hardened Fresh Diarization Orchestrator for Stage 5B.4-A.1.3.1.
    Enforces mandatory preflight checkpoint verification against the immutable allowlist,
    defeats TOCTOU races via exact in-memory byte loading,
    restricts RTTM writes strictly to canonical outputs/ root with os.O_NOFOLLOW,
    and enforces multi-state validation on overlap posteriors.
    """

    def __init__(
        self,
        cache_manager: IsolatedEmbeddingCacheManagerRemediated,
        vad_adapter: VADAdapterRemediated,
        clustering_backend: BaseClusteringBackend,
        evaluation_adapter: BaseEvaluationAdapter,
        overlap_loader: Optional[OverlapPosteriorLoaderStrict] = None,
        titanet_handle: Optional[VerifiedModelHandle] = None,
        wespeaker_handle: Optional[VerifiedModelHandle] = None,
        titanet_checkpoint_path: Optional[Path] = None,
        titanet_checkpoint_sha256: Optional[str] = None,
        wespeaker_checkpoint_path: Optional[Path] = None,
        wespeaker_checkpoint_sha256: Optional[str] = None,
        execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
        custom_permitted_checkpoint_roots: Optional[List[Path]] = None,
        custom_output_root: Optional[Path] = None,
        # Backward compatibility keyword arguments (strictly prohibited in production)
        titanet_model: Optional[Any] = None,
        wespeaker_model: Optional[Any] = None,
    ):
        self.execution_mode = execution_mode
        self.cache_mgr = cache_manager
        self.vad_adapter = vad_adapter
        self.clustering_backend = clustering_backend
        self.evaluation_adapter = evaluation_adapter
        self.overlap_loader = overlap_loader or OverlapPosteriorLoaderStrict(execution_mode=execution_mode)
        self.custom_permitted_checkpoint_roots = custom_permitted_checkpoint_roots
        self.custom_output_root = custom_output_root

        # Prohibit arbitrary preloaded models in production
        if self.execution_mode == ExecutionMode.PRODUCTION:
            if titanet_model is not None or wespeaker_model is not None:
                raise ModelFactoryError(
                    "Security violation: Arbitrary preloaded model instances (titanet_model/wespeaker_model) "
                    "are strictly prohibited in PRODUCTION mode. Use ControlledModelFactory or pass VerifiedModelHandle."
                )

        # Checkpoint paths and expected hashes
        self.titanet_ckpt_path = Path(titanet_checkpoint_path) if titanet_checkpoint_path else None
        self.titanet_ckpt_hash = titanet_checkpoint_sha256
        self.wespeaker_ckpt_path = Path(wespeaker_checkpoint_path) if wespeaker_checkpoint_path else None
        self.wespeaker_ckpt_hash = wespeaker_checkpoint_sha256

        self.titanet_handle: Optional[VerifiedModelHandle] = None
        self.wespeaker_handle: Optional[VerifiedModelHandle] = None

        # Build or validate model handles
        self._initialize_model_handles(
            titanet_handle=titanet_handle,
            wespeaker_handle=wespeaker_handle,
            mock_titanet=titanet_model,
            mock_wespeaker=wespeaker_model,
        )

    def _initialize_model_handles(
        self,
        titanet_handle: Optional[VerifiedModelHandle],
        wespeaker_handle: Optional[VerifiedModelHandle],
        mock_titanet: Optional[Any],
        mock_wespeaker: Optional[Any],
    ) -> None:
        """
        Constructs and binds model handles through ControlledModelFactory,
        or strictly validates caller-supplied VerifiedModelHandle instances.
        """
        if self.execution_mode == ExecutionMode.PRODUCTION:
            # 1. Validate caller-supplied handles if present
            if titanet_handle is not None:
                self.titanet_handle = validate_model_handle_strict(
                    titanet_handle, "titanet_large", ExecutionMode.PRODUCTION
                )
            else:
                v_tita = verify_encoder_checkpoint_strict(
                    encoder_name="titanet_large",
                    checkpoint_path=self.titanet_ckpt_path,
                    expected_sha256=self.titanet_ckpt_hash,
                    execution_mode=ExecutionMode.PRODUCTION,
                    custom_permitted_roots=self.custom_permitted_checkpoint_roots,
                )
                self.titanet_handle = ControlledModelFactory.build_production_model(v_tita)

            if wespeaker_handle is not None:
                self.wespeaker_handle = validate_model_handle_strict(
                    wespeaker_handle, "wespeaker_resnet293", ExecutionMode.PRODUCTION
                )
            else:
                v_wes = verify_encoder_checkpoint_strict(
                    encoder_name="wespeaker_resnet293",
                    checkpoint_path=self.wespeaker_ckpt_path,
                    expected_sha256=self.wespeaker_ckpt_hash,
                    execution_mode=ExecutionMode.PRODUCTION,
                    custom_permitted_roots=self.custom_permitted_checkpoint_roots,
                )
                self.wespeaker_handle = ControlledModelFactory.build_production_model(v_wes)

        elif self.execution_mode == ExecutionMode.SYNTHETIC:
            # Synthetic mode
            if titanet_handle is not None:
                self.titanet_handle = validate_model_handle_strict(
                    titanet_handle, "titanet_large", ExecutionMode.SYNTHETIC
                )
            else:
                t_mock = mock_titanet or MockTitaNetModel(return_dim=TITANET_EMBEDDING_DIM)
                if self.titanet_ckpt_path and self.titanet_ckpt_hash:
                    v_tita = verify_encoder_checkpoint_strict(
                        "titanet_large", self.titanet_ckpt_path, self.titanet_ckpt_hash, ExecutionMode.SYNTHETIC
                    )
                else:
                    v_tita = create_synthetic_checkpoint_fixture("titanet_large", Path("/tmp/mock_titanet.nemo"))
                self.titanet_handle = ControlledModelFactory.build_synthetic_model(v_tita, t_mock)

            if wespeaker_handle is not None:
                self.wespeaker_handle = validate_model_handle_strict(
                    wespeaker_handle, "wespeaker_resnet293", ExecutionMode.SYNTHETIC
                )
            else:
                w_mock = mock_wespeaker or MockWeSpeakerModel(return_dim=WESPEAKER_EMBEDDING_DIM)
                if self.wespeaker_ckpt_path and self.wespeaker_ckpt_hash:
                    v_wes = verify_encoder_checkpoint_strict(
                        "wespeaker_resnet293", self.wespeaker_ckpt_path, self.wespeaker_ckpt_hash, ExecutionMode.SYNTHETIC
                    )
                else:
                    v_wes = create_synthetic_checkpoint_fixture("wespeaker_resnet293", Path("/tmp/mock_wespeaker.onnx"))
                self.wespeaker_handle = ControlledModelFactory.build_synthetic_model(v_wes, w_mock)
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
        source_audio_override: Optional[Path] = None,
        pyannet_checkpoint_override: Optional[Path] = None,
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

        # 2. VAD Interval Generation & Multi-State Overlap Posterior Loading
        structurally_valid = False
        overlap_provenance_verified = False
        overlap_production_eligible = False
        m0_execution_eligible = False
        m4_execution_eligible = False

        if posteriors_override is not None:
            # Synthetic / override mode
            f_times, p_speech, p_ovl_raw = posteriors_override
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

            if overlap_method == "M4_conservative_abstention":
                if p_ovl_raw is None:
                    raise OverlapProvenanceError(
                        f"Condition {cond_id} requires M4 overlap posteriors, but p_overlap is None."
                    )
                # Load and validate synthetic container
                container = self.overlap_loader.load_and_validate(
                    recording_id=rec_id,
                    stream="mono",
                    expected_duration=expected_duration,
                    override_arrays=(f_times, p_ovl_raw),
                    is_synthetic=True,
                    source_audio_override=source_audio_override,
                    pyannet_checkpoint_override=pyannet_checkpoint_override,
                )
                p_ovl = container.p_overlap
                structurally_valid = container.structural_state.structurally_valid
                overlap_provenance_verified = container.provenance_state.provenance_verified
                overlap_production_eligible = container.overall_state.production_eligible
                m4_execution_eligible = overlap_production_eligible

                # Production M4 must fail closed if not production eligible
                if self.execution_mode == ExecutionMode.PRODUCTION and not overlap_production_eligible:
                    raise OverlapProvenanceError(
                        f"Production M4 rejected for '{rec_id}': Posterior is not production eligible. "
                        f"Reasons: {container.overall_state.reasons}"
                    )
            else:
                p_ovl = None
                structurally_valid = True
                overlap_provenance_verified = False
                overlap_production_eligible = False
                m0_execution_eligible = True
                m4_execution_eligible = False

        else:
            vad_intervals, vad_hash = self.vad_adapter.generate_vad_intervals(rec_id, vad_id)
            f_times, _ = self.vad_adapter.load_posteriors(rec_id, "16k_mono")

            # M4 Overlap posterior integration with strict multi-state verification
            if overlap_method == "M4_conservative_abstention":
                container = self.overlap_loader.load_and_validate(
                    recording_id=rec_id,
                    stream="mono",
                    expected_duration=expected_duration,
                    source_audio_override=source_audio_override,
                    pyannet_checkpoint_override=pyannet_checkpoint_override,
                )
                # Verify exact timestamp alignment
                if len(f_times) != len(container.frame_times) or not np.allclose(f_times, container.frame_times, atol=1e-4):
                    raise OverlapProvenanceError(
                        f"Overlap frame times misaligned with VAD frame times for recording {rec_id}"
                    )

                structurally_valid = container.structural_state.structurally_valid
                overlap_provenance_verified = container.provenance_state.provenance_verified
                overlap_production_eligible = container.overall_state.production_eligible
                m4_execution_eligible = overlap_production_eligible

                # Production M4 fails closed if not production eligible
                if self.execution_mode == ExecutionMode.PRODUCTION and not overlap_production_eligible:
                    raise OverlapProvenanceError(
                        f"Production M4 rejected for '{rec_id}': Posterior is not production eligible. "
                        f"Reasons: {container.overall_state.reasons}"
                    )
                p_ovl = container.p_overlap
            else:
                # M0 does not require overlap posterior provenance
                p_ovl = None
                structurally_valid = True
                overlap_provenance_verified = False
                overlap_production_eligible = False
                m0_execution_eligible = True
                m4_execution_eligible = False

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
            structurally_valid=structurally_valid,
            m0_execution_eligible=m0_execution_eligible,
            m4_execution_eligible=m4_execution_eligible,
            overlap_provenance_verified=overlap_provenance_verified,
            overlap_production_eligible=overlap_production_eligible,
            provenance_verified=overlap_provenance_verified,
            production_eligible=overlap_production_eligible,
        )

        if len(base_subsegs) == 0:
            return stats

        # 4. Dual-Encoder Embedding Extraction via Verified Model Handles
        tita_embs_by_scale: Dict[int, torch.Tensor] = {}
        res_embs_by_scale: Dict[int, torch.Tensor] = {}

        tita_hash = self.titanet_handle.verified_checkpoint.computed_sha256
        wespeaker_hash = self.wespeaker_handle.verified_checkpoint.computed_sha256

        for scale_k in range(5):
            subsegs_k = all_scales[scale_k]
            win_sec, shift_sec = SCALES[scale_k][1], SCALES[scale_k][2]

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

                extracted_list = []
                norms_list = []
                for sub in subsegs_k:
                    st_i = max(0, int(round(sub["start"] * sample_rate)))
                    en_i = min(len(dfn_waveform), int(round(sub["end"] * sample_rate)))
                    slice_wav = dfn_waveform[st_i:en_i]
                    res_t = extract_titanet_embedding(slice_wav, sample_rate, self.titanet_handle.model)
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

                extracted_list = []
                norms_list = []
                for sub in subsegs_k:
                    st_i = max(0, int(round(sub["start"] * sample_rate)))
                    en_i = min(len(mono_waveform), int(round(sub["end"] * sample_rate)))
                    slice_wav = mono_waveform[st_i:en_i]
                    res_w = extract_wespeaker_embedding(slice_wav, sample_rate, self.wespeaker_handle.model)
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
        A_eq = 0.5 * (A_eq + A_eq.t())
        A_eq.fill_diagonal_(1.0)

        # 7. Fresh Clustering Backend Execution
        spk_labels, k_est = self.clustering_backend.cluster(
            affinity_matrix=A_eq,
            base_subsegments=base_subsegs,
            max_num_speakers=10,
        )
        stats.estimated_k = k_est

        # 8. Single-Speaker Segments Conversion
        from rttm_adapter_remediated import subsegments_to_single_speaker_segments
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

        # 10. Canonical Output RTTM Writing with Strict Containment and os.O_NOFOLLOW
        final_rttm_obj = validate_final_rttm_path(
            recording_id=rec_id,
            condition_id=cond_id,
            execution_mode=self.execution_mode,
            custom_output_root=self.custom_output_root or output_dir,
        )
        revalidate_and_write_rttm(final_rttm_obj, final_segs)
        stats.output_rttm_path = str(final_rttm_obj.path)

        # 11. Evaluation if reference RTTM provided
        if reference_rttm_path is not None:
            eval_metrics = self.evaluation_adapter.evaluate(
                hypothesis_rttm=final_rttm_obj.path,
                reference_rttm=reference_rttm_path,
                aligned_words_json=aligned_words_json,
                reference_transcript_json=reference_transcript_json,
                k_hat=k_est,
            )
            stats.evaluation_metrics = eval_metrics

        return stats
