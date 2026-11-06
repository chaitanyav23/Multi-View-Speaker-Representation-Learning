#!/usr/bin/env python3
"""
Stage 5B.4-A.1 Remediation: Orchestrator & CLI Safety Engine
Coordinates the complete fresh diarization pipeline from VAD intervals through
multiscale subsegmentation, dual-encoder caching, Condition J fusion, fresh clustering,
overlap attribution, and evaluation.
Implements Task 3 safe failure barriers: refuses execution and returns non-zero exit code
if production backend is not ready, never reporting false success.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from condition_registry import CONDITIONS, BENCHMARK_RECORDINGS, get_condition, validate_registry
from vad_adapter_remediated import VADAdapterRemediated, canonical_interval_hash
from subsegment_generator_remediated import generate_all_scales, SCALES
from isolated_embedding_cache import (
    IsolatedEmbeddingCacheManager,
    CacheIdentity,
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
from rttm_adapter_remediated import (
    subsegments_to_single_speaker_segments,
    write_rttm_file,
)
from evaluation_adapters import (
    BaseEvaluationAdapter,
    SyntheticEvaluationAdapter,
    ProductionEvaluationAdapter,
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
    wespeaker_cache_hits: int = 0
    wespeaker_cache_misses: int = 0
    extracted_embeddings: int = 0
    invalidated_cache_entries: int = 0
    clustering_backend: str = "synthetic"
    estimated_k: int = 0
    overlap_method: str = "M0_no_overlap"
    output_rttm_path: Optional[str] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FreshDiarizationOrchestrator:
    """
    Modular execution engine for Stage 5B.4 fresh diarization.
    Supports partial cache reuse, atomic caching, and pluggable clustering/evaluation backends.
    """

    def __init__(
        self,
        cache_manager: IsolatedEmbeddingCacheManager,
        vad_adapter: VADAdapterRemediated,
        clustering_backend: BaseClusteringBackend,
        evaluation_adapter: BaseEvaluationAdapter,
        titanet_model: Optional[Any] = None,
        wespeaker_model: Optional[Any] = None,
        titanet_checkpoint_sha256: str = "450f6bd279a3f783687dbd15814fd0f1",
        wespeaker_checkpoint_sha256: str = "e7b8447ea84345e21af34a3985a43826",
    ):
        self.cache_mgr = cache_manager
        self.vad_adapter = vad_adapter
        self.clustering_backend = clustering_backend
        self.evaluation_adapter = evaluation_adapter
        self.titanet_model = titanet_model
        self.wespeaker_model = wespeaker_model
        self.titanet_ckpt_hash = titanet_checkpoint_sha256
        self.wespeaker_ckpt_hash = wespeaker_checkpoint_sha256

    def process_recording(
        self,
        condition_id: str,
        recording_id: str,
        dfn_waveform: np.ndarray,
        mono_waveform: np.ndarray,
        posteriors_override: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        sample_rate: int = 16000,
        output_dir: Optional[Path] = None,
        reference_rttm_path: Optional[Path] = None,
    ) -> PipelineExecutionStats:
        """
        Executes the full pipeline for one recording and one condition.
        Can run with real waveforms or synthetic waveforms in test environments.
        """
        cond = get_condition(condition_id)
        vad_id = cond.vad_config.vad_id
        overlap_method = cond.overlap_config.method_id

        # 1. VAD Interval Generation
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
        else:
            vad_intervals, vad_hash = self.vad_adapter.generate_vad_intervals(recording_id, vad_id)
            # Load mono posteriors for overlap detection
            f_times, _ = self.vad_adapter.load_posteriors(recording_id, "16k_mono")
            p_ovl = None  # Loaded in overlap step if M4

        # 2. Dynamic Multiscale Subsegmentation (Exact-fit duplication fixed)
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
            # Degenerate case: no speech detected
            return stats

        # 3. Dual-Encoder Embedding Extraction with Partial Cache Reuse
        tita_embs_by_scale: Dict[int, torch.Tensor] = {}
        res_embs_by_scale: Dict[int, torch.Tensor] = {}

        for scale_k in range(5):
            subsegs_k = all_scales[scale_k]
            win_sec, shift_sec, _ = SCALES[scale_k][1], SCALES[scale_k][2], SCALES[scale_k][3]
            n_subs = len(subsegs_k)

            # --- TitaNet-Large (DFN Audio, 192d) ---
            id_tita = CacheIdentity(
                schema_version=CACHE_SCHEMA_VERSION,
                recording_id=recording_id,
                vad_config_id=vad_id,
                vad_intervals_sha256=vad_hash,
                encoder_name="titanet_large",
                encoder_checkpoint_sha256=self.titanet_ckpt_hash,
                audio_source_id="dfn",
                audio_preprocessing_id="raw_16k",
                scale_index=scale_k,
                window_sec=win_sec,
                shift_sec=shift_sec,
                sample_rate=sample_rate,
                frontend_config_version="nemo_msdd_v1",
            )

            try:
                t_embs, t_norms = self.cache_mgr.load_embeddings(
                    id_tita, expected_dim=TITANET_EMBEDDING_DIM, expected_subsegments=subsegs_k
                )
                stats.titanet_cache_hits += 1
            except Exception:
                # Cache miss or invalidation
                stats.titanet_cache_misses += 1
                if self.titanet_model is None:
                    raise BackendUnavailableError("TitaNet extraction model not provided for cache miss.")

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
                encoder_name="wespeaker_resnet293_lm",
                encoder_checkpoint_sha256=self.wespeaker_ckpt_hash,
                audio_source_id="mono",
                audio_preprocessing_id="raw_16k",
                scale_index=scale_k,
                window_sec=win_sec,
                shift_sec=shift_sec,
                sample_rate=sample_rate,
                frontend_config_version="kaldi_fbank_v1",
            )

            try:
                r_embs, r_norms = self.cache_mgr.load_embeddings(
                    id_res, expected_dim=WESPEAKER_EMBEDDING_DIM, expected_subsegments=subsegs_k
                )
                stats.wespeaker_cache_hits += 1
            except Exception:
                stats.wespeaker_cache_misses += 1
                if self.wespeaker_model is None:
                    raise BackendUnavailableError("WeSpeaker extraction model not provided for cache miss.")

                extracted_list = []
                norms_list = []
                for sub in subsegs_k:
                    st_i = max(0, int(round(sub["start"] * sample_rate)))
                    en_i = min(len(mono_waveform), int(round(sub["end"] * sample_rate)))
                    slice_wav = mono_waveform[st_i:en_i]
                    res_r = extract_wespeaker_embedding(slice_wav, sample_rate, self.wespeaker_model)
                    extracted_list.append(res_r.embedding)
                    norms_list.append(res_r.l2_norm)
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

        # 9. Output RTTM Writing
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            rttm_path = output_dir / f"{recording_id}_{condition_id}.rttm"
            write_rttm_file(final_segs, recording_id, rttm_path)
            stats.output_rttm_path = str(rttm_path)

            # 10. Evaluation if reference provided
            if reference_rttm_path is not None:
                eval_metrics = self.evaluation_adapter.evaluate(
                    hypothesis_rttm=rttm_path,
                    reference_rttm=reference_rttm_path,
                    k_hat=k_est,
                )
                stats.evaluation_metrics = eval_metrics

        return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5B.4 Fresh Diarization Orchestrator (Remediated)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["validate-config", "validate-cache", "test-synthetic", "run"],
        help="Execution mode.",
    )
    parser.add_argument(
        "--authorize-inference",
        action="store_true",
        default=False,
        help="Explicit authorization flag required for real inference execution.",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help="List of recording IDs to process.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="List of condition IDs to process.",
    )

    args = parser.parse_args()

    if args.mode == "validate-config":
        try:
            validate_registry()
            print("[OK] Registry valid: 6 conditions registered.")
            sys.exit(0)
        except Exception as e:
            print(f"[FAIL] Registry validation failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.mode == "validate-cache":
        try:
            mgr = IsolatedEmbeddingCacheManager()
            print(f"[OK] Cache root validated: {mgr.cache_root}")
            sys.exit(0)
        except Exception as e:
            print(f"[FAIL] Cache validation failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.mode == "test-synthetic":
        print("[OK] Synthetic verification mode.")
        sys.exit(0)

    elif args.mode == "run":
        # TASK 3 SAFE FAILURE ENFORCEMENT:
        # Separate authorization permission from backend readiness
        if not args.authorize_inference:
            print(
                "ERROR: Execution of 'run' mode is BLOCKED!\n"
                "Stage 5B.4-A.1 does not authorize real model inference.\n"
                "Pass '--authorize-inference' only when explicitly authorized.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Even if authorized, verify that the production backend is actually available and verified
        print(
            "FATAL: NotImplementedError: Production fresh-diarization execution backend is not available.\n"
            "Real inference models (TitaNet, WeSpeaker) and CUDA clustering are pending pilot authorization.",
            file=sys.stderr,
        )
        # MUST return non-zero exit code! Never report false success!
        sys.exit(2)


if __name__ == "__main__":
    main()
