#!/usr/bin/env python3
"""
Stage 5B.4-A.1.1 Remediation: Cross-Component Integration Scenarios (A through H)
Validates all integrated behaviors, error isolation, partial caching, and security boundaries.
"""

import hashlib
import json
import numpy as np
from pathlib import Path
import pytest
import sys
from typing import Any, Dict, List, Optional, Tuple
import torch

CORRECTIONS_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(CORRECTIONS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORRECTIONS_SCRIPTS))

from path_utils import get_stage5b_4_dir
A1_SCRIPTS = get_stage5b_4_dir() / "a1_remediation" / "scripts"
if str(A1_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A1_SCRIPTS))

from cache_manager_remediated import (
    IsolatedEmbeddingCacheManagerRemediated,
    CacheSecurityError,
)
from vad_adapter_remediated import VADAdapterRemediated
from clustering_backends import SyntheticSpectralClusteringBackend
from evaluation_adapters_remediated import (
    SyntheticEvaluationAdapter,
    ProductionEvaluationAdapterRemediated,
)
from feature_extraction_interface import MockTitaNetModel, MockWeSpeakerModel
from overlap_posterior_loader import OverlapPosteriorLoader, OverlapPosteriorError
from output_path_guard import OutputSecurityError
from checkpoint_preflight import CheckpointPreflightError
from orchestrator_remediated import FreshDiarizationOrchestratorRemediated


def create_synthetic_audio(duration_sec: float = 5.0, sample_rate: int = 16000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * sample_rate)
    wav_mono = rng.standard_normal(n_samples).astype(np.float32)
    wav_dfn = rng.standard_normal(n_samples).astype(np.float32)
    return wav_dfn, wav_mono


def create_synthetic_posteriors(duration_sec: float = 5.0, has_overlap: bool = False):
    n_frames = int(duration_sec / 0.02)
    f_times = np.linspace(0.01, duration_sec, n_frames)
    p_speech = np.zeros(n_frames, dtype=np.float32)
    # Two distinct speech turns
    p_speech[int(0.5 / 0.02) : int(2.0 / 0.02)] = 0.90
    p_speech[int(2.5 / 0.02) : int(4.5 / 0.02)] = 0.90

    p_ovl = np.zeros(n_frames, dtype=np.float32)
    if has_overlap:
        p_ovl[int(1.0 / 0.02) : int(1.5 / 0.02)] = 0.85
    return f_times, p_speech, p_ovl


def setup_test_orchestrator(
    tmp_path: Path,
    tita_ckpt_hash: Optional[str] = None,
    res_ckpt_hash: Optional[str] = None,
    eval_adapter=None,
):
    cache_root = tmp_path / "cache"
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(cache_root=cache_root, allow_custom_root=True)
    vad_adapter = VADAdapterRemediated(posteriors_dir=tmp_path / "posteriors")
    overlap_loader = OverlapPosteriorLoader(posteriors_dir=tmp_path / "posteriors")
    clustering = SyntheticSpectralClusteringBackend()
    eval_ad = eval_adapter or SyntheticEvaluationAdapter()

    t_hash = tita_ckpt_hash or ("a" * 64)
    r_hash = res_ckpt_hash or ("b" * 64)

    orch = FreshDiarizationOrchestratorRemediated(
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=clustering,
        evaluation_adapter=eval_ad,
        overlap_loader=overlap_loader,
        titanet_model=MockTitaNetModel(),
        wespeaker_model=MockWeSpeakerModel(),
        titanet_checkpoint_sha256=t_hash,
        wespeaker_checkpoint_sha256=r_hash,
        skip_checkpoint_preflight=True,  # Will test preflight separately in Scenario F
        allowed_output_roots=[tmp_path],
    )
    return orch, cache_mgr


def test_scenario_a_clean_execution_m0(tmp_path):
    """Scenario A: Base Clean Execution with M0 single-speaker baseline."""
    orch, cache_mgr = setup_test_orchestrator(tmp_path)
    dfn, mono = create_synthetic_audio(4.0)
    f_times, p_speech, p_ovl = create_synthetic_posteriors(4.0, has_overlap=False)

    ref_rttm = tmp_path / "ref.rttm"
    ref_rttm.write_text("SPEAKER rec 1 0.0 4.0 <NA> <NA> spk0 <NA> <NA>\n")

    out_dir = tmp_path / "outputs_a"

    # First run: cold cache -> cache misses
    stats1 = orch.process_recording(
        condition_id="E00",
        recording_id="rec_A",
        dfn_waveform=dfn,
        mono_waveform=mono,
        posteriors_override=(f_times, p_speech, None),  # M0 does not need p_ovl
        output_dir=out_dir,
        reference_rttm_path=ref_rttm,
    )

    assert stats1.condition_id == "E00"
    assert stats1.overlap_method == "M0_no_overlap"
    assert stats1.titanet_cache_misses == 5
    assert stats1.wespeaker_cache_misses == 5
    assert stats1.titanet_cache_hits == 0
    assert stats1.wespeaker_cache_hits == 0
    assert Path(stats1.output_rttm_path).exists()
    assert stats1.evaluation_metrics is not None

    # Second run: warm cache -> 100% cache hits
    stats2 = orch.process_recording(
        condition_id="E00",
        recording_id="rec_A",
        dfn_waveform=dfn,
        mono_waveform=mono,
        posteriors_override=(f_times, p_speech, None),
        output_dir=out_dir,
        reference_rttm_path=ref_rttm,
    )
    assert stats2.titanet_cache_hits == 5
    assert stats2.wespeaker_cache_hits == 5
    assert stats2.titanet_cache_misses == 0
    assert stats2.wespeaker_cache_misses == 0


def test_scenario_b_clean_execution_m4_and_cpwer(tmp_path):
    """Scenario B: Base Clean Execution with M4 overlap and cpWER evaluation."""
    orch, cache_mgr = setup_test_orchestrator(
        tmp_path, eval_adapter=ProductionEvaluationAdapterRemediated()
    )
    dfn, mono = create_synthetic_audio(4.0)
    f_times, p_speech, p_ovl = create_synthetic_posteriors(4.0, has_overlap=True)

    ref_rttm = tmp_path / "ref.rttm"
    ref_rttm.write_text(
        "SPEAKER rec_B 1 0.00 2.00 <NA> <NA> spk1 <NA> <NA>\n"
        "SPEAKER rec_B 1 1.50 2.50 <NA> <NA> spk2 <NA> <NA>\n"
    )

    words_json = tmp_path / "words.json"
    words_json.write_text(json.dumps([
        {"word": "hello", "start": 0.2, "end": 0.8},
        {"word": "world", "start": 1.0, "end": 1.4},
        {"word": "overlapping", "start": 1.8, "end": 2.2},
        {"word": "speech", "start": 2.6, "end": 3.0},
    ]))

    ref_transcript = tmp_path / "ref_transcript.json"
    ref_transcript.write_text(json.dumps({
        "segments": [
            {"speaker": "spk1", "text": "hello world"},
            {"speaker": "spk2", "text": "overlapping speech"},
        ]
    }))

    out_dir = tmp_path / "outputs_b"

    stats = orch.process_recording(
        condition_id="E01",  # C00 x M4
        recording_id="rec_B",
        dfn_waveform=dfn,
        mono_waveform=mono,
        posteriors_override=(f_times, p_speech, p_ovl),
        output_dir=out_dir,
        reference_rttm_path=ref_rttm,
        aligned_words_json=words_json,
        reference_transcript_json=ref_transcript,
    )

    assert stats.condition_id == "E01"
    assert stats.overlap_method == "M4_conservative_abstention"
    assert Path(stats.output_rttm_path).exists()
    assert stats.evaluation_metrics is not None
    assert "der_collar0" in stats.evaluation_metrics
    assert "cpwer" in stats.evaluation_metrics
    assert isinstance(stats.evaluation_metrics["cpwer"], float)


def test_scenario_c_missing_overlap_posteriors(tmp_path):
    """Scenario C: M4 fails explicitly if p_overlap is missing; M0 succeeds."""
    orch, _ = setup_test_orchestrator(tmp_path)
    dfn, mono = create_synthetic_audio(4.0)
    f_times, p_speech, _ = create_synthetic_posteriors(4.0)

    # 1. M4 condition with p_overlap=None must fail explicitly!
    with pytest.raises(OverlapPosteriorError, match="requires M4 overlap posteriors"):
        orch.process_recording(
            condition_id="E01",  # M4
            recording_id="rec_C",
            dfn_waveform=dfn,
            mono_waveform=mono,
            posteriors_override=(f_times, p_speech, None),
        )

    # 2. M0 condition with p_overlap=None succeeds completely!
    stats_m0 = orch.process_recording(
        condition_id="E00",  # M0
        recording_id="rec_C",
        dfn_waveform=dfn,
        mono_waveform=mono,
        posteriors_override=(f_times, p_speech, None),
    )
    assert stats_m0.overlap_method == "M0_no_overlap"


def test_scenario_d_corrupt_cache_invalidation(tmp_path):
    """Scenario D: Corrupt cache entry detected as invalidation, extracted, counter incremented."""
    orch, cache_mgr = setup_test_orchestrator(tmp_path)
    dfn, mono = create_synthetic_audio(4.0)
    f_times, p_speech, _ = create_synthetic_posteriors(4.0)

    # Populate cache first
    orch.process_recording(
        condition_id="E00",
        recording_id="rec_D",
        dfn_waveform=dfn,
        mono_waveform=mono,
        posteriors_override=(f_times, p_speech, None),
    )

    # Corrupt one TitaNet cache entry
    entry_dir = cache_mgr.get_entry_dir("C00", "rec_D")
    tita_files = list(entry_dir.glob("titanet_large*scale0*.npz"))
    assert len(tita_files) > 0
    tita_files[0].write_bytes(b"corrupt payload bytes")

    # Re-run: should detect invalidation on scale 0
    stats = orch.process_recording(
        condition_id="E00",
        recording_id="rec_D",
        dfn_waveform=dfn,
        mono_waveform=mono,
        posteriors_override=(f_times, p_speech, None),
    )

    assert stats.titanet_cache_invalidations == 1
    assert stats.titanet_cache_hits == 4
    assert stats.wespeaker_cache_hits == 5


def test_scenario_e_cache_security_violation(tmp_path):
    """Scenario E: Security violation in identifier raises immediately; no extraction."""
    orch, _ = setup_test_orchestrator(tmp_path)
    dfn, mono = create_synthetic_audio(4.0)
    f_times, p_speech, _ = create_synthetic_posteriors(4.0)

    with pytest.raises(CacheSecurityError, match="Path traversal"):
        orch.process_recording(
            condition_id="E00",
            recording_id="../forbidden_escape",
            dfn_waveform=dfn,
            mono_waveform=mono,
            posteriors_override=(f_times, p_speech, None),
        )


def test_scenario_f_checkpoint_hash_mismatch(tmp_path):
    """Scenario F: Checkpoint hash mismatch caught during preflight before execution."""
    ckpt = tmp_path / "mock_titanet.nemo"
    ckpt.write_bytes(b"some model weights")
    wrong_hash = "0" * 64

    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(cache_root=tmp_path / "cache", allow_custom_root=True)
    vad_adapter = VADAdapterRemediated(posteriors_dir=tmp_path)

    with pytest.raises(CheckpointPreflightError, match="Hash mismatch"):
        FreshDiarizationOrchestratorRemediated(
            cache_manager=cache_mgr,
            vad_adapter=vad_adapter,
            clustering_backend=SyntheticSpectralClusteringBackend(),
            evaluation_adapter=SyntheticEvaluationAdapter(),
            titanet_checkpoint_path=ckpt,
            titanet_checkpoint_sha256=wrong_hash,
            skip_checkpoint_preflight=False,  # Enforce preflight
        )


def test_scenario_g_partial_cache_reuse(tmp_path):
    """Scenario G: Changing TitaNet hash invalidates TitaNet, WeSpeaker hits 100%."""
    t_hash1 = "1" * 64
    r_hash1 = "2" * 64
    orch1, cache_mgr = setup_test_orchestrator(tmp_path, tita_ckpt_hash=t_hash1, res_ckpt_hash=r_hash1)
    dfn, mono = create_synthetic_audio(4.0)
    f_times, p_speech, _ = create_synthetic_posteriors(4.0)

    # Initial run under hash 1
    orch1.process_recording(
        condition_id="E00",
        recording_id="rec_G",
        dfn_waveform=dfn,
        mono_waveform=mono,
        posteriors_override=(f_times, p_speech, None),
    )

    # Now create new orchestrator with modified TitaNet hash, unchanged WeSpeaker hash
    t_hash2 = "9" * 64  # New TitaNet hash
    orch2 = FreshDiarizationOrchestratorRemediated(
        cache_manager=cache_mgr,
        vad_adapter=orch1.vad_adapter,
        clustering_backend=orch1.clustering_backend,
        evaluation_adapter=orch1.evaluation_adapter,
        overlap_loader=orch1.overlap_loader,
        titanet_model=MockTitaNetModel(),
        wespeaker_model=MockWeSpeakerModel(),
        titanet_checkpoint_sha256=t_hash2,
        wespeaker_checkpoint_sha256=r_hash1,
        skip_checkpoint_preflight=True,
        allowed_output_roots=[tmp_path],
    )

    stats2 = orch2.process_recording(
        condition_id="E00",
        recording_id="rec_G",
        dfn_waveform=dfn,
        mono_waveform=mono,
        posteriors_override=(f_times, p_speech, None),
    )

    # TitaNet entries had identity mismatch on checkpoint hash -> 5 invalidations
    assert stats2.titanet_cache_invalidations == 5
    assert stats2.titanet_cache_hits == 0
    # WeSpeaker entries were untouched and match completely -> 5 hits!
    assert stats2.wespeaker_cache_hits == 5
    assert stats2.wespeaker_cache_invalidations == 0


def test_scenario_h_output_path_escape(tmp_path):
    """Scenario H: Output directory escape attempt rejected by OutputSecurityError."""
    orch, _ = setup_test_orchestrator(tmp_path)
    dfn, mono = create_synthetic_audio(4.0)
    f_times, p_speech, _ = create_synthetic_posteriors(4.0)

    escaped_dir = tmp_path / "sandbox" / ".." / ".." / "system_escape"
    with pytest.raises(OutputSecurityError):
        orch.process_recording(
            condition_id="E00",
            recording_id="rec_H",
            dfn_waveform=dfn,
            mono_waveform=mono,
            posteriors_override=(f_times, p_speech, None),
            output_dir=escaped_dir,
        )
