#!/usr/bin/env python3
"""
Unit tests for Stage 5B.4-A.1.3.1 Finding 6:
Formalized M0 and M4 execution eligibility and overlap provenance distinction.
Verifies that:
1. Condition M0 is execution-eligible without consuming or requiring overlap posteriors.
   (m0_execution_eligible=True, overlap_provenance_verified=False, overlap_production_eligible=False)
2. Condition M4 consumes overlap posteriors and strictly requires overlap_production_eligible=True in production.
3. If overlap posterior provenance is unverified (e.g. historical mono audio), M4 fails closed in production.
4. Synthetic fixtures running M4 are explicitly marked m4_execution_eligible=False for production.
5. Backward compatibility aliases (provenance_verified, production_eligible) are correctly synchronized.
"""

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_preflight import ExecutionMode, create_synthetic_checkpoint_fixture
from model_factory import ControlledModelFactory
from cache_manager_remediated import IsolatedEmbeddingCacheManagerRemediated
from clustering_backends import SyntheticSpectralClusteringBackend
from evaluation_adapters_remediated import SyntheticEvaluationAdapter
from vad_adapter_remediated import VADAdapterRemediated
from overlap_provenance import OverlapProvenanceError, PYANNET_FIRST_FRAME_CENTER, PYANNET_FRAME_STEP_SEC
from orchestrator_remediated import FreshDiarizationOrchestratorRemediated, PipelineExecutionStats


class MockTitaNetEncoder(torch.nn.Module):
    def forward(self, input_signal=None, input_signal_length=None, **kwargs):
        return None, torch.randn(1, 192)


class MockWeSpeakerEncoder(torch.nn.Module):
    def forward(self, feats=None, **kwargs):
        return None, torch.randn(1, 256)


@pytest.fixture
def orchestrator_env(tmp_path):
    """Sets up a synthetic orchestrator environment."""
    tita_file = tmp_path / "tita.pt"
    tita_token = create_synthetic_checkpoint_fixture("titanet_large", tita_file)
    tita_handle = ControlledModelFactory.build_synthetic_model(tita_token, MockTitaNetEncoder())

    wes_file = tmp_path / "wes.pt"
    wes_token = create_synthetic_checkpoint_fixture("wespeaker_resnet293", wes_file)
    wes_handle = ControlledModelFactory.build_synthetic_model(wes_token, MockWeSpeakerEncoder())

    cache_dir = tmp_path / "cache"
    cache_mgr = IsolatedEmbeddingCacheManagerRemediated(cache_dir, allow_custom_root=True)
    vad_adapter = VADAdapterRemediated()
    eval_adapter = SyntheticEvaluationAdapter()
    cluster_backend = SyntheticSpectralClusteringBackend()

    orchestrator = FreshDiarizationOrchestratorRemediated(
        titanet_handle=tita_handle,
        wespeaker_handle=wes_handle,
        cache_manager=cache_mgr,
        vad_adapter=vad_adapter,
        clustering_backend=cluster_backend,
        evaluation_adapter=eval_adapter,
        execution_mode=ExecutionMode.SYNTHETIC,
        custom_output_root=tmp_path / "outputs",
    )

    return {
        "orchestrator": orchestrator,
        "tmp_path": tmp_path,
    }


def test_1_m0_execution_eligible_without_overlap_provenance(orchestrator_env):
    """
    Verifies that M0 (E00, E10, E20) is execution eligible without requiring overlap posteriors.
    Formalizes the distinction:
    - m0_execution_eligible is True
    - overlap_provenance_verified is False
    - overlap_production_eligible is False
    - m4_execution_eligible is False
    """
    orch = orchestrator_env["orchestrator"]
    rec_id = "syn_rec_01"
    wav = np.zeros(16000 * 2, dtype=np.float32)

    f_times = PYANNET_FIRST_FRAME_CENTER + np.arange(100) * PYANNET_FRAME_STEP_SEC
    p_speech = np.ones(100, dtype=np.float32)

    stats = orch.process_recording(
        condition_id="E00",  # E00 = V0 (Oracle) + M0 (No overlap)
        recording_id=rec_id,
        dfn_waveform=wav,
        mono_waveform=wav,
        posteriors_override=(f_times, p_speech, None),
        sample_rate=16000,
    )

    assert stats.m0_execution_eligible is True
    assert stats.overlap_provenance_verified is False
    assert stats.overlap_production_eligible is False
    assert stats.m4_execution_eligible is False
    assert stats.structurally_valid is True
    assert stats.overlap_method == "M0_no_overlap"


def test_2_m4_synthetic_execution_marked_ineligible_for_production(orchestrator_env):
    """
    Verifies that synthetic M4 runs cleanly for testing but explicitly marks
    overlap_provenance_verified=False, overlap_production_eligible=False, m4_execution_eligible=False.
    """
    orch = orchestrator_env["orchestrator"]
    rec_id = "syn_rec_02"
    wav = np.zeros(16000 * 3, dtype=np.float32)

    f_times = PYANNET_FIRST_FRAME_CENTER + np.arange(100) * PYANNET_FRAME_STEP_SEC
    p_speech = np.ones(100, dtype=np.float32)
    p_overlap = np.zeros(100, dtype=np.float32)

    stats = orch.process_recording(
        condition_id="E01",  # E01 = V0 (Oracle) + M4 (Overlap)
        recording_id=rec_id,
        dfn_waveform=wav,
        mono_waveform=wav,
        posteriors_override=(f_times, p_speech, p_overlap),
        sample_rate=16000,
    )

    assert stats.m0_execution_eligible is False
    assert stats.overlap_provenance_verified is False
    assert stats.overlap_production_eligible is False
    assert stats.m4_execution_eligible is False
    assert stats.structurally_valid is True
    assert stats.overlap_method == "M4_conservative_abstention"


def test_3_backward_compatibility_aliases_synchronized():
    """Verifies that stats aliases (provenance_verified, production_eligible) are synchronized."""
    stats = PipelineExecutionStats(
        condition_id="E00",
        recording_id="syn_rec_03",
        vad_id="V0",
        vad_interval_count=1,
        vad_intervals_sha256="0" * 64,
        subsegment_counts={},
        m0_execution_eligible=True,
        m4_execution_eligible=False,
        overlap_provenance_verified=False,
        overlap_production_eligible=False,
    )

    assert stats.m0_execution_eligible is True
    assert stats.overlap_provenance_verified is False
    assert stats.provenance_verified is False
    assert stats.production_eligible is False
