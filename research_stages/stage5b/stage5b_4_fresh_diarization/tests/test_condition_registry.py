#!/usr/bin/env python3
"""Tests for Stage 5B.4 Condition Registry."""

import pytest
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from condition_registry import (
    CONDITIONS,
    VAD_REGISTRY,
    OVERLAP_REGISTRY,
    get_condition,
    validate_registry,
    BENCHMARK_RECORDINGS,
)


def test_registry_size():
    assert len(CONDITIONS) == 6
    assert set(CONDITIONS.keys()) == {"E00", "E01", "E10", "E11", "E20", "E21"}


def test_vad_configs():
    assert len(VAD_REGISTRY) == 3
    assert "C00" in VAD_REGISTRY
    assert "C50" in VAD_REGISTRY
    assert "C62" in VAD_REGISTRY

    # C00: DFN baseline
    c00 = VAD_REGISTRY["C00"]
    assert c00.audio_source == "dfn"
    assert c00.onset == 0.481
    assert c00.offset == 0.481
    assert c00.min_duration_on == 0.10
    assert c00.min_duration_off == 0.15

    # C50: Mono high-recall
    c50 = VAD_REGISTRY["C50"]
    assert c50.audio_source == "16k_mono"
    assert c50.onset == 0.481
    assert c50.offset == 0.481
    assert c50.min_duration_on == 0.10
    assert c50.min_duration_off == 0.15

    # C62: Consensus
    c62 = VAD_REGISTRY["C62"]
    assert c62.audio_source == "consensus"
    assert c62.onset == 0.600
    assert c62.offset == 0.350
    assert c62.min_duration_on == 0.10
    assert c62.min_duration_off == 0.30
    assert c62.consensus_weights == {"dfn": 0.5, "16k_mono": 0.5}


def test_overlap_configs():
    assert "M0_no_overlap" in OVERLAP_REGISTRY
    assert "M4_conservative_abstention" in OVERLAP_REGISTRY

    m4 = OVERLAP_REGISTRY["M4_conservative_abstention"]
    assert m4.tau_overlap == 0.40
    assert m4.min_overlap_dur == 0.10
    assert m4.max_gap_sec == 5.0
    assert m4.min_turn_dur_sec == 0.20


def test_get_condition_and_rejection():
    c = get_condition("E00")
    assert c.vad_config.vad_id == "C00"
    assert c.overlap_config.method_id == "M0_no_overlap"

    with pytest.raises(KeyError):
        get_condition("E99")

    # Disambiguation: O02 should not be accepted as a condition ID directly
    with pytest.raises(KeyError):
        get_condition("O02")


def test_benchmark_recordings():
    assert len(BENCHMARK_RECORDINGS) == 8
    assert "R01_sara_sharif" in BENCHMARK_RECORDINGS
    assert "R08_uk_parliament" in BENCHMARK_RECORDINGS


def test_validate_registry():
    assert validate_registry() is True
