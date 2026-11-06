#!/usr/bin/env python3
"""
Stage 5B.4-A.1 Remediation: Condition Registry
Authoritative registry defining the 6 factorial experimental conditions (E00–E21),
their VAD parameterizations (C00, C50, C62), and semantic overlap attribution methods.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class VADConfig:
    vad_id: str
    description: str
    audio_source: str  # "dfn" | "16k_mono" | "consensus"
    onset: float
    offset: float
    min_duration_on: float
    min_duration_off: float
    consensus_weights: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class OverlapConfig:
    method_id: str  # "M0_no_overlap" | "M4_conservative_abstention"
    description: str
    tau_overlap: float
    min_overlap_dur: float
    max_gap_sec: float
    min_turn_dur_sec: float
    frame_resolution: float = 0.01


@dataclass(frozen=True)
class ExperimentCondition:
    condition_id: str  # "E00", "E01", "E10", "E11", "E20", "E21"
    vad_config: VADConfig
    overlap_config: OverlapConfig
    description: str
    role: str


# Authoritative VAD configurations
VAD_C00 = VADConfig(
    vad_id="C00",
    description="Pyannote Path A DFN VAD baseline",
    audio_source="dfn",
    onset=0.481,
    offset=0.481,
    min_duration_on=0.10,
    min_duration_off=0.15,
    consensus_weights=None,
)

VAD_C50 = VADConfig(
    vad_id="C50",
    description="Pyannote Raw 16k Mono VAD (high-recall)",
    audio_source="16k_mono",
    onset=0.481,
    offset=0.481,
    min_duration_on=0.10,
    min_duration_off=0.15,
    consensus_weights=None,
)

VAD_C62 = VADConfig(
    vad_id="C62",
    description="Linear Consensus VAD (DFN + Raw Mono) with hysteresis",
    audio_source="consensus",
    onset=0.600,
    offset=0.350,
    min_duration_on=0.10,
    min_duration_off=0.30,
    consensus_weights={"dfn": 0.5, "16k_mono": 0.5},
)

VAD_REGISTRY: Dict[str, VADConfig] = {
    "C00": VAD_C00,
    "C50": VAD_C50,
    "C62": VAD_C62,
}

# Authoritative Overlap configurations
OVERLAP_M0 = OverlapConfig(
    method_id="M0_no_overlap",
    description="Single-label baseline; overlap regions remain assigned to primary speaker",
    tau_overlap=0.0,
    min_overlap_dur=0.0,
    max_gap_sec=0.0,
    min_turn_dur_sec=0.0,
    frame_resolution=0.01,
)

OVERLAP_M4 = OverlapConfig(
    method_id="M4_conservative_abstention",
    description="Conservative contiguous turn alternation with abstention (M4-R2)",
    tau_overlap=0.40,
    min_overlap_dur=0.10,
    max_gap_sec=5.0,
    min_turn_dur_sec=0.20,
    frame_resolution=0.01,
)

OVERLAP_REGISTRY: Dict[str, OverlapConfig] = {
    "M0_no_overlap": OVERLAP_M0,
    "M4_conservative_abstention": OVERLAP_M4,
}

# The 6 factorial conditions
CONDITIONS: Dict[str, ExperimentCondition] = {
    "E00": ExperimentCondition("E00", VAD_C00, OVERLAP_M0, "C00 x M0", "Primary reference benchmark"),
    "E01": ExperimentCondition("E01", VAD_C00, OVERLAP_M4, "C00 x M4", "Isolated overlap effect on baseline VAD"),
    "E10": ExperimentCondition("E10", VAD_C50, OVERLAP_M0, "C50 x M0", "Isolated high-recall VAD effect"),
    "E11": ExperimentCondition("E11", VAD_C50, OVERLAP_M4, "C50 x M4", "Combined high-recall VAD + overlap attribution"),
    "E20": ExperimentCondition("E20", VAD_C62, OVERLAP_M0, "C62 x M0", "Isolated robust consensus VAD effect"),
    "E21": ExperimentCondition("E21", VAD_C62, OVERLAP_M4, "C62 x M4", "Full proposed integration system"),
}

BENCHMARK_RECORDINGS: List[str] = [
    "R01_sara_sharif",
    "R02_dennis_skinner",
    "R03_disabled_parking",
    "R04_piers_robinson",
    "R05_southwest_pilot",
    "R06_rich_karen",
    "R07_bwc_cheap_shot",
    "R08_uk_parliament",
]


def get_condition(condition_id: str) -> ExperimentCondition:
    if condition_id not in CONDITIONS:
        raise KeyError(
            f"Condition '{condition_id}' not found in registry. Authorized: {list(CONDITIONS.keys())}"
        )
    return CONDITIONS[condition_id]


def validate_registry() -> bool:
    assert len(CONDITIONS) == 6
    for cid, cond in CONDITIONS.items():
        assert cond.condition_id == cid
        assert cond.vad_config.vad_id in VAD_REGISTRY
        assert cond.overlap_config.method_id in OVERLAP_REGISTRY
    return True
