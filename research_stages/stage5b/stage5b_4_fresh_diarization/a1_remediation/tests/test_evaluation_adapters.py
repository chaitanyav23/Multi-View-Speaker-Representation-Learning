#!/usr/bin/env python3
"""Tests for Task 9: Audited Evaluation Adapters."""

import pytest
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluation_adapters import (
    SyntheticEvaluationAdapter,
    ProductionEvaluationAdapter,
)


def test_synthetic_evaluation_adapter(tmp_path):
    hyp = tmp_path / "hyp.rttm"
    ref = tmp_path / "ref.rttm"
    hyp.write_text("SPEAKER rec 1 0.0 1.0 <NA> <NA> spk0 <NA> <NA>\n")
    ref.write_text("SPEAKER rec 1 0.0 1.0 <NA> <NA> spk0 <NA> <NA>\n")

    adapter = SyntheticEvaluationAdapter()
    metrics = adapter.evaluate(hyp, ref, k_hat=2, k_true=2)

    assert metrics["eval_type"] == "synthetic_mock"
    assert "der_collar0" in metrics
    assert "der_collar025" in metrics
    assert "jer" in metrics
    assert metrics["k_error"] == 0


def test_production_evaluation_adapter_missing_files():
    adapter = ProductionEvaluationAdapter()
    with pytest.raises(FileNotFoundError, match="not found"):
        adapter.evaluate(Path("/missing/hyp.rttm"), Path("/missing/ref.rttm"))


def test_production_evaluation_adapter_with_synthetic_rttms(tmp_path):
    hyp = tmp_path / "hyp.rttm"
    ref = tmp_path / "ref.rttm"

    # Reference: spk0 for 0.0-2.0, spk1 for 2.0-4.0
    ref.write_text(
        "SPEAKER rec 1 0.00 2.00 <NA> <NA> ref_spk0 <NA> <NA>\n"
        "SPEAKER rec 1 2.00 2.00 <NA> <NA> ref_spk1 <NA> <NA>\n"
    )
    # Hypothesis: matches reference perfectly
    hyp.write_text(
        "SPEAKER rec 1 0.00 2.00 <NA> <NA> hyp_spk0 <NA> <NA>\n"
        "SPEAKER rec 1 2.00 2.00 <NA> <NA> hyp_spk1 <NA> <NA>\n"
    )

    adapter = ProductionEvaluationAdapter()
    metrics = adapter.evaluate(hyp, ref, k_hat=2, k_true=2)

    assert metrics["eval_type"] == "production_pyannote"
    assert metrics["der_collar0"] == 0.0
    assert metrics["der_collar025"] == 0.0
    assert metrics["jer"] == 0.0
    assert metrics["total_reference_time"] == 4.0
