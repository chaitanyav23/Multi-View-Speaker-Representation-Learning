#!/usr/bin/env python3
"""Tests for Stage 5B.4 Evaluation Adapter."""

import pytest
import sys
import numpy as np
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluation_adapter import (
    compute_vad_union_metrics,
    compute_speaker_count_error,
    format_cpwer_attribution_envelope,
    compute_macro_and_pooled_summary,
    paired_wilcoxon_test,
    clustered_bootstrap_ci,
)


def test_vad_union_metrics():
    # Hyp: 1.0s to 3.0s (2.0s)
    # Ref: 1.5s to 3.5s (2.0s)
    # Overlap (TP): 1.5s to 3.0s (1.5s)
    # FP: 1.0s to 1.5s (0.5s)
    # FN: 3.0s to 3.5s (0.5s)
    hyp = [{"start": 1.0, "end": 3.0}]
    ref = [{"start": 1.5, "end": 3.5}]
    metrics = compute_vad_union_metrics(hyp, ref, total_duration=5.0)

    # Precision = 1.5 / 2.0 = 0.75, Recall = 1.5 / 2.0 = 0.75, F1 = 0.75
    assert metrics["vad_precision"] == pytest.approx(0.75, abs=0.01)
    assert metrics["vad_recall"] == pytest.approx(0.75, abs=0.01)
    assert metrics["vad_f1"] == pytest.approx(0.75, abs=0.01)


def test_speaker_count_error():
    err = compute_speaker_count_error(k_hat=3, k_true=2)
    assert err["k_hat"] == 3
    assert err["k_true"] == 2
    assert err["k_error"] == 1
    assert err["k_abs_error"] == 1


def test_format_cpwer_attribution_envelope_schema():
    raw_segs = [
        {"start": 1.25, "end": 3.50, "speaker": "spk0"},
        {"start": 0.50, "end": 1.00, "speaker": "spk1"},
    ]
    env = format_cpwer_attribution_envelope(raw_segs)

    # Must be a dict with key "segments"
    assert isinstance(env, dict)
    assert "segments" in env
    assert isinstance(env["segments"], list)
    assert len(env["segments"]) == 2
    # Must be sorted by start time
    assert env["segments"][0]["speaker"] == "spk1"
    assert env["segments"][0]["start"] == 0.50


def test_macro_and_pooled_summary():
    results = [
        {"der": 0.10, "der_error_sec": 10.0, "total_speech_sec": 100.0},
        {"der": 0.20, "der_error_sec": 40.0, "total_speech_sec": 200.0},
    ]
    summary = compute_macro_and_pooled_summary(results, ["der"])
    # Macro DER: (0.10 + 0.20) / 2 = 0.15
    assert summary["macro_der"] == 0.15
    # Pooled DER: (10 + 40) / (100 + 200) = 50 / 300 = 0.1667
    assert summary["pooled_der"] == pytest.approx(0.1667, abs=0.001)


def test_wilcoxon_and_bootstrap():
    diffs = [0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.07, 0.05]
    ci = clustered_bootstrap_ci(diffs, n_bootstraps=1000)
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]
    assert ci["mean"] == pytest.approx(0.04875, abs=0.001)

    a = [0.20, 0.22, 0.19, 0.25, 0.30, 0.28, 0.21, 0.24]
    b = [0.15, 0.18, 0.14, 0.20, 0.24, 0.22, 0.17, 0.19]
    w_res = paired_wilcoxon_test(a, b)
    assert w_res["significant_005"] is True
