"""Unit tests for confidence calibration, metrics, and adaptive feedback."""
from __future__ import annotations

from adaptive.confidence import ConfidenceCalibrator
from adaptive.feedback_loop import FeedbackLoopManager


def test_confidence_calibrator_methods():
    iso_cal = ConfidenceCalibrator(method="isotonic_regression")
    temp_cal = ConfidenceCalibrator(method="temperature_scaling", temperature=1.5)

    raw_scores = [0.1, 0.5, 0.9]
    for r in raw_scores:
        iso_val = iso_cal.calibrate_score(r)
        temp_val = temp_cal.calibrate_score(r)
        assert 0.0 <= iso_val <= 1.0
        assert 0.0 <= temp_val <= 1.0


def test_brier_score_and_ece():
    # Perfect calibration test
    confidences = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]

    brier = ConfidenceCalibrator.compute_brier_score(confidences, labels)
    ece = ConfidenceCalibrator.compute_ece(confidences, labels, n_bins=5)

    assert brier >= 0.0
    assert ece >= 0.0
    assert brier < 0.1  # Very low error for close probabilities

    # Reliability diagram
    diagram = ConfidenceCalibrator.compute_reliability_diagram(confidences, labels, n_bins=5)
    assert len(diagram) == 5
    assert "confidence" in diagram[0]
    assert "accuracy" in diagram[0]

    # Risk-coverage curve
    rc_curve = ConfidenceCalibrator.compute_risk_coverage(confidences, labels, n_steps=5)
    assert len(rc_curve) == 5
    assert "coverage" in rc_curve[0]
    assert "risk" in rc_curve[0]


def test_word_correctness_label_generation():
    hyp = ["good", "mornin", "officer", "here"]
    ref = ["good", "morning", "officer"]

    result = ConfidenceCalibrator.generate_word_correctness_labels(hyp, ref)
    labels = result["labels"]
    assert len(labels) == len(hyp)
    assert labels[0] == 1  # "good" matches
    assert labels[1] == 0  # "mornin" is substitution
    assert labels[2] == 1  # "officer" matches
    assert labels[3] == 0  # "here" is insertion
    assert result["matches"] == 2
    assert result["substitutions"] == 1
    assert result["insertions"] == 1
    assert result["wer"] > 0.0


def test_feedback_loop_vocabulary_extraction():
    mgr = FeedbackLoopManager()
    res = mgr.register_correction(
        original_text="unit 24 code for",
        corrected_text="Unit 24 Code-4 all secure",
        original_speaker="Speaker 1",
        corrected_speaker="Officer Smith",
    )
    assert res["error_type"] == "compound"
    assert "Code-4" in mgr.get_vocabulary() or "Unit" in mgr.get_vocabulary()

    prompt = mgr.get_whisper_prompt(base_prompt="Traffic stop.")
    assert "Domain terms:" in prompt
