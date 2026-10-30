# Phase 12: Karen Evaluation - Generalization Stress Test

## Overview
Phase 12 focuses on the rigorous evaluation of the hardened pipeline (developed through Phase 11) on completely unseen and "wild" data. The primary objective was to validate the pipeline's ability to generalize to new acoustic environments, speaker characteristics, and conversational dynamics without manual hyper-parameter tuning or oracle speaker counts.

The "Karen" dataset—a body-worn camera (BWC) audio clip characterized by high-tension interaction and multi-speaker overlap—served as the primary test case.

## Objectives
- **Generalization Validation:** Confirm that the Stage 1-11 hardening holds up on unseen datasets.
- **Unbiased Diarization:** Evaluate the MSDD (Multi-scale Segmentation and Diarization) clusterer's performance with `oracle_num_speakers=False`.
- **Truncation Resilience:** Assess the pipeline's behavior when encountering corrupted or truncated input files (discovered during `step0`).
- **End-to-End Metrics:** Generate SA-WER, WDER, and cpWER for the unseen data to establish a baseline for "real-world" performance.

## Key Implementation Features
- **Auto-Speaker Estimation:** Utilization of TitaNet-Large and MSDD for unsupervised speaker discovery.
- **Slice-Based ASR:** WhisperX Large-v3 execution on VAD-conditioned slices to minimize hallucination.
- **Maximum Overlap Fusion:** Word-level speaker attribution based on MSDD temporal alignments.
- **Audio Provenance Audit:** Automated metadata validation and normalization (BS.1770).

## Results Summary
| Metric | Value |
| :--- | :--- |
| **VAD Accuracy** | 81.9% |
| **VAD F1-Score** | 89.9% |
| **Detected Speakers** | 2 |
| **Ground Truth Speakers** | 6 (Full) / 2 (Truncated) |
| **Diarization Error Rate (DER)** | *See METRICS_TABLES.md* |

## Directory Structure
- `scripts/`: Implementation steps from validation to final evaluation.
- `reports/`: CSV and Markdown reports for VAD, ASR, and Diarization.
- `diagnostics/`: Detailed error analysis for speaker attribution and counts.
- `outputs/`: Final conversational timelines and intermediate artifacts.
