# Phase 12 Error Analysis

This document categorizes and analyzes the primary sources of error observed during the Phase 12 Karen evaluation.

## 1. Categorization of Errors

### A. Major: Input Data Truncation
- **Description:** The input `karen.mp3` had a physical size and decoded duration of 52 seconds, despite metadata claiming 7 minutes.
- **Impact:** This caused a catastrophic failure in "Global" metrics (WER/cpWER) because 87% of the ground truth text had no corresponding audio.
- **Mitigation implemented:** `step0` validation and `step1` stream repair ensured the pipeline didn't crash, but it could not recover the missing data.

### B. Minor: VAD Fragmentation
- **Description:** Rapid conversational exchanges occasionally caused the VAD to produce many short segments.
- **Impact:** This can lead to "choppy" ASR output if not handled properly.
- **Mitigation implemented:** The 150ms merge gap in `step2_to_6.py` successfully unified these fragments into coherent turns.

### C. Minor: Speaker Discovery Limitation
- **Description:** While the pipeline correctly found 2 speakers for the 52s clip, it cannot discover speakers who only appear in the missing 6 minutes of audio.
- **Impact:** Leads to a discrepancy between "Expected Speakers" (6) and "Found Speakers" (2).
- **Analysis:** This is an environmental error (missing data) rather than an algorithmic failure.

## 2. Statistical Error Profile (VAD)

Based on `vad_metrics.csv` and diagnostics:
- **Miss Rate (10.2%)**: These are frames where the GT says there is speech, but the VAD did not trigger. Often occurs at the very start or end of a word where energy is low.
- **False Alarm Rate (High calculation)**: Likely reflects the ratio of false-positives to total silent frames. In this high-tension clip, background noise from the vehicle or sirens may have caused minor triggers in non-speech regions.

## 3. Attribution Errors (Diarization)
- **Overlap Confusion:** In sections where both speakers spoke simultaneously (e.g., the "Why are you being a jerk?" exchange), the MSDD model occasionally favored the louder speaker (the officer) for both word-sets.
- **Resolution:** The word-level fusion logic partially corrected this by looking at fine-grained temporal overlaps, but some confusion remains in highly overlapping frames.

## 4. Lessons Learned
1.  **Validation is Non-Negotiable:** Without `step0`, the evaluation results would have been misinterpreted as a massive ASR failure rather than a data truncation issue.
2.  **Unbiased Clustering Works:** Even with limited data (52s), TitaNet-Large generates sufficiently distinct embeddings for robust separation of two speakers.
