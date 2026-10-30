# Phase 1 Evaluation Analysis

## Key Metrics Definitions

### Diarization Metrics (DER)
*   **DER (Diarization Error Rate):** The primary metric for speaker attribution.
*   **Miss:** Speech present in ground truth but not in prediction. High Miss rates usually indicate over-aggressive VAD.
*   **FA (False Alarm):** Speech present in prediction but not in ground truth. High FA usually indicates VAD sensitivity to background noise.
*   **Confusion:** Speech attributed to the wrong speaker.

### Conversational ASR Metrics
*   **WER (Word Error Rate):** Standard measure of transcription accuracy.
*   **cpWER (Concatenated minimum-Permutation Word Error Rate):** Measures ASR accuracy while accounting for speaker assignment. It is highly sensitive to diarization errors.
*   **SA-WER (Speaker-Attributed WER):** Similar to cpWER, but focuses on the accuracy of words assigned to specific speakers.
*   **WDER (Word Diarization Error Rate):** Percentage of words assigned to the wrong speaker.

## Script Interpretation: `evaluate_conversational_timeline.py`
This script computes metrics by:
1.  Mapping predicted speakers to ground truth speakers to minimize cpWER.
2.  Aggregating segments from the same speaker.
3.  Calculating Levenshtein distance between ground truth and predicted text for each mapped speaker pair.

## Tradeoffs in VAD
*   **Aggressive VAD (e.g., Silero):** Lowers FA (better for noisy environments like BWC) but increases Miss (potentially losing important speech).
*   **Permissive VAD (e.g., Pyannote):** Lowers Miss (captures more speech) but increases FA (transcribing background noise or static as "hallucinated" words).
