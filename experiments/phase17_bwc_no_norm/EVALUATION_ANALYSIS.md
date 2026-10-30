# Evaluation Analysis: Phase 17

## Methodology
The evaluation suite in Phase 17 is designed to provide a multi-layered view of performance, ranging from low-level signal detection to high-level conversational structure.

## Primary Metrics

### 1. Voice Activity Detection (Step 8)
- **Miss Rate:** The percentage of ground truth speech that was not detected by the VAD.
    - *Phase 17 Result:* 20.46% (Significant improvement over baseline).
- **False Alarm (FA) Rate:** The percentage of non-speech audio that was incorrectly flagged as speech.
    - *Phase 17 Result:* 6.25% (Internal VAD metric) / 16.74% (Global Diarization metric).
- **F1-Score:** The harmonic mean of Precision and Recall for speech detection.

### 2. Lexical Accuracy (Step 9)
- **Word Error Rate (WER):** Standard metric for ASR accuracy.
    - *Formula:* (Substitutions + Deletions + Insertions) / Total Reference Words.
    - *Phase 17 Result:* 33.68%.
    - *Observation:* The increase in WER compared to Phase 11A (30.0%) suggests that WhisperX's acoustic model benefits from normalized amplitude levels during transcription.

### 3. Diarization Accuracy (Step 10)
- **Diarization Error Rate (DER):** The sum of Miss, False Alarm, and Speaker Confusion.
    - *Phase 17 Result:* 44.93%.
- **Speaker Confusion:** The percentage of speech time attributed to the wrong speaker.
    - *Phase 17 Result:* 7.70% (Improvement from 10.35% in baseline).

### 4. Speaker-Attributed ASR (Step 11-13)
- **SA-WER:** Word Error Rate calculated per speaker and then averaged.
- **WDER (Word Diarization Error Rate):** The percentage of correctly transcribed words that were assigned to the wrong speaker.
    - *Phase 17 Result:* 10.66%.
- **cpWER (Concatenated Permutation WER):** A metric that handles speaker permutations to find the best-fit mapping between hypothesis and reference transcripts.
    - *Phase 17 Result:* 48.14%.

## Comparative Context
All metrics are benchmarked against **Phase 11A**, which represents the "Golden Baseline" utilizing the full pipeline including `LoudNorm`. The comparison is captured in `comparisons/bwc_no_norm_vs_baseline.csv`, which is the primary artifact for interpreting the success of this experiment.
