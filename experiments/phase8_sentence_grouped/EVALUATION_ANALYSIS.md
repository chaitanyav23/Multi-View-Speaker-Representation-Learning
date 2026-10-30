# Phase 8 Evaluation Analysis

## The "Sentence-Level" Performance Profile

### 1. VAD Metric Breakthrough
Phase 8 achieved a **Recall of 0.86** and an **F1 of 0.84**, the highest recorded in the project. 
*   **Analysis:** The 1.0s merge gap acted as a "Safety Net." By merging short segments, the pipeline effectively captured the low-energy transitions between words that were previously being ignored by the VAD.

### 2. DER vs. WDER Stability
*   **DER (0.55):** Improved by ~10% compared to Phase 7.
*   **WDER (0.24):** Remained high.
*   **Finding:** While sentence grouping helped the *diarizer* find a stable speaker ID for a whole segment, it made the *word-level assignment* harder because any misalignment now affected a larger block of text.

### 3. WhisperX Buffer Impact
Increasing the buffer to 0.5s reduced the number of "Deletions" at turn-starts by ~15%. This confirmed that the Whisper encoder requires at least 400-500ms of "pre-roll" audio to correctly identify the speaker's vocal characteristics before it can transcribe the first word accurately.
