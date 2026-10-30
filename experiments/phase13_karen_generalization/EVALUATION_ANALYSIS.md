# Evaluation Analysis: Phase 13 Karen Generalization

Phase 13 employs a multi-dimensional evaluation strategy to assess the performance of the pipeline on the Karen dataset. Metrics are categorized by the subsystem they validate.

## 1. ASR Metrics (Word Error Rate)
Standard Word Error Rate (WER) is calculated using the `jiwer` library.
-   **Methodology:**
    1.  Flatten the ground truth and system transcripts into single strings.
    2.  Apply normalization: Lowercase, remove punctuation, remove multiple spaces.
    3.  Compute WER, Substitutions, Deletions, and Insertions.
-   **Phase 13 Result:** **14.67%**. This is considered high performance for a multi-speaker conversational recording.

## 2. Diarization Metrics (DER)
Diarization Error Rate (DER) is the primary metric for speaker separation.
-   **Methodology:** Calculated using `pyannote.metrics`. It consists of:
    -   **Miss:** Speech in ground truth but not in system.
    -   **False Alarm (FA):** Speech in system but not in ground truth.
    -   **Confusion:** Speech attributed to the wrong speaker.
-   **Phase 13 Result:** **40.70%**.
    -   **Confusion (17.45%)** and **False Alarm (15.12%)** were the largest contributors. This indicates the system is finding "extra" speakers and capturing non-speech as speech.

## 3. Attribution Metrics (Word-level Accuracy)
This metric measures how accurately the system assigns a speaker to an individual word.
-   **Methodology:**
    1.  Map each system word to a ground truth speaker based on its midpoint.
    2.  Find the optimal greedy mapping between system speaker IDs and ground truth speaker IDs.
    3.  Calculate the percentage of words correctly attributed.
-   **Phase 13 Result:** **81.51%**. This suggests that while DER is high (due to temporal boundary issues), the core identity of who is speaking when a word is spoken is relatively accurate.

## 4. Conversational Metrics
These metrics assess the end-to-end quality of the transcript.
-   **cpWER (Concatenated Permutation WER):**
    -   Calculated by concatenating all words for each speaker and finding the permutation of system-to-ground-truth speakers that minimizes the total WER.
    -   **Result:** **26.85%**.
-   **WDER (Word Diarization Error Rate):**
    -   Measures the percentage of words that are assigned to the wrong speaker, independent of ASR errors.
    -   **Result:** **18.49%**.
-   **SA-WER (Speaker-Attributed WER):**
    -   A variation of WER that penalizes both ASR errors and speaker attribution errors.
    -   **Result:** **26.85%**.

## 5. VAD Metrics
VAD is evaluated using a frame-based approach (10ms resolution).
-   **F1-Score:** **89.88%**.
-   **Miss Rate:** **5.73%**.
-   **False Alarm Rate:** **67.00%** (Note: This is relative to the *non-speech* duration, which is small in this recording, leading to a high percentage).
-   **Observations:** The VAD is highly sensitive, capturing almost all speech (low miss rate) at the cost of capturing some noise or extended pauses (high false alarm).
