# Script Analysis: Phase 13 Karen Generalization

This document provides a technical breakdown of the scripts used in Phase 13, detailing their functions, inputs, and outputs.

## Pipeline Steps

### 1. `step2_vad_grounding.py`
-   **Purpose:** Executes Voice Activity Detection (VAD) to establish speech grounding.
-   **Logic:** Uses Pyannote 3.1 VAD. Implements a custom merging logic for segments with gaps < 0.15s.
-   **Input:** `audio/karen_normalized.wav`
-   **Output:** `grounding/karen_grounding.json`

### 2. `step3_oracle_rttm.py`
-   **Purpose:** Converts the VAD grounding JSON into an RTTM format.
-   **Logic:** Maps each speech segment to a generic "speaker" label. This RTTM is used as "Oracle VAD" for the MSDD model.
-   **Input:** `grounding/karen_grounding.json`
-   **Output:** `grounding/karen_oracle.rttm`

### 3. `step4_whisperx.py`
-   **Purpose:** Performs transcription and word-level alignment.
-   **Logic:**
    -   Loads the audio slice for each VAD segment.
    -   Transcribes using WhisperX `large-v3` (with `condition_on_previous_text=False`).
    -   Aligns the transcriptions using Wav2Vec2 to obtain high-precision word timestamps.
-   **Input:** `audio/karen_normalized.wav`, `grounding/karen_grounding.json`
-   **Output:** `outputs/aligned_words.json`

### 4. `step5_msdd.py`
-   **Purpose:** Executes Multi-scale Diarization Decoder (MSDD).
-   **Logic:**
    -   Initializes NeMo NeuralDiarizer with a multi-scale configuration.
    -   Uses the `karen_oracle.rttm` to define speech boundaries.
    -   Performs speaker embedding extraction (TitaNet_Large) and clustering.
-   **Input:** `audio/karen_normalized.wav`, `grounding/karen_oracle.rttm`
-   **Output:** `diarization/karen_msdd.rttm`

### 5. `step6_fusion.py`
-   **Purpose:** Merges ASR words with MSDD speaker turns.
-   **Logic:**
    -   Implements `assign_speakers_to_words` using maximum temporal overlap.
    -   Groups attributed words into readable conversational blocks.
    -   Maps internal speaker IDs (e.g., `speaker_0`) to friendly labels (e.g., `Speaker_1`).
-   **Input:** `outputs/aligned_words.json`, `diarization/karen_msdd.rttm`
-   **Output:** `outputs/karen_final.json`, `outputs/karen_final.txt`

### 6. `step7_process_gt.py`
-   **Purpose:** Prepares ground truth data for evaluation.
-   **Input:** `ground_truth/karen.txt`
-   **Output:** `ground_truth_json/karen.json`, `ground_truth_rttm/karen.rttm`

## Evaluation Scripts

### 7. `step8_vad_eval.py`
-   **Purpose:** Calculates VAD metrics (Accuracy, Precision, Recall, F1, Miss Rate, FA Rate).
-   **Logic:** Uses a mask-based approach with 10ms resolution. Generates diagnostic files for Miss and False Alarm regions.

### 8. `step9_asr_eval.py`
-   **Purpose:** Calculates Word Error Rate (WER) using `jiwer`.
-   **Logic:** Normalizes text (lowercase, no punctuation) before comparison.

### 9. `step10_diar_eval.py`
-   **Purpose:** Calculates Diarization Error Rate (DER).
-   **Logic:** Uses `pyannote.metrics.diarization.DiarizationErrorRate` to compute DER, Miss, FA, and Confusion.

### 10. `step11_attribution_eval.py`
-   **Purpose:** Evaluates speaker attribution accuracy at the word level.
-   **Logic:** Performs a greedy mapping between system speakers and ground truth speakers to find the optimal alignment before calculating accuracy.

### 11. `step12_13_conversational_eval.py`
-   **Purpose:** Computes advanced conversational metrics.
-   **Metrics:** cpWER (Concatenated Permutation WER), WDER (Word Diarization Error Rate).

### 12. `step14_comparison.py`
-   **Purpose:** Generates a comparison report between the Karen dataset and the BWC Phase 11A baseline.
-   **Output:** `comparisons/karen_vs_bwc.csv`
