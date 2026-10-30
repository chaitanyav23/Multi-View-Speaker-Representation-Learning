# Phase 15: Evaluation Analysis

## Evaluation Framework
Phase 15 uses a multi-dimensional evaluation framework to assess the pipeline's performance on the `karen.wav` dataset. Evaluation is performed across four primary categories: VAD, ASR, Diarization, and Conversational (Fused).

### 1. Voice Activity Detection (VAD)
- **Tool:** Custom script `step8_vad_eval.py`.
- **Metrics:** Accuracy, Precision, Recall, F1, Miss Rate, False Alarm Rate.
- **Methodology:** Frame-level comparison between the VAD grounding segments and the Ground Truth segments.

### 2. Automatic Speech Recognition (ASR)
- **Tool:** `jiwer` library in `step9_asr_eval.py`.
- **Metrics:** Word Error Rate (WER), Substitutions, Deletions, Insertions.
- **Methodology:** Compares the "raw" concatenated text from Step 4 with the Ground Truth text. This measures the transcription accuracy without considering speaker labels.

### 3. Diarization
- **Tool:** `pyannote.metrics.diarization` in `step10_diar_eval.py`.
- **Metrics:** Diarization Error Rate (DER), Miss, False Alarm, Confusion.
- **Methodology:** Standard RTTM vs. RTTM comparison with a 0.25s collar. DER = (Miss + False Alarm + Confusion) / Total Speech Duration.

### 4. Conversational (Speaker-Attributed ASR)
- **Tool:** Custom script `step12_13_conversational_eval.py`.
- **Metrics:**
  - **cpWER (Concatenated Permutation WER):** The most rigorous metric. It permutes the speaker labels to find the best match with the ground truth and then calculates WER per speaker.
  - **WDER (Word Diarization Error Rate):** Measures the percentage of correctly transcribed words that were assigned to the wrong speaker.
  - **SA-WER (Speaker-Attributed WER):** Similar to cpWER, measuring the WER of the final attributed transcript.

## Ground Truth Processing
Step 7 (`step7_process_gt.py`) is vital for evaluation. It takes the "Karen" ground truth and generates:
- An RTTM file for DER calculation.
- A JSON file for word-level attribution comparison.
- A normalized text file for WER calculation.
This ensures that the "No-Normalization" system output is compared against a consistent and reliable reference.
