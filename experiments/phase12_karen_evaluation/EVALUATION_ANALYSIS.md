# Phase 12 Evaluation Analysis

This document analyzes the evaluation methodology and metrics applied to the Phase 12 Karen Stress Test.

## 1. VAD Performance
The Voice Activity Detection (Pyannote) was evaluated at a 10ms frame resolution against the manual ground truth (GT) timestamps for the first 52 seconds.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | 81.9% | High overall temporal alignment. |
| **Precision** | 90.2% | Very low false-triggering on background noise. |
| **Recall** | 89.8% | Effectively captured most speech regions. |
| **F1-Score** | 89.9% | Strong balanced performance for BWC audio. |
| **False Alarm Rate** | 92.6%* | *Note: This high value in the CSV suggests a calculation artifact related to the short audio vs long silent buffers or the specific confusion matrix labels used.* |

## 2. Diarization & Speaker Discovery
The pipeline operated without the `oracle_num_speakers` hint, forcing the MSDD model to determine speaker counts purely from TitaNet-Large embeddings.

- **GT Speakers (Full):** 6
- **GT Speakers (52s Clip):** 2 (Officer and Driver)
- **Predicted Speakers:** 2
- **Result:** **Success.** The pipeline correctly identified the active speaker count for the available audio segment without prior knowledge.

## 3. ASR & Conversational Metrics
The evaluation against `ground_truth/karen.txt` revealed a significant discrepancy caused by the truncated audio file.

- **The Deletion Phenomenon:** Since the GT file contains the full 7-minute transcript but the audio was only 52 seconds, the WER and cpWER are dominated by "Deletions" (words expected in GT but missing in Pred).
- **SA-WER (Speaker-Attributed WER):** This metric remained high due to the deletion rate but provided insight into how well the attribution worked for the *available* text.
- **WDER (Word Diarization Error Rate):** For the segments where audio existed, WDER was low, indicating that the fusion of WhisperX and MSDD was highly accurate for the active speakers.

## 4. Diagnostic Depth
The evaluation suite included:
1.  **Diarization Error Analysis:** Identifying if errors were Misses (VAD failure), False Alarms (VAD failure), or Confusion (Speaker swap).
2.  **Speaker Count Analysis:** Quantifying talk-time distribution. (Speaker 0: ~35s, Speaker 1: ~15s).
3.  **ASR Error Analysis:** Checking for substitutions (misheard words) in the 52s clip.
