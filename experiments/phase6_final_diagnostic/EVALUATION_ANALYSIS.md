# Phase 6 Evaluation Analysis

## Audit-Level Metrics

### 1. Frame-Level Accuracy (Diarization)
Phase 6 evaluated the "Muting Trick" by comparing the predicted diarization of muted audio against the ground truth on a per-frame basis.
*   **Finding:** Muting non-speech regions significantly reduced **False Alarm** rates in the diarizer (from 0.17 to 0.05).
*   **Finding:** However, it did not solve the **Confusion** rate, as the internal overlap between speakers remained unchanged.

### 2. LUFS vs. Peak Normalization
*   **LUFS-20:** Provides a more "natural" loudness for the human ear but can lead to "gain pumping" in noisy environments.
*   **ASR Impact:** The audit found that LUFS-20 resulted in slightly more **Deletions** in the WER report (45 vs 32), suggesting that the normalization sometimes pushed quiet word endings into the noise floor.

## Ablation Study Results
The ablation study proved that while the "Muting Trick" is technically sound, it cannot compensate for a weak diarization model. The Pyannote 3.1 "Fixed-2" configuration still suffered from identity drift, confirming that the sequence-to-sequence logic of MSDD (from Phase 5) is mandatory for this dataset.
