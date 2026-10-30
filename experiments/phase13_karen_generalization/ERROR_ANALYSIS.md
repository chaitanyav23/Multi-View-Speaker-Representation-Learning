# Error Analysis: Phase 13 Karen Generalization

Through detailed diagnostic auditing, three primary failure modes were identified in the Phase 13 pipeline as applied to the Karen dataset.

## 1. Speaker Over-Clustering (Primary Diarization Error)
The most significant technical issue was the discovery of **6 speakers** by the MSDD model, while the ground truth contains only **4**.
-   **Impact:** This leads to "Identity Fragmentation," where a single real speaker is split into two or more system speaker IDs.
-   **Cause:** The TitaNet_Large embeddings for certain speakers likely exhibited high intra-speaker variance (potentially due to distance from microphone or emotional inflection changes), causing the clustering algorithm to treat them as distinct individuals.
-   **Metric Correlation:** This is the primary driver of the **17.45% Confusion** rate in DER and the **18.49% WDER**.

## 2. VAD Over-Sensitivity (False Alarm Source)
The Pyannote 3.1 VAD, while excellent at capturing all speech, is prone to false alarms in this environment.
-   **Impact:** Non-speech segments (e.g., loud breaths, paper rustling, or table thumps) are classified as speech. These segments are then passed to ASR (which might hallucinate short words) and Diarization (which attributes them to a speaker).
-   **Metric Correlation:** Results in a **15.12% False Alarm** rate in DER.
-   **Evidence:** `reports/vad_metrics.csv` shows a 67% False Alarm rate relative to the non-speech duration.

## 3. ASR Deletions (Secondary ASR Error)
Despite the low WER (14.67%), the ASR engine is still failing to capture a significant number of words (156 deletions).
-   **Pattern:** Deletions tend to occur during:
    -   Very short interjections (e.g., "Yeah", "Uh-huh").
    -   Regions of low-volume "muttered" speech.
    -   Rapid conversational transitions where the end of one turn and the start of another are closely packed.
-   **Mechanism:** The VAD-based segmentation might be clipping the very start or end of these short bursts, or WhisperX might be failing to decode them as valid tokens within the context of the segment.

## 4. Word-Speaker Alignment Jitter
There are instances where words are assigned to the wrong speaker turn even when the turn is correctly identified.
-   **Cause:** "Temporal Jitter" in the Wav2Vec2 alignment. If a word's timestamp is offset by even 100-200ms, it might fall into a neighboring speaker's turn in a fast-paced conversation.
-   **Metric Correlation:** Contributes to the delta between WER and SA-WER.

## 5. Diagnostic Summary
The pipeline is currently "Speech-Aggressive" (prioritizing Recall over Precision). This is beneficial for ensuring no content is lost but detrimental to the clarity of the speaker timeline. Future tuning must focus on **clustering stability** and **non-speech rejection**.
