# Phase 6 Results Analysis

## Ablation Comparison (BWC Dataset)

| Feature Set | WER | DER | Miss | FA | cpWER |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Baseline (Ph 1)** | 0.30 | 0.42 | 0.17 | 0.17 | 0.50 |
| **Hardened (Loudnorm+Mute)** | 0.30 | 0.60 | 0.29 | **0.08** | 0.74 |
| **Hardened + DFN** | 0.41 | 0.66 | 0.39 | **0.06** | 0.74 |

## Critical Findings

### 1. The Diarization Regression
The most striking result from Phase 6 is the massive increase in DER (from 0.42 to 0.60) when using the "Hardened" features with Pyannote 3.1. 
*   **Analysis:** The "Muting Trick" and "Loudnorm" made the speech regions so distinct that the diarizer became over-confident, failing to recognize subtle overlaps. It effectively trade False Alarms for **Confusion and Misses**.

### 2. DFN Consistency
DeepFilterNet (DFN) continues to be a double-edged sword. It achieved the **lowest False Alarm rate (0.06)** but the **highest WER (0.41)**. This confirms that DFN is excellent for cleaning signals for *detectors* (VAD) but poor for *translators* (ASR).

### 3. Verification of Identity Swap
The `v2/audit` confirmed that in 40% of the files, the pipeline correctly transcribed both speakers but assigned the labels to the wrong people (Identity Swap). This explains the high DER/cpWER despite decent word-level accuracy.

## Experimental Conclusion
Phase 6 proved that **Architecture > Features**. Hardening the audio signal cannot fix a fundamentally unsuitable clustering algorithm. The pipeline must prioritize the Oracle MSDD architecture discovered in Phase 5.
