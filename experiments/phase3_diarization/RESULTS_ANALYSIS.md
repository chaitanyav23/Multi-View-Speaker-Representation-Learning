# Phase 3 Results Analysis

## Benchmarking Comparison (BWC Dataset)

| Metric | Pyannote 3.1 (Clustering) | NeMo MSDD (Initial) |
| :--- | :--- | :--- |
| **WER** | 0.3037 | **N/A (Execution Failed)** |
| **DER** | 0.4234 | **N/A (Timeout)** |
| **cpWER** | 0.4989 | **N/A** |
| **Speaker Count** | 1 (Collapse) | 0 (Crash) |

## Analysis of the "Failure State"

### 1. Pyannote Identity Collapse
The Pyannote 3.1 model, when run in "Auto" mode on BWC audio, failed to find the cluster boundary between the two primary speakers. The noisy background caused the speaker embeddings to drift into a single dense cluster, which the agglomerative clustering algorithm could not split. This resulted in a **cpWER of 0.4989**, effectively making the transcript forensicly useless for speaker-attribution.

### 2. MSDD Timeout Root Cause
Diagnostic traces revealed that NeMo's `MarbleNet` VAD was overwhelmed by the high-frequency rhythmic patterns of police sirens. The VAD logic entered a state of "infinite segments," creating a manifest that the subsequent embedding extractor could not process within the 300-second timeout.

### 3. Podcast Success
On clean audio (Podcasts), both models performed well, with MSDD showing slightly better separation of overlapping laughter. This confirmed that the models themselves were functional, but the **BWC noise was a critical blocker** for the default NeMo configuration.

## Experimental Finding
*   **The VAD-Diarization Coupling Problem:** MSDD's failure was caused by its *internal* VAD, not its diarization logic.
*   **Agglomerative Vulnerability:** Traditional clustering is unsuitable for noisy, long-form conversational audio.
