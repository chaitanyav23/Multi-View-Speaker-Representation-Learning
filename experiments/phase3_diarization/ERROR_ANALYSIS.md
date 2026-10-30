# Phase 3 Error Analysis

## Identity Collapse (Pyannote)
*   **Symptom:** The prediction contains only one speaker label (`SPEAKER_00`), even though multiple people are speaking.
*   **Root Cause:** The `AgglomerativeClustering` algorithm used a distance threshold that was too high for the noisy embeddings generated from BWC audio. The noise "pulled" different speakers' embeddings toward a common centroid.

## Infinite VAD Loop (NeMo)
*   **Symptom:** `pipeline_msdd.py` hangs and consumes 100% CPU without generating an RTTM.
*   **Root Cause:** NeMo's `MarbleNet` VAD is a lightweight model that struggles with out-of-distribution sounds like sirens. It produced a massive number of micro-segments (fragments <0.1s), which caused the multiscale embedding windowing logic to stall.

## Midpoint Fusion Errors
*   **Symptom:** Words assigned to the wrong speaker during the merge.
*   **Root Cause:** In Phase 3, the alignment between WhisperX word timestamps and MSDD segments was not perfect. A 100ms drift could cause a word at a turn-taking boundary to be "captured" by the wrong speaker's segment.

## Embedding Drift
*   **Symptom:** A speaker's identity changes halfway through the file.
*   **Root Cause:** As a police officer moves from a quiet car to a loud street, their voice's acoustic profile shifts. Global clustering fails to realize it is the same person.
