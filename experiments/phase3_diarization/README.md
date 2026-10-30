# Phase 3: Neural Diarization (MSDD Study)

## Phase Objective
Phase 3 was designed to address the "Identity Collapse" problem observed in earlier tests, where standard agglomerative clustering (Pyannote) would often group multiple speakers into a single identity in noisy environments. The objective was to integrate and benchmark NeMo's **Multi-Scale Diarization Decoder (MSDD)**.

## Historical Motivation
BWC audio is characterized by high embedding drift—where a speaker's voice characteristics change significantly due to movement, background noise, or police radio interference. Agglomerative clustering is mathematically prone to failure in these non-stationary conditions. MSDD, a sequence-to-sequence model, was hypothesized to be more robust.

## Hypothesis being Tested
NeMo MSDD, utilizing TitaNet embeddings and a multiscale decoding architecture, will provide more accurate speaker boundaries and prevent identity collapse compared to Pyannote 3.1 clustering.

## What Changed from Previous Phase
*   **Diarization Engine:** Transitioned from Pyannote Agglomerative Clustering to NeMo NeuralDiarizer (MSDD).
*   **Clustering Logic:** Moved from distance-based clustering to a sequence-to-sequence decoding approach.

## Outcomes (The "Phase 3 Failure")
*   **MSDD Timeout:** The initial implementation of MSDD failed on BWC data. The internal `MarbleNet` VAD component in NeMo entered an infinite computation loop when processing high-intensity sirens and overlapping background noise.
*   **Identity Collapse:** Baseline Pyannote runs during this phase confirmed the "Identity Collapse" issue, where the model artificially minimized DER by assigning everyone to Speaker 0.
*   **Result:** This phase was technically unsuccessful but provided the critical insight that MSDD's internal VAD must be bypassed (leading to the "Oracle MSDD" breakthrough in Phase 5).
