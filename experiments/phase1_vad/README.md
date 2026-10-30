# Phase 1: VAD Benchmarking (Silero vs. Pyannote)

## Phase Objective
The primary objective of Phase 1 was to establish a robust Voice Activity Detection (VAD) baseline for the project. By comparing two industry-standard VAD models—Silero VAD and Pyannote Segmentation 3.0—the goal was to determine which model provided better "grounding" for downstream ASR and Diarization tasks, particularly in challenging audio environments (BWC and Podcasts).

## Historical Motivation
Initial pipeline tests indicated that errors in the ASR/Diarization stages were often propagated from poor VAD segmentation (e.g., cutting off speech, including long silences, or failing to detect overlapping speakers).

## Hypothesis being Tested
Pyannote's segmentation model (v3.0) will outperform Silero VAD in multi-speaker environments due to its more sophisticated neural architecture trained on diverse diarization datasets, leading to lower Miss rates and better downstream DER (Diarization Error Rate).

## What Changed from Previous Phase
This is the initial benchmarking phase. It introduced the concept of "Grounding JSON" files which define the segments to be processed by the subsequent pipeline stages.

## Outcomes
*   **Silero VAD:** Demonstrated high speed and low False Alarm rates but occasionally missed faint speech or cut off endings.
*   **Pyannote VAD:** Showed higher precision in speech boundaries and better handling of low-SNR (Signal-to-Noise Ratio) segments.
*   **Best Performer:** Pyannote VAD was selected as the default grounding mechanism for subsequent phases.
