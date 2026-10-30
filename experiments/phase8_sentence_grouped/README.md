# Phase 8: Sentence-Level Context Grouping

## Phase Objective
Phase 8 explored whether increasing the temporal context provided to the ASR and Diarization models could improve transcription accuracy and identity separation. By merging closely-spaced VAD segments into larger "logical sentences," the team aimed to reduce the fragmentation observed in previous sliced-ASR attempts.

## Historical Motivation
Phase 7 results showed that very tight VAD segments (0.0s merge gap) caused fragmentation, making it difficult for WhisperX to maintain conversational flow even with stateful prompting. Phase 8 hypothesized that logically grouping these segments would provide the models with more natural linguistic and acoustic context.

## Hypothesis being Tested
Merging speech segments with a 1.0s gap (to form "Sentences") will improve VAD Recall and lower DER by providing MSDD with more continuous acoustic information, leading to a more coherent final transcript.

## What Changed from Previous Phase
*   **Merge Gap:** Increased `vad-merge-gap` from 0.0s to 1.0s.
*   **ASR Buffer:** Increased `asr-buffer` from 0.2s to 0.5s to ensure the WhisperX engine has sufficient "pre-roll" audio to stabilize its decoder.
*   **Segment Logic:** Transitioned from "Segment-at-a-Time" to "Sentence-at-a-Time" processing.

## Outcomes
*   **VAD Breakthrough:** Achieved the project's highest **VAD Recall (0.86)**. Merging segments effectively "filled in the gaps" between closely spaced words that individual VAD passes were missing.
*   **Diarization Improvement:** DER improved from 0.61 to 0.55. Longer segments provided MSDD with more embedding data per turn, leading to more stable identity predictions.
*   **WER Stagnation:** WER remained high (0.37) because the audio was still being processed through DeepFilterNet. This confirmed that no amount of contextual grouping can overcome the spectral damage of aggressive neural denoising.
