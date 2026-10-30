# Phase 9 Error Analysis

## False Alarm Hallucinations
*   **Symptom:** Long strings of repeated filler words ("and," "you," "the") assigned to a speaker.
*   **Root Cause:** The high-recall VAD Grounding correctly captures extremely quiet speech but also captures rhythmic siren pulses. WhisperX tries to "decode" these pulses, resulting in hallucinations.

## Turn-Boundary "Jitter"
*   **Symptom:** Speaker A says a word, but it is assigned to Speaker B.
*   **Root Cause:** The midpoint fusion logic. If the word's midpoint falls 10ms outside of the MSDD segment, it gets attributed to the wrong person. This was common in Phase 9 during overlapping cross-talk.

## Embedding Conflict
*   In the BWC dataset, there are moments where the officer is shouting. The acoustic profile of shouting is very different from normal speech, causing MSDD to occasionally mis-identify the shouting officer as a new, unknown speaker.

## Normalization "Artifacts"
*   `ffmpeg loudnorm` can introduce a slight "breathing" effect where the background noise volume rises and falls based on the speaker's activity. This dynamic noise floor was found to be slightly confusing for the Whisper Large-v3 encoder.
