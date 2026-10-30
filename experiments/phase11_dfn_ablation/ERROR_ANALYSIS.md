# Phase 11 Error Analysis

## The "Trailing Consonant" Deletion
*   **Symptom:** Words like "Stop" becoming "Sto", or "Right" becoming "Righ".
*   **Root Cause:** DFN's recurrent neural network attempts to minimize the loss against a clean speech target. Unvoiced consonants (fricatives, plosives) look very similar to broadband noise in a spectrogram. DFN suppresses them, causing WhisperX to delete the word entirely because it no longer matches the expected acoustic token.

## The VAD Shatter Effect
*   **Symptom:** A single 5-second sentence is broken into 4 smaller segments.
*   **Root Cause:** Human speech naturally contains micro-pauses (e.g., between words). DFN pushes these micro-pauses to absolute zero (-inf dB). Pyannote 3.1 is sensitive to absolute silence and immediately closes its VAD gate, resulting in highly fragmented grounding.

## Synergistic Failure (11B)
*   Pipeline 11B had the worst performance of all because it combined the VAD Shatter Effect with the Trailing Consonant Deletion. 
*   **Impact:** WhisperX was fed a fragmented audio slice that was *also* missing its consonants. This resulted in an enormous spike in Deletions (87 vs 46 in the control) and Substitutions (86 vs 84).
