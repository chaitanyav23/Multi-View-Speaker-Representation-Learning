# Phase 6 Error Analysis

## VAD/Diarization Sync Errors
*   **Symptom:** A word is correctly transcribed but has no speaker label (assigned to 'UNK').
*   **Root Cause:** The manual ASR slicing occasionally generated words that extended slightly beyond the binary muting mask. Since the diarizer only processed the non-zero parts of the mask, these "tail words" fell into a temporal void.

## The "Gain Pumping" Effect
*   **Symptom:** Increase in Deletions during loud environmental noise.
*   **Root Cause:** `loudnorm` attempts to maintain a consistent integrated LUFS. When a siren blares, the algorithm reduces the gain of the entire segment to compensate, which often makes the human voice in that segment too quiet for WhisperX to decode.

## Identity Swap (The "Officer-Suspect Flip")
*   **Symptom:** High cpWER despite correct transcription and separation.
*   **Root Cause:** Pyannote 3.1 correctly finds two speakers but assigns the "Suspect" label to the "Officer" and vice versa. Without a fixed reference embedding, the model has a 50/50 chance of flipping the identities in any given file.

## Over-Muting
*   **Symptom:** Sentences being cut off at the start or end.
*   **Root Cause:** The Pyannote VAD segments were sometimes too tight. When the binary mask was applied, it chopped off the first and last 50ms of speech, which is where critical consonant information (plosives) resides.
