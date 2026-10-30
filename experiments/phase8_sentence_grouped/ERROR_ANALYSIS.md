# Phase 8 Error Analysis

## The "Turn-Merge" Error
*   **Symptom:** A single sentence segment contains speech from two different people.
*   **Root Cause:** If Speaker A stops and Speaker B starts within 1.0s, the `vad-merge-gap` logic combines them into a single segment. 
*   **Impact:** While MSDD (Oracle Mode) can still technically find the identity switch within that segment, the ASR stage now has to handle a multi-speaker turn, which increases the likelihood of identity confusion during fusion.

## Buffer Interference
*   **Symptom:** The ASR segment starts with the last word of the *previous* speaker.
*   **Root Cause:** A 0.5s buffer is large enough to capture the end of a previous turn if the gap was small. WhisperX sometimes transcribes this buffer text, leading to redundant words in the final timeline.

## False Alarm Bloat
*   By merging segments with a 1.0s gap, any two False Alarms occurring near each other were combined into a larger "hallucination block." 
*   **Result:** The transcript contains longer runs of nonsensical text compared to the fragmented False Alarms of Phase 7.

## DeepFilterNet "Phasing"
*   In the sentence-level chunks, the artifacts of DFN were even more apparent. Long continuous segments of denoised audio sometimes exhibited "spectral drift," where the voice quality changed over the course of the sentence, confusing the ASR decoder.
