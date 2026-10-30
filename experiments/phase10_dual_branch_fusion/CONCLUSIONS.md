# Phase 10 Conclusions

## Summary of Results
Phase 10 was an ambitious attempt to formalize a "Dual-Branch" architecture. While the infrastructure worked, the results were severely compromised by a flawed Custom Fusion script that utilized a segment-level majority vote. The phase resulted in a cpWER of 0.50, failing to beat the Phase 5 baseline of 0.45.

## Hypothesis Success
The hypothesis was **rejected**. Using DFN audio for the diarization branch did not improve performance when Oracle segments were already provided. Furthermore, the custom overlap logic failed to improve upon simple midpoint assignment due to the overriding majority vote.

## Key Takeaways
*   **Word-Level Attribution is Mandatory:** Never roll up speaker identities to the segment level in conversational audio.
*   **Raw Audio Wins Again:** When provided with Oracle boundaries, MSDD performs better on Raw-Normalized audio than on DFN-Enhanced audio.
*   **The Baseline Holds:** Phase 9 remains the most stable, and Phase 5 remains the most accurate.

## Preservation
The concept of the "Dual-Branch" architecture is sound, but the execution was flawed. The `custom_fusion.py` script will be discarded or heavily refactored to strictly enforce word-level assignments in future phases.
