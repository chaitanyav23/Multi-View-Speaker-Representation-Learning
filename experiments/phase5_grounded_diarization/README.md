# Phase 5: Grounded Diarization (The MSDD Breakthrough)

## Phase Objective
Phase 5 represents the turning point of the project. Its primary goal was to fix the failures of Phase 3 by decoupling the Diarization model (MSDD) from its internal VAD and forcing it to work within the high-quality speech boundaries established in Phase 1 and refined in Phase 4.

## Historical Motivation
Previous phases identified two major blockers:
1.  **Pyannote Identity Collapse:** Unconstrained clustering fails to separate speakers in high-noise environments.
2.  **MSDD Timeout:** NeMo's internal VAD (MarbleNet) hangs infinitely on police sirens.
The team hypothesized that if MSDD was provided with "Oracle" (perfect or near-perfect) speech segments, its sequence-to-sequence decoder would be able to disentangle the speakers effectively without timing out.

## Hypothesis being Tested
Injecting Pyannote 3.1 VAD segments as "Oracle RTTMs" into the NeMo MSDD pipeline will bypass the MarbleNet timeout and solve the identity collapse problem, resulting in the best cpWER achieved so far.

## What Changed from Previous Phase
*   **Oracle Injection:** Developed a mechanism to convert VAD Grounding JSONs into RTTM format for injection into NeMo.
*   **MSDD Configuration:** Bypassed internal VAD (`oracle_vad: True`).
*   **Diagnostic Audit:** Introduced a detailed `diarization_diagnostics.py` script to visualize Miss/FA/Confusion at the audio level.

## Outcomes
*   **Success:** MSDD successfully processed the BWC files in under 40 seconds (down from >300s timeouts).
*   **Performance:** Achieved a **cpWER of 0.4582** on BWC, a 9.3% relative improvement over the Pyannote baseline (0.5054).
*   **Breakthrough:** Confirmed that sequence-to-sequence diarization is the solution to forensic identity collapse, provided it is "grounded" by reliable VAD.
