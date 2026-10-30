# Phase 6: Final Diagnostic Audit & Ablation Study

## Phase Objective
Phase 6 was a rigorous forensic audit of the pipeline's core components. Its primary goal was to perform a "Diagnostic Stress Test" on the hardened pipeline features—including LUFS normalization and VAD-based audio muting—to identify exactly where the remaining errors originate and whether the "enhancements" from previous phases provide consistent value.

## Historical Motivation
After the breakthrough in Phase 5, the team needed to verify if the individual features (like DFN and Normalization) were truly synergistic. Phase 6 aimed to "ablate" the system by testing these features in various combinations using a standardized, frame-by-frame diagnostic tool.

## Hypothesis being Tested
A "Hardened" pipeline utilizing LUFS-20 normalization and a binary VAD muting mask will provide more stable speaker embeddings for Pyannote 3.1, potentially allowing it to compete with the MSDD performance from Phase 5.

## What Changed from Previous Phase
*   **Normalization Strategy:** Switched from simple Peak Normalization to `ffmpeg loudnorm` (-20 LUFS).
*   **Muting Trick:** Implemented a NumPy-based binary mask that silences the audio during non-speech segments *before* the diarization stage.
*   **Manual ASR Slicing:** Modified the ASR stage to process audio in discrete segments rather than a continuous stream to minimize context drift.

## Outcomes
*   **Diarization Regression:** The audit revealed that even with the "Muting Trick," Pyannote 3.1 (Fixed-2) remained significantly worse than the Oracle MSDD path from Phase 5, with DER values exceeding 0.60.
*   **VAD/ASR Coupling:** Discovered that LUFS normalization, while good for human listening, actually slightly increased WER on BWC data (from 0.29 to 0.30) due to how it affected the background noise floor.
*   **Diagnostic Visualization:** Successfully generated frame-level spectrogram audits that visually proved that False Alarms (sirens) were still being processed as speech.
