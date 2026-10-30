# Phase 3 Conclusions

## Summary of Results
Phase 3 was a **productive failure**. It demonstrated that state-of-the-art neural diarizers (NeMo MSDD) cannot be used "out of the box" on high-noise BWC audio due to VAD computation loops. Simultaneously, it proved that baseline clustering (Pyannote 3.1) is fundamentally unsuitable for forensic audio due to the Identity Collapse problem.

## Hypothesis Success
The hypothesis was **inconclusive** in this phase. While MSDD promised better performance, the implementation was too fragile to generate valid metrics on the BWC dataset.

## Key Takeaways
*   **Identity Collapse** is the primary challenge for BWC speaker attribution.
*   **Bypass the Internal VAD:** To make MSDD work, we must find a way to feed it external VAD boundaries (the "Oracle VAD" path).
*   **Decoupled Fusion** via `run_hardened_with_custom_diar.py` is the correct architectural path forward.

## Preservation
The `run_hardened_with_custom_diar.py` script is a key architectural artifact and will be integrated into the core pipeline. The MSDD configuration will be significantly modified in future phases to enable "Oracle VAD" injection.
