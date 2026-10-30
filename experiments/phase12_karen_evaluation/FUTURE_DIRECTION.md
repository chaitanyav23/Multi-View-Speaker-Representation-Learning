# Phase 12 Future Direction

Building on the successes and challenges identified during the Karen evaluation, the following areas have been prioritized for future development.

## 1. Advanced Diarization Models (EEND)
While MSDD is highly effective, it still relies on a separate VAD and embedding stage.
- **Direction:** Explore **End-to-End Neural Diarization (EEND)** or **EEND-EDA** (Encoder-Decoder Based Attractors). These models can handle an unknown number of speakers and overlapping speech in a single forward pass, potentially further reducing the WDER.

## 2. LLM-Based Post-Correction
The conversational timelines generated in Phase 12 are acoustically accurate but lack "semantic polish."
- **Direction:** Integrate a Large Language Model (e.g., GPT-4o or Gemini 1.5 Pro) to:
    - Correct ASR substitutions based on conversational context.
    - Resolve "UNKNOWN" speakers by analyzing dialogue flow (e.g., if Speaker B answers a question addressed to "Officer," Speaker B is likely an officer).
    - Provide higher-level summaries and entity extraction.

## 3. Streaming & Real-Time Adaptation
The current pipeline is batch-oriented.
- **Direction:** Adapt the `step2_to_6.py` logic for **streaming inference**. This would involve moving from full-file VAD to sliding-window VAD and implementing online diarization.

## 4. Robustness to Non-Speech Audio
Phase 12 handled background noise well, but complex acoustic environments (sirens, radio chatter, shouting in the background) still present a challenge.
- **Direction:** Investigate **Audio Source Separation** (e.g., Demucs) as a pre-processing step to isolate human voice from environmental noise before it reaches the ASR and Diarization models.

## 5. Automated "Full Data" Recovery
The truncation issue in Karen highlighted a need for better data ingestion.
- **Direction:** Implement automated retry logic or multi-source fetching to ensure that the physical audio payload matches the expected metadata duration before processing begins.

## 6. Phase 13 Target
Phase 13 should focus on **Generalization Part 2**, using a full-duration dataset that challenges the system with 4+ speakers in a non-telephonic environment (e.g., a meeting or a multi-officer scene) to further push the boundaries of the auto-speaker estimation logic.
