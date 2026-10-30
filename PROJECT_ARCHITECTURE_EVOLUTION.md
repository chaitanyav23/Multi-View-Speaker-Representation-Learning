# PROJECT ARCHITECTURE EVOLUTION: BWC PIPELINE

This document tracks the structural evolution of the BWC Speech Pipeline, explaining what changed, why it changed, and the architectural diagrams for each major epoch.

---

## 1. Epoch 1: The Linear Legacy (Phase 1-2)
Initially, the pipeline followed a standard serial processing model commonly used in academic datasets (AMI, LibriSpeech).

### Architecture Diagram
```text
Raw Audio -> [Enhancer (DFN)] -> [WhisperX VAD] -> [Whisper ASR] -> [Pyannote Clustering] -> Output
```

### What Changed?
*   **The Problem:** WhisperX's internal VAD was too aggressive for BWC noise, skipping quiet speech. Aggressive denoising (DeepFilterNet) destroyed the acoustic features needed for ASR.
*   **The Rationale:** We realized that BWC audio requires "Recall over Precision." It is better to transcribe a noisy segment than to skip it entirely.
*   **Outcome:** Shifted to standalone, high-recall VAD (Pyannote 3.1) and abandoned pre-ASR enhancement.

---

## 2. Epoch 2: The Oracle MSDD Pivot (Phase 3-5)
The most critical architectural shift occurred when we decoupled VAD from Diarization.

### Architecture Diagram
```text
Raw Audio -> [Pyannote 3.1 VAD (Master Grounding)]
                     |
        ┌────────────┴────────────┐
        ▼                         ▼
[WhisperX (ASR Branch)]    [NeMo MSDD (Diarization Branch)]
                                  ^
                                  | (Oracle VAD Injection)
```

### What Changed?
*   **The Problem:** NeMo MSDD would "time out" or crash when processing raw BWC audio because its internal VAD (MarbleNet) was confused by sirens. Pyannote's clustering collapsed into one speaker.
*   **The Rationale:** By injecting Pyannote's VAD segments as "Oracle" RTTMs into MSDD, we forced the diarizer to only look at known speech.
*   **Outcome:** Achieved a stable 2-speaker separation in noisy field audio for the first time.

---

## 3. Epoch 3: The Dual-Branch Experiment (Phase 10)
Attempted to optimize the audio representation for each branch independently.

### Architecture Diagram
```text
              Raw Audio
                  |
        ┌─────────┴─────────┐
        ▼                   ▼
[Normalized Audio]  [Denoised Audio (DFN)]
        |                   |
        ▼                   ▼
[Diarization Branch]  [ASR Branch]
        |                   |
        └─────────┬─────────┘
                  ▼
          [Timeline Fusion]
```

### What Changed?
*   **The Problem:** Fusion failed. The Denoised audio had slight phase shifts and sub-millisecond temporal drift compared to the Normalized audio.
*   **The Rationale:** Even a 10ms drift is enough to misalign a word's start time with a speaker's boundary.
*   **Outcome:** Unified the pipeline back to a single "Normalized" representation for both branches.

---

## 4. Epoch 4: The Hardened Master Grounding (Phase 18+)
The current state-of-the-art architecture, optimizing for forensic stability.

### Architecture Diagram
```text
           [Raw BWC Audio]
                  |
        [RMS Normalization (-20dB)]
                  |
        [Pyannote 3.1 VAD Grounding]
                  |
        [NumPy Binary Muting Mask] -- (Zeros out all non-speech)
                  |
        ┌─────────┴─────────┐
        ▼                   ▼
[Faster-Whisper (Large-v3)] [NeMo Oracle MSDD]
[Wav2Vec2 Alignment]        [TitaNet Embeddings]
        |                   |
        └─────────┬─────────┘
                  ▼
      [Word-Level Midpoint Fusion]
                  |
        [Forensic JSON Transcript]
```

### What Changed?
*   **The Problem:** Whisper hallucinations in silent/siren regions and unstable alignment.
*   **The Rationale:** The "Muting Trick" (Epoch 4) physically removes non-speech before it hits the transformer, preventing the model from "hearing" speech in sirens. Manual slicing ensures the 30s context window is strictly controlled.
*   **Outcome:** This is the most stable and accurate version of the pipeline to date, achieving the lowest cpWER and best forensic integrity.

---

## Summary of Evolutionary Drivers

| Factor | Change | Reason |
| :--- | :--- | :--- |
| **Acoustics** | Raw -> Norm -> Muted | Clipping and siren-hallucinations. |
| **VAD** | Internal -> Pyannote 3.1 | High recall is required for forensic safety. |
| **Diarizer** | AHC Clustering -> Oracle MSDD | Embedding drift in dynamic BWC environments. |
| **ASR** | Full-file -> Manual Slicing | Stabilizes timestamps and prevents hallucination loops. |
| **Fusion** | Segment-level -> Word-Midpoint | Prevents boundary-jitter from causing misattribution. |
