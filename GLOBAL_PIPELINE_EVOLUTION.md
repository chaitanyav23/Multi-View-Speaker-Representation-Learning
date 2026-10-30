# GLOBAL PIPELINE EVOLUTION

This document provides a high-level overview of the architectural evolution of the BWC (Body-Worn Camera) Conversational Intelligence Pipeline. It tracks the project's journey from a basic linear model to a sophisticated, hardened dual-path system.

## The Evolutionary Epochs

The project progressed through four distinct "Epochs," each representing a major shift in design philosophy based on experimental findings.

### Epoch 1: The Linear Legacy (Phases 1-2)
*   **Architecture:** `Raw Audio -> [Enhancer] -> [WhisperX VAD] -> [Whisper ASR] -> [Pyannote Clustering]`
*   **Key Discovery:** Traditional linear processing fails in hostile acoustic environments.
*   **The Problem:** DeepFilterNet (DFN) denoising, while making audio "sound" better to humans, catastrophically degraded Whisper's ASR accuracy (WER increased by ~24%). WhisperX's internal VAD was too aggressive, often skipping critical but quiet speech.
*   **Outcome:** Abandoned pre-ASR neural enhancement; shifted focus to VAD recall.

### Epoch 2: The MSDD & Oracle Pivot (Phases 3-5)
*   **Architecture:** `Raw Audio -> [Pyannote 3.1 VAD] -> [ASR Branch] & [NeMo MSDD Branch (Oracle Injection)]`
*   **Key Discovery:** Decoupling VAD from Diarization is essential for stability.
*   **The Problem:** Standard Agglomerative Hierarchical Clustering (AHC) collapsed in noisy audio due to embedding drift. NeMo's MSDD was powerful but its internal VAD (MarbleNet) was unreliable.
*   **The Solution:** Introduced **Oracle VAD Injection**. Pyannote 3.1 was used as a master grounding tool to provide speech segments directly to the MSDD diarizer, bypassing its internal VAD.
*   **Outcome:** Solved the clustering collapse issue and stabilized speaker separation.

### Epoch 3: The Dual-Branch Experiment (Phase 10)
*   **Architecture:** `Raw Audio -> [Normalized Branch] & [Denoised Branch] -> Fusion`
*   **Key Discovery:** Temporal synchronization is fragile.
*   **The Problem:** Attempted to use Denoised audio for Diarization and Normalized audio for ASR. 
*   **Failure:** Sub-millisecond temporal drift between the two processed audio streams caused word-to-speaker misattribution.
*   **Outcome:** Unified the pipeline back to a single "Normalized" audio representation for all branches to ensure sample-perfect alignment.

### Epoch 4: The Hardened Master Grounding (Phases 12-18)
*   **Architecture:** `Raw Audio -> RMS Norm -> Pyannote VAD -> Muting Mask -> [ASR/MSDD] -> Word-Midpoint Fusion`
*   **Key Discovery:** "Muting" is the ultimate protection against hallucinations.
*   **The Innovation:** The "Muting Trick" zeros out all non-speech regions (as identified by VAD) before the audio reaches the ASR and Diarization models.
*   **The Fusion:** Moved from segment-level overlap to **Word-Level Midpoint Fusion**, mapping the center of each word-token to the diarization timeline.
*   **Outcome:** Current State-of-the-Art (SOTA) for the project. Achieved the most robust forensic performance on the "Karen" and "Tich" benchmarks.

## Core Architectural Themes

| Theme | Evolution | Why? |
| :--- | :--- | :--- |
| **Pre-processing** | Denoising -> RMS Normalization | Denoising hurts ASR; Normalization is "safe" and stabilizes embeddings. |
| **VAD Strategy** | Internal -> Standalone (Pyannote 3.1) | Forensic applications require high recall over precision. |
| **Diarization** | AHC Clustering -> MSDD (Oracle) | Sequential context (MSDD) handles embedding drift better than clustering. |
| **ASR Control** | Full-file -> Manual Slicing + Muting | Prevents hallucination loops in silent or noisy regions. |
| **Fusion Logic** | Segment Overlap -> Word-Midpoint | Eliminates attribution errors caused by boundary jitter. |
