# Phase 10 Results Analysis

## Ablation Comparison (BWC Dataset)

| Metric | Phase 10C (Raw/Raw + Custom Fusion) | Phase 10B (DFN/Raw + Custom Fusion) | Phase 5 (Raw/Raw + Midpoint Fusion) |
| :--- | :--- | :--- | :--- |
| **WER** | 0.3244 | 0.3781 | **0.2975** |
| **DER** | 0.4703 | 0.5318 | **0.4492** |
| **cpWER** | 0.5032 | 0.5301 | **0.4582** |

## Key Findings

### 1. The Majority Vote Penalty
Comparing Phase 10C to Phase 5 highlights the penalty of the `custom_fusion.py` script. Both used the exact same audio (Raw Normalized) and the exact same models (WhisperX + Oracle MSDD). The only difference was the fusion logic. The Segment-Level Majority Vote increased cpWER by ~10% relative.

### 2. DFN Diarization Penalty
Phase 10B tested using DFN audio for the *ASR* branch, which degraded WER as expected. However, previous phases suggested DFN might help *Diarization*. Phase 10 results do not support this in the context of Oracle MSDD. When MSDD has perfect Oracle boundaries, giving it DFN-flattened audio actually increases DER (0.53) compared to raw audio (0.47).

### 3. Pyannote 3.1 Confirmation
Phase 10A (Control) yielded a cpWER of 0.74, reaffirming that Pyannote 3.1 is entirely unsuitable for high-noise forensic diarization without MSDD's sequence decoding.

## Experimental Conclusion
The "Dual-Branch" architecture is mechanically viable but was sabotaged by flawed fusion logic. The core lesson is that **Word-Level granularity must be preserved**; any attempt to "smooth" or "majority vote" at the segment level will destroy the conversational timeline.
