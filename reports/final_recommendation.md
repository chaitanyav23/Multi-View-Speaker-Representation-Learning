# BWC Pipeline Optimization Study: Final Recommendation

## Executive Summary
This study evaluated three critical axes of the BWC pipeline: Voice Activity Detection (Phase 1), Speech Enhancement (Phase 2), and Diarization Architecture (Phase 3). 

### 1. Is VAD the dominant bottleneck?
**YES.** 
- The current **Silero VAD** misses 18% of speech in BWC footage.
- **Pyannote VAD** reduces misses to 11% but doubles False Alarms (23%).
- This "Grounding" step determines the context window for ASR. Errors here (Misses) lead directly to Deletions in the final transcript.

### 2. Is speech enhancement the dominant bottleneck?
**NO.** 
- In fact, traditional neural enhancement (**SpeechBrain MetricGAN+**) made the pipeline **significantly worse**, increasing WER from 30% to 45%. 
- The artifacts introduced by enhancement are more damaging to Whisper's transformer attention than the original environmental noise.

### 3. Is clustering-based diarization the dominant bottleneck?
**YES (for attribution).**
- Speaker Confusion remains at ~7-10%. While Normalization (Pipeline B) helped stabilize embeddings, simple clustering still struggles with overlapping speech and distance-varying acoustics.

### 4. Largest Improvements
| Metric | Winner | Modification |
| :--- | :--- | :--- |
| **DER** | **Pyannote VAD** | Reducing Miss Rate (though FA increased) |
| **WDER** | **Normalization** | Stabilizing speaker embeddings |
| **SA-WER** | **Normalization** | Best balance of text and speaker accuracy |
| **WER** | **Raw Audio** | No enhancement performed better than neural enhancement |

## Final Recommended Production Pipeline
The new production standard should follow the **"Stabilized Grounded Pipeline"**:

1.  **Loudness Normalization (-20dB Peak):** Essential for reducing Speaker Confusion.
2.  **Pyannote 3.1 VAD:** Use for grounding to minimize Miss Rate (prioritize coverage over precision).
3.  **Muted Diarization Trick:** Continue zeroing out non-speech to focus the clusterer.
4.  **WhisperX (large-v3) ASR:** Use on raw (normalized) audio only—skip neural enhancement.

## Future Path
To break the current ceiling:
- Implement **DeepFilterNet 3** properly once environment issues are resolved (it is generally more conservative than SpeechBrain).
- Move to **Overlap-Aware Neural Diarization** (NeMo MSDD) once the complex configuration overhead is automated.

