# PHASE 9 PRODUCTION CANDIDATE PIPELINE: FINAL REPORT

## 1. Pipeline Architecture
The Phase 9 pipeline was designed as the "production candidate", assembling the winning findings from Phases 1 through 7 into a single unified architecture. The architecture processes the BWC audio systematically:

- **Stage 0**: Input (`raw_audio/bwc.wav`)
- **Stage 1 - Normalization**: LUFS Normalization (-20 LUFS) with peak protection via `ffmpeg`.
- **Stage 2 - Master Grounding**: Voice Activity Detection via Pyannote 3.1 using a high-recall configuration (min_duration=0.05s, padding=0.1s, merge_gap=0.0s).
- **Stage 3 - Grounded Audio**: Non-speech regions are heavily muted based on the Pyannote VAD mask to prevent ghost hallucination. 
- **Stage 4 - Hardened ASR**: WhisperX Large-v3 operated via manual VAD-grounded slicing with `condition_on_previous_text=False` to strictly limit ungrounded hallucinations.
- **Stage 5 - Alignment**: WhisperX word-level alignment against the canonical normalized audio.
- **Stage 6 - Diarization (Oracle-VAD MSDD)**: NeMo's Multi-scale Diarization Decoder (MSDD) telephonic model, using the Master Grounding as the Oracle VAD source.
- **Stage 7 - Speaker Attribution**: Assigning speakers to aligned transcript segments.

## 2. Configuration Used
- **Audio Preprocessing**: `-af loudnorm=I=-20:TP=-1.5:LRA=11 -ar 16000 -ac 1`
- **VAD (Pyannote 3.1)**: `min_duration=0.05`, `padding=0.1`, `merge_gap=0.0`
- **ASR (WhisperX Large-v3)**: `language="en"`, `condition_on_previous_text=False`
- **Diarization (NeMo MSDD)**: `oracle_vad=True`, `max_num_speakers=2`, `multiscale_weights=[1, 2, 3, 3, 3]`, `collar=0.25`, `ignore_overlap=False`.

## 3. Rationale for Component Selection
- **Pyannote VAD**: Selected over Silero because previous phases revealed its significantly lower miss rate on dynamic body-worn camera audio. High recall was favored over precision to capture transient speech.
- **Neural Enhancement Exclusion**: DeepFilterNet and SpeechBrain were explicitly excluded from this pipeline because Phase 4 and Phase 7 findings consistently proved they severely degraded Word Error Rate (WER) and exacerbated artifacts on non-studio audio. 
- **Grounded ASR & Slicing**: BWC audio is highly prone to conversational hallucination. Conditioning was disabled to prevent run-away ASR loops, and manual segment slicing was utilized to force transcription exclusively within verified speech bounds.
- **Oracle-VAD MSDD**: NeMo MSDD previously provided better clustered conversational utility than simple embedding clustering models, making it the top candidate for final speaker segmenting.

## 4. VAD Analysis
- **Accuracy**: 74.47%
- **Precision**: 83.55%
- **Recall**: 80.88%
- **F1 Score**: 82.19%
- **Miss Rate**: 19.12%
- **False Alarm Rate**: 42.68%
*Observations*: Pyannote effectively captured the majority of speech (high precision/recall). However, the False Alarm rate ballooned due to the intentional high-recall parameters and ambient noise in BWC audio.

## 5. ASR Analysis
- **Word Error Rate (WER)**: 31.58%
- **Substitutions**: 67
- **Deletions**: 48
- **Insertions**: 47
- **Total Reference Words**: 513
*Observations*: The ASR heavily reduced hallucinated insertions compared to earlier unconstrained phases, driven largely by the unconditioned manual VAD slicing strategy. The baseline WER sits at ~31.58% which reflects the inherent acoustic challenge of BWC.

## 6. Diarization Analysis
- **Diarization Error Rate (DER)**: 52.96%
- **Missed Detection**: 27.47%
- **False Alarm**: 8.40%
- **Confusion**: 17.09%
*Observations*: MSDD struggled significantly with the body-worn camera acoustic environment. The DER sits unacceptably high. The Oracle VAD minimized FA (8.40%), but Missed Detections skyrocketed (27.47%) as MSDD failed to assign embeddings to noisy, overlapped, or muffled speech frames.

## 7. Conversational Intelligence Analysis
- **Speaker-Attributed WER (SA-WER)**: 52.69%
- **Concatenated-Poly WER (cpWER)**: 59.74%
- **Word Diarization Error Rate (WDER)**: 19.42%
*Observations*: The true impact of the high DER is visible here. Despite a respectable baseline ASR WER of 31.58%, the poor diarization inflated the final conversational intelligence metrics by over 20 points.

## 8. Delta Analysis vs Previous Phases
| Metric | Phase 4/5 Baseline | Phase 9 Production | Absolute Δ | Relative Δ |
|--------|--------------------|--------------------|------------|------------|
| WER | 29.75% | 31.58% | +1.83% | +6.15% |
| DER | 41.14% | 52.96% | +11.82% | +28.73% |
| SA-WER | 40.08% | 52.69% | +12.61% | +31.46% |
| cpWER | 50.54% | 59.74% | +9.20% | +18.20% |

## 9. Error Analysis
The attempt to combine "best of" components introduced catastrophic pipeline friction. Specifically:
1. **ASR Slicing**: Hard-cutting audio frames via VAD directly severed conversational continuity. 
2. **Oracle VAD MSDD Disconnect**: MSDD operates poorly when fed highly fragmented Oracle VAD from Pyannote. The rapid switching and padded bounds confused the multi-scale embedding extraction, causing heavy speaker confusion and massive drops (Missed Detections: 27.47%). 

## 10. Diagnostic Examples
- **Miss Regions**: Short, transient officer commands masked by wind or movement were heavily missed during the speaker assignment phase.
- **Confusion Regions**: High-stress back-and-forth between two speakers located similarly far from the body-camera microphone led to 17.09% confusion.

## 11. Key Findings
- **Did Phase 9 outperform previous phases?** No. Despite individual components showing promise in isolation during earlier phases, their strict integration (Hardened VAD slicing + Oracle MSDD) fundamentally broke the context window, performing significantly worse than the raw Phase 4/5 Baseline. 
- **Which metric improved the most?** The False Alarm rate on diarization improved, dropping to 8.40%, heavily gated by the Oracle VAD structure.
- **Which metric remains the largest bottleneck?** Speaker Confusion & Missed Diarization Detection (DER 52.96%). The diarizer cannot disentangle speakers in BWC audio.
- **Is the pipeline production-ready?** No. A cpWER of 59.74% indicates nearly 60% of the transcribed conversational timeline is misattributed, missed, or hallucinated. 

## 12. Final Conclusions & Next Steps
Combining isolated winning strategies does not yield a winning composite pipeline. The aggressive "hardened" approach (slicing ASR, muting audio, enforcing high-recall Pyannote limits over NeMo MSDD) fractured the temporal context that these neural models rely on. 

**What should Phase 10 focus on?**
Phase 10 must pivot away from aggressive manual VAD slicing. It should focus entirely on **Joint ASR-Diarization Models** or **End-to-End Speaker Turn Models** (like Pyannote 3.1 combined natively with WhisperX's built-in contextual alignment) rather than attempting to duct-tape Pyannote, NeMo MSDD, and WhisperX into a rigid, non-contextual sequence. Overlap detection and spatial (multi-channel) awareness must be explored if BWC speaker attribution is to be solved.
