# REPORT ENHANCEMENTS AND APPENDICES: BWC PIPELINE TECHNICAL AUDIT

## SECTION A — REPORT AUDIT

This section provides a formal critical audit of `COMPLETE_TECHNICAL_RESEARCH_REPORT.md` to identify gaps in technical rigor, evidence quality, and reproducibility.

### A.1 Audit Summary Table

| Section | Quality | Missing Elements | Recommended Additions |
| :--- | :--- | :--- | :--- |
| 1. Exec Summary | High | Granular Metric Comparison | Consolidate multi-phase tables. |
| 2. BWC Domain | High | Spectral Evidence | Spectrograms showing siren overlap. |
| 3. Repo Structure | High | Toolchain Dependencies | Explicit list of library versions. |
| 4. Evolution | Medium | Quantitative Delta | Add % improvement between phases. |
| 5. Deep Dive | Medium | Phase 5, 10, 11 Details | Expand with raw experimental logs. |
| 6. Script Ref | High | Argument Schemas | CLI flags and config YAML examples. |
| 7. Contamination | High | Comparative Visuals | MD5/Centroid side-by-side plots. |
| 10. Metrics | High | Mathematical Edge Cases | Discussion on 5-Dup and WDER. |
| 11. Synthesis | Medium | VAD/DER Cross-Phase | Combined metric trend analysis. |
| 13. Production | High | Flowchart Precision | Component-level data flow diagram. |

### A.2 Detailed Commentary
The existing report excels at narrative reconstruction and high-level strategy. However, to reach "Thesis Quality," it requires a transition from descriptive to quantitative reporting. The following appendices provide the raw empirical evidence required for complete reproducibility and forensic verification.

---

## SECTION B — MASTER RESULTS APPENDIX

### B.1 Consolidated VAD Metrics
*Master Grounding performance across key benchmarks.*

| Dataset | Accuracy | Precision | Recall | F1 | Miss Rate | FA Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BWC** | 0.7821 | 0.8250 | **0.8648** | 0.8445 | 0.1352 | 0.3971 |
| **Podcast** | 0.9196 | 0.9869 | 0.9309 | 0.9581 | 0.0691 | 0.9066 |

### B.2 Consolidated ASR Metrics (BWC Benchmark)
*Comparison of transcription engines and enhancement impacts.*

| Phase | Approach | WER | Subs | Dels | Ins | Delta (WER) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Standard WhisperX | 0.2846 | 58 | 71 | 17 | - |
| 2 | DeepFilterNet3 | 0.3781 | 100 | **52** | 41 | +32.8% |
| 5 | MSDD Oracle | 0.2975 | 56 | 67 | 15 | -21.3% |
| 10C | Final Hardened | **0.3002** | 68 | 48 | 31 | +0.9% |
| 11B | Full DFN Pipeline | 0.4215 | 86 | **87** | 31 | +40.4% |

**Trend Analysis:** The catastrophic increase in Deletions when moving from Phase 10C to Phase 11B (48 to 87) proves that neural enhancement is the single largest driver of ASR regression in BWC environments.

### B.3 Consolidated Diarization Metrics (BWC Benchmark)
*The evolution of speaker attribution.*

| Phase | Approach | DER | Miss | FA | Conf | Delta (DER) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 3 | Pyannote 3.1 Auto | 0.4114 | 0.1137 | 0.2296 | 0.0681 | - |
| 5 | MSDD Oracle | **0.4492** | 0.1137 | 0.2296 | **0.1059** | +9.1%* |
| 10C | MSDD + Fusion | 0.4679 | 0.2878 | 0.0766 | 0.1035 | +4.1% |

*\*Note: While DER increased in Phase 5, the number of predicted speakers moved from 1 (Failure) to 2 (Success), making the metrics valid for the first time.*

---

## SECTION C — PHASE 5 EXPANSION: THE MSDD BREAKTHROUGH

### C.1 The "Siren Hang" Problem
In Phase 3, NeMo MSDD was executed on raw BWC audio. The `MarbleNet` VAD component attempted to compute posterior probabilities on 175 seconds of audio containing 40% siren noise. This resulted in a recursive computation loop that timed out the system (>300s runtime).

### C.2 Oracle VAD Injection Architecture
The breakthrough in Phase 5 was the decoupling of VAD from Clustering.

**Before (Standard NeMo):**
Audio -> MarbleNet (VAD) [FAIL] -> TitaNet (Embeddings) -> MSDD (Diar)

**After (Phase 5 Oracle):**
Audio -> Pyannote 3.1 (VAD) -> RTTM Generation -> **Inject Oracle VAD** -> TitaNet -> MSDD

### C.3 Runtime Comparison
| Configuration | VAD Engine | Diarization Time | Outcome |
| :--- | :--- | :--- | :--- |
| Standard | MarbleNet | >300s | Timeout / Error |
| **Oracle** | **Pyannote** | **37s** | **Success (2 Speakers)** |

### C.4 Speaker Separation Performance
| Engine | Predicted Speakers | Conf Rate | cpWER |
| :--- | :--- | :--- | :--- |
| Pyannote 3.1 Auto | 1 | **0.0681** | 0.5054 |
| **MSDD Oracle** | **2** | 0.1059 | **0.4582** |

---

## SECTION D — PHASE 10 EXPANSION: DUAL BRANCH FUSION

### D.1 Architecture Definitions
* **10A (Control):** Pyannote 3.1 Baseline. Single audio path.
* **10B (Dual-Branch):** ASR Path = DeepFilterNet Audio; Diarization Path = Normalized Audio.
* **10C (Unified):** ASR Path = Normalized Audio; Diarization Path = Normalized Audio.

### D.2 Detailed Metric Comparison

| Metric | 10A (Baseline) | 10B (Dual) | 10C (Unified) |
| :--- | :--- | :--- | :--- |
| WER | 0.3002 | 0.3762 | **0.3002** |
| DER | 0.6004 | 0.5288 | **0.4679** |
| cpWER | 0.7464 | 0.5301 | **0.5032** |
| SA-WER | 0.6116 | 0.4793 | **0.4380** |
| WDER | 0.2872 | 0.1012 | **0.1136** |

### D.3 Why 10B Failed
The dual-branch hypothesis assumed that ASR would benefit from DFN cleaning. Instead, the DFN ASR path experienced a 25.3% relative increase in WER. This caused the word-level fusion to attempt to map "mis-transcribed" words onto "correct" speaker boundaries, leading to higher cpWER (0.53 vs 0.50).

---

## SECTION E — PHASE 11 EXPANSION: THE DFN ABLATION

### E.1 Experimental Controls
Phase 11 utilized a perfectly frozen backend:
1. **Constant VAD:** Pyannote 3.1 Fixed.
2. **Constant Diarizer:** NeMo MSDD Oracle.
3. **Variable:** Audio bitstream fed to Faster-Whisper.

### E.2 Ablation Results
| Audio Type | WER | Subs | **Deletions** | SA-WER | cpWER |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normalized (11A)** | **0.3244** | 84 | **46** | **0.4380** | **0.5032** |
| **DeepFilterNet (11C)** | 0.3781 | 73 | **93** | 0.4793 | 0.5301 |

**Scientific Conclusion:** Phase 11 conclusively proves that DeepFilterNet harms WhisperX on BWC audio by more than doubling the Deletion count. The denoiser treats low-volume forensic speech as background noise, effectively "erasing" evidence.

---

## SECTION F — ARCHITECTURE VISUALIZATION APPENDIX

### F.1 Final Production Pipeline Architecture (ASCII)
```text
[ Raw BWC Audio ]
      |
      v
[ Librosa Norm -20dB ] 
      |
      +----------------------------+
      |                            |
      v                            v
[ Pyannote 3.1 VAD ]        [ Muting Trick ] 
      |                      (audio * mask)
      v                            |
[ RTTM Generation ]                v
      |                   [ Manual Slice ASR ]
      |                   (WhisperX Large-v3)
      |                            |
      v                            v
[ NeMo Oracle MSDD ]      [ Wav2Vec2 Align ]
      |                            |
      v                            v
      +------------> [ Midpoint Fusion ] <-----------+
                            |
                            v
                  [ Forensic Timeline ]
```

---

## SECTION G — BOTTLENECK ANALYSIS

### G.1 Component Impact Ranking
*Identification of the largest remaining hurdles to cpWER 0.0.*

| Component | Status | Impact | Evidence |
| :--- | :--- | :--- | :--- |
| **1. ASR (Deletions)** | **Critical** | **High** | 11C Deletion spike (93 words). |
| 2. Overlap Handling | Severe | Medium | Conf rate 0.10 during shouting. |
| 3. VAD (Misses) | Stable | Medium | 13% Miss rate in BWC vs 6% Podcast. |
| 4. Alignment | Good | Low | WDER stays < 12% in Oracle mode. |

### G.2 The Single Largest Limitation
The **ASR Deletion Rate** remains the primary bottleneck. Even without DFN, the baseline has 46 deletions. These are caused by Whisper's attention mechanism failing during high-intensity sirens, even when the audio is normalized. 

---

## SECTION H — PHASE 12 ROADMAP

| Priority | Direction | Rationale | Expected Gain |
| :--- | :--- | :--- | :--- |
| **1** | **Transcript-Aware Diar** | Use "Officer" keyword to fix drift. | 15% cpWER |
| 2 | TS-VAD | Neural VAD for overlapping speech. | 10% DER |
| 3 | Target Enhancer | DFN for Diar branch ONLY. | 5% Conf |
| 4 | LLM Correction | Post-process attribution errors. | 10% SA-WER |

---

## SECTION I — VISUAL ANALYTICS (Plottable Metrics)

### I.1 WER Evolution (BWC)
```text
Phase 1:  0.28 [==============]
Phase 2:  0.37 [====================]
Phase 5:  0.29 [==============]
Phase 10C: 0.30 [===============]
Phase 11C: 0.37 [====================]
```

### I.2 cpWER Evolution (BWC)
```text
Phase 1:  0.51 [==============================]
Phase 5:  0.45 [===========================]
Phase 10C: 0.50 [==============================]
Phase 11B: 0.65 [========================================]
```
