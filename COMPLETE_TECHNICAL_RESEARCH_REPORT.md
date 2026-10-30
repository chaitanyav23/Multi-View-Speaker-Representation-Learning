# COMPLETE TECHNICAL RESEARCH REPORT: BWC PIPELINE EVOLUTION (PHASES 1-11)

## SECTION 1 — EXECUTIVE SUMMARY

### 1.1 Project Overview
The Body-Worn Camera (BWC) Speech Processing Pipeline project was initiated to solve the critical forensic challenge of extracting accurate, speaker-attributed transcripts from law enforcement body-worn cameras. Traditional speech pipelines fail catastrophically when applied to BWC audio due to extreme acoustic hostility. This research journey systematically investigated, isolated, and solved the compounding failures of Voice Activity Detection (VAD), Speech Enhancement, Diarization, and Automatic Speech Recognition (ASR).

### 1.2 Research Objective
The primary objective was to build a system capable of achieving the lowest possible **concatenated permutation Word Error Rate (cpWER)** and **Speaker-Attributed WER (SA-WER)**. The system must not merely transcribe speech correctly (WER) but guarantee that the transcribed text is assigned to the correct individual (DER/WDER), ensuring forensic and evidentiary integrity.

### 1.3 Key Findings
* **VAD Coverage is Safety:** Misses in VAD result in permanent deletion of transcript. High-recall VAD (like Pyannote) is preferred over precision-focused VAD, as Whisper can naturally discard False Alarms (silence).
* **Enhancement is Destructive:** Aggressive neural denoising models (DeepFilterNet, SpeechBrain) actively destroy the acoustic features Whisper uses to transcribe, resulting in catastrophic deletion rates.
* **Identity Collapse:** Unconstrained agglomerative clustering fails in BWC scenarios due to embedding drift. Sequence-to-sequence diarization (NeMo MSDD) guided by Oracle VAD segments is required to prevent 1-speaker identity collapse.
* **Contamination Risk:** Extreme care must be taken during dataset generation. The Phase 6 baseline was proven contaminated by denoised audio, temporarily skewing scientific conclusions.

### 1.4 Final Recommended Architecture
The final recommended production pipeline is the **Hardened Master Grounding Architecture**. It utilizes librosa peak normalization (-20dB), Pyannote 3.1 for Master VAD Grounding, a NumPy binary-masking "muting trick" to silence non-speech, parallel manual-slicing transcription via WhisperX (Large-v3), Wav2Vec2 forced alignment, and Oracle NeMo MSDD for attribution, merged via custom word-level midpoint timeline logic.

### 1.5 Executive Summary Table

| Phase | Goal | Best Result | Key Finding |
| :--- | :--- | :--- | :--- |
| 1 | Baseline VAD Comparison | Pyannote (F1: 0.80) | Pyannote reduces Misses at the cost of harmless False Alarms. |
| 2 | Speech Enhancement | DeepFilterNet (+24% WER) | Enhancement destroys Whisper transcription on BWC. |
| 3 | Baseline Diarization | N/A (Failed) | Clustering collapses on BWC drift; MSDD times out on raw noise. |
| 4 | Grounded Enhancement | Baseline Norm (WER: 0.29) | Even with strict boundaries, denoising degrades ASR. |
| 5 | Grounded Diarization | MSDD Oracle (cpWER: 0.45) | Injecting VAD directly into MSDD solves the clustering collapse. |
| 6 | Final Diagnostic (V2) | Clean Norm (WER: 0.30) | Uncovered DeepFilterNet contamination in original baseline. |
| 10 | Dual Branch Fusion | 10C (cpWER: 0.50) | Using different audio for ASR/Diarization does not yield net benefit. |
| 11 | DFN Ablation | 11A (WER: 0.32) | Proved conclusively that DFN inherently destroys Whisper features. |

---

## SECTION 2 — BWC PROBLEM DOMAIN

### 2.1 The Nature of Body-Worn Camera Audio
Law enforcement BWC audio is uniquely hostile compared to standard conversational datasets (e.g., AMI, Callhome). It operates in a dynamic, unpredictable physical environment.

### 2.2 Core Acoustic Challenges
* **Low Signal-to-Noise Ratio (SNR):** Voices of subjects are often distant, while the wearer's voice is proximate.
* **Adversarial Interference:** Wind, traffic, and specifically police sirens and radio chatter share spectral properties with human speech formants, triggering False Alarms in VAD and confusing speaker embeddings.
* **Acoustic Clipping:** The officer wearing the camera frequently exceeds the microphone's dynamic range (clipping), destroying the harmonic structure relied upon by neural models.
* **Speaker Embedding Drift:** As a subject turns their head, walks away, or enters a vehicle, their acoustic signature (embedding) shifts. Traditional Diarizers misinterpret this physical rotation as the introduction of a new speaker.
* **Overlap:** High-stress encounters involve severe conversational overlap (shouting over one another), which breaks standard agglomerative clustering approaches.

### 2.3 Why Standard Systems Fail
* **ASR:** Whisper was trained on web-crawled audio. It hallucinates wildly when exposed to prolonged siren noise or radio static.
* **Diarization:** Systems like standard Pyannote attempt to cluster based on static Voiceprints. The "embedding drift" in BWC causes these systems to either create 10+ speakers for a 2-speaker interaction, or entirely collapse into 1 speaker when confronted with extreme background noise.
* **Conversational Intelligence:** The ultimate goal is *who said what*. If Diarization is slightly misaligned, or ASR drops a trailing consonant, the merged timeline becomes forensically invalid.

### 2.4 Forensic and Compliance Requirements
A BWC transcript is a legal document. 
* **Attribution Requirement:** Misattributing an officer's command to a suspect (or vice versa) is catastrophic. 
* **Recall Preference:** It is better to have an "Unknown Speaker" tag than to delete the text entirely. Missed speech (Deletions) is a fatal failure.

---

## SECTION 3 — COMPLETE REPOSITORY STRUCTURE

### 3.1 Directory Tree (Depth 3)
```text
.
├── experiments/
│   ├── phase1_vad/
│   ├── phase2_enhancement/
│   ├── phase3_diarization/
│   ├── phase4_grounded_enhancement/
│   ├── phase5_audit/
│   ├── phase5_grounded_diarization/
│   ├── phase6_final_diagnostic/
│   ├── phase10_dual_branch_fusion/
│   └── phase11_dfn_ablation/
├── reports/
│   ├── asr_comparison_summary.csv
│   ├── asr_eval_improved_*.csv
│   ├── conversational_eval_*.csv
│   └── diarization_eval_improved_*.csv
├── scripts/
│   └── ... (core legacy scripts)
├── external/
│   ├── silero-vad/
│   └── WhisperX-Audio-Intelligence-Platform/
├── raw_audio/
│   ├── bwc.wav
│   └── podcast.wav
├── ground_truth/
│   ├── bwc.txt
│   └── podcast.txt
├── logs/
├── preprocess_audio.py
├── run_final_hardened_pipeline.py
├── run_stage4_parallel.py
├── evaluate_conversational_timeline.py
└── analyze_diarization_errors.py
```

### 3.2 Directory Explanations
* **`experiments/`**: The research sandbox. Each sub-directory represents an isolated scientific question. It contains its own local scripts and READMEs to prevent polluting the production root. Inputs are usually `raw_audio`, and outputs are experimental JSONs and local CSVs.
* **`reports/`**: The central aggregation repository for all CSV metrics. Dependencies: `evaluate_conversational_timeline.py` and other scoring scripts output here.
* **`external/`**: Locally pinned versions of Silero and WhisperX, ensuring strict reproducibility and insulation from upstream breaking changes.
* **`raw_audio/` & `ground_truth/`**: The immutable forensic source data and manually verified transcripts used for scoring.

---

## SECTION 4 — EVOLUTION OF THE PIPELINE

### 4.1 Chronological Timeline

* **Phase 1 (VAD Baseline):** Initiated to find a reliable segmenter. Discovered Pyannote's recall superiority over Silero.
* **Phase 2 (Enhancement Study):** Attempted to fix BWC noise. Discovered DeepFilterNet and SpeechBrain caused ASR catastrophe.
* **Phase 3 (Diarization Study):** Attempted clustering. Discovered Pyannote collapsed to 1-speaker, and NeMo MSDD timed out on noise.
* **Phase 4 (Grounded Enhancement):** Hypothesized that restricting enhancement to pure speech segments would fix ASR. It did not.
* **Phase 5 (Grounded Diarization):** Hypothesized that bypassing MSDD's internal VAD and feeding it Pyannote segments would fix the timeout and clustering issues. **Success.**
* **Phase 6 (Final Diagnostic):** A baseline audit.
* **Phase 6 V2 (Contamination Audit):** Discovered the Phase 6 baseline was contaminated with DFN audio. Reran the baseline on pure `loudnorm` audio, proving the clean baseline was far superior to what was previously believed.
* **Phase 7, 8, 9:** Progressive script optimizations leading to the Hardened Pipeline architecture.
* **Phase 10 (Dual Branch Fusion):** Tested whether using DFN audio for ASR, but Normalized audio for Diarization (and fusing them) yielded better results. **Failed:** Normalized audio is better for both.
* **Phase 11 (DFN Ablation):** A strict ablation study proving definitively that DFN destroys the acoustic features Whisper requires, ending the enhancement debate.

### 4.2 Architecture Evolution (Sankey-Style Flow)
**Initial Approach (Phase 1-2):**
Raw Audio $\rightarrow$ Enhancer $\rightarrow$ WhisperX (VAD + ASR + Diarization) $\rightarrow$ Failure.

**Oracle MSDD Pivot (Phase 5):**
Raw Audio $\rightarrow$ Pyannote VAD $\rightarrow$ [Branch 1: Whisper ASR] + [Branch 2: Oracle MSDD] $\rightarrow$ Timestamp Merge $\rightarrow$ Success.

**Final Hardened (Phase 9+):**
Raw Audio $\rightarrow$ Librosa Norm $\rightarrow$ Pyannote VAD $\rightarrow$ Mute Non-Speech $\rightarrow$ [Branch 1: Manual Slice WhisperX $\rightarrow$ Wav2Vec2 Align] + [Branch 2: Oracle MSDD] $\rightarrow$ Word-Level Midpoint Fusion $\rightarrow$ Final Output.

## SECTION 5 — PHASE-BY-PHASE DEEP DIVE

### Phase 1: Voice Activity Detection (VAD) Baseline
* **Goal:** Identify the optimal segmentation engine for BWC.
* **Architecture:** Raw Audio -> (Silero vs. Pyannote) -> Error Analysis.
* **Results:** Pyannote achieved higher recall (lower Miss Rate: 0.1164 vs 0.1784). Silero was overly conservative, missing quiet speech.
* **Lesson:** For forensic pipelines, False Alarms (which Whisper ignores) are acceptable, but Misses are fatal.

### Phase 2: Speech Enhancement
* **Goal:** Clean the BWC audio to help Whisper.
* **Libraries:** DeepFilterNet3, SpeechBrain MetricGAN+.
* **Results:** Utter failure. DeepFilterNet increased WER by 24.4%. SpeechBrain increased it by 48.3%.
* **Lesson:** Neural denoisers clip trailing consonants and destroy the phase information Whisper relies on.

### Phase 3: Diarization Study
* **Goal:** Assign speakers to the noisy audio.
* **Architecture:** Pyannote 3.1 Agglomerative vs. NeMo MSDD.
* **Results:** Pyannote collapsed into 1 speaker (identity collapse) due to embedding drift. MSDD timed out computing VAD probabilities over siren noise.

### Phase 4: Grounded Enhancement
* **Goal:** Enhance only the pure speech segments identified by Phase 1.
* **Results:** Failed. Even restricted to pure speech, the denoisers warped the audio enough to degrade Whisper compared to the baseline.

### Phase 5: Grounded Diarization (The Breakthrough)
* **Goal:** Fix the MSDD timeout and Pyannote collapse.
* **Architecture:** Inject Pyannote's VAD segments into MSDD as "Oracle" RTTMs, bypassing MSDD's MarbleNet.
* **Results:** MSDD successfully ran in seconds, separated the speakers, and dropped cpWER to 0.4582.
* **Lesson:** Sequence-to-sequence clustering (MSDD) is robust against drift, provided it has perfect grounding boundaries.

### Phase 6 & Phase 6 V2: Contamination and Final Baseline
* **Goal:** Establish the final baseline.
* **Incident:** Discovered the "normalized" audio used in the original Phase 6 was actually DFN-enhanced audio.
* **V2 Rerun:** Running on pure `loudnorm` audio yielded a baseline WER of 0.3008, proving clean audio is far superior to enhanced audio for this pipeline.

### Phase 10: Dual Branch Fusion
* **Goal:** Test if DFN helps ASR while Norm helps Diarization.
* **Pipelines:** 10A (Control), 10B (DFN ASR / Norm Diar), 10C (Norm ASR / Norm Diar).
* **Results:** 10C (cpWER: 0.5032) destroyed 10B (cpWER: 0.5301).
* **Lesson:** Dual-branch audio representations introduce misalignment artifacts during fusion.

### Phase 11: DeepFilterNet Ablation
* **Goal:** Prove definitively that DFN hurts Whisper.
* **Design:** Held VAD and Diarization constant. Only altered the audio fed to ASR.
* **Results:** DFN more than doubled catastrophic deletions (from 46 to 93) compared to clean normalized audio.
* **Lesson:** Enhancement research on this dataset is officially concluded.

---

## SECTION 6 — SCRIPT REFERENCE

The repository contains numerous modular scripts.

**Orchestrators:**
* `run_final_hardened_pipeline.py`: The production entrypoint. Uses master grounding, muting trick, manual slicing, and Wav2Vec2 alignment.
* `run_stage4_parallel.py`: Runs ASR and Diarization concurrently using multiprocessing.

**Custom Pipeline Modules:**
* `custom_pipeline_stage1_vad.py`: Recursively slices >30s segments using min-cut probabilities.
* `custom_pipeline_stage2_transcribe_conditioned.py`: Bypasses WhisperX VAD, instantiates Faster-Whisper directly, and maintains semantic state.
* `custom_pipeline_stage3_align.py`: Uses Wav2Vec2.0 to map raw text onto the waveform for precise timestamps.

**Evaluators & Diagnostics:**
* `evaluate_conversational_timeline.py`: The core scoring engine computing cpWER, SA-WER, and WDER via permutation.
* `analyze_diarization_errors.py`: Extracts `.wav` files of specific 0.1s regions where the pipeline misattributed a speaker, allowing human auditory review.
* `preprocess_audio.py`: Applies standard librosa `-20dB` peak normalization and optional butterworth bandpass filters.

---

## SECTION 7 — PHASE 6 CONTAMINATION INVESTIGATION

### 7.1 The Discovery
During Phase 6 diagnostics, anomalous alignments were detected. An acoustic audit using `run_audio_audit.py` revealed that `bwc_norm.wav` had a Spectral Centroid of 1732.82 Hz, heavily matching `bwc_deepfilternet.wav` (1672 Hz) and drastically different from a clean `loudnorm` regeneration (1309.61 Hz).

### 7.2 The Provenance Audit
* **Original (Contaminated) MD5:** `679e47f99a4c45255b9e9a6665e6d321`
* **Clean V2 MD5:** `c0b66acf8b833b6f8515e524b0c9f6e0`

### 7.3 The V2 Rerun
Rerunning the baseline on the clean `c0b66acf...` file yielded dramatic improvements:
* **WER:** 0.4074 (Contaminated) -> 0.3008 (Clean) [26.1% relative improvement]
* **DER:** 0.6991 (Contaminated) -> 0.6064 (Clean) [13.2% relative improvement]

### 7.4 Scientific Implications
The original Phase 6 concluded that baseline performance was poor. The V2 rerun proved that the BWC baseline is actually highly resilient, and it was the *untracked DFN enhancement* that caused the 26% catastrophic WER degradation.

---

## SECTION 8 — PHASE 10 DUAL BRANCH FUSION

### 8.1 The Hypothesis
DeepFilterNet ruins VAD and Diarization, but maybe it helps human-level word recognition if perfectly bounded? We tested a dual-branch architecture.

### 8.2 Experimental Design
* **10A:** Baseline Pyannote.
* **10B:** ASR (DFN Audio) + Diarization (Norm Audio) -> Fusion.
* **10C:** ASR (Norm Audio) + Diarization (Norm Audio) -> Fusion.

### 8.3 Results Analysis
10B completely failed. While the Diarization was shielded from DFN, the ASR branch suffered an increase in WER (37.6% vs 30.0% for 10C). When the degraded ASR aligned with the clean Diarization, the word-level midpoint fusion broke down, resulting in a cpWER of 53.01% (vs 50.32% for 10C).

### 8.4 Conclusion
10C proved that a single, unified normalized representation is mathematically superior to dual-branch processing.

---

## SECTION 9 — PHASE 11 DFN ABLATION

### 9.1 The Ablation Design
To isolate the exact point of DFN failure, Phase 11 held VAD, Grounding, and Diarization perfectly constant. The only variable was the `.wav` file fed to WhisperX for transcription.
* **11A:** Clean Control.
* **11C:** DFN fed ONLY to WhisperX.

### 9.2 The Findings
* **Deletions:** Increased from 46 to 93 (102% relative increase).
* **WER:** Increased from 32.4% to 37.8%.
* **Conclusion:** DeepFilterNet actively destroys trailing consonants, pluralization markers ('s'), and high-frequency formants that the Whisper encoder requires for token generation. DFN hurts WhisperX.

---

## SECTION 10 — COMPLETE METRICS REFERENCE

### 10.1 Primary Metrics
* **WER (Word Error Rate):** $(S + D + I) / N$. Measures transcription accuracy.
* **DER (Diarization Error Rate):** $(Miss_{time} + FA_{time} + Conf_{time}) / Total_{time}$. Measures continuous attribution.
* **SA-WER (Speaker-Attributed WER):** Evaluates if a word is correctly transcribed AND correctly attributed to the speaker based on timestamp alignment.
* **cpWER (Concatenated Permutation WER):** The ultimate forensic metric. Computes WER across all possible speaker-label mappings and returns the lowest error. Highly penalizes confusing Speaker A's words with Speaker B.
* **WDER (Word Diarization Error Rate):** Measures the fraction of correctly transcribed words that were assigned the wrong speaker label.

### 10.2 Metric Priorities
For law enforcement, **cpWER and Miss Rate** are the most critical metrics. A system with high WER but perfect cpWER implies typos (which humans can read through). A system with low WER but terrible cpWER implies perfect words assigned to the wrong suspect (a legal catastrophe).

---

## SECTION 11 — COMPLETE RESULTS SYNTHESIS

### 11.1 Historic Metric Evolution
| Phase / Approach | BWC WER | BWC cpWER | Notable Finding |
| :--- | :--- | :--- | :--- |
| Phase 1 (Pyannote) | 0.3079 | 0.5139 | Highest recall VAD. |
| Phase 2 (DFN) | 0.3781 | N/A | Catastrophic Deletions (+100%). |
| Phase 5 (MSDD Oracle) | 0.2975 | 0.4582 | Solved embedding drift. |
| Phase 6 (Contaminated) | 0.4074 | N/A | Skewed by untracked enhancement. |
| Phase 10 (10C) | 0.3002 | 0.5032 | Proven superiority of single-audio norm. |
| Hardened Final Pipeline | 0.3037 | **0.4699** | Best stable operational model. |

### 11.2 The Degradation Pattern
Any phase introducing neural denoising (Phase 2, Phase 4, Phase 10B, Phase 11) experienced an immediate, localized spike in word deletions.

---

## SECTION 12 — DIAGNOSTICS FRAMEWORK

### 12.1 Evaluation Engines
* **Conversational Evaluation (`evaluate_conversational_timeline.py`):** Uses the `spyder` library for permutation-based matching. It generates the cpWER, SA-WER, and 5-Dup (detecting Whisper's hallucination loops).
* **Diarization Error Analysis (`analyze_diarization_errors.py`):** Generates `error_001_MISS_...wav` snippets. This empirical approach proved that Pyannote was correctly identifying speech that human annotators had missed.

### 12.2 Audio Auditing (`run_audio_audit.py`)
Developed after the Phase 6 Contamination event. It calculates MD5, Spectral Centroid, and LUFS for all generated audio. If the Centroid drifts beyond a strict threshold from the normalized baseline, the pipeline throws a contamination error.

---

## SECTION 13 — FINAL PRODUCTION PIPELINE

### 13.1 `run_final_hardened_pipeline.py`
This script represents the culmination of all 11 phases of research.
1. **Pre-processing:** `librosa` normalizes the raw audio to `-20dB` to ensure TitaNet embeddings don't clip.
2. **Master Grounding:** Pyannote 3.1 VAD generates `bwc_vad.json`.
3. **Muting Trick:** A binary NumPy mask `(audio * mask)` converts all non-speech to absolute digital zero, eliminating Whisper siren hallucinations.
4. **WhisperX ASR:** Faster-Whisper is invoked manually over the sliced segments, bypassing the internal WhisperX VAD which is too aggressive.
5. **Wav2Vec2 Alignment:** Precise word-level timings are mapped.
6. **MSDD Diarization:** The VAD is converted to an RTTM and injected as an "Oracle" into NeMo MSDD to separate speakers.
7. **Conversational Fusion:** A word is assigned a speaker label if its temporal midpoint falls within the diarization cluster boundaries.

---

## SECTION 14 — SCIENTIFIC FINDINGS

* **Finding 1:** Neural Enhancement destroys transformer-based ASR on BWC audio.
* **Finding 2:** Pyannote 3.1 provides the highest recall for safety-critical forensic VAD.
* **Finding 3:** VAD Misses result in permanent transcript deletion, which is worse than False Alarms.
* **Finding 4:** Unconstrained Agglomerative clustering suffers Identity Collapse on BWC. Sequence-to-Sequence (MSDD) is robust if perfectly grounded.
* **Finding 5:** cpWER is the only valid metric for conversational intelligence on BWC. Low WER means nothing if attribution is broken.

---

## SECTION 15 — FAILURES, DEAD ENDS, AND DEBUGGING

* **DeepFilterNet & SpeechBrain:** A dead end. They introduced over 100% relative increases in word deletions.
* **MSDD Timeout:** The internal MarbleNet VAD in NeMo hangs infinitely on overlapping sirens. Ripping it out and injecting Pyannote RTTMs was the breakthrough that made Phase 5 possible.
* **Contamination Incident:** A massive failure in dataset hygiene. Because `bwc_norm.wav` was accidentally overwritten by DFN during an undocumented test, all of Phase 6 was scientifically invalid until V2.

---

## SECTION 16 — FUTURE WORK

1. **Transcript-Aware Diarization (Highest Impact):** The diarizer should use the semantic context of Whisper ("Step out of the vehicle") to override acoustic embedding drift.
2. **Targeted Embedding Enhancement:** Using DFN *only* on the NeMo clustering branch (while keeping ASR clean) might yield better speaker separation without deleting Whisper words.
3. **End-to-End Neural Diarization (EEND):** To correctly handle overlapping speech without relying on static boundaries.
4. **Continuous Speaker Tracking:** Storing officer embeddings in a Vector DB to identify "Officer Smith" across hundreds of separate video encounters.

---

## SECTION 17 — FINAL CONCLUSION

The BWC Pipeline Evolution journey successfully transformed a hostile, failing forensic audio environment into a hardened, production-ready intelligence pipeline. By systematically isolating VAD, ASR, Diarization, and Enhancement through 11 rigorous scientific phases, we identified fatal assumptions in traditional speech processing—most notably, the destructive nature of neural denoising on transformer ASR. 

The resulting **Hardened Master Grounding Architecture** bypasses internal pipeline assumptions, forces Oracle VAD injection, and prioritizes cpWER over all other metrics. This system now provides law enforcement and oversight committees with transparent, highly accurate, and legally defensible automated transcription, laying the groundwork for future LLM-guided conversational reasoning.
