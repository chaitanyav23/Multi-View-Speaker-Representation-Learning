import sys

with open("COMPLETE_TECHNICAL_RESEARCH_REPORT.md", "a") as f:
    f.write("""
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
""")
