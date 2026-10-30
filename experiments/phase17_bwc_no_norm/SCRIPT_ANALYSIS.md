# Script Analysis: Phase 17

## Orchestration
### `run_phase17.py`
- **Role:** Main entry point and workflow manager.
- **Functionality:** 
    - Defines the experimental directory structure.
    - Executes the FFmpeg audio repair command.
    - Sequences the execution of all 13 sub-steps.
    - Handles command-line arguments for input audio and overwrite flags.

## Implementation Steps

### `step2_vad_grounding.py`
- **Implementation:** Wraps Pyannote 3.1 VAD.
- **Key Logic:** Downloads the segmentation model from HuggingFace, performs inference, and applies post-processing (merge_gap=0.15s, min_duration=0.1s).
- **Critical Change:** Processes the `bwc_repaired.wav` directly, without LoudNorm.

### `step3_oracle_rttm.py`
- **Implementation:** Utility to convert VAD segments into a dummy RTTM format.
- **Role:** Provides a "voice vs non-voice" RTTM for intermediate validation steps.

### `step4_whisperx.py`
- **Implementation:** WhisperX integration.
- **Key Logic:** Loads the `large-v2` model, performs transcription, and uses the VAD segments from Step 2 to filter/align results.

### `step5_msdd.py`
- **Implementation:** NeMo MSDD Diarization.
- **Key Logic:** Configures the `diar_msdd_nlp` model. Uses the VAD grounding as input to bypass internal VAD and focus purely on speaker attribution.

### `step6_fusion.py`
- **Implementation:** Custom speaker-to-word attribution logic.
- **Key Logic:** Uses a temporal overlap algorithm to assign MSDD speaker IDs to WhisperX word objects. Implements a 0.5s fallback tolerance for non-overlapping words.

### `step7_process_gt.py`
- **Implementation:** Ground Truth parser.
- **Role:** Normalizes the human-annotated transcripts and RTTMs for comparison with pipeline output.

## Evaluation & Metrics

### `step8_vad_eval.py`
- **Metric Logic:** Compares VAD grounding against GT using `pyannote.metrics`.
- **Calculates:** Precision, Recall, F1, Miss, and False Alarm.

### `step9_asr_eval.py`
- **Metric Logic:** Uses `jiwer` to calculate Word Error Rate (WER).

### `step10_diar_eval.py`
- **Metric Logic:** Standard DER calculation using `pyannote.metrics.diarization`.

### `step11_attribution_eval.py`
- **Metric Logic:** Calculates Speaker-Attributed WER (SA-WER).

### `step12_13_conversational_eval.py`
- **Metric Logic:** Complex metrics including WDER and Concatenated Permutation WER (cpWER).

### `step14_comparison.py`
- **Implementation:** Result synthesis.
- **Role:** Compares Phase 17 metrics against the Phase 11A baseline and generates `comparisons/bwc_no_norm_vs_baseline.csv`.
