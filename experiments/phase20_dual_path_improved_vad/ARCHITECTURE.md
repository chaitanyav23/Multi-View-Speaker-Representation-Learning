# Phase 20 Dual-Path Improved VAD Architecture

This experiment implements a self-contained dual-path speech processing framework for `raw_audio/bwc.wav`.
The design is derived from Phase 19: diarization-oriented audio conditioning and transcription-oriented audio conditioning remain separated, then both paths are fused at word level using maximum temporal overlap.
The only architectural change is the VAD stage, which now uses probability-guided recursive min-cut splitting for long speech regions.

## Pipeline Framework

```text
================================================================================
                    PHASE 20 DUAL-PATH IMPROVED VAD PIPELINE
================================================================================

                              raw_audio/bwc.wav
                                      |
                                      v
                    +-------------------------------------+
                    | Step 1: Preprocess                  |
                    | - mono                              |
                    | - 16 kHz                            |
                    | - waveform preserving               |
                    | - no denoising                      |
                    |                                     |
                    | Output: audio/bwc_repaired.wav      |
                    +------------------+------------------+
                                       |
                    +------------------+------------------+
                    |                                     |
                    v                                     v
     +--------------------------------+     +------------------------------------+
     | PATH A: Diarization           |     | PATH B: Transcription              |
     |                                |     |                                    |
     | Step 2A: DeepFilterNet         |     | Step 2B: Loudness Normalization    |
     | Output: audio/bwc_dfn.wav      |     | Target: -20 LUFS                   |
     |                                |     | Output: audio/bwc_normalized.wav   |
     +---------------+----------------+     +----------------+-------------------+
                     |                                       |
                     v                                       v
     +--------------------------------+     +------------------------------------+
     | Step 3A: Improved Pyannote VAD |     | Step 3B: Improved Pyannote VAD     |
     | Input: bwc_dfn.wav             |     | Input: bwc_normalized.wav          |
     | Output: grounding/pathA_vad    |     | Output: grounding/pathB_vad        |
     +---------------+----------------+     +----------------+-------------------+
                     |                                       |
                     v                                       v
     +--------------------------------+     +------------------------------------+
     | Step 4A: VAD Evaluation        |     | Step 4B: VAD Evaluation            |
     | Metrics: P/R/F1/FA/Miss        |     | Metrics: P/R/F1/FA/Miss            |
     | Output: pathA_vad_metrics      |     | Output: pathB_vad_metrics          |
     +---------------+----------------+     +----------------+-------------------+
                     |                                       |
                     v                                       v
     +--------------------------------+     +------------------------------------+
     | Step 5A: NeMo MSDD             |     | Step 5B: WhisperX Large-v3         |
     | Oracle VAD: pathA_vad          |     | VAD guide: pathB_vad               |
     | Embeddings: TitaNet Large      |     | Forced alignment enabled           |
     | Output: pathA_msdd.rttm        |     | Output: aligned_words.json         |
     +---------------+----------------+     +----------------+-------------------+
                     |                                       |
                     v                                       v
     +--------------------------------+     +------------------------------------+
     | Step 6A: Diarization Eval      |     | Step 6B: ASR Eval                  |
     | Metrics: DER/FA/Miss/Conf/JER  |     | Metrics: WER/S/I/D/word counts     |
     | Output: diarization_metrics    |     | Output: asr_metrics                |
     +---------------+----------------+     +----------------+-------------------+
                     |                                       |
                     +-------------------+-------------------+
                                         |
                                         v
                    +---------------------------------------+
                    | Step 7: Maximum-Overlap Fusion        |
                    |                                       |
                    | Inputs:                               |
                    | - diarization/pathA_msdd.rttm         |
                    | - transcription/aligned_words.json    |
                    |                                       |
                    | For each word:                        |
                    | - find diarization segment overlap    |
                    | - assign speaker with max overlap     |
                    | - preserve start/end/confidence       |
                    |                                       |
                    | Outputs:                              |
                    | - fusion/attributed_words.json        |
                    | - fusion/transcript.txt               |
                    +-------------------+-------------------+
                                        |
                                        v
                    +---------------------------------------+
                    | Step 8: Fusion Evaluation             |
                    | Metrics:                              |
                    | - cpWER                               |
                    | - SA-WER                              |
                    | - WDER                                |
                    | - speaker attribution accuracy        |
                    | - word attribution error              |
                    | - turn accuracy                       |
                    | - timeline accuracy                   |
                    |                                       |
                    | Outputs:                              |
                    | - metrics/fusion_metrics.json         |
                    | - reports/fusion_report.md            |
                    +-------------------+-------------------+
                                        |
                                        v
                    +---------------------------------------+
                    | Step 9: Aggregation + Final Report    |
                    |                                       |
                    | Outputs:                              |
                    | - metrics/all_metrics.json            |
                    | - reports/final_report.md             |
                    +---------------------------------------+

================================================================================
```

## Stage Contract

Every stage follows the same framework contract:

```text
+----------------+-------------------------------------------------------------+
| Contract Field | Description                                                 |
+----------------+-------------------------------------------------------------+
| inputs         | Explicit input paths passed by argparse or defaults          |
| outputs        | Concrete intermediate artifact paths written by the stage     |
| evaluation     | Stage-specific metrics when the stage is evaluative           |
| logging        | Per-stage log file under logs/ plus console output             |
+----------------+-------------------------------------------------------------+
```

## Artifact Flow

```text
audio/
  bwc_repaired.wav
  bwc_dfn.wav
  bwc_normalized.wav

grounding/
  pathA_vad.json
  pathA_oracle.rttm
  pathB_vad.json

diarization/
  pathA_msdd.rttm

transcription/
  aligned_words.json

fusion/
  attributed_words.json
  transcript.txt

metrics/
  pathA_vad_metrics.json
  pathB_vad_metrics.json
  asr_metrics.json
  diarization_metrics.json
  fusion_metrics.json
  all_metrics.json

reports/
  pathA_vad_report.md
  pathB_vad_report.md
  asr_report.md
  diarization_report.md
  fusion_report.md
  final_report.md
```

## Execution Entry Points

The full framework is executed by:

```bash
python experiments/phase19_dual_path_evaluation/run_phase19.py
```

Each stage is independently executable:

```bash
python experiments/phase19_dual_path_evaluation/scripts/step01_preprocess.py --help
python experiments/phase19_dual_path_evaluation/scripts/step03_vad.py --help
python experiments/phase19_dual_path_evaluation/scripts/step07_fusion.py --help
```

## Improved VAD Logic

```text
Pyannote neural VAD
  -> retain frame-level speech probabilities
  -> extract raw speech timeline
  -> merge gaps < 150 ms
  -> remove regions < 100 ms
  -> recursively split segments > 30 s
       search only 15-30 s from segment start
       choose local minimum with lowest speech probability
       fallback to minimum RMS only if probabilities are unavailable
  -> final merge only when gap < 150 ms and merged duration <= 28 s
```

## Design Rationale

Path A favors diarization robustness by applying DeepFilterNet before VAD and MSDD speaker discovery.
Path B favors lexical transcription by applying loudness normalization before VAD-guided WhisperX transcription and forced alignment.
Fusion is intentionally late and word-level, so diarization and transcription can be evaluated independently before speaker attribution is applied.
