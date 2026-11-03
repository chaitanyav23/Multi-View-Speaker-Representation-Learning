# Stage 5B.0 — Repository Reconnaissance & Forensic Audit: Codebase, Assets & Artifacts

**Date**: 2026-11-03  
**Stage**: Stage 5B.0  
**Status**: COMPLETE & VERIFIED  
**Working Directory**: `research_stages/stage5b/`  
**Git Baseline Commit**: `c09dd6751f2b77a81672fb7cb0fb1c4b41a7ecb5`  
**Frozen Baseline Commit**: `ddf089de986f9e7cf026aa507324f3fccaf0cd8f`  

---

## 1. Executive Summary & Audit Scope

This document provides a comprehensive forensic inventory of all software pipelines, pretrained neural network model checkpoints, local audio datasets, ground truth annotations, feature caches, configuration files, and verification test suites across the repository.

The repository investigates multi-view speaker representation learning on noisy, real-world body-worn-camera (BWC) and multi-speaker audio recordings using two complementary frozen pretrained encoders:
1. **NVIDIA TitaNet-Large** (192-dimensional embeddings)
2. **WeSpeaker ResNet293-LM** (256-dimensional embeddings)

Stages 1 through 5A.1.2 established that while reliability-aware and equal affinity fusion reduce speaker error, the overall Diarization Error Rate (DER) remains dominated by upstream errors:
- **Upstream VAD errors**: False alarms (14.28% macro) and missed speech (21.90% macro) account for ~36.18% out of 46.45% macro DER (77.9% of all diarization errors).
- **Overlapping speech**: In high-overlap sessions (R02 with 34.6% overlap speech, R04 with 51.1% overlap speech), missed speech surges to 49.33% and 28.52% due to single-label assignment.
- **Speaker-count collapse**: Extreme speaker duration imbalance causes NeMo NME-SC spectral clustering to collapse to $k=1$ (R07 has 87.0% dominant speaker), driving speaker error to 52.34%.

This audit maps every file, tool, environment, and asset required to formulate and execute Stage 5B.

---

## 2. Environment & System Architecture

### 2.1 Hardware Environment
- **Host**: Linux x86_64 (`priyanka-Precision-7920-Tower`, Ubuntu / Debian-based)
- **CPU**: Intel Xeon Gold multi-core workstation architecture
- **GPU**: NVIDIA GeForce RTX 4090 (24,564 MiB VRAM), CUDA 12.x compatible
- **Local Storage**: High-speed NVMe/SSD workspace at `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`

### 2.2 Python Virtual Environments
1. **Primary Research Environment (`research_stages/.venv_phase20`)**:
   - Python: 3.10.12
   - Key packages:
     - `torch`: 2.6.0+cu124 (PyTorch with CUDA 12.4 acceleration)
     - `torchaudio`: 2.6.0+cu124
     - `pyannote.audio`: 3.3.2 (Includes pretrained segmentation and VAD models)
     - `nemo_toolkit`: Installed with ASR / Speaker Diarization submodules
     - `wespeaker`: 1.2.0 (ResNet293-LM inference)
     - `deepfilternet`: 0.5.6 (DFN3 enhancement)
     - `scipy`, `numpy`, `scikit-learn`, `pandas`, `soundfile`, `librosa`
2. **Legacy Standalone Environment (`.venv`)**:
   - Python 3.10 virtual environment used in early Stage 0/1 development. Maintained for backward compatibility.

---

## 3. Pretrained Neural Models & Checkpoint Inventory

All neural models are hosted locally or cached in user/system directories, eliminating mandatory cloud dependencies during offline execution.

| Model Name | Local Path / Source | Parameter Count / Architecture | Modality / Dimensionality | Used in Stages |
|---|---|---|---|---|
| **NVIDIA TitaNet-Large** | NeMo Cache / HuggingFace `nvidia/speakerverification_en_titanet_large` | 25.3M parameters, 1D Depthwise-Separable Convolutions + Squeeze-and-Excitation | 192-dim L2-normalized speaker embedding | Stage 3, 4, 4.1, 5A, 5A.1 |
| **WeSpeaker ResNet293-LM** | Pretrained checkpoint `wespeaker/models/voxceleb_resnet293` | ~29.3M parameters, Deep ResNet with Large-Margin Cosine Loss | 256-dim L2-normalized speaker embedding | Stage 3, 4, 4.1, 5A, 5A.1 |
| **Pyannote VAD (PyanNet)** | `assets/pytorch_model.bin` / HuggingFace `pyannote/voice-activity-detection` | Multi-layer SincNet + Bidirectional LSTM + Linear | Frame-level speech posterior (stride 16ms) | Stage 1, 2, Phase 20 |
| **Pyannote Segmentation 3.0** | Cached in torch hub / `pyannote/segmentation-3.0` | SincNet + Multi-stage LSTM with multi-speaker overlap output | Up to 3 simultaneous speaker posteriors | Planned for Stage 5B.2 |
| **DeepFilterNet3** | Pretrained weights via `df` library (`DeepFilterNet3`) | Deep complex convolutional recurrent network (Perceptual loss) | Audio enhancement (spectral masking + periodic synthesis) | Stage 1, 3, 4, 5A |
| **OpenAI Whisper / WhisperX** | Cached weights (`large-v2` / `medium.en`) | Transformer Encoder-Decoder + Phoneme Alignment | Word-level speech-to-text with precise timestamps | Baseline Phase 20 / Stage 2 |

---

## 4. Dataset & Annotation Inventory

The canonical evaluation suite consists of 8 diverse, challenging real-world audio recordings.

| Recording ID | Alias / Filename | Source / Acoustic Domain | Audio Duration | GT Speakers | Total Speech Duration | Overlap Duration (%) |
|---|---|---|---|---|---|---|
| **R01** | `R01_clean_dialogue_01.wav` | Synthetic/Clean Studio Conversation | 16.03 s | 2 | 15.35 s | 0.00 s (0.0%) |
| **R02** | `R02_parliament_debate_01.wav` | Broad Parliamentary Acoustic / Shouting | 185.00 s | 3 | 162.00 s | 56.00 s (34.6%) |
| **R03** | `R03_interview_podcast_01.wav` | Clean Multi-Mic Studio Podcast | 190.00 s | 2 | 188.00 s | 2.00 s (1.1%) |
| **R04** | `R04_talk_show_argument_01.wav` | Televised Cross-Talk / Heated Debate | 190.00 s | 4 | 186.00 s | 95.00 s (51.1%) |
| **R05** | `R05_bodycam_traffic_stop_01.wav` | Axon BWC / Heavy Engine Noise & Siren | 100.00 s | 2 | 75.00 s | 0.00 s (0.0%) |
| **R06** | `R06_bodycam_domestic_call_01.wav` | Axon BWC / Domestic Disturbance, 9 Speakers | 385.00 s | 9 | 371.00 s | 2.00 s (0.5%) |
| **R07** | `R07_bwc_cheap_shot.wav` | Axon BWC / High Wind, Distortion, Heavy Imbalance | 124.00 s | 3 | 123.00 s | 1.00 s (0.8%) |
| **R08** | `R08_synthetic_noisy_two_speaker.wav` | Noisy Synthetic Mixed Dialogue | 20.00 s | 2 | 19.38 s | 0.00 s (0.0%) |

### 4.1 Ground Truth Annotation File Locations
- Unified canonical JSON annotations: `research_stages/stage1_data_prep/annotations/uniform_json/{rec_id}.json`
- Canonical RTTM references: `research_stages/stage1_data_prep/annotations/rttm/{rec_id}.rttm`
- Source audit reports: `research_stages/stage1_data_prep/reports/stage1_data_prep_report.json`

All files are verified bitwise consistent with the Stage 5A.1 freeze manifest.

---

## 5. Software Pipeline & Implementation Audit

### 5.1 Stage 1: Non-Destructive Data Preparation (`research_stages/stage1_data_prep/`)
- **Scripts**:
  - `run_stage1.py`: Orchestrates parsing of source JSON/TXT/RTTM annotations, generates standardized 16kHz mono audio, and applies DeepFilterNet3 enhancement without overwriting source files.
- **Outputs**:
  - `audio/16k_mono/`: Clean 16kHz mono 16-bit PCM WAV.
  - `audio/dfn/`: DFN3-enhanced 16kHz mono audio.
  - `annotations/uniform_json/`: Strict schema validation (schema verified in Stage 4.1).

### 5.2 Stage 2: Independent Baseline Reproduction (`research_stages/stage2_baseline/`)
- **Scripts**:
  - `run_stage2_baseline.py`: Runs baseline Pyannote VAD, WhisperX word-alignment, single-encoder TitaNet and ResNet extraction, and baseline clustering without oracle fallback.
- **Outputs**:
  - `vad_segments/`: Detected speech boundaries.
  - `eval_results/`: Baseline DER and JER metrics establishing the initial 46.2% DER baseline.

### 5.3 Stage 3: Feature Extraction & Representation Cache (`research_stages/stage3_features/`)
- **Scripts**:
  - `extract_features.py`: Extracts sliding window embeddings (1.5s subsegments with 0.75s shift) across Path A (TitaNet-Large on DFN audio) and Path B (ResNet293-LM on 16k_mono audio).
- **Extracted Tensors**:
  - `features/{rec_id}_titanet_dfn.npz` (192-dim)
  - `features/{rec_id}_resnet_16k.npz` (256-dim)
  - Pre-computed frame timestamps, segment durations, SNR estimates, and preliminary overlap posteriors.

### 5.4 Stage 4 & 4.1: Representation Complementarity & Statistical Leakage Audit (`research_stages/stage4_complementarity/`, `stage4_1_audit/`)
- **Scripts**:
  - `run_stage4_complementarity.py`: Implements spectral clustering (`nme_spectral_clustering`), affinity fusion functions (`equal_fusion`, `snr_gated_fusion`, `logistic_fusion`), and evaluation.
  - `run_stage4_1_audit.py`: Validated 5-fold cross-validation, verified absence of test-train data leakage, proved normalization properties, and established bootstrap significance.

### 5.5 Stage 5A, 5A.1, and 5A.1.2: Controlled Fusion Freeze
- **Scripts & Freeze Records**:
  - `research_stages/stage5a_1/run_stage5a_1.py`: Condition A through Condition K evaluation.
  - `research_stages/stage5a_1_final_audit/`: Code-level proof of NME-SC diagonal zeroing invariance.
  - `research_stages/stage5a_1_2_freeze_closure/STAGE5A_1_2_FINAL_FREEZE_REPORT.md`: Formally froze representation-fusion findings with Condition J (Equal Fusion) achieving DER = 46.45%, DER Speaker Error = 10.27%, Cluster Purity = 0.8125, NMI = 0.5369.

### 5.6 Legacy Baseline Pipeline (`phase20_standalone/`)
- **Scripts**:
  - `run_phase20_pipeline.py`: Monolithic execution script wrapping Pyannote VAD, WhisperX, NeMo clustering, and transcription attribution.
  - `evaluate_vad_whisperx.py`: Original script reporting VAD metrics on `R07_bwc_cheap_shot` (F1 = 0.844, Miss = 13.5%, False Alarm = 39.7%).

---

## 6. Identification of Upstream Diarization Bottlenecks

A comprehensive forensic audit of Stage 5A.1 frozen results reveals the exact breakdown of the 46.45% Macro DER:

```
Total Macro DER (Condition J): 46.45%
├── Upstream VAD Missed Speech: 21.90%  (47.1% of total error)
├── Upstream VAD False Alarms:  14.28%  (30.7% of total error)
└── Speaker Confusion Error:    10.27%  (22.1% of total error)
```

**Key Takeaway**: Upstream speech activity and boundary detection account for **77.9%** of the entire system's diarization error. Only 22.1% is attributable to embedding clustering and representation fusion.

In particular:
1. **Environmental Noise False Alarms**: R05 (FA = 37.79%) and R06 (FA = 41.33%) suffer from excessive non-speech false triggers.
2. **Overlap-Induced Missed Speech**: R02 (Miss = 49.33%) and R04 (Miss = 28.52%) suffer catastrophic speech misses because single-label segmentation ignores concurrent speakers.
3. **Imbalance-Induced Count Collapse**: R07 (SpkErr = 52.34%) collapses to $k=1$ because NeMo NME-SC's maximum eigengap heuristic fails when one speaker occupies 87% of the recording duration.

These three bottlenecks form the explicit target of Stage 5B.
