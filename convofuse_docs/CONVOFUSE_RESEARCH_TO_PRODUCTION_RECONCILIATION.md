# ConvoFuse — Research-to-Production Reconciliation Report

**Project**: `CS787_Project`  
**Framework**: `ConvoFuse`  
**Date**: 2026-11-13  

---

## 1. Repository Discovery & Structural Reconciliation

The `ConvoFuse` unified conversational intelligence framework bridges two physical repositories in the project workspace:

1. **Phase20 Application Repository**: `/home/priyanka/chaitanya/Video-Transcription-with-Confidence-Analytics` (`HEAD = 3f1b744`)
   - Hosts the original Phase20 application under `app/` (`backend/`, `frontend/`, `engine/`, `adaptive/`, `storage/`, `tests/`).
   - Provides the FastAPI REST server (`app/backend/main.py`), SQLite job/recording/transcript repositories (`app/backend/database/`), media ingestion and 16 kHz audio normalization (`app/backend/services/audio_service.py`), Faster-Whisper `large-v3` + Wav2Vec2 CTC alignment + confidence calibration (`app/engine/phase20_engine.py`), and the accessible Streamlit dashboard (`app/frontend/streamlit_app.py`).
2. **Multi-View Speaker Representation & Generative Refinement Research Repository**: `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning` (`HEAD = 8f47c7b`)
   - Hosts the frozen Stage 1–5B.4 research audit (`research_audit/`, `research_stages/stage1_data_prep/`, `stage2_baseline/`, `stage3_extraction/`, `stage5a_1/`, `stage5b/`) and Stage 6 / Stage 6.1 generative refinement studies (`research_stages/stage6_generative_refinement/`).
   - Hosts the unified `convofuse/` Python package (`contracts.py`, `audio_diarization_adapter.py`, `generative_refiner.py`, `transcript_reconstructor.py`, `evaluator.py`, `pipeline.py`, `cli.py`), which is imported directly by `app/backend/services/convofuse_service.py`.

---

## 2. Authoritative Reconciliation of Research Decisions (Part 1.3)

### A. Canonical Audio Extraction & Sample-Rate Policy
- **Validated Finding (`STAGE1_DATA_PREPARATION_REPORT.md`)**: All recordings (`R01`–`R08`) are extracted to canonical 16 kHz, 16-bit mono PCM WAV (`pcm_s16le`, `-ar 16000 -ac 1`) with `-af aresample=async=1:first_pts=0` to preserve exact temporal origin `t = 0.000s` for alignment with reference annotations and multi-model streams.
- **ConvoFuse Production Rule**: `CanonicalAudioProcessor` (`convofuse/audio_diarization_adapter.py`) enforces 16 kHz mono PCM extraction, verifies finite samples (`np.isfinite`), records sample count and duration, and computes SHA-256 digests before downstream inference.

### B. Task-Specific Dual-Path Audio Conditioning
- **Validated Finding (`STAGE1_DATA_PREPARATION_REPORT.md`, `STAGE3_EXTRACTION_REPORT.md`, `STAGE5A_1_FINAL_RECOMMENDATION.md`, `STAGE5B_CONSOLIDATED_RESULTS_REPORT.md`)**:
  - **TitaNet-Large (192-D)** benefits from DeepFilterNet3 (DFN) speech enhancement on noisy BWC audio (`dfn.wav`), achieving `28.59%` DER solo on DFN vs higher error on raw noisy audio.
  - **WeSpeaker ResNet293-LM (256-D)** is trained with heavy additive noise augmentation and suffers spectral distortion when fed DFN-filtered audio (`33.36%` DER on DFN vs `24.16%` DER on unenhanced `16k_mono.wav`, a `-9.20 pp` improvement on raw mono).
  - **Faster-Whisper Large-v3 ASR & Wav2Vec2 CTC Alignment** operate on EBU R128 loudness-normalized audio (`loudnorm=I=-20:TP=-1.5:LRA=11`, `normalized.wav`).
  - **Pyannote Overlap Posteriors** are extracted primarily from raw `16k_mono.wav` (with DFN overlap posteriors cached separately in Stage 5B.2).
- **ConvoFuse Production Rule**: `TaskSpecificAudioPaths` explicitly routes DFN-conditioned audio to TitaNet-Large, unenhanced 16 kHz mono audio to WeSpeaker ResNet293-LM and overlap detection, and EBU R128 loudness-normalized audio to ASR and CTC forced alignment.

### C. Speaker Embedding Models & Complementary Views
- **Validated Finding (`STAGE3_EXTRACTION_REPORT.md`, `STAGE4_DECISION_GATE_REPORT.md`)**:
  - **View 1**: `nvidia/speakerverification_en_titanet_large` (192-D L2-normalized embeddings, 1.5s window / 0.75s step).
  - **View 2**: `WeSpeaker ResNet293-LM` (`wespeaker_en_voxceleb_CAM++_LM` / ResNet293-LM ONNX, 256-D L2-normalized embeddings, 1.5s window / 0.75s step).
  - Combining complementary architectures across complementary acoustic views (DFN TitaNet + Raw Mono WeSpeaker) outperforms any single encoder.

### D. Affinity Normalization, Multiscale Fusion & Spectral Clustering
- **Validated Finding (`STAGE5A_1_FINAL_RECOMMENDATION.md`, `STAGE5A_1_FREEZE_RECORD.md`, `R01_PILOT_FINAL_REPORT.md`)**:
  - Raw cosine affinity matrices from TitaNet and WeSpeaker have incompatible off-diagonal score distributions (`WeSpeaker` raw affinities are shifted higher than `TitaNet`). Unnormalized fusion collapses speaker count estimation on 5 of 8 recordings.
  - **Off-Diagonal Z-Score Row Normalization** (`normalize_affinity_matrix`) followed by **Equal-Weight Fusion** (`0.5 * S_titanet_norm + 0.5 * S_wespeaker_norm`, Condition J) or **LORO-Clipped Reliability-Gated Fusion** (`w_t in [0.35, 0.65]`, Condition K) resolves scale mismatch and achieves **16.64% macro DER** (Condition J) and **15.86% macro DER** (Condition K) across R01–R08 without oracle speaker count.
  - **Clustering**: Calibrated Normalized Maximum Eigengap Spectral Clustering (`NME-SC`, `max_RP_threshold = 0.15`, `sparse_search = True`, `max_num_speakers = 8`) estimates speaker count automatically without ground-truth supervision.

### E. VAD Modernization & Overlap Attribution Findings
- **Validated Finding (`STAGE5B_CONSOLIDATED_RESULTS_REPORT.md`, `R01_PILOT_FINAL_REPORT.md`)**:
  - Stage 5B.1 tuned Silero VAD (`C62`: `threshold = 0.35`, `min_silence_duration_ms = 250`, `speech_pad_ms = 100`) reduces missed speech on `R01_sara_sharif` (`4.29%` False Alarm, `5.11%` Missed Speech).
  - Stage 5B.2 V4.2.1 and Stage 5B.4 (`E20/M0` vs `E21/M4-DFN`) demonstrated that while second-speaker heuristic overlap assignment (`M4`) slightly lowers acoustic DER on R01 (`30.58% -> 30.25%`, `-0.33 pp`), it leaves `cpWER` (`52.30%`) and `WDER` (`20.08%`) completely unchanged because single-stream WhisperX ASR emits one linear word stream per timestamp interval.
- **ConvoFuse Production Rule**: ConvoFuse uses `M0` (`E20` on R01 / `Condition J` across R01–R08) as the primary word-attribution diarization input while attaching explicit multi-speaker RTTM overlap intervals and Stage 5B.2 frame-level overlap posterior statistics (`frame_overlap_prob_mean`, `frame_overlap_prob_max`) to every `ConvoFuseWord`.

### F. Generative Speaker Refinement (`DiarizationLM-Gemma-4-E4B-v1`)
- **Validated Finding (`STAGE6_1_BENCHMARK_REPORT.md` & ConvoFuse Remediation)**:
  - Local `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (Q4_K_M GGUF via Ollama on CUDA) executes at ~146 tokens/sec with zero API cost or truncation issues when segmented cleanly.
  - **Defect Fixed in ConvoFuse**: In the initial Stage 6.1 prototype (`diarizationlm_adapter.py`), `UNK` was mapped to `<speaker:1>` (`spk_to_tag["UNK"] = "1"`), which caused `DiarizationLM-Gemma` to reassign two `UNK` words (`"Yes sir."` at `2.052s–2.377s` on R01) to `speaker_1`, degrading `cpWER` by `+0.70 pp` (`52.30% -> 53.00%`). Furthermore, `extract_text_and_spk(..., skip_meaningless_speaker=True)` silently fell back to `previous_spk` on invalid tags.
  - **ConvoFuse Remediation**: `SpeakerIdentityMapper` (`convofuse/contracts.py`) and `ConvoFuseGenerativeRefiner` (`convofuse/generative_refiner.py`) isolate `UNK` (`<speaker:0>`), strictly forbid mapping `UNK` to any valid speaker (`REJECTED_UNKNOWN_SPEAKER_MAPPING` / `ABSTAINED_PRESERVE_UNK`), and use `extract_text_and_spk_strict` to explicitly reject missing, malformed, or out-of-inventory tags. As a result, the `+0.70 pp` R01 cpWER degradation is completely eliminated (`cpWER = 52.30%`, `Δ = 0.00 pp`), while corpus-wide macro `cpWER` across R01–R08 improves from `37.23%` to `37.00%` (`-0.23 pp`) and macro Lexical WDER improves from `13.29%` to `13.22%` (`-0.08 pp`).
