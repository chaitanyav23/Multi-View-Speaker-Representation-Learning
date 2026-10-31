# 01 — Local Repository Discovery & Inventory

**Audit Status:** `VERIFIED`  
**Audit Date:** 2026-10-31  
**Repository Root:** `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`

---

## 1. Git State & Remote Verification

All repository discovery commands were executed directly in `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning` (`VERIFIED`):

| Property | Verified Value | Status |
|---|---|---|
| **Working Directory (`pwd`)** | `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning` | `VERIFIED` |
| **Git Top-Level (`git rev-parse --show-toplevel`)** | `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning` | `VERIFIED` |
| **Remote (`git remote -v`)** | `origin https://github.com/chaitanyav23/Multi-View-Speaker-Representation-Learning.git (fetch/push)` | `VERIFIED` |
| **Current Branch (`git branch --show-current`)** | `main` | `VERIFIED` |
| **Local `HEAD` Commit** | `f2ac2e4011ec31896023c270ac1f7d08c929d54b` | `VERIFIED` |
| **Remote `origin/main` (`git ls-remote origin HEAD`)** | `f2ac2e4011ec31896023c270ac1f7d08c929d54b` | `VERIFIED` |
| **Working Tree Status (`git status --short`)** | Empty (0 modified, 0 staged, 0 untracked prior to `research_audit/` generation) | `VERIFIED` |

### 1.1 Commit History (`git log -5 --oneline`)

| Commit Hash | Commit Message | Key Changes Verified via `git show --stat` |
|---|---|---|
| `f2ac2e4` | `Track 8 raw audio .mp3 files and update .gitignore` | Updated [`.gitignore`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/.gitignore#L1-L26) (`!raw_audio/*.mp3`) and tracked 8 `.mp3` files in `raw_audio/`. |
| `14ea1c6` | `updated raw_audio` | Removed legacy untracked/ignored audio references. |
| `09428e8` | `renamed GT files` | Renamed `ground_truth/*.txt` files to full descriptive titles (12 files). |
| `aa04fcc` | `Added base paper` | Added [`papers/2607.24958v1.pdf`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/papers/2607.24958v1.pdf) (*Towards Operational Conversational Intelligence: A Speech Intelligence Framework*). |
| `30bd35b` | `Repo Replicated and ground_truth updated` | Initial import of the Phase 1–20 experimental codebase, reports, and `phase20_standalone/`. |

### 1.2 Local vs. Remote Comparison (`Phase A2`)

- **Parity (`VERIFIED`):** Local `HEAD` (`f2ac2e4011ec31896023c270ac1f7d08c929d54b`) is byte-for-byte identical to `origin/main` on GitHub (`https://github.com/chaitanyav23/Multi-View-Speaker-Representation-Learning.git`).
- **Historical Provenance Discrepancy (`VERIFIED`):** All pre-computed outputs inside [`phase20_standalone/outputs/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/outputs), [`phase20_standalone/manifests/bwc_pathA_msdd_manifest.json`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/manifests/bwc_pathA_msdd_manifest.json#L1), and [`phase20_standalone/environment/environment_hashes.json`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/environment/environment_hashes.json#L1-L36) were generated on an external AWS EC2 instance (`/home/ubuntu/marston/phase20_standalone` with an NVIDIA Tesla T4 GPU) and committed to Git. They were **not** generated on the current local machine (`/home/priyanka/chaitanya/...`, NVIDIA GeForce RTX 4090).

---

## 2. `.gitignore` Audit & Missing Ignored Artifacts

Inspecting [`.gitignore`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/.gitignore#L1-L26) reveals why several critical runtime artifacts are absent from the local repository (`VERIFIED`):

```gitignore
# Audio and large media
*.wav
*.mp3
!raw_audio/*.mp3
*.m4a
*.flac
audio_chunks/
preprocessed_audio/
temp_audio/
vad_segments/

# Model caches and weights
.cache/
models/
pretrained_models/
*.pt
*.pth
*.bin
*.onnx
*.ckpt
*.nemo
```

### Critical Consequences of `.gitignore` Rules (`VERIFIED`):
1. **Zero `.wav` files exist anywhere in the repository (`find . -name "*.wav"` returns 0 files):**
   - `phase20_standalone/raw_audio/bwc.wav` does **not** exist (`phase20_standalone/raw_audio/` is not even a directory).
   - `phase20_standalone/audio/bwc_16k_mono.wav`, `bwc_dfn.wav`, and `bwc_normalized.wav` do **not** exist (`phase20_standalone/audio/` is absent).
2. **Zero `.nemo`, `.pt`, `.ckpt`, or `.onnx` model checkpoints exist in the repository:**
   - Neither `phase20_standalone/.cache/` nor `~/.cache/torch/NeMo/` exists on this machine (`VERIFIED`).
3. **Broken Virtual Environment Symlink (`VERIFIED`):**
   - [`phase20_standalone/.venv`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/.venv) is a tracked symlink/pointer (`../.venv_phase20_44e9bb7`) pointing to a non-existent directory outside the repository.

---

## 3. Concise Directory Tree & Component Inventory

```text
/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/
├── .gitignore                         # Excludes *.wav, *.pt, *.ckpt, *.nemo, .cache/ (except !raw_audio/*.mp3)
├── README.md                          # Root project overview and Phase 1-20 summary
├── MASTER_METRICS_TABLE.csv           # Historical metrics across Phases 1-20 on 'bwc' and 'karen'
├── MASTER_PHASE_COMPARISON.md         # Narrative comparison of Phases 1-20
├── MASTER_PROJECT_REPORT.md           # High-level project report
├── GLOBAL_PIPELINE_EVOLUTION.md       # Architectural evolution across phases
├── GLOBAL_RESULTS_ANALYSIS.md         # Cross-phase metric analysis
├── CODE_DERIVED_PIPELINE_AUDIT.md     # Earlier code audit of Phase 11-19 scripts
├── MSDD_11A_CONFIG_AUDIT.md           # Config audit of Phase 11A NeMo MSDD parameters
├── MSDD_KAREN_AUDIT_REPORT.md         # Audit of NeMo MSDD behavior on 'karen' recording
├── KAREN_VS_PHASE11A_MSDD_AUDIT.md    # Comparison of MSDD on 'karen' vs 'bwc'
├── verify_phase20.py                  # Root script comparing Phase 19 and Phase 20 outputs
├── run_final_hardened_pipeline.py     # Legacy Phase 11/12 single-file pipeline (uses pyannote 3.1 + zero-masking)
├── run_karen_audit.py                 # Standalone MSDD audit runner for 'karen'
├── papers/
│   └── 2607.24958v1.pdf               # Base paper: "Towards Operational Conversational Intelligence..." (34 pp.)
├── raw_audio/                         # 8 tracked stereo 44.1 kHz .mp3 files (4 of 12 GT recordings missing audio)
├── ground_truth/                      # 12 raw human-annotated .txt transcript/timestamp files
├── ground_truth_json/                 # Pre-parsed JSON GT for 'bwc.json' and legacy 'karen.json'
├── ground_truth_rttm/                 # Pre-parsed RTTM GT for 'bwc.rttm' and legacy 'karen.rttm'
├── configs/                           # YAML configs for historical phases (phase5-phase8, vad_config.yaml)
├── scripts/                           # Legacy pipeline scripts (01_preprocess..06_evaluate, calculate_snr.py, utils/)
├── reports/                           # JSON/CSV/MD evaluation reports from historical runs
├── vad_aug_temp/                      # Synthetic BWC noise augmentation & VAD benchmark subproject
├── experiments/                       # Historical experimental artifacts (Phases 5-20)
│   ├── karen_11A_audit/               # Saved NeMo MSDD run on 'karen' WITH TitaNet .pkl embeddings!
│   ├── phase5_grounded_diarization/   # Saved NeMo MSDD runs on 'bwc' & 'karen' WITH TitaNet .pkl embeddings!
│   ├── phase6_tuning/                 # Parameter sweep experiments
│   ├── phase7_sota/                   # SOTA baseline comparisons
│   ├── phase8_hardened/               # Hardened pipeline outputs
│   ├── phase9_production/             # Production test WITH saved TitaNet .pkl embeddings!
│   ├── phase10_clustering/            # Clustering threshold experiments
│   ├── phase11_dfn_ablation/          # DeepFilterNet ablation (11A/11B/11C) WITH saved TitaNet .pkl embeddings!
│   ├── phase12_whisper_tuning/        # WhisperX decoding experiments
│   ├── phase13_vad_smoothing/         # VAD post-processing experiments
│   ├── phase14_15_16_nudges/          # Boundary/collar/attribution nudges
│   ├── phase17_tich/                  # TICH experiments on 'bwc'
│   ├── phase18_tich_karen/            # TICH experiments on 'karen' (includes analyze_embeddings.py)
│   ├── phase19_11A_repro/             # Reproduction of Phase 11A
│   └── phase20_clean_room/            # Clean-room verification artifacts
└── phase20_standalone/                # Documented reproducible reference release
    ├── run_phase20.py                 # Main sequential orchestrator (15 subprocess calls)
    ├── download_models.py             # Pre-flight model downloader (DFN3, Pyannote VAD, NeMo MSDD, WhisperX)
    ├── requirements_frozen.txt        # 257 frozen pip packages from AWS T4 environment
    ├── constraints.txt                # Exact package constraints
    ├── environment/                   # Hardware, CUDA, package, and model SHA256 manifests
    ├── ground_truth/bwc.txt           # Copy of 'bwc' ground truth transcript
    ├── ground_truth_json/bwc.json     # Curated 'bwc' ground truth JSON (48 segments)
    ├── ground_truth_rttm/bwc.rttm     # Curated 'bwc' ground truth RTTM (37 positive-duration segments)
    ├── manifests/                     # NeMo MSDD input manifest & resolved MSDD config JSON
    ├── outputs/                       # Pre-computed VAD, WhisperX, and MSDD outputs for 'bwc'
    ├── fusion/                        # Word-level and segment-level speaker-attributed ASR outputs for 'bwc'
    ├── metrics/                       # JSON and CSV evaluation metrics for 'bwc'
    ├── reports/                       # Markdown summary reports and Phase 19 comparison
    └── scripts/                       # Modular step00..step10 Python scripts and metric modules
```

---

## 4. Structured Inventory by Artifact Type

### 4.1 Python Source Files (`phase20_standalone/` vs. Root/Legacy)

| Category | File Path | Role | Status |
|---|---|---|---|
| **Phase 20 Entrypoint** | [`phase20_standalone/run_phase20.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/run_phase20.py#L1-L63) | Orchestrates 15 sequential subprocess steps for recording `bwc`. | `VERIFIED` |
| **Phase 20 Model Loader** | [`phase20_standalone/download_models.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/download_models.py#L1-L71) | Downloads DeepFilterNet3, Pyannote segmentation-3.0, NeMo `diar_msdd_telephonic`, WhisperX `large-v2`, and `WAV2VEC2_ASR_LARGE_LV60K_960H`. | `VERIFIED` |
| **Phase 20 Shared Core** | [`phase20_standalone/scripts/common.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/common.py#L1-L853) | Defines `RecordingPaths`, `default_msdd_config()`, RTTM/JSON I/O, `evaluate_diarization()`, `evaluate_asr()`, `fuse_transcript_and_diarization()`, `evaluate_fusion()`. | `VERIFIED` |
| **Phase 20 Step 00** | [`phase20_standalone/scripts/step00_ground_truth.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step00_ground_truth.py#L1-L165) | Parses `ground_truth/<stem>.txt`, applies `apply_legacy_reference_curation()`, writes JSON and RTTM. | `VERIFIED` |
| **Phase 20 Step 01** | [`phase20_standalone/scripts/step01_preprocess.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step01_preprocess.py#L1-L24) | Converts `raw_audio/<stem>.wav` to 16 kHz mono PCM WAV via `ffmpeg`. | `VERIFIED` |
| **Phase 20 Step 02a** | [`phase20_standalone/scripts/step02a_deepfilternet.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step02a_deepfilternet.py#L1-L39) | Runs `deepFilter --no-suffix` on 16 kHz mono WAV (with silent copy fallback). | `VERIFIED` |
| **Phase 20 Step 02b** | [`phase20_standalone/scripts/step02b_loudnorm.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step02b_loudnorm.py#L1-L24) | Runs `ffmpeg` `loudnorm=I=-16:TP=-1.5:LRA=11` on 16 kHz mono WAV. | `VERIFIED` |
| **Phase 20 Step 03** | [`phase20_standalone/scripts/step03_vad.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step03_vad.py#L1-L407) | Runs Pyannote `VoiceActivityDetection` (`onset=0.6`, `offset=0.4`) + custom post-processing (`min_duration_off=0.25`, `pad_onset=0.05`, `pad_offset=0.05`, `min_duration_on=0.10`), writes JSON and NeMo oracle RTTM. | `VERIFIED` |
| **Phase 20 Step 04** | [`phase20_standalone/scripts/step04_vad_eval.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step04_vad_eval.py#L1-L33) | Evaluates VAD segments against reference RTTM (`DetectionPrecisionRecallFMeasure`). | `VERIFIED` |
| **Phase 20 Step 05a** | [`phase20_standalone/scripts/step05a_msdd.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step05a_msdd.py#L1-L72) | Runs NeMo `NeuralDiarizer` (`diar_msdd_telephonic` + `titanet_large`) on Path A (`bwc_dfn.wav` + `bwc_pathA_oracle.rttm`). | `VERIFIED` |
| **Phase 20 Step 05b** | [`phase20_standalone/scripts/step05b_whisperx.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step05b_whisperx.py#L1-L89) | Runs WhisperX `large-v2` + Wav2Vec2 forced alignment on Path B (`bwc_normalized.wav`). | `VERIFIED` |
| **Phase 20 Steps 06–10** | [`step06a_diar_eval.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step06a_diar_eval.py), [`step06b_asr_eval.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step06b_asr_eval.py), [`step07_fusion.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step07_fusion.py), [`step08_fusion_eval.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step08_fusion_eval.py), [`step09_aggregate_report.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step09_aggregate_report.py), [`step10_compare_phase19.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step10_compare_phase19.py) | Evaluates diarization, ASR, drift-aware word-to-speaker fusion, and Phase 19 vs. Phase 20 parity. | `VERIFIED` |
| **Phase 20 Metric Modules** | [`speaker_mapping.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/speaker_mapping.py#L1-L599), [`cpwer.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/cpwer.py#L1-L170), [`sawer.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/sawer.py#L1-L145), [`attribution_metrics.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/attribution_metrics.py#L1-L231), [`timeline_metrics.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/timeline_metrics.py#L1-L158), [`fusion_metrics.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/fusion_metrics.py#L1-L244) | Implements extended metrics (cpWER, saWER, WAE, SACC, timeline DER/JER) — **not** invoked by `run_phase20.py`, which uses `common.py` instead! | `VERIFIED` |
| **Embedding Analysis** | [`experiments/phase18_tich_karen/analyze_embeddings.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/experiments/phase18_tich_karen/analyze_embeddings.py#L1-L120) | Loads saved `subsegments_scale4_embeddings.pkl` (`[2278, 192]`), matches subsegments to GT speaker labels, and computes intra/inter-speaker cosine similarities and silhouette score. | `VERIFIED` |

### 4.2 Saved TitaNet Embedding Pickle Artifacts (`*.pkl`) (`VERIFIED`)

Although `.pt` and `.nemo` checkpoints are git-ignored, **35 multiscale TitaNet embedding `.pkl` files** (`subsegments_scale{0..4}_embeddings.pkl`) are tracked in Git across historical experiment directories (`VERIFIED`):

1. `experiments/phase11_dfn_ablation/outputs_phase11A/diarization/speaker_outputs/embeddings/subsegments_scale{0..4}_embeddings.pkl` (Recording: `bwc` on `bwc_dfn.wav`, 192-D tensors of shapes `[118, 192]`, `[134, 192]`, `[153, 192]`, `[194, 192]`, `[300, 192]`).
2. `experiments/phase11_dfn_ablation/outputs_phase11B/diarization/speaker_outputs/embeddings/subsegments_scale{0..4}_embeddings.pkl` (Recording: `bwc` on `bwc_16k_mono.wav` without DFN3, same segment counts `[118..300, 192]`).
3. `experiments/karen_11A_audit/msdd_out/speaker_outputs/embeddings/subsegments_scale{0..4}_embeddings.pkl` (Recording: `karen`, 192-D tensors of shapes `[865, 192]`, `[1051, 192]`, `[1211, 192]`, `[1466, 192]`, `[2278, 192]`).
4. `experiments/phase9_production/diarization_test/speaker_outputs/embeddings/subsegments_scale{0..4}_embeddings.pkl` (Recording: `bwc`).
5. `experiments/phase5_grounded_diarization/{diarization,diarization_pilot,diarization_tuned}/speaker_outputs/embeddings/subsegments_scale{0..4}_embeddings.pkl` (Both `bwc` and `karen` in single manifest).

These saved `.pkl` files allowed direct empirical verification of TitaNet's exact tensor shapes, data types (`torch.float32`), L2 norms, and cosine similarity distributions without running inference (`VERIFIED`).
