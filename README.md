# Marston: Operational Conversational Intelligence for Body-Worn Camera Audio

Marston is a research repository for **Operational Conversational Intelligence** on noisy Body-Worn Camera (BWC) audio. It preserves the full progression from exploratory speech-processing experiments through publication-grade dual-path pipelines and the final reproducible Phase20 standalone release.

The repository includes:

- Research phases and historical experiments.
- ASR, VAD, diarization, speaker-attribution, and fusion evaluation.
- Phase19 and Phase20 dual-path research pipelines.
- A standalone reproducible Phase20 release artifact.
- Technical reports, audits, metric tables, plots, and documentation.

The recommended entry point for reproducing published experiments is:

```bash
cd phase20_standalone
```

Then follow [phase20_standalone/README.md](phase20_standalone/README.md).

---

## 🧭 Repository Overview

Marston now contains multiple generations of research artifacts:

| Generation | Location | Role |
|---|---|---|
| Legacy research | Root scripts, reports, `experiments/phase1_*` through `experiments/phase18_*` | Exploratory pipeline development, benchmark construction, ablations, and early production candidates. |
| Phase19 | `experiments/phase19_dual_path_evaluation/` | Publication-grade dual-path evaluation pipeline before Phase20 VAD changes. |
| Phase20 | `experiments/phase20_dual_path_improved_vad/` | Improved probability-guided VAD research pipeline with complete evaluation artifacts. |
| Phase20 Standalone | `phase20_standalone/` | Frozen, reproducible, publication-ready release for independent reproduction. |

Conceptually, the repository evolved as:

```text
Legacy research
  ↓
Phase 19
  ↓
Phase 20
  ↓
Phase20 Standalone reproducible release
```

The root repository preserves the research history. The standalone directory freezes the final reproducible artifact.

---

## 🏛️ High-Level Architecture (HLA)

The pipeline has evolved from a simple linear sequence into a sophisticated **Dual-Path Architecture** that optimizes for both acoustic clarity (Transcription) and noise-resilience (Diarization).

```text
                                  ┌──────────────────────────┐
                                  │   Raw BWC Audio Input    │
                                  └─────────────┬────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
          [Path A: Diarization Branch]                      [Path B: Transcription Branch]
          (Focus: Noise Resilience)                          (Focus: Acoustic Fidelity)
          ┌────────────────────────┐                        ┌────────────────────────┐
          │ DeepFilterNet (DFN)    │                        │ RMS Normalization      │
          │ Spectral Denoising     │                        │ (-20dB LUFS)           │
          └────────────┬───────────┘                        └────────────┬───────────┘
                       │                                                 │
                       |                                    ┌────────────▼───────────┐
                       |                                    | Pyannote 3.1 VAD       | 
                       |                                    | Voice Activity Mask    |
                       |                                    └────────────┬───────────┘
                       |                                                 |
          ┌────────────▼───────────┐                        ┌────────────▼───────────┐
          │ Pyannote 3.1 VAD       │                        │ WhisperX (Large-v3)    │
          │ Voice Activity Mask    │                        │ ASR & Forced Alignment │
          └────────────┬───────────┘                        └────────────┬───────────┘
                       │                                                 │
          ┌────────────▼───────────┐                        ┌────────────▼───────────┐
          │ NeMo MSDD (Oracle-VAD) │                        │ Word-level Timestamps  │
          │ TitaNet Embeddings     │                        │ Phoneme Alignment      │
          └────────────┬───────────┘                        └────────────┬───────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                ▼
                                    ┌────────────────────────┐
                                    │  Maximum Overlap Fusion│
                                    │ (Speaker Attribution)  │
                                    └───────────┬────────────┘
                                                ▼
                                    ┌────────────────────────┐
                                    │   Final Attributed     │
                                    │   Transcript (JSON/TXT)│
                                    └────────────────────────┘
```

---

## 📁 Repository Layout

The current full root tree is archived in [ROOT_DIRECTORY_TREE.md](ROOT_DIRECTORY_TREE.md). The curated layout below shows the repository organization at a readable level.

```text
marston/
├── asr_outputs_wx/                         # WhisperX ASR outputs from the standard legacy pipeline.
├── asr_outputs_wx_custom/                  # WhisperX aligned outputs produced by custom VAD and slicing experiments.
├── asr_outputs_wx_custom_conditioned/      # Prompt-conditioned custom WhisperX alignment outputs.
├── config/                                 # Local tool and runtime configuration files.
├── configs/                                # Pipeline configuration files such as stage-level YAML settings.
├── diarization_grounded_test/              # Ground-truth-aware diarization comparison outputs.
├── diarization_outputs_pyannote_3.1/       # Diarization outputs produced by Pyannote 3.1 experiments.
├── diarization_outputs_wx/                 # Diarization outputs produced by WhisperX-associated workflows.
├── documentation/                          # Documentation validation and maintenance helpers.
├── error_analysis_A/ [IGNORED]             # Placeholder for diarization error snippets from experiment branch A.
├── error_analysis_B/ [IGNORED]             # Placeholder for normalized-audio error snippets from branch B.
├── error_analysis_C/ [IGNORED]             # Placeholder for normalized-and-bandpassed error snippets from branch C.
├── error_analysis_hardened/ [IGNORED]      # Placeholder for final hardened-pipeline error snippets.
├── experiments/                            # Historical research phases, ablations, benchmark runs, and phase outputs.
│   ├── phase1_vad/                         # Foundational VAD benchmarking experiments.
│   ├── phase2_enhancement/                 # Speech enhancement and denoising experiments.
│   ├── phase3_diarization/                 # Neural diarization and MSDD experiments.
│   ├── phase4_grounded_enhancement/        # Grounded enhancement studies against reference annotations.
│   ├── phase5_audit/                       # Oracle and audit-oriented diarization diagnostics.
│   ├── phase5_grounded_diarization/        # Grounded diarization experiments with reference-guided inputs.
│   ├── phase6_final_diagnostic/            # Spectral and diagnostic analysis experiments.
│   ├── phase7_optimal/                     # Early optimal architecture candidate experiments.
│   ├── phase8_sentence_grouped/            # Sentence-grouping experiments for ASR context.
│   ├── phase9_production/                  # Hardened production-candidate experiments.
│   ├── phase10_dual_branch_fusion/         # Dual-branch fusion research with separate ASR and diarization paths.
│   ├── phase11_dfn_ablation/               # DeepFilterNet ablation experiments.
│   ├── phase12_karen_evaluation/           # Karen benchmark evaluation experiments.
│   ├── phase13_karen_generalization/       # Generalization experiments on Karen benchmark data.
│   ├── phase14_bwc_fix/                    # BWC-specific fixes and normalization studies.
│   ├── phase15_no_norm/                    # No-normalization ablation experiments.
│   ├── phase16_no_norm_dfn/                # No-normalization with DeepFilterNet ablation experiments.
│   ├── phase17_bwc_no_norm/                # BWC no-normalization follow-up experiments.
│   ├── phase18_tich_karen/                 # Dual-path breakthrough experiments on the Rich Karen benchmark.
│   ├── phase19_dual_path_evaluation/       # Phase19 publication-grade dual-path evaluation pipeline.
│   ├── phase20_dual_path_improved_vad/     # Phase20 improved probability-guided VAD research pipeline.
│   └── karen_11A_audit/                    # Focused Karen-vs-Phase11A MSDD audit artifacts.
├── external/                               # External code and model repositories used by the research pipelines.
│   ├── MS-SNSD/                            # External speech/noise dataset dependency.
│   ├── silero-vad/                         # External Silero VAD code or model assets.
│   └── WhisperX-Audio-Intelligence-Platform/ # External/custom WhisperX platform integration.
├── final_bwc_outputs/                      # Final hardened output transcripts for an early root-level pipeline.
├── final_bwc_outputs_B/                    # Branch B hardened output transcripts.
├── final_bwc_outputs_C/                    # Branch C hardened output transcripts.
├── final_bwc_outputs_v2/                   # Second iteration of hardened final outputs.
├── final_bwc_outputs_v3/                   # Third iteration of hardened final outputs.
├── ground_truth/                           # Human-readable transcripts and source reference files.
├── ground_truth_json/                      # Structured JSON ground truth used by evaluation scripts.
├── ground_truth_rttm/                      # RTTM diarization references.
├── logs/                                   # Root-level pipeline and evaluation logs.
├── papers/                                 # Research papers and supporting technical references.
├── phase20_standalone/                     # Frozen, validated, reproducible Phase20 release artifact.
│   ├── environment/                        # Frozen package, system, model, and hash manifests.
│   ├── reports/                            # Scientific validation, forensic, and release reports.
│   ├── scripts/                            # Standalone Phase20 stage implementations and metrics.
│   ├── metrics/                            # Regenerated publication metrics for the standalone release.
│   ├── README.md                           # Standalone release installation and execution guide.
│   ├── REPRODUCIBILITY.md                  # Step-by-step reproduction guide.
│   ├── install.sh                          # Frozen-environment installer.
│   └── results_to_be_published.md          # Publication-ready final Phase20 results.
├── preprocessed_B_normalized/              # Loudness-normalized audio from branch B experiments.
├── preprocessed_C_normalized_bandpass/     # Loudness-normalized and bandpass-filtered audio from branch C experiments.
├── raw_audio/ [IGNORED]                    # Source audio and benchmark WAV/MP3 assets.
├── reports/                                # Root-level CSV, Markdown, and JSON evaluation reports.
├── scripts/                                # Shared root-level utility modules.
│   └── utils/                              # Shared helper utilities for parsing, diagnostics, and support code.
├── speech_segments/ [IGNORED]              # Sliced speech segments from root-level VAD experiments.
├── speech_segments_whisperx/ [IGNORED]     # WhisperX VAD-generated speech segments.
├── transcripts_custom/                     # Custom raw transcript JSON outputs.
├── transcripts_custom_conditioned/         # Prompt-conditioned custom raw transcript JSON outputs.
├── vad_aug_temp/                           # Temporary VAD augmentation package and evaluation material.
├── vad_custom_grounding/                   # Custom VAD grounding outputs.
├── vad_custom_grounding_B/                 # Branch B custom VAD grounding outputs.
├── vad_custom_grounding_C/                 # Branch C custom VAD grounding outputs.
├── vad_outputs/                            # Root-level VAD JSON outputs.
├── vad_outputs_whisperx/                   # WhisperX VAD JSON outputs.
├── .cache/ [IGNORED]                       # Local model and package caches; not part of source release.
├── .venv_phase20_44e9bb7/ [IGNORED]        # Local recreated historical Phase20 virtual environment.
├── marston-venv/ [IGNORED]                 # Legacy local virtual environment.
├── README.md                               # Root repository overview.
├── ROOT_DIRECTORY_TREE.md                  # Current generated full root tree.
├── requirements.txt                        # Root-level Python dependency list for legacy workflows.
└── requirements-exact.txt                  # Root-level exact dependency snapshot for legacy workflows.
```

---

## 🗂️ Repository Organization

| Area | Primary locations | Description |
|---|---|---|
| Research | `experiments/`, root reports | Historical phase-by-phase investigation and benchmark development. |
| Experiments | `experiments/phase*`, `final_bwc_outputs*`, `preprocessed_*` | Hypothesis-driven runs, ablations, branch comparisons, and candidate outputs. |
| Standalone Release | `phase20_standalone/` | Frozen reproducible Phase20 artifact for publication and independent validation. |
| Documentation | Root `*.md`, `documentation/`, `phase20_standalone/*.md` | Research narratives, audits, architecture notes, and release instructions. |
| Evaluation | Root `evaluate_*.py`, `reports/`, `phase20_standalone/reports/` | ASR, VAD, diarization, conversational, and fusion metric analysis. |
| Ground Truth | `ground_truth/`, `ground_truth_json/`, `ground_truth_rttm/` | Human references and structured evaluation inputs. |
| Utilities | Root scripts, `scripts/`, `phase20_standalone/scripts/` | Processing, parsing, scoring, preprocessing, and diagnostics. |
| External Dependencies | `external/`, `papers/` | External codebases, datasets, and research references. |
| Models and Caches | `.cache/`, `phase20_standalone/.cache/` | Local model caches excluded from source control; standalone model versions are documented in `phase20_standalone/environment/`. |

---

## 🔬 Research Evolution

Marston is intentionally historical. Earlier experiments remain available so that methodological choices can be audited.

| Phase range | Summary |
|---|---|
| Phase 1-9 | Foundational research covering VAD benchmarking, speech enhancement, diarization, sentence grouping, and hardened production candidates. |
| Phase 10-18 | Dual-path architecture research, DeepFilterNet ablations, normalization studies, fusion experiments, and benchmark generalization. |
| Phase 19 | Publication-grade dual-path pipeline that stabilized separate diarization and transcription branches for evaluation. |
| Phase 20 | Improved probability-guided VAD and complete evaluation framework for VAD, ASR, diarization, and fusion. |
| Phase20 Standalone | Frozen reproducible release with archived environment, model manifest, scientific validation, and publication-ready metrics. |

The detailed historical narrative is preserved in the root reports and in the phase directories under `experiments/`.

---

## 📦 Phase20 Standalone

`phase20_standalone/` is the recommended entry point for reproducing published experiments. It is a self-contained release artifact extracted from the broader Marston research tree.

It provides:

- A frozen historical Python environment.
- Exact package and model manifests.
- Real DeepFilterNet, Pyannote, WhisperX, and NeMo execution.
- Scientific validation reports.
- Reproducibility instructions.
- Publication-ready metric tables.
- A validated `python run_phase20.py --overwrite` pipeline.

Important files:

| File or directory | Purpose |
|---|---|
| `phase20_standalone/README.md` | Installation, model download, execution, and troubleshooting guide for the release artifact. |
| `phase20_standalone/REPRODUCIBILITY.md` | Step-by-step reproduction procedure for another researcher. |
| `phase20_standalone/install.sh` | Installer for the frozen Phase20 environment. |
| `phase20_standalone/environment/` | Package freeze, CUDA/GPU/system reports, model manifest, dependency tree, and hashes. |
| `phase20_standalone/reports/` | Scientific validation, forensic analysis, regression analysis, and final reports. |
| `phase20_standalone/results_to_be_published.md` | Publication-ready final Phase20 results and reproducibility statement. |

The standalone release is the frozen, publication-ready state. Root-level scripts and experiments remain research history.

---

## ⚙️ Installation

The root repository contains multiple historical environments and experiment generations. For reproducible published experiments, do not attempt to install from the root.

Use the standalone artifact:

```bash
cd phase20_standalone
cat README.md
cat REPRODUCIBILITY.md
./install.sh
python run_phase20.py --overwrite
```

The standalone README documents CUDA, HuggingFace authentication, system packages, package pinning, model downloads, and expected outputs.

---

## 📚 Documentation

| Document | Description |
|---|---|
| `BWC_PIPELINE_RESEARCH_REPORT.md` | High-level BWC pipeline research summary. |
| `COMPLETE_TECHNICAL_RESEARCH_REPORT.md` | Comprehensive technical report covering pipeline design and findings. |
| `CODE_DERIVED_PIPELINE_AUDIT.md` | Audit of code-derived pipeline behavior and implementation logic. |
| `MASTER_PROJECT_REPORT.md` | Consolidated project-level report across research phases. |
| `MASTER_PHASE_COMPARISON.md` | Cross-phase comparison and historical phase summary. |
| `MASTER_METRICS_TABLE.csv` | Root-level aggregate metrics table. |
| `GLOBAL_PIPELINE_EVOLUTION.md` | Narrative of pipeline evolution across major research generations. |
| `GLOBAL_RESULTS_ANALYSIS.md` | Global analysis of experiment results and comparisons. |
| `PROJECT_ARCHITECTURE_EVOLUTION.md` | Architecture evolution and design rationale. |
| `REPORT_ENHANCEMENTS_AND_APPENDICES.md` | Supplemental details and report appendices. |
| `KAREN_VS_PHASE11A_MSDD_AUDIT.md` | Focused audit comparing Karen benchmark behavior with Phase11A MSDD. |
| `MSDD_11A_CONFIG_AUDIT.md` | Configuration audit for the Phase11A MSDD setup. |
| `MSDD_KAREN_AUDIT_REPORT.md` | MSDD audit report for the Karen benchmark. |
| `ROOT_DIRECTORY_TREE.md` | Generated full root tree for the current repository structure. |
| `phase20_standalone/README.md` | Mature standalone release README for reproducing Phase20. |
| `phase20_standalone/REPRODUCIBILITY.md` | Formal reproducibility guide for Phase20 Standalone. |
| `phase20_standalone/results_to_be_published.md` | Publication-ready Phase20 final results. |
| `phase20_standalone/reports/scientific_validation_report.md` | Scientific validation of VAD, ASR, diarization, and fusion metrics. |
| `phase20_standalone/reports/environment_root_cause.md` | Environment regression root-cause report for the Phase20 ASR regression. |

---

## 🧠 Repository Philosophy

Marston is organized as a research archive plus a frozen release artifact.

- Research history is preserved so that decisions, failures, ablations, and improvements remain auditable.
- The production-grade reproducible pipeline is frozen in `phase20_standalone/`.
- Future work should continue in newer phase directories or new standalone releases, not by rewriting historical phases.
- Evaluation logic and published artifacts should remain traceable to their corresponding phase and environment.

This structure keeps experimentation open while protecting the reproducibility of published results.

---

## ✅ Current State

| Status item | Current state |
|---|---|
| Latest reproducible release | Phase20 Standalone |
| Environment | Frozen and archived in `phase20_standalone/environment/` |
| Scientific validation | Completed and documented in `phase20_standalone/reports/` |
| Publication readiness | Publication-ready metrics and report in `phase20_standalone/results_to_be_published.md` |
| Recommended reproduction path | `cd phase20_standalone` and follow the standalone README |
| Historical research | Preserved in root scripts, reports, and `experiments/` |

The current root repository is therefore best understood as the complete Marston research archive, with `phase20_standalone/` serving as the final reproducible Phase20 research artifact.
