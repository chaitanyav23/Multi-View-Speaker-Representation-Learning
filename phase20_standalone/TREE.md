# Phase20 Standalone Repository Tree

Generated from the repository root after the final reproducibility validation run.

Excluded from this documentation tree: `.git/`, `.cache/`, virtual environments, `__pycache__/`, `.ipynb_checkpoints/`, and `diarization/msdd_outputs/` runtime internals.

```text
phase20_standalone/
├── audio/
│   ├── bwc_dfn.wav
│   ├── bwc_normalized.wav
│   ├── bwc_repaired.wav
│   └── dfn_tmp
├── diarization/
│   └── pathA_msdd.rttm
├── environment/
│   ├── conda_list.txt
│   ├── cuda_version.txt
│   ├── cudnn_version.txt
│   ├── dependency_tree.txt
│   ├── environment_hashes.json
│   ├── environment_report.md
│   ├── environment_snapshot.json
│   ├── ffmpeg_version.txt
│   ├── gpu_information.txt
│   ├── model_manifest.md
│   ├── package_hashes.txt
│   ├── pip_freeze.txt
│   ├── pip_version.txt
│   ├── pipdeptree.txt
│   ├── python_version.txt
│   ├── recreated_environment.md
│   ├── requirements.lock
│   ├── requirements_frozen.txt
│   ├── system_dependencies.md
│   └── system_information.txt
├── external/
│   └── WhisperX-Audio-Intelligence-Platform/
│       └── utils.py
├── fusion/
│   ├── attributed_words.json
│   └── transcript.txt
├── ground_truth_json/
│   └── bwc.json
├── ground_truth_rttm/
│   └── bwc.rttm
├── grounding/
│   ├── pathA_oracle.rttm
│   ├── pathA_vad.json
│   ├── pathA_vad_probabilities.json
│   ├── pathB_vad.json
│   └── pathB_vad_probabilities.json
├── logs/
│   ├── step01_preprocess.log
│   ├── step02a_deepfilternet.log
│   ├── step02b_loudnorm.log
│   ├── step03_pathA_vad.log
│   ├── step03_pathB_vad.log
│   ├── step04_pathA_vad_eval.log
│   ├── step04_pathB_vad_eval.log
│   ├── step05a_msdd.log
│   ├── step05b_whisperx.log
│   ├── step06a_diar_eval.log
│   ├── step06b_asr_eval.log
│   ├── step07_fusion.log
│   ├── step08_fusion_eval.log
│   ├── step09_aggregate_report.log
│   └── step10_compare_phase19.log
├── manifests/
│   └── pathA_msdd_manifest.json
├── metrics/
│   ├── all_metrics.json
│   ├── asr_metrics.json
│   ├── diarization_metrics.json
│   ├── fusion_metrics.json
│   ├── pathA_vad.json
│   ├── pathA_vad_metrics.json
│   ├── pathB_vad.json
│   ├── pathB_vad_metrics.json
│   ├── phase20_vs_phase19_comparison.json
│   └── pipeline_metrics.json
├── outputs/
│   ├── step01_preprocess.json
│   ├── step02a_deepfilternet.json
│   ├── step02b_loudnorm.json
│   ├── step03_pathA_vad.json
│   ├── step03_pathB_vad.json
│   ├── step04_pathA_vad_eval.json
│   ├── step04_pathB_vad_eval.json
│   ├── step05a_msdd.json
│   ├── step05b_whisperx.json
│   ├── step06a_diar_eval.json
│   ├── step06b_asr_eval.json
│   ├── step07_fusion.json
│   ├── step08_fusion_eval.json
│   ├── step09_aggregate_report.json
│   └── step10_compare_phase19.json
├── phase19_reference/
│   ├── grounding/
│   │   ├── pathA_oracle.rttm
│   │   ├── pathA_vad.json
│   │   └── pathB_vad.json
│   └── metrics/
│       ├── all_metrics.json
│       ├── asr_metrics.json
│       ├── diarization_metrics.json
│       ├── fusion_metrics.json
│       ├── pathA_vad_metrics.json
│       └── pathB_vad_metrics.json
├── plots/
│   ├── missing_word_heatmap.png
│   ├── reference_vs_vad.png
│   ├── reference_vs_whisperx.png
│   ├── segment_word_histogram.png
│   ├── timeline_comparison.png
│   ├── timeline_reference_vs_whisperx.png
│   └── word_density.png
├── raw_audio/
│   └── bwc.wav
├── reports/
│   ├── alignment_analysis.md
│   ├── api_regression.md
│   ├── asr_forensic_report.md
│   ├── asr_pipeline_coverage.md
│   ├── asr_regression_report.md
│   ├── asr_report.md
│   ├── chunking_analysis.md
│   ├── coverage_metrics.md
│   ├── deleted_word_diagnostics.csv
│   ├── diarization_report.md
│   ├── environment_regression.md
│   ├── environment_root_cause.md
│   ├── environment_three_way_comparison.md
│   ├── final_pipeline_report.md
│   ├── final_report.md
│   ├── fusion_report.md
│   ├── missing_regions.md
│   ├── missing_word_analysis.md
│   ├── model_versions.md
│   ├── original_environment.md
│   ├── package_regression.md
│   ├── pathA_vad_report.md
│   ├── pathB_vad_report.md
│   ├── phase19_comparison.md
│   ├── phase19_phase20_asr_comparison.md
│   ├── phase19_phase20_whisper_comparison.md
│   ├── phase20_vs_phase19_comparison.md
│   ├── publication_tables.md
│   ├── regression_code_diff.md
│   ├── reproduction_results.md
│   ├── scientific_validation_report.md
│   ├── segment_loss_analysis.csv
│   ├── vad_difference_report.md
│   ├── vad_segment_analysis.csv
│   ├── validation_asr.md
│   ├── validation_asr_table.csv
│   ├── validation_diarization.md
│   ├── validation_diarization_table.csv
│   ├── validation_fusion.md
│   ├── validation_fusion_table.csv
│   ├── validation_vad.md
│   ├── validation_vad_table.csv
│   ├── whisper_api_analysis.md
│   ├── whisper_api_comparison.md
│   ├── whisper_runtime.csv
│   ├── whisperx_statistics.md
│   └── word_by_word_alignment_report.csv
├── scripts/
│   ├── attribution_metrics.py
│   ├── common.py
│   ├── cpwer.py
│   ├── fusion_metrics.py
│   ├── sawer.py
│   ├── speaker_mapping.py
│   ├── step01_preprocess.py
│   ├── step02a_deepfilternet.py
│   ├── step02b_loudnorm.py
│   ├── step03_vad.py
│   ├── step04_vad_eval.py
│   ├── step05a_msdd.py
│   ├── step05b_whisperx.py
│   ├── step06a_diar_eval.py
│   ├── step06b_asr_eval.py
│   ├── step07_fusion.py
│   ├── step08_fusion_eval.py
│   ├── step09_aggregate_report.py
│   ├── step10_compare_phase19.py
│   └── timeline_metrics.py
├── transcription/
│   └── aligned_words.json
├── .env
├── .env.example
├── DEPENDENCY_REPORT.md
├── download_models.py
├── environment.yml
├── evaluation
├── IMPLEMENTATION_REPORT.md
├── install.sh
├── models.md
├── README.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── results_to_be_published.md
└── run_phase20.py
```
