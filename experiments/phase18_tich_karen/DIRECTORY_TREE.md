# phase18_tich_karen Directory Tree

```
phase18_tich_karen/
├── analyze_embeddings.py
├── analyze_embeddings_interactive.py
├── asr_grounded
│   ├── audio
│   ├── diagnostics
│   ├── diarization
│   │   ├── asr_grounded_msdd.rttm
│   │   └── msdd_outputs
│   │       ├── pred_rttms
│   │       │   └── rich_karen_normalized.rttm
│   │       └── speaker_outputs
│   │           ├── embeddings
│   │           │   ├── subsegments_scale0_embeddings.pkl
│   │           │   ├── subsegments_scale1_embeddings.pkl
│   │           │   ├── subsegments_scale2_embeddings.pkl
│   │           │   ├── subsegments_scale3_embeddings.pkl
│   │           │   └── subsegments_scale4_embeddings.pkl
│   │           ├── oracle_vad_manifest.json
│   │           ├── subsegments_scale0.json
│   │           ├── subsegments_scale1.json
│   │           ├── subsegments_scale2.json
│   │           ├── subsegments_scale3.json
│   │           ├── subsegments_scale4.json
│   │           └── subsegments_scale4_cluster.label
│   ├── embedding_analysis
│   ├── grounding
│   │   ├── asr_derived_grounding.json
│   │   └── asr_derived_oracle.rttm
│   ├── manifests
│   │   └── asr_grounded_manifest.json
│   ├── outputs
│   │   ├── aligned_words_initial.json
│   │   └── asr_grounded_transcript.json
│   ├── reports
│   │   └── eval_summary.json
│   ├── step1_asr_initial.py
│   ├── step2_rttm.py
│   ├── step3_msdd.py
│   ├── step4_fusion.py
│   └── step5_eval.py
├── audio
│   └── rich_karen_normalized.wav
├── compute_asr_details.py
├── compute_conv_metrics.py
├── dfn_no_norm
│   ├── analyze_embeddings.py
│   ├── audio
│   │   └── rich_karen_dfn_no_norm.wav
│   ├── diagnostics
│   ├── diarization
│   │   ├── msdd_outputs
│   │   │   ├── pred_rttms
│   │   │   │   └── rich_karen_dfn_no_norm.rttm
│   │   │   └── speaker_outputs
│   │   │       ├── embeddings
│   │   │       │   ├── subsegments_scale0_embeddings.pkl
│   │   │       │   ├── subsegments_scale1_embeddings.pkl
│   │   │       │   ├── subsegments_scale2_embeddings.pkl
│   │   │       │   ├── subsegments_scale3_embeddings.pkl
│   │   │       │   └── subsegments_scale4_embeddings.pkl
│   │   │       ├── oracle_vad_manifest.json
│   │   │       ├── subsegments_scale0.json
│   │   │       ├── subsegments_scale1.json
│   │   │       ├── subsegments_scale2.json
│   │   │       ├── subsegments_scale3.json
│   │   │       ├── subsegments_scale4.json
│   │   │       └── subsegments_scale4_cluster.label
│   │   └── rich_karen_msdd.rttm
│   ├── embedding_analysis
│   │   ├── combined
│   │   │   └── all_scales_comparison.png
│   │   ├── metrics
│   │   │   ├── scale0_metrics.json
│   │   │   ├── scale1_metrics.json
│   │   │   ├── scale2_metrics.json
│   │   │   ├── scale3_metrics.json
│   │   │   └── scale4_metrics.json
│   │   ├── pca
│   │   │   ├── scale0_pca.png
│   │   │   ├── scale0_projections.csv
│   │   │   ├── scale1_pca.png
│   │   │   ├── scale1_projections.csv
│   │   │   ├── scale2_pca.png
│   │   │   ├── scale2_projections.csv
│   │   │   ├── scale3_pca.png
│   │   │   ├── scale3_projections.csv
│   │   │   ├── scale4_pca.png
│   │   │   └── scale4_projections.csv
│   │   ├── reports
│   │   │   └── analysis_report.md
│   │   ├── trajectories
│   │   │   ├── scale0_pca_trajectory.png
│   │   │   ├── scale0_umap_trajectory.png
│   │   │   ├── scale1_pca_trajectory.png
│   │   │   ├── scale1_umap_trajectory.png
│   │   │   ├── scale2_pca_trajectory.png
│   │   │   ├── scale2_umap_trajectory.png
│   │   │   ├── scale3_pca_trajectory.png
│   │   │   ├── scale3_umap_trajectory.png
│   │   │   ├── scale4_pca_trajectory.png
│   │   │   └── scale4_umap_trajectory.png
│   │   └── umap
│   │       ├── scale0_umap.png
│   │       ├── scale1_umap.png
│   │       ├── scale2_umap.png
│   │       ├── scale3_umap.png
│   │       └── scale4_umap.png
│   ├── grounding
│   │   ├── rich_karen_grounding.json
│   │   ├── rich_karen_gt.json
│   │   ├── rich_karen_gt.rttm
│   │   └── rich_karen_oracle.rttm
│   ├── manifests
│   │   └── rich_karen_manifest.json
│   ├── outputs
│   │   ├── aligned_words.json
│   │   └── rich_karen_transcript.json
│   ├── patch_scripts.py
│   ├── reports
│   │   ├── DFN_SUMMARY.md
│   │   ├── asr_eval.json
│   │   ├── diarization_eval.json
│   │   ├── diarization_metrics.csv
│   │   └── vad_eval.json
│   ├── run_pipeline.py
│   ├── step10_diar_eval.py
│   ├── step2_process_gt.py
│   ├── step3_vad_grounding.py
│   ├── step4_oracle_rttm.py
│   ├── step5_whisperx.py
│   ├── step6_msdd.py
│   ├── step7_fusion.py
│   ├── step8_vad_eval.py
│   └── step9_asr_eval.py
├── diagnostics
├── diarization
│   ├── msdd_outputs
│   │   ├── pred_rttms
│   │   │   └── rich_karen_normalized.rttm
│   │   └── speaker_outputs
│   │       ├── embeddings
│   │       │   ├── subsegments_scale0_embeddings.pkl
│   │       │   ├── subsegments_scale1_embeddings.pkl
│   │       │   ├── subsegments_scale2_embeddings.pkl
│   │       │   ├── subsegments_scale3_embeddings.pkl
│   │       │   └── subsegments_scale4_embeddings.pkl
│   │       ├── oracle_vad_manifest.json
│   │       ├── subsegments_scale0.json
│   │       ├── subsegments_scale1.json
│   │       ├── subsegments_scale2.json
│   │       ├── subsegments_scale3.json
│   │       ├── subsegments_scale4.json
│   │       └── subsegments_scale4_cluster.label
│   └── rich_karen_msdd.rttm
├── dual_path
│   ├── audio
│   │   ├── rich_karen_dfn.wav
│   │   └── rich_karen_normalized.wav
│   ├── diarization
│   │   ├── dual_path_msdd.rttm
│   │   └── msdd_outputs
│   │       ├── pred_rttms
│   │       │   └── rich_karen_dfn.rttm
│   │       └── speaker_outputs
│   │           ├── embeddings
│   │           │   ├── subsegments_scale0_embeddings.pkl
│   │           │   ├── subsegments_scale1_embeddings.pkl
│   │           │   ├── subsegments_scale2_embeddings.pkl
│   │           │   ├── subsegments_scale3_embeddings.pkl
│   │           │   └── subsegments_scale4_embeddings.pkl
│   │           ├── oracle_vad_manifest.json
│   │           ├── subsegments_scale0.json
│   │           ├── subsegments_scale1.json
│   │           ├── subsegments_scale2.json
│   │           ├── subsegments_scale3.json
│   │           ├── subsegments_scale4.json
│   │           └── subsegments_scale4_cluster.label
│   ├── grounding
│   │   ├── rich_karen_grounding.json
│   │   ├── rich_karen_gt.json
│   │   ├── rich_karen_gt.rttm
│   │   └── rich_karen_oracle.rttm
│   ├── manifests
│   │   └── dual_path_manifest.json
│   ├── outputs
│   │   ├── aligned_words.json
│   │   └── dual_path_transcript.json
│   ├── reports
│   │   ├── DUAL_PATH_SUMMARY.md
│   │   └── eval_summary.json
│   ├── run_pipeline.py
│   ├── step2_process_gt.py
│   ├── step3_vad_grounding.py
│   ├── step4_oracle_rttm.py
│   ├── step5_whisperx.py
│   ├── step6_msdd.py
│   ├── step7_fusion.py
│   └── step8_eval.py
├── embedding_analysis
│   ├── combined
│   │   └── all_scales_comparison.png
│   ├── metrics
│   │   ├── scale0_metrics.json
│   │   ├── scale1_metrics.json
│   │   ├── scale2_metrics.json
│   │   ├── scale3_metrics.json
│   │   └── scale4_metrics.json
│   ├── pca
│   │   ├── scale0_pca.png
│   │   ├── scale0_projections.csv
│   │   ├── scale1_pca.png
│   │   ├── scale1_projections.csv
│   │   ├── scale2_pca.png
│   │   ├── scale2_projections.csv
│   │   ├── scale3_pca.png
│   │   ├── scale3_projections.csv
│   │   ├── scale4_pca.png
│   │   └── scale4_projections.csv
│   ├── reports
│   │   └── analysis_report.md
│   ├── trajectories
│   │   ├── scale0_pca_trajectory.png
│   │   ├── scale0_umap_trajectory.png
│   │   ├── scale1_pca_trajectory.png
│   │   ├── scale1_umap_trajectory.png
│   │   ├── scale2_pca_trajectory.png
│   │   ├── scale2_umap_trajectory.png
│   │   ├── scale3_pca_trajectory.png
│   │   ├── scale3_umap_trajectory.png
│   │   ├── scale4_pca_trajectory.png
│   │   └── scale4_umap_trajectory.png
│   └── umap
│       ├── scale0_umap.png
│       ├── scale1_umap.png
│       ├── scale2_umap.png
│       ├── scale3_umap.png
│       └── scale4_umap.png
├── grounded
│   ├── analyze_embeddings.py
│   ├── audio
│   │   └── rich_karen_normalized.wav
│   ├── diagnostics
│   ├── diarization
│   │   ├── msdd_outputs
│   │   │   ├── pred_rttms
│   │   │   │   └── rich_karen_normalized.rttm
│   │   │   └── speaker_outputs
│   │   │       ├── embeddings
│   │   │       │   ├── subsegments_scale0_embeddings.pkl
│   │   │       │   ├── subsegments_scale1_embeddings.pkl
│   │   │       │   ├── subsegments_scale2_embeddings.pkl
│   │   │       │   ├── subsegments_scale3_embeddings.pkl
│   │   │       │   └── subsegments_scale4_embeddings.pkl
│   │   │       ├── oracle_vad_manifest.json
│   │   │       ├── subsegments_scale0.json
│   │   │       ├── subsegments_scale1.json
│   │   │       ├── subsegments_scale2.json
│   │   │       ├── subsegments_scale3.json
│   │   │       ├── subsegments_scale4.json
│   │   │       └── subsegments_scale4_cluster.label
│   │   └── rich_karen_msdd.rttm
│   ├── embedding_analysis
│   │   ├── combined
│   │   │   └── all_scales_comparison.png
│   │   ├── metrics
│   │   │   ├── scale0_metrics.json
│   │   │   ├── scale1_metrics.json
│   │   │   ├── scale2_metrics.json
│   │   │   ├── scale3_metrics.json
│   │   │   └── scale4_metrics.json
│   │   ├── pca
│   │   │   ├── scale0_pca.png
│   │   │   ├── scale0_projections.csv
│   │   │   ├── scale1_pca.png
│   │   │   ├── scale1_projections.csv
│   │   │   ├── scale2_pca.png
│   │   │   ├── scale2_projections.csv
│   │   │   ├── scale3_pca.png
│   │   │   ├── scale3_projections.csv
│   │   │   ├── scale4_pca.png
│   │   │   └── scale4_projections.csv
│   │   ├── reports
│   │   │   └── analysis_report.md
│   │   ├── trajectories
│   │   │   ├── scale0_pca_trajectory.png
│   │   │   ├── scale0_umap_trajectory.png
│   │   │   ├── scale1_pca_trajectory.png
│   │   │   ├── scale1_umap_trajectory.png
│   │   │   ├── scale2_pca_trajectory.png
│   │   │   ├── scale2_umap_trajectory.png
│   │   │   ├── scale3_pca_trajectory.png
│   │   │   ├── scale3_umap_trajectory.png
│   │   │   ├── scale4_pca_trajectory.png
│   │   │   └── scale4_umap_trajectory.png
│   │   └── umap
│   │       ├── scale0_umap.png
│   │       ├── scale1_umap.png
│   │       ├── scale2_umap.png
│   │       ├── scale3_umap.png
│   │       └── scale4_umap.png
│   ├── grounding
│   │   ├── rich_karen_gt.json
│   │   └── rich_karen_gt.rttm
│   ├── manifests
│   │   └── rich_karen_manifest.json
│   ├── outputs
│   │   ├── aligned_words.json
│   │   └── rich_karen_transcript.json
│   ├── reports
│   │   ├── asr_eval.json
│   │   ├── diarization_eval.json
│   │   ├── diarization_metrics.csv
│   │   └── vad_eval.json
│   ├── step10_diar_eval.py
│   ├── step5_whisperx.py
│   ├── step6_msdd.py
│   ├── step7_fusion.py
│   ├── step8_vad_eval.py
│   └── step9_asr_eval.py
├── grounding
│   ├── rich_karen_grounding.json
│   ├── rich_karen_gt.json
│   ├── rich_karen_gt.rttm
│   └── rich_karen_oracle.rttm
├── manifests
│   └── rich_karen_manifest.json
├── no_norm
│   ├── analyze_embeddings.py
│   ├── audio
│   │   └── rich_karen_no_norm.wav
│   ├── diagnostics
│   ├── diarization
│   │   ├── msdd_outputs
│   │   │   ├── pred_rttms
│   │   │   │   └── rich_karen_no_norm.rttm
│   │   │   └── speaker_outputs
│   │   │       ├── embeddings
│   │   │       │   ├── subsegments_scale0_embeddings.pkl
│   │   │       │   ├── subsegments_scale1_embeddings.pkl
│   │   │       │   ├── subsegments_scale2_embeddings.pkl
│   │   │       │   ├── subsegments_scale3_embeddings.pkl
│   │   │       │   └── subsegments_scale4_embeddings.pkl
│   │   │       ├── oracle_vad_manifest.json
│   │   │       ├── subsegments_scale0.json
│   │   │       ├── subsegments_scale1.json
│   │   │       ├── subsegments_scale2.json
│   │   │       ├── subsegments_scale3.json
│   │   │       ├── subsegments_scale4.json
│   │   │       └── subsegments_scale4_cluster.label
│   │   └── rich_karen_msdd.rttm
│   ├── embedding_analysis
│   │   ├── combined
│   │   │   └── all_scales_comparison.png
│   │   ├── metrics
│   │   │   ├── scale0_metrics.json
│   │   │   ├── scale1_metrics.json
│   │   │   ├── scale2_metrics.json
│   │   │   ├── scale3_metrics.json
│   │   │   └── scale4_metrics.json
│   │   ├── pca
│   │   │   ├── scale0_pca.png
│   │   │   ├── scale0_projections.csv
│   │   │   ├── scale1_pca.png
│   │   │   ├── scale1_projections.csv
│   │   │   ├── scale2_pca.png
│   │   │   ├── scale2_projections.csv
│   │   │   ├── scale3_pca.png
│   │   │   ├── scale3_projections.csv
│   │   │   ├── scale4_pca.png
│   │   │   └── scale4_projections.csv
│   │   ├── reports
│   │   │   └── analysis_report.md
│   │   ├── trajectories
│   │   │   ├── scale0_pca_trajectory.png
│   │   │   ├── scale0_umap_trajectory.png
│   │   │   ├── scale1_pca_trajectory.png
│   │   │   ├── scale1_umap_trajectory.png
│   │   │   ├── scale2_pca_trajectory.png
│   │   │   ├── scale2_umap_trajectory.png
│   │   │   ├── scale3_pca_trajectory.png
│   │   │   ├── scale3_umap_trajectory.png
│   │   │   ├── scale4_pca_trajectory.png
│   │   │   └── scale4_umap_trajectory.png
│   │   └── umap
│   │       ├── scale0_umap.png
│   │       ├── scale1_umap.png
│   │       ├── scale2_umap.png
│   │       ├── scale3_umap.png
│   │       └── scale4_umap.png
│   ├── grounding
│   │   ├── rich_karen_grounding.json
│   │   ├── rich_karen_gt.json
│   │   ├── rich_karen_gt.rttm
│   │   └── rich_karen_oracle.rttm
│   ├── manifests
│   │   └── rich_karen_manifest.json
│   ├── no_norm
│   │   └── audio
│   │       └── rich_karen_no_norm.wav
│   ├── outputs
│   │   ├── aligned_words.json
│   │   └── rich_karen_transcript.json
│   ├── reports
│   │   ├── asr_eval.json
│   │   ├── diarization_eval.json
│   │   ├── diarization_metrics.csv
│   │   └── vad_eval.json
│   ├── run_pipeline.py
│   ├── step10_diar_eval.py
│   ├── step1_preprocess.py
│   ├── step2_process_gt.py
│   ├── step3_vad_grounding.py
│   ├── step4_oracle_rttm.py
│   ├── step5_whisperx.py
│   ├── step6_msdd.py
│   ├── step7_fusion.py
│   ├── step8_vad_eval.py
│   └── step9_asr_eval.py
├── outputs
│   ├── aligned_words.json
│   └── rich_karen_transcript.json
├── reports
│   ├── PHASE18_FINAL_COMPARISON.md
│   ├── PHASE18_SUMMARY.md
│   ├── asr_eval.json
│   ├── diarization_eval.json
│   ├── diarization_metrics.csv
│   └── vad_eval.json
├── run_pipeline.py
├── step10_diar_eval.py
├── step1_preprocess.py
├── step2_process_gt.py
├── step3_vad_grounding.py
├── step4_oracle_rttm.py
├── step5_whisperx.py
├── step6_msdd.py
├── step7_fusion.py
├── step8_vad_eval.py
└── step9_asr_eval.py
```
