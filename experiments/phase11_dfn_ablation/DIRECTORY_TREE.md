# phase11_dfn_ablation Directory Tree

```
phase11_dfn_ablation/
├── audio_variants
│   ├── bwc_deepfilternet.wav
│   └── bwc_normalized.wav
├── comparisons
├── diagnostics
│   ├── asr_errors
│   │   ├── 11A_asr_segments.json
│   │   ├── 11B_asr_segments.json
│   │   └── 11C_asr_segments.json
│   ├── diarization_errors
│   │   ├── 11A
│   │   ├── 11B
│   │   └── 11C
│   └── vad_errors
│       ├── 11A_vad_errors.json
│       └── 11B_vad_errors.json
├── grounding
│   ├── 11A_grounding.json
│   ├── 11A_oracle.rttm
│   ├── 11B_grounding.json
│   └── 11B_oracle.rttm
├── outputs_phase11A
│   ├── bwc_final.json
│   ├── bwc_final.txt
│   ├── bwc_normalized
│   │   ├── pred_rttms
│   │   │   └── bwc_normalized.rttm
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
│   └── bwc_normalized_manifest.json
├── outputs_phase11B
│   ├── bwc_final.json
│   ├── bwc_final.txt
│   ├── bwc_normalized
│   │   ├── pred_rttms
│   │   │   └── bwc_normalized.rttm
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
│   └── bwc_normalized_manifest.json
├── outputs_phase11C
│   ├── bwc_final.json
│   └── bwc_final.txt
├── reports
│   ├── asr_eval_improved_phase11_11A_asr.csv
│   ├── asr_eval_improved_phase11_11B_asr.csv
│   ├── asr_eval_improved_phase11_11C_asr.csv
│   ├── audio_audit.csv
│   ├── conversational_eval_phase11_11A_conv.csv
│   ├── conversational_eval_phase11_11B_conv.csv
│   ├── conversational_eval_phase11_11C_conv.csv
│   ├── diarization_eval_improved_phase11_11A_diar.csv
│   ├── diarization_eval_improved_phase11_11B_diar.csv
│   ├── diarization_eval_improved_phase11_11C_diar.csv
│   ├── phase11_dfn_ablation_report.md
│   └── vad_metrics.csv
├── step1_generate_audio.py
├── step2_generate_grounding.py
├── step3_run_pipelines.py
├── step4_evaluate.py
├── step5_compile_report.py
└── visualizations
```
