# phase15_no_norm Directory Tree

```
phase15_no_norm/
├── audio
│   └── karen_repaired.wav
├── comparisons
│   └── karen_vs_bwc.csv
├── diagnostics
│   ├── asr
│   │   └── asr_comparison.txt
│   ├── attribution
│   │   └── attribution_confusion.csv
│   ├── conversational
│   ├── diarization
│   └── vad
│       ├── fa_regions.json
│       └── miss_regions.json
├── diarization
│   ├── karen_msdd.rttm
│   └── msdd_outputs
│       ├── pred_rttms
│       │   └── karen_repaired.rttm
│       └── speaker_outputs
│           ├── embeddings
│           │   ├── subsegments_scale0_embeddings.pkl
│           │   ├── subsegments_scale1_embeddings.pkl
│           │   ├── subsegments_scale2_embeddings.pkl
│           │   ├── subsegments_scale3_embeddings.pkl
│           │   └── subsegments_scale4_embeddings.pkl
│           ├── oracle_vad_manifest.json
│           ├── subsegments_scale0.json
│           ├── subsegments_scale1.json
│           ├── subsegments_scale2.json
│           ├── subsegments_scale3.json
│           ├── subsegments_scale4.json
│           └── subsegments_scale4_cluster.label
├── grounding
│   ├── karen_grounding.json
│   └── karen_oracle.rttm
├── manifests
│   └── karen_manifest.json
├── outputs
│   ├── aligned_words.json
│   ├── karen_final.json
│   └── karen_final.txt
├── reports
│   ├── asr
│   ├── asr_metrics.csv
│   ├── attribution
│   ├── attribution_metrics.csv
│   ├── conversational
│   ├── conversational_metrics.csv
│   ├── diarization
│   ├── diarization_metrics.csv
│   ├── vad
│   └── vad_metrics.csv
├── run_phase15.py
├── step10_diar_eval.py
├── step11_attribution_eval.py
├── step12_13_conversational_eval.py
├── step14_comparison.py
├── step2_vad_grounding.py
├── step3_oracle_rttm.py
├── step4_whisperx.py
├── step5_msdd.py
├── step6_fusion.py
├── step7_process_gt.py
├── step8_vad_eval.py
├── step9_asr_eval.py
└── visualizations
```
