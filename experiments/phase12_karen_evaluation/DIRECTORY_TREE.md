# phase12_karen_evaluation Directory Tree

```
phase12_karen_evaluation/
├── audio_preprocessed
│   ├── karen_muted.wav
│   └── karen_normalized.wav
├── audits
│   ├── audio_metadata.md
│   └── audio_provenance_report.md
├── diagnostics
│   ├── asr_errors
│   ├── attribution_errors
│   ├── diarization_errors
│   ├── speaker_count_analysis
│   │   └── speaker_count_analysis.md
│   └── vad_errors
├── grounding
│   ├── karen_system_vad.rttm
│   └── karen_vad.json
├── outputs
│   ├── asr
│   │   ├── karen_aligned.json
│   │   └── karen_asr.json
│   ├── conversational_timeline
│   │   ├── karen_conversation.json
│   │   └── karen_conversation.txt
│   ├── diarization
│   │   ├── karen_diarization.rttm
│   │   ├── karen_manifest.json
│   │   ├── pred_rttms
│   │   │   └── karen_normalized.rttm
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
│   └── fusion
│       ├── karen_conversation.json
│       └── karen_fused.json
├── reports
│   └── vad_metrics.csv
├── scripts
│   ├── step0_audio_validation.py
│   ├── step11_to_15.py
│   ├── step1_normalization.py
│   ├── step2_to_6.py
│   └── step7_to_10.py
├── vad_segments
│   ├── index.csv
│   ├── segment_0001.wav
│   ├── segment_0002.wav
│   ├── segment_0003.wav
│   ├── segment_0004.wav
│   ├── segment_0005.wav
│   └── segment_0006.wav
└── visualizations
```
