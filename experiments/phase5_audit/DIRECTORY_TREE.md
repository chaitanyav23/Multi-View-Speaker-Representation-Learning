# phase5_audit Directory Tree

```
phase5_audit/
├── msdd_bwc_test
│   ├── audit_timing.json
│   ├── diagnostics_msdd
│   │   ├── bwc_conf_0.json
│   │   ├── bwc_conf_0.wav
│   │   ├── bwc_conf_1.json
│   │   ├── bwc_conf_1.wav
│   │   ├── bwc_conf_2.json
│   │   ├── bwc_conf_2.wav
│   │   ├── bwc_conf_3.json
│   │   ├── bwc_conf_3.wav
│   │   ├── bwc_conf_4.json
│   │   ├── bwc_conf_4.wav
│   │   ├── bwc_fa_0.json
│   │   ├── bwc_fa_0.wav
│   │   ├── bwc_fa_1.json
│   │   ├── bwc_fa_1.wav
│   │   ├── bwc_fa_2.json
│   │   ├── bwc_fa_2.wav
│   │   ├── bwc_fa_3.json
│   │   ├── bwc_fa_3.wav
│   │   ├── bwc_fa_4.json
│   │   ├── bwc_fa_4.wav
│   │   ├── bwc_miss_0.json
│   │   ├── bwc_miss_0.wav
│   │   ├── bwc_miss_1.json
│   │   ├── bwc_miss_1.wav
│   │   ├── bwc_miss_2.json
│   │   ├── bwc_miss_2.wav
│   │   ├── bwc_miss_3.json
│   │   ├── bwc_miss_3.wav
│   │   ├── bwc_miss_4.json
│   │   ├── bwc_miss_4.wav
│   │   └── diarization_diagnostics.csv
│   ├── msdd_raw
│   │   ├── bwc
│   │   │   ├── pred_rttms
│   │   │   │   └── bwc.rttm
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
│   │   ├── bwc_diarization.json
│   │   ├── bwc_manifest.json
│   │   └── timing_report.json
│   └── outputs_msdd
│       └── bwc_hardened.json
├── msdd_raw
│   ├── bwc
│   │   ├── manifest_vad_input.json
│   │   ├── pred_rttms
│   │   ├── speaker_outputs
│   │   └── vad_outputs
│   │       ├── bwc.frame
│   │       └── overlap_smoothing_output_mean_0.5
│   └── bwc_manifest.json
├── oracle_rttms
│   ├── bwc.rttm
│   └── podcast.rttm
├── pipeline_msdd_oracle.py
├── pyannote_auto
│   ├── bwc_hardened.json
│   └── bwc_hardened.txt
├── pyannote_fixed2
│   ├── bwc_hardened.json
│   └── bwc_hardened.txt
├── run_audit_msdd_bwc.py
└── temp_audio
    └── bwc.wav
```
