# phase7_optimal Directory Tree

```
phase7_optimal/
├── audio
│   ├── _tmp_deepfilternet
│   ├── bwc.wav
│   ├── bwc_norm.wav
│   └── podcast_norm.wav
├── diarization
│   ├── bwc
│   │   ├── pred_rttms
│   │   │   └── bwc.rttm
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
│   ├── bwc_diarization.json
│   ├── bwc_manifest.json
│   └── bwc_soft_vad.rttm
├── error_audit
│   ├── asr_deletions
│   │   ├── bwc_deletion_10.00_11.00.wav
│   │   ├── bwc_deletion_12.00_23.00.wav
│   │   ├── bwc_deletion_126.00_133.00.wav
│   │   ├── bwc_deletion_134.00_135.00.wav
│   │   ├── bwc_deletion_135.00_136.00.wav
│   │   ├── bwc_deletion_137.00_138.00.wav
│   │   ├── bwc_deletion_138.00_139.00.wav
│   │   ├── bwc_deletion_139.00_146.00.wav
│   │   ├── bwc_deletion_161.00_163.00.wav
│   │   ├── bwc_deletion_34.00_35.00.wav
│   │   ├── bwc_deletion_36.00_37.00.wav
│   │   ├── bwc_deletion_37.00_38.00.wav
│   │   ├── bwc_deletion_40.00_41.00.wav
│   │   ├── bwc_deletion_41.00_42.00.wav
│   │   ├── bwc_deletion_56.00_59.00.wav
│   │   ├── bwc_deletion_60.00_61.00.wav
│   │   ├── bwc_deletion_69.00_70.00.wav
│   │   ├── bwc_deletion_72.00_73.00.wav
│   │   └── bwc_deletion_82.00_83.00.wav
│   ├── diar_confusion
│   │   ├── bwc_diar_confusion_10.00_10.10.wav
│   │   ├── bwc_diar_confusion_100.00_100.10.wav
│   │   ├── bwc_diar_confusion_110.00_110.10.wav
│   │   ├── bwc_diar_confusion_115.00_115.10.wav
│   │   ├── bwc_diar_confusion_120.00_120.10.wav
│   │   ├── bwc_diar_confusion_135.00_135.10.wav
│   │   ├── bwc_diar_confusion_140.00_140.10.wav
│   │   ├── bwc_diar_confusion_145.00_145.10.wav
│   │   ├── bwc_diar_confusion_15.00_15.10.wav
│   │   ├── bwc_diar_confusion_150.00_150.10.wav
│   │   ├── bwc_diar_confusion_155.00_155.10.wav
│   │   ├── bwc_diar_confusion_160.00_160.10.wav
│   │   ├── bwc_diar_confusion_20.00_20.10.wav
│   │   ├── bwc_diar_confusion_25.00_25.10.wav
│   │   ├── bwc_diar_confusion_30.00_30.10.wav
│   │   ├── bwc_diar_confusion_5.00_5.10.wav
│   │   ├── bwc_diar_confusion_65.00_65.10.wav
│   │   ├── bwc_diar_confusion_75.00_75.10.wav
│   │   ├── bwc_diar_confusion_80.00_80.10.wav
│   │   └── bwc_diar_confusion_90.00_90.10.wav
│   ├── diar_false_alarm
│   │   ├── bwc_diar_false_alarm_125.00_125.10.wav
│   │   ├── bwc_diar_false_alarm_35.00_35.10.wav
│   │   ├── bwc_diar_false_alarm_70.00_70.10.wav
│   │   └── bwc_diar_false_alarm_95.00_95.10.wav
│   └── diar_miss
│       ├── bwc_diar_miss_100.00_100.10.wav
│       ├── bwc_diar_miss_120.00_120.10.wav
│       ├── bwc_diar_miss_130.00_130.10.wav
│       ├── bwc_diar_miss_135.00_135.10.wav
│       ├── bwc_diar_miss_40.00_40.10.wav
│       ├── bwc_diar_miss_50.00_50.10.wav
│       └── bwc_diar_miss_60.00_60.10.wav
├── eval_phase7_metrics.py
├── extract_phase7_audit.py
├── grounding
│   └── bwc_soft_vad.json
├── outputs
│   ├── asr
│   │   └── bwc_asr.json
│   ├── bwc_optimal_final.json
│   └── bwc_optimal_final.txt
├── reports
│   ├── asr_metrics.csv
│   ├── cp_sawer_metrics.csv
│   ├── diarization_metrics.csv
│   ├── final_summary.csv
│   └── vad_metrics.csv
└── run_phase7_optimal.py
```
