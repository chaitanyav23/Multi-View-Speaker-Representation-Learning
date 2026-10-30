# phase14_bwc_fix Directory Tree

```
phase14_bwc_fix/
├── audio
│   ├── bwc_normalized.wav
│   └── bwc_repaired.wav
├── diarization
│   ├── bwc_msdd.rttm
│   └── msdd_outputs
│       ├── pred_rttms
│       │   └── bwc_normalized.rttm
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
│   ├── bwc_grounding.json
│   └── bwc_oracle.rttm
├── manifests
│   └── bwc_normalized_manifest.json
├── outputs
│   ├── bwc_aligned_words.json
│   ├── bwc_final.json
│   └── bwc_final.txt
├── run_pipeline.py
├── step2_vad_grounding.py
├── step3_oracle_rttm.py
├── step4_whisperx.py
├── step5_msdd.py
└── step6_fusion.py
```
