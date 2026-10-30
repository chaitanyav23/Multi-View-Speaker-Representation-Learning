# phase10_dual_branch_fusion Directory Tree

```
phase10_dual_branch_fusion/
├── audio_variants
│   ├── bwc.wav
│   ├── bwc_deepfilternet.wav
│   ├── bwc_normalized.wav
│   └── bwc_speechbrain.wav
├── custom_fusion.py
├── error_breakdown
│   ├── Phase10A
│   │   ├── bwc
│   │   │   ├── error_000_MISS_1.76_2.05.wav
│   │   │   ├── error_001_MISS_3.75_3.77.wav
│   │   │   ├── error_002_MISS_5.96_6.26.wav
│   │   │   ├── error_003_CONFUSION_10.25_10.55.wav
│   │   │   ├── error_004_MISS_10.55_10.75.wav
│   │   │   ├── error_005_FALSE_ALARM_11.41_11.75.wav
│   │   │   ├── error_006_MISS_14.18_14.69.wav
│   │   │   ├── error_007_FALSE_ALARM_23.25_23.34.wav
│   │   │   ├── error_008_CONFUSION_24.25_24.75.wav
│   │   │   ├── error_009_MISS_25.25_25.91.wav
│   │   │   ├── error_010_MISS_26.45_27.03.wav
│   │   │   ├── error_011_MISS_32.86_32.89.wav
│   │   │   ├── error_012_FALSE_ALARM_35.29_35.75.wav
│   │   │   ├── error_013_MISS_37.25_37.34.wav
│   │   │   ├── error_014_MISS_39.25_39.30.wav
│   │   │   ├── error_015_CONFUSION_39.30_39.75.wav
│   │   │   ├── error_016_MISS_40.25_40.75.wav
│   │   │   ├── error_017_CONFUSION_41.25_41.40.wav
│   │   │   ├── error_018_MISS_41.40_41.44.wav
│   │   │   ├── error_019_CONFUSION_41.44_41.75.wav
│   │   │   ├── error_020_CONFUSION_47.25_50.80.wav
│   │   │   ├── error_021_MISS_50.80_50.82.wav
│   │   │   ├── error_022_CONFUSION_50.82_51.75.wav
│   │   │   ├── error_023_MISS_56.25_58.75.wav
│   │   │   ├── error_024_MISS_60.25_60.75.wav
│   │   │   ├── error_025_CONFUSION_61.25_62.89.wav
│   │   │   ├── error_026_MISS_62.89_62.97.wav
│   │   │   ├── error_027_CONFUSION_62.97_63.84.wav
│   │   │   ├── error_028_MISS_63.84_64.97.wav
│   │   │   ├── error_029_CONFUSION_64.97_66.75.wav
│   │   │   ├── error_030_FALSE_ALARM_67.25_67.66.wav
│   │   │   ├── error_031_MISS_69.25_69.75.wav
│   │   │   ├── error_032_FALSE_ALARM_70.25_70.87.wav
│   │   │   ├── error_033_MISS_72.25_72.75.wav
│   │   │   ├── error_034_MISS_74.25_74.45.wav
│   │   │   ├── error_035_CONFUSION_74.45_75.20.wav
│   │   │   ├── error_036_MISS_75.20_77.70.wav
│   │   │   ├── error_037_CONFUSION_77.70_77.75.wav
│   │   │   ├── error_038_MISS_78.25_78.75.wav
│   │   │   ├── error_039_MISS_79.25_79.47.wav
│   │   │   ├── error_040_CONFUSION_79.47_80.75.wav
│   │   │   ├── error_041_FALSE_ALARM_81.36_81.75.wav
│   │   │   ├── error_042_MISS_82.25_82.55.wav
│   │   │   ├── error_043_FALSE_ALARM_83.25_83.57.wav
│   │   │   ├── error_044_FALSE_ALARM_83.61_84.21.wav
│   │   │   ├── error_045_FALSE_ALARM_85.84_86.34.wav
│   │   │   ├── error_046_FALSE_ALARM_86.76_86.95.wav
│   │   │   ├── error_047_MISS_88.25_88.34.wav
│   │   │   ├── error_048_MISS_90.84_92.87.wav
│   │   │   ├── error_049_FALSE_ALARM_95.25_95.75.wav
│   │   │   ├── error_050_MISS_97.61_98.11.wav
│   │   │   ├── error_051_FALSE_ALARM_103.25_103.66.wav
│   │   │   ├── error_052_FALSE_ALARM_106.21_106.41.wav
│   │   │   ├── error_053_CONFUSION_109.25_110.51.wav
│   │   │   ├── error_054_MISS_110.51_111.61.wav
│   │   │   ├── error_055_CONFUSION_111.61_111.75.wav
│   │   │   ├── error_056_MISS_113.01_113.69.wav
│   │   │   ├── error_057_CONFUSION_113.69_113.75.wav
│   │   │   ├── error_058_MISS_114.27_114.31.wav
│   │   │   ├── error_059_MISS_116.19_116.21.wav
│   │   │   ├── error_060_MISS_116.99_117.01.wav
│   │   │   ├── error_061_MISS_117.63_117.67.wav
│   │   │   ├── error_062_MISS_119.64_120.10.wav
│   │   │   ├── error_063_MISS_121.52_121.75.wav
│   │   │   ├── error_064_FALSE_ALARM_122.25_123.09.wav
│   │   │   ├── error_065_FALSE_ALARM_125.04_125.66.wav
│   │   │   ├── error_066_CONFUSION_126.25_128.17.wav
│   │   │   ├── error_067_MISS_128.17_130.44.wav
│   │   │   ├── error_068_CONFUSION_130.44_131.65.wav
│   │   │   ├── error_069_MISS_131.65_132.75.wav
│   │   │   ├── error_070_FALSE_ALARM_133.25_133.75.wav
│   │   │   ├── error_071_CONFUSION_134.25_134.75.wav
│   │   │   ├── error_072_MISS_135.25_135.60.wav
│   │   │   ├── error_073_FALSE_ALARM_136.25_136.42.wav
│   │   │   ├── error_074_CONFUSION_137.25_137.49.wav
│   │   │   ├── error_075_MISS_137.49_137.51.wav
│   │   │   ├── error_076_MISS_138.25_138.28.wav
│   │   │   ├── error_077_CONFUSION_138.28_138.75.wav
│   │   │   ├── error_078_CONFUSION_139.25_139.51.wav
│   │   │   ├── error_079_MISS_139.51_139.93.wav
│   │   │   ├── error_080_MISS_141.64_143.18.wav
│   │   │   ├── error_081_CONFUSION_143.18_145.75.wav
│   │   │   ├── error_082_MISS_150.25_154.75.wav
│   │   │   ├── error_083_CONFUSION_154.75_155.99.wav
│   │   │   ├── error_084_MISS_155.99_156.58.wav
│   │   │   ├── error_085_CONFUSION_156.58_157.75.wav
│   │   │   ├── error_086_FALSE_ALARM_158.25_158.64.wav
│   │   │   ├── error_087_MISS_160.67_160.75.wav
│   │   │   ├── error_088_CONFUSION_161.25_161.72.wav
│   │   │   ├── error_089_MISS_161.72_162.75.wav
│   │   │   └── error_090_FALSE_ALARM_164.32_164.58.wav
│   │   └── diarization_errors.csv
│   ├── Phase10B
│   │   ├── bwc
│   │   │   ├── error_000_MISS_1.76_2.15.wav
│   │   │   ├── error_001_MISS_5.18_9.75.wav
│   │   │   ├── error_002_MISS_10.25_10.75.wav
│   │   │   ├── error_003_FALSE_ALARM_11.41_11.75.wav
│   │   │   ├── error_004_MISS_14.12_14.69.wav
│   │   │   ├── error_005_FALSE_ALARM_23.25_23.30.wav
│   │   │   ├── error_006_MISS_24.74_24.75.wav
│   │   │   ├── error_007_MISS_25.50_25.93.wav
│   │   │   ├── error_008_MISS_26.45_27.03.wav
│   │   │   ├── error_009_MISS_28.96_29.12.wav
│   │   │   ├── error_010_MISS_29.98_30.02.wav
│   │   │   ├── error_011_MISS_31.26_31.34.wav
│   │   │   ├── error_012_MISS_31.66_31.70.wav
│   │   │   ├── error_013_MISS_32.58_32.89.wav
│   │   │   ├── error_014_MISS_34.25_34.75.wav
│   │   │   ├── error_015_MISS_36.25_36.75.wav
│   │   │   ├── error_016_MISS_37.25_37.30.wav
│   │   │   ├── error_017_MISS_39.25_39.30.wav
│   │   │   ├── error_018_MISS_40.25_40.75.wav
│   │   │   ├── error_019_MISS_41.74_41.75.wav
│   │   │   ├── error_020_MISS_49.94_50.08.wav
│   │   │   ├── error_021_MISS_50.80_51.75.wav
│   │   │   ├── error_022_MISS_56.25_58.75.wav
│   │   │   ├── error_023_MISS_60.25_60.75.wav
│   │   │   ├── error_024_MISS_61.25_61.44.wav
│   │   │   ├── error_025_MISS_63.08_64.53.wav
│   │   │   ├── error_026_FALSE_ALARM_67.25_67.66.wav
│   │   │   ├── error_027_MISS_69.25_69.75.wav
│   │   │   ├── error_028_FALSE_ALARM_70.25_70.85.wav
│   │   │   ├── error_029_MISS_72.25_72.75.wav
│   │   │   ├── error_030_MISS_74.25_74.43.wav
│   │   │   ├── error_031_MISS_75.14_77.70.wav
│   │   │   ├── error_032_CONFUSION_78.25_78.27.wav
│   │   │   ├── error_033_MISS_78.27_78.75.wav
│   │   │   ├── error_034_MISS_79.25_79.47.wav
│   │   │   ├── error_035_FALSE_ALARM_81.38_81.75.wav
│   │   │   ├── error_036_MISS_82.25_82.55.wav
│   │   │   ├── error_037_FALSE_ALARM_83.25_83.75.wav
│   │   │   ├── error_038_FALSE_ALARM_85.84_86.53.wav
│   │   │   ├── error_039_FALSE_ALARM_86.71_86.97.wav
│   │   │   ├── error_040_MISS_88.25_88.36.wav
│   │   │   ├── error_041_MISS_90.82_92.87.wav
│   │   │   ├── error_042_FALSE_ALARM_95.25_95.75.wav
│   │   │   ├── error_043_MISS_102.54_102.75.wav
│   │   │   ├── error_044_FALSE_ALARM_106.19_106.37.wav
│   │   │   ├── error_045_CONFUSION_109.25_110.37.wav
│   │   │   ├── error_046_MISS_110.37_111.63.wav
│   │   │   ├── error_047_CONFUSION_111.63_111.75.wav
│   │   │   ├── error_048_MISS_112.25_112.27.wav
│   │   │   ├── error_049_MISS_113.03_113.67.wav
│   │   │   ├── error_050_CONFUSION_114.25_114.35.wav
│   │   │   ├── error_051_MISS_114.35_114.37.wav
│   │   │   ├── error_052_CONFUSION_114.37_116.19.wav
│   │   │   ├── error_053_MISS_116.19_116.21.wav
│   │   │   ├── error_054_MISS_117.13_117.15.wav
│   │   │   ├── error_055_MISS_117.57_117.59.wav
│   │   │   ├── error_056_MISS_119.60_120.08.wav
│   │   │   ├── error_057_MISS_121.50_121.75.wav
│   │   │   ├── error_058_FALSE_ALARM_122.25_123.07.wav
│   │   │   ├── error_059_FALSE_ALARM_125.04_125.58.wav
│   │   │   ├── error_060_MISS_128.17_130.44.wav
│   │   │   ├── error_061_MISS_131.63_132.75.wav
│   │   │   ├── error_062_FALSE_ALARM_133.25_133.75.wav
│   │   │   ├── error_063_CONFUSION_134.25_134.75.wav
│   │   │   ├── error_064_MISS_135.25_135.60.wav
│   │   │   ├── error_065_FALSE_ALARM_136.25_136.42.wav
│   │   │   ├── error_066_CONFUSION_137.25_137.59.wav
│   │   │   ├── error_067_MISS_137.59_137.61.wav
│   │   │   ├── error_068_CONFUSION_137.61_137.75.wav
│   │   │   ├── error_069_MISS_138.25_138.28.wav
│   │   │   ├── error_070_CONFUSION_138.28_138.75.wav
│   │   │   ├── error_071_CONFUSION_139.25_139.51.wav
│   │   │   ├── error_072_MISS_139.51_139.91.wav
│   │   │   ├── error_073_CONFUSION_139.91_141.64.wav
│   │   │   ├── error_074_MISS_141.64_143.20.wav
│   │   │   ├── error_075_CONFUSION_143.20_145.75.wav
│   │   │   ├── error_076_MISS_150.25_154.77.wav
│   │   │   ├── error_077_MISS_155.99_156.86.wav
│   │   │   ├── error_078_FALSE_ALARM_158.25_158.66.wav
│   │   │   ├── error_079_MISS_160.63_160.75.wav
│   │   │   ├── error_080_MISS_161.25_161.37.wav
│   │   │   ├── error_081_CONFUSION_161.37_161.69.wav
│   │   │   ├── error_082_MISS_161.69_162.75.wav
│   │   │   ├── error_083_CONFUSION_163.25_163.75.wav
│   │   │   └── error_084_FALSE_ALARM_164.25_164.56.wav
│   │   └── diarization_errors.csv
│   └── Phase10C
│       ├── bwc
│       │   ├── error_000_MISS_1.76_2.05.wav
│       │   ├── error_001_MISS_3.75_3.77.wav
│       │   ├── error_002_MISS_5.96_6.26.wav
│       │   ├── error_003_CONFUSION_10.25_10.55.wav
│       │   ├── error_004_MISS_10.55_10.75.wav
│       │   ├── error_005_FALSE_ALARM_11.41_11.75.wav
│       │   ├── error_006_MISS_14.18_14.69.wav
│       │   ├── error_007_FALSE_ALARM_23.25_23.34.wav
│       │   ├── error_008_MISS_25.25_25.91.wav
│       │   ├── error_009_MISS_26.45_27.03.wav
│       │   ├── error_010_MISS_32.86_32.89.wav
│       │   ├── error_011_FALSE_ALARM_35.29_35.75.wav
│       │   ├── error_012_MISS_37.25_37.34.wav
│       │   ├── error_013_MISS_39.25_39.30.wav
│       │   ├── error_014_MISS_40.25_40.75.wav
│       │   ├── error_015_MISS_41.40_41.44.wav
│       │   ├── error_016_MISS_50.80_50.82.wav
│       │   ├── error_017_MISS_56.25_58.75.wav
│       │   ├── error_018_MISS_60.25_60.75.wav
│       │   ├── error_019_MISS_62.89_62.97.wav
│       │   ├── error_020_MISS_63.84_64.97.wav
│       │   ├── error_021_FALSE_ALARM_67.25_67.66.wav
│       │   ├── error_022_MISS_69.25_69.75.wav
│       │   ├── error_023_FALSE_ALARM_70.25_70.87.wav
│       │   ├── error_024_MISS_72.25_72.75.wav
│       │   ├── error_025_MISS_74.25_74.45.wav
│       │   ├── error_026_MISS_75.20_77.70.wav
│       │   ├── error_027_MISS_78.25_78.75.wav
│       │   ├── error_028_MISS_79.25_79.47.wav
│       │   ├── error_029_FALSE_ALARM_81.36_81.75.wav
│       │   ├── error_030_MISS_82.25_82.55.wav
│       │   ├── error_031_FALSE_ALARM_83.25_83.57.wav
│       │   ├── error_032_FALSE_ALARM_83.61_84.21.wav
│       │   ├── error_033_FALSE_ALARM_85.84_86.34.wav
│       │   ├── error_034_FALSE_ALARM_86.76_86.95.wav
│       │   ├── error_035_MISS_88.25_88.34.wav
│       │   ├── error_036_MISS_90.84_92.87.wav
│       │   ├── error_037_FALSE_ALARM_95.25_95.75.wav
│       │   ├── error_038_MISS_97.61_98.11.wav
│       │   ├── error_039_FALSE_ALARM_103.25_103.66.wav
│       │   ├── error_040_FALSE_ALARM_106.21_106.41.wav
│       │   ├── error_041_CONFUSION_109.25_110.51.wav
│       │   ├── error_042_MISS_110.51_111.61.wav
│       │   ├── error_043_CONFUSION_111.61_111.75.wav
│       │   ├── error_044_MISS_113.01_113.69.wav
│       │   ├── error_045_CONFUSION_114.25_114.27.wav
│       │   ├── error_046_MISS_114.27_114.31.wav
│       │   ├── error_047_CONFUSION_114.31_116.19.wav
│       │   ├── error_048_MISS_116.19_116.21.wav
│       │   ├── error_049_MISS_116.99_117.01.wav
│       │   ├── error_050_MISS_117.63_117.67.wav
│       │   ├── error_051_MISS_119.64_120.10.wav
│       │   ├── error_052_MISS_121.52_121.75.wav
│       │   ├── error_053_FALSE_ALARM_122.25_123.09.wav
│       │   ├── error_054_FALSE_ALARM_125.04_125.66.wav
│       │   ├── error_055_MISS_128.17_130.44.wav
│       │   ├── error_056_MISS_131.65_132.75.wav
│       │   ├── error_057_FALSE_ALARM_133.25_133.75.wav
│       │   ├── error_058_CONFUSION_134.25_134.75.wav
│       │   ├── error_059_MISS_135.25_135.60.wav
│       │   ├── error_060_FALSE_ALARM_136.25_136.42.wav
│       │   ├── error_061_CONFUSION_137.25_137.49.wav
│       │   ├── error_062_MISS_137.49_137.51.wav
│       │   ├── error_063_CONFUSION_137.51_137.75.wav
│       │   ├── error_064_MISS_138.25_138.28.wav
│       │   ├── error_065_CONFUSION_138.28_138.75.wav
│       │   ├── error_066_CONFUSION_139.25_139.51.wav
│       │   ├── error_067_MISS_139.51_139.93.wav
│       │   ├── error_068_CONFUSION_139.93_141.64.wav
│       │   ├── error_069_MISS_141.64_143.18.wav
│       │   ├── error_070_CONFUSION_143.18_145.75.wav
│       │   ├── error_071_MISS_150.25_154.75.wav
│       │   ├── error_072_MISS_155.99_156.58.wav
│       │   ├── error_073_FALSE_ALARM_158.25_158.64.wav
│       │   ├── error_074_MISS_160.67_160.75.wav
│       │   ├── error_075_CONFUSION_161.25_161.72.wav
│       │   ├── error_076_MISS_161.72_162.75.wav
│       │   ├── error_077_CONFUSION_163.25_163.75.wav
│       │   └── error_078_FALSE_ALARM_164.32_164.58.wav
│       └── diarization_errors.csv
├── grounding_master
│   ├── bwc_oracle.rttm
│   └── bwc_vad.json
├── outputs_phase10A
│   ├── bwc_final.json
│   └── bwc_final.txt
├── outputs_phase10B
│   ├── bwc_diagnostics.json
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
│   ├── bwc_normalized_manifest.json
│   └── bwc_word_metadata.json
├── outputs_phase10C
│   ├── bwc_diagnostics.json
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
│   ├── bwc_normalized_manifest.json
│   └── bwc_word_metadata.json
├── reports
│   ├── audio_audit_report.csv
│   ├── audio_audit_report.md
│   └── phase10_dual_branch_report.md
├── step1_generate_audio.py
├── step2_generate_grounding.py
├── step3_run_pipelines.py
└── step4_evaluate.py
```
