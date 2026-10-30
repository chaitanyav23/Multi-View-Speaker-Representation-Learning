# phase13_karen_generalization Directory Tree

```
phase13_karen_generalization/
├── Untitled.ipynb
├── analyze_embeddings.py
├── analyze_embeddings_interactive.py
├── audio
│   └── karen_normalized.wav
├── comparisons
│   └── karen_vs_bwc.csv
├── diagnostics
│   ├── asr
│   │   └── asr_comparison.txt
│   ├── attribution
│   │   └── attribution_confusion.csv
│   ├── diarization
│   ├── fusion
│   ├── overlap
│   └── vad
│       ├── fa_regions.json
│       └── miss_regions.json
├── diarization
│   ├── karen_msdd.rttm
│   └── msdd_outputs
│       ├── pred_rttms
│       │   └── karen_normalized.rttm
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
├── embedding_analysis
│   ├── combined
│   │   └── all_scales_comparison.png
│   ├── interactive_plots
│   │   ├── global_pca_master.html
│   │   └── global_umap_master.html
│   ├── local_windows
│   │   ├── csv
│   │   │   ├── local_scale1_0.0_5.0.csv
│   │   │   ├── local_scale1_10.0_15.0.csv
│   │   │   ├── local_scale1_100.0_105.0.csv
│   │   │   ├── local_scale1_102.5_107.5.csv
│   │   │   ├── local_scale1_105.0_110.0.csv
│   │   │   ├── local_scale1_107.5_112.5.csv
│   │   │   ├── local_scale1_110.0_115.0.csv
│   │   │   ├── local_scale1_112.5_117.5.csv
│   │   │   ├── local_scale1_115.0_120.0.csv
│   │   │   ├── local_scale1_117.5_122.5.csv
│   │   │   ├── local_scale1_12.5_17.5.csv
│   │   │   ├── local_scale1_120.0_125.0.csv
│   │   │   ├── local_scale1_122.5_127.5.csv
│   │   │   ├── local_scale1_125.0_130.0.csv
│   │   │   ├── local_scale1_127.5_132.5.csv
│   │   │   ├── local_scale1_130.0_135.0.csv
│   │   │   ├── local_scale1_132.5_137.5.csv
│   │   │   ├── local_scale1_135.0_140.0.csv
│   │   │   ├── local_scale1_137.5_142.5.csv
│   │   │   ├── local_scale1_140.0_145.0.csv
│   │   │   ├── local_scale1_142.5_147.5.csv
│   │   │   ├── local_scale1_145.0_150.0.csv
│   │   │   ├── local_scale1_147.5_152.5.csv
│   │   │   ├── local_scale1_15.0_20.0.csv
│   │   │   ├── local_scale1_150.0_155.0.csv
│   │   │   ├── local_scale1_152.5_157.5.csv
│   │   │   ├── local_scale1_155.0_160.0.csv
│   │   │   ├── local_scale1_157.5_162.5.csv
│   │   │   ├── local_scale1_160.0_165.0.csv
│   │   │   ├── local_scale1_162.5_167.5.csv
│   │   │   ├── local_scale1_165.0_170.0.csv
│   │   │   ├── local_scale1_167.5_172.5.csv
│   │   │   ├── local_scale1_17.5_22.5.csv
│   │   │   ├── local_scale1_170.0_175.0.csv
│   │   │   ├── local_scale1_172.5_177.5.csv
│   │   │   ├── local_scale1_177.5_182.5.csv
│   │   │   ├── local_scale1_180.0_185.0.csv
│   │   │   ├── local_scale1_182.5_187.5.csv
│   │   │   ├── local_scale1_185.0_190.0.csv
│   │   │   ├── local_scale1_187.5_192.5.csv
│   │   │   ├── local_scale1_190.0_195.0.csv
│   │   │   ├── local_scale1_192.5_197.5.csv
│   │   │   ├── local_scale1_195.0_200.0.csv
│   │   │   ├── local_scale1_197.5_202.5.csv
│   │   │   ├── local_scale1_2.5_7.5.csv
│   │   │   ├── local_scale1_20.0_25.0.csv
│   │   │   ├── local_scale1_200.0_205.0.csv
│   │   │   ├── local_scale1_202.5_207.5.csv
│   │   │   ├── local_scale1_205.0_210.0.csv
│   │   │   ├── local_scale1_207.5_212.5.csv
│   │   │   ├── local_scale1_210.0_215.0.csv
│   │   │   ├── local_scale1_212.5_217.5.csv
│   │   │   ├── local_scale1_215.0_220.0.csv
│   │   │   ├── local_scale1_217.5_222.5.csv
│   │   │   ├── local_scale1_22.5_27.5.csv
│   │   │   ├── local_scale1_220.0_225.0.csv
│   │   │   ├── local_scale1_222.5_227.5.csv
│   │   │   ├── local_scale1_225.0_230.0.csv
│   │   │   ├── local_scale1_227.5_232.5.csv
│   │   │   ├── local_scale1_230.0_235.0.csv
│   │   │   ├── local_scale1_232.5_237.5.csv
│   │   │   ├── local_scale1_235.0_240.0.csv
│   │   │   ├── local_scale1_237.5_242.5.csv
│   │   │   ├── local_scale1_240.0_245.0.csv
│   │   │   ├── local_scale1_242.5_247.5.csv
│   │   │   ├── local_scale1_245.0_250.0.csv
│   │   │   ├── local_scale1_247.5_252.5.csv
│   │   │   ├── local_scale1_25.0_30.0.csv
│   │   │   ├── local_scale1_250.0_255.0.csv
│   │   │   ├── local_scale1_252.5_257.5.csv
│   │   │   ├── local_scale1_255.0_260.0.csv
│   │   │   ├── local_scale1_257.5_262.5.csv
│   │   │   ├── local_scale1_260.0_265.0.csv
│   │   │   ├── local_scale1_262.5_267.5.csv
│   │   │   ├── local_scale1_265.0_270.0.csv
│   │   │   ├── local_scale1_267.5_272.5.csv
│   │   │   ├── local_scale1_27.5_32.5.csv
│   │   │   ├── local_scale1_270.0_275.0.csv
│   │   │   ├── local_scale1_272.5_277.5.csv
│   │   │   ├── local_scale1_275.0_280.0.csv
│   │   │   ├── local_scale1_277.5_282.5.csv
│   │   │   ├── local_scale1_280.0_285.0.csv
│   │   │   ├── local_scale1_282.5_287.5.csv
│   │   │   ├── local_scale1_285.0_290.0.csv
│   │   │   ├── local_scale1_287.5_292.5.csv
│   │   │   ├── local_scale1_290.0_295.0.csv
│   │   │   ├── local_scale1_292.5_297.5.csv
│   │   │   ├── local_scale1_295.0_300.0.csv
│   │   │   ├── local_scale1_297.5_302.5.csv
│   │   │   ├── local_scale1_30.0_35.0.csv
│   │   │   ├── local_scale1_300.0_305.0.csv
│   │   │   ├── local_scale1_302.5_307.5.csv
│   │   │   ├── local_scale1_305.0_310.0.csv
│   │   │   ├── local_scale1_307.5_312.5.csv
│   │   │   ├── local_scale1_310.0_315.0.csv
│   │   │   ├── local_scale1_312.5_317.5.csv
│   │   │   ├── local_scale1_315.0_320.0.csv
│   │   │   ├── local_scale1_317.5_322.5.csv
│   │   │   ├── local_scale1_32.5_37.5.csv
│   │   │   ├── local_scale1_320.0_325.0.csv
│   │   │   ├── local_scale1_327.5_332.5.csv
│   │   │   ├── local_scale1_330.0_335.0.csv
│   │   │   ├── local_scale1_332.5_337.5.csv
│   │   │   ├── local_scale1_335.0_340.0.csv
│   │   │   ├── local_scale1_337.5_342.5.csv
│   │   │   ├── local_scale1_340.0_345.0.csv
│   │   │   ├── local_scale1_342.5_347.5.csv
│   │   │   ├── local_scale1_345.0_350.0.csv
│   │   │   ├── local_scale1_347.5_352.5.csv
│   │   │   ├── local_scale1_35.0_40.0.csv
│   │   │   ├── local_scale1_350.0_355.0.csv
│   │   │   ├── local_scale1_352.5_357.5.csv
│   │   │   ├── local_scale1_355.0_360.0.csv
│   │   │   ├── local_scale1_357.5_362.5.csv
│   │   │   ├── local_scale1_360.0_365.0.csv
│   │   │   ├── local_scale1_362.5_367.5.csv
│   │   │   ├── local_scale1_365.0_370.0.csv
│   │   │   ├── local_scale1_367.5_372.5.csv
│   │   │   ├── local_scale1_37.5_42.5.csv
│   │   │   ├── local_scale1_370.0_375.0.csv
│   │   │   ├── local_scale1_372.5_377.5.csv
│   │   │   ├── local_scale1_375.0_380.0.csv
│   │   │   ├── local_scale1_377.5_382.5.csv
│   │   │   ├── local_scale1_380.0_385.0.csv
│   │   │   ├── local_scale1_382.5_387.5.csv
│   │   │   ├── local_scale1_385.0_390.0.csv
│   │   │   ├── local_scale1_387.5_392.5.csv
│   │   │   ├── local_scale1_390.0_395.0.csv
│   │   │   ├── local_scale1_392.5_397.5.csv
│   │   │   ├── local_scale1_395.0_400.0.csv
│   │   │   ├── local_scale1_397.5_402.5.csv
│   │   │   ├── local_scale1_40.0_45.0.csv
│   │   │   ├── local_scale1_400.0_405.0.csv
│   │   │   ├── local_scale1_402.5_407.5.csv
│   │   │   ├── local_scale1_405.0_410.0.csv
│   │   │   ├── local_scale1_407.5_412.5.csv
│   │   │   ├── local_scale1_410.0_415.0.csv
│   │   │   ├── local_scale1_412.5_417.5.csv
│   │   │   ├── local_scale1_415.0_420.0.csv
│   │   │   ├── local_scale1_417.5_422.5.csv
│   │   │   ├── local_scale1_42.5_47.5.csv
│   │   │   ├── local_scale1_420.0_425.0.csv
│   │   │   ├── local_scale1_422.5_427.5.csv
│   │   │   ├── local_scale1_45.0_50.0.csv
│   │   │   ├── local_scale1_47.5_52.5.csv
│   │   │   ├── local_scale1_5.0_10.0.csv
│   │   │   ├── local_scale1_50.0_55.0.csv
│   │   │   ├── local_scale1_50.0_70.0.csv
│   │   │   ├── local_scale1_52.5_57.5.csv
│   │   │   ├── local_scale1_55.0_60.0.csv
│   │   │   ├── local_scale1_57.5_62.5.csv
│   │   │   ├── local_scale1_60.0_65.0.csv
│   │   │   ├── local_scale1_62.5_67.5.csv
│   │   │   ├── local_scale1_65.0_70.0.csv
│   │   │   ├── local_scale1_67.5_72.5.csv
│   │   │   ├── local_scale1_7.5_12.5.csv
│   │   │   ├── local_scale1_75.0_80.0.csv
│   │   │   ├── local_scale1_77.5_82.5.csv
│   │   │   ├── local_scale1_80.0_85.0.csv
│   │   │   ├── local_scale1_82.5_87.5.csv
│   │   │   ├── local_scale1_85.0_90.0.csv
│   │   │   ├── local_scale1_87.5_92.5.csv
│   │   │   ├── local_scale1_90.0_95.0.csv
│   │   │   ├── local_scale1_92.5_97.5.csv
│   │   │   ├── local_scale1_95.0_100.0.csv
│   │   │   └── local_scale1_97.5_102.5.csv
│   │   ├── global_coords
│   │   │   ├── csv
│   │   │   │   ├── local_scale1_0.0_5.0.csv
│   │   │   │   ├── local_scale1_10.0_15.0.csv
│   │   │   │   ├── local_scale1_100.0_105.0.csv
│   │   │   │   ├── local_scale1_102.5_107.5.csv
│   │   │   │   ├── local_scale1_105.0_110.0.csv
│   │   │   │   ├── local_scale1_107.5_112.5.csv
│   │   │   │   ├── local_scale1_110.0_115.0.csv
│   │   │   │   ├── local_scale1_112.5_117.5.csv
│   │   │   │   ├── local_scale1_115.0_120.0.csv
│   │   │   │   ├── local_scale1_117.5_122.5.csv
│   │   │   │   ├── local_scale1_12.5_17.5.csv
│   │   │   │   ├── local_scale1_120.0_125.0.csv
│   │   │   │   ├── local_scale1_122.5_127.5.csv
│   │   │   │   ├── local_scale1_125.0_130.0.csv
│   │   │   │   ├── local_scale1_127.5_132.5.csv
│   │   │   │   ├── local_scale1_130.0_135.0.csv
│   │   │   │   ├── local_scale1_132.5_137.5.csv
│   │   │   │   ├── local_scale1_135.0_140.0.csv
│   │   │   │   ├── local_scale1_137.5_142.5.csv
│   │   │   │   ├── local_scale1_140.0_145.0.csv
│   │   │   │   ├── local_scale1_142.5_147.5.csv
│   │   │   │   ├── local_scale1_145.0_150.0.csv
│   │   │   │   ├── local_scale1_147.5_152.5.csv
│   │   │   │   ├── local_scale1_15.0_20.0.csv
│   │   │   │   ├── local_scale1_150.0_155.0.csv
│   │   │   │   ├── local_scale1_152.5_157.5.csv
│   │   │   │   ├── local_scale1_155.0_160.0.csv
│   │   │   │   ├── local_scale1_157.5_162.5.csv
│   │   │   │   ├── local_scale1_160.0_165.0.csv
│   │   │   │   ├── local_scale1_162.5_167.5.csv
│   │   │   │   ├── local_scale1_165.0_170.0.csv
│   │   │   │   ├── local_scale1_167.5_172.5.csv
│   │   │   │   ├── local_scale1_17.5_22.5.csv
│   │   │   │   ├── local_scale1_170.0_175.0.csv
│   │   │   │   ├── local_scale1_172.5_177.5.csv
│   │   │   │   ├── local_scale1_177.5_182.5.csv
│   │   │   │   ├── local_scale1_180.0_185.0.csv
│   │   │   │   ├── local_scale1_182.5_187.5.csv
│   │   │   │   ├── local_scale1_185.0_190.0.csv
│   │   │   │   ├── local_scale1_187.5_192.5.csv
│   │   │   │   ├── local_scale1_190.0_195.0.csv
│   │   │   │   ├── local_scale1_192.5_197.5.csv
│   │   │   │   ├── local_scale1_195.0_200.0.csv
│   │   │   │   ├── local_scale1_197.5_202.5.csv
│   │   │   │   ├── local_scale1_2.5_7.5.csv
│   │   │   │   ├── local_scale1_20.0_25.0.csv
│   │   │   │   ├── local_scale1_200.0_205.0.csv
│   │   │   │   ├── local_scale1_202.5_207.5.csv
│   │   │   │   ├── local_scale1_205.0_210.0.csv
│   │   │   │   ├── local_scale1_207.5_212.5.csv
│   │   │   │   ├── local_scale1_210.0_215.0.csv
│   │   │   │   ├── local_scale1_212.5_217.5.csv
│   │   │   │   ├── local_scale1_215.0_220.0.csv
│   │   │   │   ├── local_scale1_217.5_222.5.csv
│   │   │   │   ├── local_scale1_22.5_27.5.csv
│   │   │   │   ├── local_scale1_220.0_225.0.csv
│   │   │   │   ├── local_scale1_222.5_227.5.csv
│   │   │   │   ├── local_scale1_225.0_230.0.csv
│   │   │   │   ├── local_scale1_227.5_232.5.csv
│   │   │   │   ├── local_scale1_230.0_235.0.csv
│   │   │   │   ├── local_scale1_232.5_237.5.csv
│   │   │   │   ├── local_scale1_235.0_240.0.csv
│   │   │   │   ├── local_scale1_237.5_242.5.csv
│   │   │   │   ├── local_scale1_240.0_245.0.csv
│   │   │   │   ├── local_scale1_242.5_247.5.csv
│   │   │   │   ├── local_scale1_245.0_250.0.csv
│   │   │   │   ├── local_scale1_247.5_252.5.csv
│   │   │   │   ├── local_scale1_25.0_30.0.csv
│   │   │   │   ├── local_scale1_250.0_255.0.csv
│   │   │   │   ├── local_scale1_252.5_257.5.csv
│   │   │   │   ├── local_scale1_255.0_260.0.csv
│   │   │   │   ├── local_scale1_257.5_262.5.csv
│   │   │   │   ├── local_scale1_260.0_265.0.csv
│   │   │   │   ├── local_scale1_262.5_267.5.csv
│   │   │   │   ├── local_scale1_265.0_270.0.csv
│   │   │   │   ├── local_scale1_267.5_272.5.csv
│   │   │   │   ├── local_scale1_27.5_32.5.csv
│   │   │   │   ├── local_scale1_270.0_275.0.csv
│   │   │   │   ├── local_scale1_272.5_277.5.csv
│   │   │   │   ├── local_scale1_275.0_280.0.csv
│   │   │   │   ├── local_scale1_277.5_282.5.csv
│   │   │   │   ├── local_scale1_280.0_285.0.csv
│   │   │   │   ├── local_scale1_282.5_287.5.csv
│   │   │   │   ├── local_scale1_285.0_290.0.csv
│   │   │   │   ├── local_scale1_287.5_292.5.csv
│   │   │   │   ├── local_scale1_290.0_295.0.csv
│   │   │   │   ├── local_scale1_292.5_297.5.csv
│   │   │   │   ├── local_scale1_295.0_300.0.csv
│   │   │   │   ├── local_scale1_297.5_302.5.csv
│   │   │   │   ├── local_scale1_30.0_35.0.csv
│   │   │   │   ├── local_scale1_300.0_305.0.csv
│   │   │   │   ├── local_scale1_302.5_307.5.csv
│   │   │   │   ├── local_scale1_305.0_310.0.csv
│   │   │   │   ├── local_scale1_307.5_312.5.csv
│   │   │   │   ├── local_scale1_310.0_315.0.csv
│   │   │   │   ├── local_scale1_312.5_317.5.csv
│   │   │   │   ├── local_scale1_315.0_320.0.csv
│   │   │   │   ├── local_scale1_317.5_322.5.csv
│   │   │   │   ├── local_scale1_32.5_37.5.csv
│   │   │   │   ├── local_scale1_320.0_325.0.csv
│   │   │   │   ├── local_scale1_327.5_332.5.csv
│   │   │   │   ├── local_scale1_330.0_335.0.csv
│   │   │   │   ├── local_scale1_332.5_337.5.csv
│   │   │   │   ├── local_scale1_335.0_340.0.csv
│   │   │   │   ├── local_scale1_337.5_342.5.csv
│   │   │   │   ├── local_scale1_340.0_345.0.csv
│   │   │   │   ├── local_scale1_342.5_347.5.csv
│   │   │   │   ├── local_scale1_345.0_350.0.csv
│   │   │   │   ├── local_scale1_347.5_352.5.csv
│   │   │   │   ├── local_scale1_35.0_40.0.csv
│   │   │   │   ├── local_scale1_350.0_355.0.csv
│   │   │   │   ├── local_scale1_352.5_357.5.csv
│   │   │   │   ├── local_scale1_355.0_360.0.csv
│   │   │   │   ├── local_scale1_357.5_362.5.csv
│   │   │   │   ├── local_scale1_360.0_365.0.csv
│   │   │   │   ├── local_scale1_362.5_367.5.csv
│   │   │   │   ├── local_scale1_365.0_370.0.csv
│   │   │   │   ├── local_scale1_367.5_372.5.csv
│   │   │   │   ├── local_scale1_37.5_42.5.csv
│   │   │   │   ├── local_scale1_370.0_375.0.csv
│   │   │   │   ├── local_scale1_372.5_377.5.csv
│   │   │   │   ├── local_scale1_375.0_380.0.csv
│   │   │   │   ├── local_scale1_377.5_382.5.csv
│   │   │   │   ├── local_scale1_380.0_385.0.csv
│   │   │   │   ├── local_scale1_382.5_387.5.csv
│   │   │   │   ├── local_scale1_385.0_390.0.csv
│   │   │   │   ├── local_scale1_387.5_392.5.csv
│   │   │   │   ├── local_scale1_390.0_395.0.csv
│   │   │   │   ├── local_scale1_392.5_397.5.csv
│   │   │   │   ├── local_scale1_395.0_400.0.csv
│   │   │   │   ├── local_scale1_397.5_402.5.csv
│   │   │   │   ├── local_scale1_40.0_45.0.csv
│   │   │   │   ├── local_scale1_400.0_405.0.csv
│   │   │   │   ├── local_scale1_402.5_407.5.csv
│   │   │   │   ├── local_scale1_405.0_410.0.csv
│   │   │   │   ├── local_scale1_407.5_412.5.csv
│   │   │   │   ├── local_scale1_410.0_415.0.csv
│   │   │   │   ├── local_scale1_412.5_417.5.csv
│   │   │   │   ├── local_scale1_415.0_420.0.csv
│   │   │   │   ├── local_scale1_417.5_422.5.csv
│   │   │   │   ├── local_scale1_42.5_47.5.csv
│   │   │   │   ├── local_scale1_420.0_425.0.csv
│   │   │   │   ├── local_scale1_422.5_427.5.csv
│   │   │   │   ├── local_scale1_45.0_50.0.csv
│   │   │   │   ├── local_scale1_47.5_52.5.csv
│   │   │   │   ├── local_scale1_5.0_10.0.csv
│   │   │   │   ├── local_scale1_50.0_55.0.csv
│   │   │   │   ├── local_scale1_52.5_57.5.csv
│   │   │   │   ├── local_scale1_55.0_60.0.csv
│   │   │   │   ├── local_scale1_57.5_62.5.csv
│   │   │   │   ├── local_scale1_60.0_65.0.csv
│   │   │   │   ├── local_scale1_62.5_67.5.csv
│   │   │   │   ├── local_scale1_65.0_70.0.csv
│   │   │   │   ├── local_scale1_67.5_72.5.csv
│   │   │   │   ├── local_scale1_7.5_12.5.csv
│   │   │   │   ├── local_scale1_75.0_80.0.csv
│   │   │   │   ├── local_scale1_77.5_82.5.csv
│   │   │   │   ├── local_scale1_80.0_85.0.csv
│   │   │   │   ├── local_scale1_82.5_87.5.csv
│   │   │   │   ├── local_scale1_85.0_90.0.csv
│   │   │   │   ├── local_scale1_87.5_92.5.csv
│   │   │   │   ├── local_scale1_90.0_95.0.csv
│   │   │   │   ├── local_scale1_92.5_97.5.csv
│   │   │   │   ├── local_scale1_95.0_100.0.csv
│   │   │   │   └── local_scale1_97.5_102.5.csv
│   │   │   ├── metrics
│   │   │   │   ├── local_scale1_0.0_5.0.json
│   │   │   │   ├── local_scale1_10.0_15.0.json
│   │   │   │   ├── local_scale1_100.0_105.0.json
│   │   │   │   ├── local_scale1_102.5_107.5.json
│   │   │   │   ├── local_scale1_105.0_110.0.json
│   │   │   │   ├── local_scale1_107.5_112.5.json
│   │   │   │   ├── local_scale1_110.0_115.0.json
│   │   │   │   ├── local_scale1_112.5_117.5.json
│   │   │   │   ├── local_scale1_115.0_120.0.json
│   │   │   │   ├── local_scale1_117.5_122.5.json
│   │   │   │   ├── local_scale1_12.5_17.5.json
│   │   │   │   ├── local_scale1_120.0_125.0.json
│   │   │   │   ├── local_scale1_122.5_127.5.json
│   │   │   │   ├── local_scale1_125.0_130.0.json
│   │   │   │   ├── local_scale1_127.5_132.5.json
│   │   │   │   ├── local_scale1_130.0_135.0.json
│   │   │   │   ├── local_scale1_132.5_137.5.json
│   │   │   │   ├── local_scale1_135.0_140.0.json
│   │   │   │   ├── local_scale1_137.5_142.5.json
│   │   │   │   ├── local_scale1_140.0_145.0.json
│   │   │   │   ├── local_scale1_142.5_147.5.json
│   │   │   │   ├── local_scale1_145.0_150.0.json
│   │   │   │   ├── local_scale1_147.5_152.5.json
│   │   │   │   ├── local_scale1_15.0_20.0.json
│   │   │   │   ├── local_scale1_150.0_155.0.json
│   │   │   │   ├── local_scale1_152.5_157.5.json
│   │   │   │   ├── local_scale1_155.0_160.0.json
│   │   │   │   ├── local_scale1_157.5_162.5.json
│   │   │   │   ├── local_scale1_160.0_165.0.json
│   │   │   │   ├── local_scale1_162.5_167.5.json
│   │   │   │   ├── local_scale1_165.0_170.0.json
│   │   │   │   ├── local_scale1_167.5_172.5.json
│   │   │   │   ├── local_scale1_17.5_22.5.json
│   │   │   │   ├── local_scale1_170.0_175.0.json
│   │   │   │   ├── local_scale1_172.5_177.5.json
│   │   │   │   ├── local_scale1_177.5_182.5.json
│   │   │   │   ├── local_scale1_180.0_185.0.json
│   │   │   │   ├── local_scale1_182.5_187.5.json
│   │   │   │   ├── local_scale1_185.0_190.0.json
│   │   │   │   ├── local_scale1_187.5_192.5.json
│   │   │   │   ├── local_scale1_190.0_195.0.json
│   │   │   │   ├── local_scale1_192.5_197.5.json
│   │   │   │   ├── local_scale1_195.0_200.0.json
│   │   │   │   ├── local_scale1_197.5_202.5.json
│   │   │   │   ├── local_scale1_2.5_7.5.json
│   │   │   │   ├── local_scale1_20.0_25.0.json
│   │   │   │   ├── local_scale1_200.0_205.0.json
│   │   │   │   ├── local_scale1_202.5_207.5.json
│   │   │   │   ├── local_scale1_205.0_210.0.json
│   │   │   │   ├── local_scale1_207.5_212.5.json
│   │   │   │   ├── local_scale1_210.0_215.0.json
│   │   │   │   ├── local_scale1_212.5_217.5.json
│   │   │   │   ├── local_scale1_215.0_220.0.json
│   │   │   │   ├── local_scale1_217.5_222.5.json
│   │   │   │   ├── local_scale1_22.5_27.5.json
│   │   │   │   ├── local_scale1_220.0_225.0.json
│   │   │   │   ├── local_scale1_222.5_227.5.json
│   │   │   │   ├── local_scale1_225.0_230.0.json
│   │   │   │   ├── local_scale1_227.5_232.5.json
│   │   │   │   ├── local_scale1_230.0_235.0.json
│   │   │   │   ├── local_scale1_232.5_237.5.json
│   │   │   │   ├── local_scale1_235.0_240.0.json
│   │   │   │   ├── local_scale1_237.5_242.5.json
│   │   │   │   ├── local_scale1_240.0_245.0.json
│   │   │   │   ├── local_scale1_242.5_247.5.json
│   │   │   │   ├── local_scale1_245.0_250.0.json
│   │   │   │   ├── local_scale1_247.5_252.5.json
│   │   │   │   ├── local_scale1_25.0_30.0.json
│   │   │   │   ├── local_scale1_250.0_255.0.json
│   │   │   │   ├── local_scale1_252.5_257.5.json
│   │   │   │   ├── local_scale1_255.0_260.0.json
│   │   │   │   ├── local_scale1_257.5_262.5.json
│   │   │   │   ├── local_scale1_260.0_265.0.json
│   │   │   │   ├── local_scale1_262.5_267.5.json
│   │   │   │   ├── local_scale1_265.0_270.0.json
│   │   │   │   ├── local_scale1_267.5_272.5.json
│   │   │   │   ├── local_scale1_27.5_32.5.json
│   │   │   │   ├── local_scale1_270.0_275.0.json
│   │   │   │   ├── local_scale1_272.5_277.5.json
│   │   │   │   ├── local_scale1_275.0_280.0.json
│   │   │   │   ├── local_scale1_277.5_282.5.json
│   │   │   │   ├── local_scale1_280.0_285.0.json
│   │   │   │   ├── local_scale1_282.5_287.5.json
│   │   │   │   ├── local_scale1_285.0_290.0.json
│   │   │   │   ├── local_scale1_287.5_292.5.json
│   │   │   │   ├── local_scale1_290.0_295.0.json
│   │   │   │   ├── local_scale1_292.5_297.5.json
│   │   │   │   ├── local_scale1_295.0_300.0.json
│   │   │   │   ├── local_scale1_297.5_302.5.json
│   │   │   │   ├── local_scale1_30.0_35.0.json
│   │   │   │   ├── local_scale1_300.0_305.0.json
│   │   │   │   ├── local_scale1_302.5_307.5.json
│   │   │   │   ├── local_scale1_305.0_310.0.json
│   │   │   │   ├── local_scale1_307.5_312.5.json
│   │   │   │   ├── local_scale1_310.0_315.0.json
│   │   │   │   ├── local_scale1_312.5_317.5.json
│   │   │   │   ├── local_scale1_315.0_320.0.json
│   │   │   │   ├── local_scale1_317.5_322.5.json
│   │   │   │   ├── local_scale1_32.5_37.5.json
│   │   │   │   ├── local_scale1_320.0_325.0.json
│   │   │   │   ├── local_scale1_327.5_332.5.json
│   │   │   │   ├── local_scale1_330.0_335.0.json
│   │   │   │   ├── local_scale1_332.5_337.5.json
│   │   │   │   ├── local_scale1_335.0_340.0.json
│   │   │   │   ├── local_scale1_337.5_342.5.json
│   │   │   │   ├── local_scale1_340.0_345.0.json
│   │   │   │   ├── local_scale1_342.5_347.5.json
│   │   │   │   ├── local_scale1_345.0_350.0.json
│   │   │   │   ├── local_scale1_347.5_352.5.json
│   │   │   │   ├── local_scale1_35.0_40.0.json
│   │   │   │   ├── local_scale1_350.0_355.0.json
│   │   │   │   ├── local_scale1_352.5_357.5.json
│   │   │   │   ├── local_scale1_355.0_360.0.json
│   │   │   │   ├── local_scale1_357.5_362.5.json
│   │   │   │   ├── local_scale1_360.0_365.0.json
│   │   │   │   ├── local_scale1_362.5_367.5.json
│   │   │   │   ├── local_scale1_365.0_370.0.json
│   │   │   │   ├── local_scale1_367.5_372.5.json
│   │   │   │   ├── local_scale1_37.5_42.5.json
│   │   │   │   ├── local_scale1_370.0_375.0.json
│   │   │   │   ├── local_scale1_372.5_377.5.json
│   │   │   │   ├── local_scale1_375.0_380.0.json
│   │   │   │   ├── local_scale1_377.5_382.5.json
│   │   │   │   ├── local_scale1_380.0_385.0.json
│   │   │   │   ├── local_scale1_382.5_387.5.json
│   │   │   │   ├── local_scale1_385.0_390.0.json
│   │   │   │   ├── local_scale1_387.5_392.5.json
│   │   │   │   ├── local_scale1_390.0_395.0.json
│   │   │   │   ├── local_scale1_392.5_397.5.json
│   │   │   │   ├── local_scale1_395.0_400.0.json
│   │   │   │   ├── local_scale1_397.5_402.5.json
│   │   │   │   ├── local_scale1_40.0_45.0.json
│   │   │   │   ├── local_scale1_400.0_405.0.json
│   │   │   │   ├── local_scale1_402.5_407.5.json
│   │   │   │   ├── local_scale1_405.0_410.0.json
│   │   │   │   ├── local_scale1_407.5_412.5.json
│   │   │   │   ├── local_scale1_410.0_415.0.json
│   │   │   │   ├── local_scale1_412.5_417.5.json
│   │   │   │   ├── local_scale1_415.0_420.0.json
│   │   │   │   ├── local_scale1_417.5_422.5.json
│   │   │   │   ├── local_scale1_42.5_47.5.json
│   │   │   │   ├── local_scale1_420.0_425.0.json
│   │   │   │   ├── local_scale1_422.5_427.5.json
│   │   │   │   ├── local_scale1_45.0_50.0.json
│   │   │   │   ├── local_scale1_47.5_52.5.json
│   │   │   │   ├── local_scale1_5.0_10.0.json
│   │   │   │   ├── local_scale1_50.0_55.0.json
│   │   │   │   ├── local_scale1_52.5_57.5.json
│   │   │   │   ├── local_scale1_55.0_60.0.json
│   │   │   │   ├── local_scale1_57.5_62.5.json
│   │   │   │   ├── local_scale1_60.0_65.0.json
│   │   │   │   ├── local_scale1_62.5_67.5.json
│   │   │   │   ├── local_scale1_65.0_70.0.json
│   │   │   │   ├── local_scale1_67.5_72.5.json
│   │   │   │   ├── local_scale1_7.5_12.5.json
│   │   │   │   ├── local_scale1_75.0_80.0.json
│   │   │   │   ├── local_scale1_77.5_82.5.json
│   │   │   │   ├── local_scale1_80.0_85.0.json
│   │   │   │   ├── local_scale1_82.5_87.5.json
│   │   │   │   ├── local_scale1_85.0_90.0.json
│   │   │   │   ├── local_scale1_87.5_92.5.json
│   │   │   │   ├── local_scale1_90.0_95.0.json
│   │   │   │   ├── local_scale1_92.5_97.5.json
│   │   │   │   ├── local_scale1_95.0_100.0.json
│   │   │   │   └── local_scale1_97.5_102.5.json
│   │   │   ├── pca
│   │   │   │   ├── local_scale1_0.0_5.0_pca.png
│   │   │   │   ├── local_scale1_10.0_15.0_pca.png
│   │   │   │   ├── local_scale1_100.0_105.0_pca.png
│   │   │   │   ├── local_scale1_102.5_107.5_pca.png
│   │   │   │   ├── local_scale1_105.0_110.0_pca.png
│   │   │   │   ├── local_scale1_107.5_112.5_pca.png
│   │   │   │   ├── local_scale1_110.0_115.0_pca.png
│   │   │   │   ├── local_scale1_112.5_117.5_pca.png
│   │   │   │   ├── local_scale1_115.0_120.0_pca.png
│   │   │   │   ├── local_scale1_117.5_122.5_pca.png
│   │   │   │   ├── local_scale1_12.5_17.5_pca.png
│   │   │   │   ├── local_scale1_120.0_125.0_pca.png
│   │   │   │   ├── local_scale1_122.5_127.5_pca.png
│   │   │   │   ├── local_scale1_125.0_130.0_pca.png
│   │   │   │   ├── local_scale1_127.5_132.5_pca.png
│   │   │   │   ├── local_scale1_130.0_135.0_pca.png
│   │   │   │   ├── local_scale1_132.5_137.5_pca.png
│   │   │   │   ├── local_scale1_135.0_140.0_pca.png
│   │   │   │   ├── local_scale1_137.5_142.5_pca.png
│   │   │   │   ├── local_scale1_140.0_145.0_pca.png
│   │   │   │   ├── local_scale1_142.5_147.5_pca.png
│   │   │   │   ├── local_scale1_145.0_150.0_pca.png
│   │   │   │   ├── local_scale1_147.5_152.5_pca.png
│   │   │   │   ├── local_scale1_15.0_20.0_pca.png
│   │   │   │   ├── local_scale1_150.0_155.0_pca.png
│   │   │   │   ├── local_scale1_152.5_157.5_pca.png
│   │   │   │   ├── local_scale1_155.0_160.0_pca.png
│   │   │   │   ├── local_scale1_157.5_162.5_pca.png
│   │   │   │   ├── local_scale1_160.0_165.0_pca.png
│   │   │   │   ├── local_scale1_162.5_167.5_pca.png
│   │   │   │   ├── local_scale1_165.0_170.0_pca.png
│   │   │   │   ├── local_scale1_167.5_172.5_pca.png
│   │   │   │   ├── local_scale1_17.5_22.5_pca.png
│   │   │   │   ├── local_scale1_170.0_175.0_pca.png
│   │   │   │   ├── local_scale1_172.5_177.5_pca.png
│   │   │   │   ├── local_scale1_177.5_182.5_pca.png
│   │   │   │   ├── local_scale1_180.0_185.0_pca.png
│   │   │   │   ├── local_scale1_182.5_187.5_pca.png
│   │   │   │   ├── local_scale1_185.0_190.0_pca.png
│   │   │   │   ├── local_scale1_187.5_192.5_pca.png
│   │   │   │   ├── local_scale1_190.0_195.0_pca.png
│   │   │   │   ├── local_scale1_192.5_197.5_pca.png
│   │   │   │   ├── local_scale1_195.0_200.0_pca.png
│   │   │   │   ├── local_scale1_197.5_202.5_pca.png
│   │   │   │   ├── local_scale1_2.5_7.5_pca.png
│   │   │   │   ├── local_scale1_20.0_25.0_pca.png
│   │   │   │   ├── local_scale1_200.0_205.0_pca.png
│   │   │   │   ├── local_scale1_202.5_207.5_pca.png
│   │   │   │   ├── local_scale1_205.0_210.0_pca.png
│   │   │   │   ├── local_scale1_207.5_212.5_pca.png
│   │   │   │   ├── local_scale1_210.0_215.0_pca.png
│   │   │   │   ├── local_scale1_212.5_217.5_pca.png
│   │   │   │   ├── local_scale1_215.0_220.0_pca.png
│   │   │   │   ├── local_scale1_217.5_222.5_pca.png
│   │   │   │   ├── local_scale1_22.5_27.5_pca.png
│   │   │   │   ├── local_scale1_220.0_225.0_pca.png
│   │   │   │   ├── local_scale1_222.5_227.5_pca.png
│   │   │   │   ├── local_scale1_225.0_230.0_pca.png
│   │   │   │   ├── local_scale1_227.5_232.5_pca.png
│   │   │   │   ├── local_scale1_230.0_235.0_pca.png
│   │   │   │   ├── local_scale1_232.5_237.5_pca.png
│   │   │   │   ├── local_scale1_235.0_240.0_pca.png
│   │   │   │   ├── local_scale1_237.5_242.5_pca.png
│   │   │   │   ├── local_scale1_240.0_245.0_pca.png
│   │   │   │   ├── local_scale1_242.5_247.5_pca.png
│   │   │   │   ├── local_scale1_245.0_250.0_pca.png
│   │   │   │   ├── local_scale1_247.5_252.5_pca.png
│   │   │   │   ├── local_scale1_25.0_30.0_pca.png
│   │   │   │   ├── local_scale1_250.0_255.0_pca.png
│   │   │   │   ├── local_scale1_252.5_257.5_pca.png
│   │   │   │   ├── local_scale1_255.0_260.0_pca.png
│   │   │   │   ├── local_scale1_257.5_262.5_pca.png
│   │   │   │   ├── local_scale1_260.0_265.0_pca.png
│   │   │   │   ├── local_scale1_262.5_267.5_pca.png
│   │   │   │   ├── local_scale1_265.0_270.0_pca.png
│   │   │   │   ├── local_scale1_267.5_272.5_pca.png
│   │   │   │   ├── local_scale1_27.5_32.5_pca.png
│   │   │   │   ├── local_scale1_270.0_275.0_pca.png
│   │   │   │   ├── local_scale1_272.5_277.5_pca.png
│   │   │   │   ├── local_scale1_275.0_280.0_pca.png
│   │   │   │   ├── local_scale1_277.5_282.5_pca.png
│   │   │   │   ├── local_scale1_280.0_285.0_pca.png
│   │   │   │   ├── local_scale1_282.5_287.5_pca.png
│   │   │   │   ├── local_scale1_285.0_290.0_pca.png
│   │   │   │   ├── local_scale1_287.5_292.5_pca.png
│   │   │   │   ├── local_scale1_290.0_295.0_pca.png
│   │   │   │   ├── local_scale1_292.5_297.5_pca.png
│   │   │   │   ├── local_scale1_295.0_300.0_pca.png
│   │   │   │   ├── local_scale1_297.5_302.5_pca.png
│   │   │   │   ├── local_scale1_30.0_35.0_pca.png
│   │   │   │   ├── local_scale1_300.0_305.0_pca.png
│   │   │   │   ├── local_scale1_302.5_307.5_pca.png
│   │   │   │   ├── local_scale1_305.0_310.0_pca.png
│   │   │   │   ├── local_scale1_307.5_312.5_pca.png
│   │   │   │   ├── local_scale1_310.0_315.0_pca.png
│   │   │   │   ├── local_scale1_312.5_317.5_pca.png
│   │   │   │   ├── local_scale1_315.0_320.0_pca.png
│   │   │   │   ├── local_scale1_317.5_322.5_pca.png
│   │   │   │   ├── local_scale1_32.5_37.5_pca.png
│   │   │   │   ├── local_scale1_320.0_325.0_pca.png
│   │   │   │   ├── local_scale1_327.5_332.5_pca.png
│   │   │   │   ├── local_scale1_330.0_335.0_pca.png
│   │   │   │   ├── local_scale1_332.5_337.5_pca.png
│   │   │   │   ├── local_scale1_335.0_340.0_pca.png
│   │   │   │   ├── local_scale1_337.5_342.5_pca.png
│   │   │   │   ├── local_scale1_340.0_345.0_pca.png
│   │   │   │   ├── local_scale1_342.5_347.5_pca.png
│   │   │   │   ├── local_scale1_345.0_350.0_pca.png
│   │   │   │   ├── local_scale1_347.5_352.5_pca.png
│   │   │   │   ├── local_scale1_35.0_40.0_pca.png
│   │   │   │   ├── local_scale1_350.0_355.0_pca.png
│   │   │   │   ├── local_scale1_352.5_357.5_pca.png
│   │   │   │   ├── local_scale1_355.0_360.0_pca.png
│   │   │   │   ├── local_scale1_357.5_362.5_pca.png
│   │   │   │   ├── local_scale1_360.0_365.0_pca.png
│   │   │   │   ├── local_scale1_362.5_367.5_pca.png
│   │   │   │   ├── local_scale1_365.0_370.0_pca.png
│   │   │   │   ├── local_scale1_367.5_372.5_pca.png
│   │   │   │   ├── local_scale1_37.5_42.5_pca.png
│   │   │   │   ├── local_scale1_370.0_375.0_pca.png
│   │   │   │   ├── local_scale1_372.5_377.5_pca.png
│   │   │   │   ├── local_scale1_375.0_380.0_pca.png
│   │   │   │   ├── local_scale1_377.5_382.5_pca.png
│   │   │   │   ├── local_scale1_380.0_385.0_pca.png
│   │   │   │   ├── local_scale1_382.5_387.5_pca.png
│   │   │   │   ├── local_scale1_385.0_390.0_pca.png
│   │   │   │   ├── local_scale1_387.5_392.5_pca.png
│   │   │   │   ├── local_scale1_390.0_395.0_pca.png
│   │   │   │   ├── local_scale1_392.5_397.5_pca.png
│   │   │   │   ├── local_scale1_395.0_400.0_pca.png
│   │   │   │   ├── local_scale1_397.5_402.5_pca.png
│   │   │   │   ├── local_scale1_40.0_45.0_pca.png
│   │   │   │   ├── local_scale1_400.0_405.0_pca.png
│   │   │   │   ├── local_scale1_402.5_407.5_pca.png
│   │   │   │   ├── local_scale1_405.0_410.0_pca.png
│   │   │   │   ├── local_scale1_407.5_412.5_pca.png
│   │   │   │   ├── local_scale1_410.0_415.0_pca.png
│   │   │   │   ├── local_scale1_412.5_417.5_pca.png
│   │   │   │   ├── local_scale1_415.0_420.0_pca.png
│   │   │   │   ├── local_scale1_417.5_422.5_pca.png
│   │   │   │   ├── local_scale1_42.5_47.5_pca.png
│   │   │   │   ├── local_scale1_420.0_425.0_pca.png
│   │   │   │   ├── local_scale1_422.5_427.5_pca.png
│   │   │   │   ├── local_scale1_45.0_50.0_pca.png
│   │   │   │   ├── local_scale1_47.5_52.5_pca.png
│   │   │   │   ├── local_scale1_5.0_10.0_pca.png
│   │   │   │   ├── local_scale1_50.0_55.0_pca.png
│   │   │   │   ├── local_scale1_52.5_57.5_pca.png
│   │   │   │   ├── local_scale1_55.0_60.0_pca.png
│   │   │   │   ├── local_scale1_57.5_62.5_pca.png
│   │   │   │   ├── local_scale1_60.0_65.0_pca.png
│   │   │   │   ├── local_scale1_62.5_67.5_pca.png
│   │   │   │   ├── local_scale1_65.0_70.0_pca.png
│   │   │   │   ├── local_scale1_67.5_72.5_pca.png
│   │   │   │   ├── local_scale1_7.5_12.5_pca.png
│   │   │   │   ├── local_scale1_75.0_80.0_pca.png
│   │   │   │   ├── local_scale1_77.5_82.5_pca.png
│   │   │   │   ├── local_scale1_80.0_85.0_pca.png
│   │   │   │   ├── local_scale1_82.5_87.5_pca.png
│   │   │   │   ├── local_scale1_85.0_90.0_pca.png
│   │   │   │   ├── local_scale1_87.5_92.5_pca.png
│   │   │   │   ├── local_scale1_90.0_95.0_pca.png
│   │   │   │   ├── local_scale1_92.5_97.5_pca.png
│   │   │   │   ├── local_scale1_95.0_100.0_pca.png
│   │   │   │   └── local_scale1_97.5_102.5_pca.png
│   │   │   ├── reports
│   │   │   │   ├── all_windows_metrics.csv
│   │   │   │   └── local_analysis_summary.md
│   │   │   ├── trajectories
│   │   │   └── umap
│   │   │       ├── local_scale1_0.0_5.0_umap.png
│   │   │       ├── local_scale1_10.0_15.0_umap.png
│   │   │       ├── local_scale1_100.0_105.0_umap.png
│   │   │       ├── local_scale1_102.5_107.5_umap.png
│   │   │       ├── local_scale1_105.0_110.0_umap.png
│   │   │       ├── local_scale1_107.5_112.5_umap.png
│   │   │       ├── local_scale1_110.0_115.0_umap.png
│   │   │       ├── local_scale1_112.5_117.5_umap.png
│   │   │       ├── local_scale1_115.0_120.0_umap.png
│   │   │       ├── local_scale1_117.5_122.5_umap.png
│   │   │       ├── local_scale1_12.5_17.5_umap.png
│   │   │       ├── local_scale1_120.0_125.0_umap.png
│   │   │       ├── local_scale1_122.5_127.5_umap.png
│   │   │       ├── local_scale1_125.0_130.0_umap.png
│   │   │       ├── local_scale1_127.5_132.5_umap.png
│   │   │       ├── local_scale1_130.0_135.0_umap.png
│   │   │       ├── local_scale1_132.5_137.5_umap.png
│   │   │       ├── local_scale1_135.0_140.0_umap.png
│   │   │       ├── local_scale1_137.5_142.5_umap.png
│   │   │       ├── local_scale1_140.0_145.0_umap.png
│   │   │       ├── local_scale1_142.5_147.5_umap.png
│   │   │       ├── local_scale1_145.0_150.0_umap.png
│   │   │       ├── local_scale1_147.5_152.5_umap.png
│   │   │       ├── local_scale1_15.0_20.0_umap.png
│   │   │       ├── local_scale1_150.0_155.0_umap.png
│   │   │       ├── local_scale1_152.5_157.5_umap.png
│   │   │       ├── local_scale1_155.0_160.0_umap.png
│   │   │       ├── local_scale1_157.5_162.5_umap.png
│   │   │       ├── local_scale1_160.0_165.0_umap.png
│   │   │       ├── local_scale1_162.5_167.5_umap.png
│   │   │       ├── local_scale1_165.0_170.0_umap.png
│   │   │       ├── local_scale1_167.5_172.5_umap.png
│   │   │       ├── local_scale1_17.5_22.5_umap.png
│   │   │       ├── local_scale1_170.0_175.0_umap.png
│   │   │       ├── local_scale1_172.5_177.5_umap.png
│   │   │       ├── local_scale1_177.5_182.5_umap.png
│   │   │       ├── local_scale1_180.0_185.0_umap.png
│   │   │       ├── local_scale1_182.5_187.5_umap.png
│   │   │       ├── local_scale1_185.0_190.0_umap.png
│   │   │       ├── local_scale1_187.5_192.5_umap.png
│   │   │       ├── local_scale1_190.0_195.0_umap.png
│   │   │       ├── local_scale1_192.5_197.5_umap.png
│   │   │       ├── local_scale1_195.0_200.0_umap.png
│   │   │       ├── local_scale1_197.5_202.5_umap.png
│   │   │       ├── local_scale1_2.5_7.5_umap.png
│   │   │       ├── local_scale1_20.0_25.0_umap.png
│   │   │       ├── local_scale1_200.0_205.0_umap.png
│   │   │       ├── local_scale1_202.5_207.5_umap.png
│   │   │       ├── local_scale1_205.0_210.0_umap.png
│   │   │       ├── local_scale1_207.5_212.5_umap.png
│   │   │       ├── local_scale1_210.0_215.0_umap.png
│   │   │       ├── local_scale1_212.5_217.5_umap.png
│   │   │       ├── local_scale1_215.0_220.0_umap.png
│   │   │       ├── local_scale1_217.5_222.5_umap.png
│   │   │       ├── local_scale1_22.5_27.5_umap.png
│   │   │       ├── local_scale1_220.0_225.0_umap.png
│   │   │       ├── local_scale1_222.5_227.5_umap.png
│   │   │       ├── local_scale1_225.0_230.0_umap.png
│   │   │       ├── local_scale1_227.5_232.5_umap.png
│   │   │       ├── local_scale1_230.0_235.0_umap.png
│   │   │       ├── local_scale1_232.5_237.5_umap.png
│   │   │       ├── local_scale1_235.0_240.0_umap.png
│   │   │       ├── local_scale1_237.5_242.5_umap.png
│   │   │       ├── local_scale1_240.0_245.0_umap.png
│   │   │       ├── local_scale1_242.5_247.5_umap.png
│   │   │       ├── local_scale1_245.0_250.0_umap.png
│   │   │       ├── local_scale1_247.5_252.5_umap.png
│   │   │       ├── local_scale1_25.0_30.0_umap.png
│   │   │       ├── local_scale1_250.0_255.0_umap.png
│   │   │       ├── local_scale1_252.5_257.5_umap.png
│   │   │       ├── local_scale1_255.0_260.0_umap.png
│   │   │       ├── local_scale1_257.5_262.5_umap.png
│   │   │       ├── local_scale1_260.0_265.0_umap.png
│   │   │       ├── local_scale1_262.5_267.5_umap.png
│   │   │       ├── local_scale1_265.0_270.0_umap.png
│   │   │       ├── local_scale1_267.5_272.5_umap.png
│   │   │       ├── local_scale1_27.5_32.5_umap.png
│   │   │       ├── local_scale1_270.0_275.0_umap.png
│   │   │       ├── local_scale1_272.5_277.5_umap.png
│   │   │       ├── local_scale1_275.0_280.0_umap.png
│   │   │       ├── local_scale1_277.5_282.5_umap.png
│   │   │       ├── local_scale1_280.0_285.0_umap.png
│   │   │       ├── local_scale1_282.5_287.5_umap.png
│   │   │       ├── local_scale1_285.0_290.0_umap.png
│   │   │       ├── local_scale1_287.5_292.5_umap.png
│   │   │       ├── local_scale1_290.0_295.0_umap.png
│   │   │       ├── local_scale1_292.5_297.5_umap.png
│   │   │       ├── local_scale1_295.0_300.0_umap.png
│   │   │       ├── local_scale1_297.5_302.5_umap.png
│   │   │       ├── local_scale1_30.0_35.0_umap.png
│   │   │       ├── local_scale1_300.0_305.0_umap.png
│   │   │       ├── local_scale1_302.5_307.5_umap.png
│   │   │       ├── local_scale1_305.0_310.0_umap.png
│   │   │       ├── local_scale1_307.5_312.5_umap.png
│   │   │       ├── local_scale1_310.0_315.0_umap.png
│   │   │       ├── local_scale1_312.5_317.5_umap.png
│   │   │       ├── local_scale1_315.0_320.0_umap.png
│   │   │       ├── local_scale1_317.5_322.5_umap.png
│   │   │       ├── local_scale1_32.5_37.5_umap.png
│   │   │       ├── local_scale1_320.0_325.0_umap.png
│   │   │       ├── local_scale1_327.5_332.5_umap.png
│   │   │       ├── local_scale1_330.0_335.0_umap.png
│   │   │       ├── local_scale1_332.5_337.5_umap.png
│   │   │       ├── local_scale1_335.0_340.0_umap.png
│   │   │       ├── local_scale1_337.5_342.5_umap.png
│   │   │       ├── local_scale1_340.0_345.0_umap.png
│   │   │       ├── local_scale1_342.5_347.5_umap.png
│   │   │       ├── local_scale1_345.0_350.0_umap.png
│   │   │       ├── local_scale1_347.5_352.5_umap.png
│   │   │       ├── local_scale1_35.0_40.0_umap.png
│   │   │       ├── local_scale1_350.0_355.0_umap.png
│   │   │       ├── local_scale1_352.5_357.5_umap.png
│   │   │       ├── local_scale1_355.0_360.0_umap.png
│   │   │       ├── local_scale1_357.5_362.5_umap.png
│   │   │       ├── local_scale1_360.0_365.0_umap.png
│   │   │       ├── local_scale1_362.5_367.5_umap.png
│   │   │       ├── local_scale1_365.0_370.0_umap.png
│   │   │       ├── local_scale1_367.5_372.5_umap.png
│   │   │       ├── local_scale1_37.5_42.5_umap.png
│   │   │       ├── local_scale1_370.0_375.0_umap.png
│   │   │       ├── local_scale1_372.5_377.5_umap.png
│   │   │       ├── local_scale1_375.0_380.0_umap.png
│   │   │       ├── local_scale1_377.5_382.5_umap.png
│   │   │       ├── local_scale1_380.0_385.0_umap.png
│   │   │       ├── local_scale1_382.5_387.5_umap.png
│   │   │       ├── local_scale1_385.0_390.0_umap.png
│   │   │       ├── local_scale1_387.5_392.5_umap.png
│   │   │       ├── local_scale1_390.0_395.0_umap.png
│   │   │       ├── local_scale1_392.5_397.5_umap.png
│   │   │       ├── local_scale1_395.0_400.0_umap.png
│   │   │       ├── local_scale1_397.5_402.5_umap.png
│   │   │       ├── local_scale1_40.0_45.0_umap.png
│   │   │       ├── local_scale1_400.0_405.0_umap.png
│   │   │       ├── local_scale1_402.5_407.5_umap.png
│   │   │       ├── local_scale1_405.0_410.0_umap.png
│   │   │       ├── local_scale1_407.5_412.5_umap.png
│   │   │       ├── local_scale1_410.0_415.0_umap.png
│   │   │       ├── local_scale1_412.5_417.5_umap.png
│   │   │       ├── local_scale1_415.0_420.0_umap.png
│   │   │       ├── local_scale1_417.5_422.5_umap.png
│   │   │       ├── local_scale1_42.5_47.5_umap.png
│   │   │       ├── local_scale1_420.0_425.0_umap.png
│   │   │       ├── local_scale1_422.5_427.5_umap.png
│   │   │       ├── local_scale1_45.0_50.0_umap.png
│   │   │       ├── local_scale1_47.5_52.5_umap.png
│   │   │       ├── local_scale1_5.0_10.0_umap.png
│   │   │       ├── local_scale1_50.0_55.0_umap.png
│   │   │       ├── local_scale1_52.5_57.5_umap.png
│   │   │       ├── local_scale1_55.0_60.0_umap.png
│   │   │       ├── local_scale1_57.5_62.5_umap.png
│   │   │       ├── local_scale1_60.0_65.0_umap.png
│   │   │       ├── local_scale1_62.5_67.5_umap.png
│   │   │       ├── local_scale1_65.0_70.0_umap.png
│   │   │       ├── local_scale1_67.5_72.5_umap.png
│   │   │       ├── local_scale1_7.5_12.5_umap.png
│   │   │       ├── local_scale1_75.0_80.0_umap.png
│   │   │       ├── local_scale1_77.5_82.5_umap.png
│   │   │       ├── local_scale1_80.0_85.0_umap.png
│   │   │       ├── local_scale1_82.5_87.5_umap.png
│   │   │       ├── local_scale1_85.0_90.0_umap.png
│   │   │       ├── local_scale1_87.5_92.5_umap.png
│   │   │       ├── local_scale1_90.0_95.0_umap.png
│   │   │       ├── local_scale1_92.5_97.5_umap.png
│   │   │       ├── local_scale1_95.0_100.0_umap.png
│   │   │       └── local_scale1_97.5_102.5_umap.png
│   │   ├── metrics
│   │   │   ├── local_scale1_0.0_5.0.json
│   │   │   ├── local_scale1_10.0_15.0.json
│   │   │   ├── local_scale1_100.0_105.0.json
│   │   │   ├── local_scale1_102.5_107.5.json
│   │   │   ├── local_scale1_105.0_110.0.json
│   │   │   ├── local_scale1_107.5_112.5.json
│   │   │   ├── local_scale1_110.0_115.0.json
│   │   │   ├── local_scale1_112.5_117.5.json
│   │   │   ├── local_scale1_115.0_120.0.json
│   │   │   ├── local_scale1_117.5_122.5.json
│   │   │   ├── local_scale1_12.5_17.5.json
│   │   │   ├── local_scale1_120.0_125.0.json
│   │   │   ├── local_scale1_122.5_127.5.json
│   │   │   ├── local_scale1_125.0_130.0.json
│   │   │   ├── local_scale1_127.5_132.5.json
│   │   │   ├── local_scale1_130.0_135.0.json
│   │   │   ├── local_scale1_132.5_137.5.json
│   │   │   ├── local_scale1_135.0_140.0.json
│   │   │   ├── local_scale1_137.5_142.5.json
│   │   │   ├── local_scale1_140.0_145.0.json
│   │   │   ├── local_scale1_142.5_147.5.json
│   │   │   ├── local_scale1_145.0_150.0.json
│   │   │   ├── local_scale1_147.5_152.5.json
│   │   │   ├── local_scale1_15.0_20.0.json
│   │   │   ├── local_scale1_150.0_155.0.json
│   │   │   ├── local_scale1_152.5_157.5.json
│   │   │   ├── local_scale1_155.0_160.0.json
│   │   │   ├── local_scale1_157.5_162.5.json
│   │   │   ├── local_scale1_160.0_165.0.json
│   │   │   ├── local_scale1_162.5_167.5.json
│   │   │   ├── local_scale1_165.0_170.0.json
│   │   │   ├── local_scale1_167.5_172.5.json
│   │   │   ├── local_scale1_17.5_22.5.json
│   │   │   ├── local_scale1_170.0_175.0.json
│   │   │   ├── local_scale1_172.5_177.5.json
│   │   │   ├── local_scale1_177.5_182.5.json
│   │   │   ├── local_scale1_180.0_185.0.json
│   │   │   ├── local_scale1_182.5_187.5.json
│   │   │   ├── local_scale1_185.0_190.0.json
│   │   │   ├── local_scale1_187.5_192.5.json
│   │   │   ├── local_scale1_190.0_195.0.json
│   │   │   ├── local_scale1_192.5_197.5.json
│   │   │   ├── local_scale1_195.0_200.0.json
│   │   │   ├── local_scale1_197.5_202.5.json
│   │   │   ├── local_scale1_2.5_7.5.json
│   │   │   ├── local_scale1_20.0_25.0.json
│   │   │   ├── local_scale1_200.0_205.0.json
│   │   │   ├── local_scale1_202.5_207.5.json
│   │   │   ├── local_scale1_205.0_210.0.json
│   │   │   ├── local_scale1_207.5_212.5.json
│   │   │   ├── local_scale1_210.0_215.0.json
│   │   │   ├── local_scale1_212.5_217.5.json
│   │   │   ├── local_scale1_215.0_220.0.json
│   │   │   ├── local_scale1_217.5_222.5.json
│   │   │   ├── local_scale1_22.5_27.5.json
│   │   │   ├── local_scale1_220.0_225.0.json
│   │   │   ├── local_scale1_222.5_227.5.json
│   │   │   ├── local_scale1_225.0_230.0.json
│   │   │   ├── local_scale1_227.5_232.5.json
│   │   │   ├── local_scale1_230.0_235.0.json
│   │   │   ├── local_scale1_232.5_237.5.json
│   │   │   ├── local_scale1_235.0_240.0.json
│   │   │   ├── local_scale1_237.5_242.5.json
│   │   │   ├── local_scale1_240.0_245.0.json
│   │   │   ├── local_scale1_242.5_247.5.json
│   │   │   ├── local_scale1_245.0_250.0.json
│   │   │   ├── local_scale1_247.5_252.5.json
│   │   │   ├── local_scale1_25.0_30.0.json
│   │   │   ├── local_scale1_250.0_255.0.json
│   │   │   ├── local_scale1_252.5_257.5.json
│   │   │   ├── local_scale1_255.0_260.0.json
│   │   │   ├── local_scale1_257.5_262.5.json
│   │   │   ├── local_scale1_260.0_265.0.json
│   │   │   ├── local_scale1_262.5_267.5.json
│   │   │   ├── local_scale1_265.0_270.0.json
│   │   │   ├── local_scale1_267.5_272.5.json
│   │   │   ├── local_scale1_27.5_32.5.json
│   │   │   ├── local_scale1_270.0_275.0.json
│   │   │   ├── local_scale1_272.5_277.5.json
│   │   │   ├── local_scale1_275.0_280.0.json
│   │   │   ├── local_scale1_277.5_282.5.json
│   │   │   ├── local_scale1_280.0_285.0.json
│   │   │   ├── local_scale1_282.5_287.5.json
│   │   │   ├── local_scale1_285.0_290.0.json
│   │   │   ├── local_scale1_287.5_292.5.json
│   │   │   ├── local_scale1_290.0_295.0.json
│   │   │   ├── local_scale1_292.5_297.5.json
│   │   │   ├── local_scale1_295.0_300.0.json
│   │   │   ├── local_scale1_297.5_302.5.json
│   │   │   ├── local_scale1_30.0_35.0.json
│   │   │   ├── local_scale1_300.0_305.0.json
│   │   │   ├── local_scale1_302.5_307.5.json
│   │   │   ├── local_scale1_305.0_310.0.json
│   │   │   ├── local_scale1_307.5_312.5.json
│   │   │   ├── local_scale1_310.0_315.0.json
│   │   │   ├── local_scale1_312.5_317.5.json
│   │   │   ├── local_scale1_315.0_320.0.json
│   │   │   ├── local_scale1_317.5_322.5.json
│   │   │   ├── local_scale1_32.5_37.5.json
│   │   │   ├── local_scale1_320.0_325.0.json
│   │   │   ├── local_scale1_327.5_332.5.json
│   │   │   ├── local_scale1_330.0_335.0.json
│   │   │   ├── local_scale1_332.5_337.5.json
│   │   │   ├── local_scale1_335.0_340.0.json
│   │   │   ├── local_scale1_337.5_342.5.json
│   │   │   ├── local_scale1_340.0_345.0.json
│   │   │   ├── local_scale1_342.5_347.5.json
│   │   │   ├── local_scale1_345.0_350.0.json
│   │   │   ├── local_scale1_347.5_352.5.json
│   │   │   ├── local_scale1_35.0_40.0.json
│   │   │   ├── local_scale1_350.0_355.0.json
│   │   │   ├── local_scale1_352.5_357.5.json
│   │   │   ├── local_scale1_355.0_360.0.json
│   │   │   ├── local_scale1_357.5_362.5.json
│   │   │   ├── local_scale1_360.0_365.0.json
│   │   │   ├── local_scale1_362.5_367.5.json
│   │   │   ├── local_scale1_365.0_370.0.json
│   │   │   ├── local_scale1_367.5_372.5.json
│   │   │   ├── local_scale1_37.5_42.5.json
│   │   │   ├── local_scale1_370.0_375.0.json
│   │   │   ├── local_scale1_372.5_377.5.json
│   │   │   ├── local_scale1_375.0_380.0.json
│   │   │   ├── local_scale1_377.5_382.5.json
│   │   │   ├── local_scale1_380.0_385.0.json
│   │   │   ├── local_scale1_382.5_387.5.json
│   │   │   ├── local_scale1_385.0_390.0.json
│   │   │   ├── local_scale1_387.5_392.5.json
│   │   │   ├── local_scale1_390.0_395.0.json
│   │   │   ├── local_scale1_392.5_397.5.json
│   │   │   ├── local_scale1_395.0_400.0.json
│   │   │   ├── local_scale1_397.5_402.5.json
│   │   │   ├── local_scale1_40.0_45.0.json
│   │   │   ├── local_scale1_400.0_405.0.json
│   │   │   ├── local_scale1_402.5_407.5.json
│   │   │   ├── local_scale1_405.0_410.0.json
│   │   │   ├── local_scale1_407.5_412.5.json
│   │   │   ├── local_scale1_410.0_415.0.json
│   │   │   ├── local_scale1_412.5_417.5.json
│   │   │   ├── local_scale1_415.0_420.0.json
│   │   │   ├── local_scale1_417.5_422.5.json
│   │   │   ├── local_scale1_42.5_47.5.json
│   │   │   ├── local_scale1_420.0_425.0.json
│   │   │   ├── local_scale1_422.5_427.5.json
│   │   │   ├── local_scale1_45.0_50.0.json
│   │   │   ├── local_scale1_47.5_52.5.json
│   │   │   ├── local_scale1_5.0_10.0.json
│   │   │   ├── local_scale1_50.0_55.0.json
│   │   │   ├── local_scale1_50.0_70.0.json
│   │   │   ├── local_scale1_52.5_57.5.json
│   │   │   ├── local_scale1_55.0_60.0.json
│   │   │   ├── local_scale1_57.5_62.5.json
│   │   │   ├── local_scale1_60.0_65.0.json
│   │   │   ├── local_scale1_62.5_67.5.json
│   │   │   ├── local_scale1_65.0_70.0.json
│   │   │   ├── local_scale1_67.5_72.5.json
│   │   │   ├── local_scale1_7.5_12.5.json
│   │   │   ├── local_scale1_75.0_80.0.json
│   │   │   ├── local_scale1_77.5_82.5.json
│   │   │   ├── local_scale1_80.0_85.0.json
│   │   │   ├── local_scale1_82.5_87.5.json
│   │   │   ├── local_scale1_85.0_90.0.json
│   │   │   ├── local_scale1_87.5_92.5.json
│   │   │   ├── local_scale1_90.0_95.0.json
│   │   │   ├── local_scale1_92.5_97.5.json
│   │   │   ├── local_scale1_95.0_100.0.json
│   │   │   └── local_scale1_97.5_102.5.json
│   │   ├── pca
│   │   │   ├── local_scale1_0.0_5.0_pca.png
│   │   │   ├── local_scale1_10.0_15.0_pca.png
│   │   │   ├── local_scale1_100.0_105.0_pca.png
│   │   │   ├── local_scale1_102.5_107.5_pca.png
│   │   │   ├── local_scale1_105.0_110.0_pca.png
│   │   │   ├── local_scale1_107.5_112.5_pca.png
│   │   │   ├── local_scale1_110.0_115.0_pca.png
│   │   │   ├── local_scale1_112.5_117.5_pca.png
│   │   │   ├── local_scale1_115.0_120.0_pca.png
│   │   │   ├── local_scale1_117.5_122.5_pca.png
│   │   │   ├── local_scale1_12.5_17.5_pca.png
│   │   │   ├── local_scale1_120.0_125.0_pca.png
│   │   │   ├── local_scale1_122.5_127.5_pca.png
│   │   │   ├── local_scale1_125.0_130.0_pca.png
│   │   │   ├── local_scale1_127.5_132.5_pca.png
│   │   │   ├── local_scale1_130.0_135.0_pca.png
│   │   │   ├── local_scale1_132.5_137.5_pca.png
│   │   │   ├── local_scale1_135.0_140.0_pca.png
│   │   │   ├── local_scale1_137.5_142.5_pca.png
│   │   │   ├── local_scale1_140.0_145.0_pca.png
│   │   │   ├── local_scale1_142.5_147.5_pca.png
│   │   │   ├── local_scale1_145.0_150.0_pca.png
│   │   │   ├── local_scale1_147.5_152.5_pca.png
│   │   │   ├── local_scale1_15.0_20.0_pca.png
│   │   │   ├── local_scale1_150.0_155.0_pca.png
│   │   │   ├── local_scale1_152.5_157.5_pca.png
│   │   │   ├── local_scale1_155.0_160.0_pca.png
│   │   │   ├── local_scale1_157.5_162.5_pca.png
│   │   │   ├── local_scale1_160.0_165.0_pca.png
│   │   │   ├── local_scale1_162.5_167.5_pca.png
│   │   │   ├── local_scale1_165.0_170.0_pca.png
│   │   │   ├── local_scale1_167.5_172.5_pca.png
│   │   │   ├── local_scale1_17.5_22.5_pca.png
│   │   │   ├── local_scale1_170.0_175.0_pca.png
│   │   │   ├── local_scale1_172.5_177.5_pca.png
│   │   │   ├── local_scale1_177.5_182.5_pca.png
│   │   │   ├── local_scale1_180.0_185.0_pca.png
│   │   │   ├── local_scale1_182.5_187.5_pca.png
│   │   │   ├── local_scale1_185.0_190.0_pca.png
│   │   │   ├── local_scale1_187.5_192.5_pca.png
│   │   │   ├── local_scale1_190.0_195.0_pca.png
│   │   │   ├── local_scale1_192.5_197.5_pca.png
│   │   │   ├── local_scale1_195.0_200.0_pca.png
│   │   │   ├── local_scale1_197.5_202.5_pca.png
│   │   │   ├── local_scale1_2.5_7.5_pca.png
│   │   │   ├── local_scale1_20.0_25.0_pca.png
│   │   │   ├── local_scale1_200.0_205.0_pca.png
│   │   │   ├── local_scale1_202.5_207.5_pca.png
│   │   │   ├── local_scale1_205.0_210.0_pca.png
│   │   │   ├── local_scale1_207.5_212.5_pca.png
│   │   │   ├── local_scale1_210.0_215.0_pca.png
│   │   │   ├── local_scale1_212.5_217.5_pca.png
│   │   │   ├── local_scale1_215.0_220.0_pca.png
│   │   │   ├── local_scale1_217.5_222.5_pca.png
│   │   │   ├── local_scale1_22.5_27.5_pca.png
│   │   │   ├── local_scale1_220.0_225.0_pca.png
│   │   │   ├── local_scale1_222.5_227.5_pca.png
│   │   │   ├── local_scale1_225.0_230.0_pca.png
│   │   │   ├── local_scale1_227.5_232.5_pca.png
│   │   │   ├── local_scale1_230.0_235.0_pca.png
│   │   │   ├── local_scale1_232.5_237.5_pca.png
│   │   │   ├── local_scale1_235.0_240.0_pca.png
│   │   │   ├── local_scale1_237.5_242.5_pca.png
│   │   │   ├── local_scale1_240.0_245.0_pca.png
│   │   │   ├── local_scale1_242.5_247.5_pca.png
│   │   │   ├── local_scale1_245.0_250.0_pca.png
│   │   │   ├── local_scale1_247.5_252.5_pca.png
│   │   │   ├── local_scale1_25.0_30.0_pca.png
│   │   │   ├── local_scale1_250.0_255.0_pca.png
│   │   │   ├── local_scale1_252.5_257.5_pca.png
│   │   │   ├── local_scale1_255.0_260.0_pca.png
│   │   │   ├── local_scale1_257.5_262.5_pca.png
│   │   │   ├── local_scale1_260.0_265.0_pca.png
│   │   │   ├── local_scale1_262.5_267.5_pca.png
│   │   │   ├── local_scale1_265.0_270.0_pca.png
│   │   │   ├── local_scale1_267.5_272.5_pca.png
│   │   │   ├── local_scale1_27.5_32.5_pca.png
│   │   │   ├── local_scale1_270.0_275.0_pca.png
│   │   │   ├── local_scale1_272.5_277.5_pca.png
│   │   │   ├── local_scale1_275.0_280.0_pca.png
│   │   │   ├── local_scale1_277.5_282.5_pca.png
│   │   │   ├── local_scale1_280.0_285.0_pca.png
│   │   │   ├── local_scale1_282.5_287.5_pca.png
│   │   │   ├── local_scale1_285.0_290.0_pca.png
│   │   │   ├── local_scale1_287.5_292.5_pca.png
│   │   │   ├── local_scale1_290.0_295.0_pca.png
│   │   │   ├── local_scale1_292.5_297.5_pca.png
│   │   │   ├── local_scale1_295.0_300.0_pca.png
│   │   │   ├── local_scale1_297.5_302.5_pca.png
│   │   │   ├── local_scale1_30.0_35.0_pca.png
│   │   │   ├── local_scale1_300.0_305.0_pca.png
│   │   │   ├── local_scale1_302.5_307.5_pca.png
│   │   │   ├── local_scale1_305.0_310.0_pca.png
│   │   │   ├── local_scale1_307.5_312.5_pca.png
│   │   │   ├── local_scale1_310.0_315.0_pca.png
│   │   │   ├── local_scale1_312.5_317.5_pca.png
│   │   │   ├── local_scale1_315.0_320.0_pca.png
│   │   │   ├── local_scale1_317.5_322.5_pca.png
│   │   │   ├── local_scale1_32.5_37.5_pca.png
│   │   │   ├── local_scale1_320.0_325.0_pca.png
│   │   │   ├── local_scale1_327.5_332.5_pca.png
│   │   │   ├── local_scale1_330.0_335.0_pca.png
│   │   │   ├── local_scale1_332.5_337.5_pca.png
│   │   │   ├── local_scale1_335.0_340.0_pca.png
│   │   │   ├── local_scale1_337.5_342.5_pca.png
│   │   │   ├── local_scale1_340.0_345.0_pca.png
│   │   │   ├── local_scale1_342.5_347.5_pca.png
│   │   │   ├── local_scale1_345.0_350.0_pca.png
│   │   │   ├── local_scale1_347.5_352.5_pca.png
│   │   │   ├── local_scale1_35.0_40.0_pca.png
│   │   │   ├── local_scale1_350.0_355.0_pca.png
│   │   │   ├── local_scale1_352.5_357.5_pca.png
│   │   │   ├── local_scale1_355.0_360.0_pca.png
│   │   │   ├── local_scale1_357.5_362.5_pca.png
│   │   │   ├── local_scale1_360.0_365.0_pca.png
│   │   │   ├── local_scale1_362.5_367.5_pca.png
│   │   │   ├── local_scale1_365.0_370.0_pca.png
│   │   │   ├── local_scale1_367.5_372.5_pca.png
│   │   │   ├── local_scale1_37.5_42.5_pca.png
│   │   │   ├── local_scale1_370.0_375.0_pca.png
│   │   │   ├── local_scale1_372.5_377.5_pca.png
│   │   │   ├── local_scale1_375.0_380.0_pca.png
│   │   │   ├── local_scale1_377.5_382.5_pca.png
│   │   │   ├── local_scale1_380.0_385.0_pca.png
│   │   │   ├── local_scale1_382.5_387.5_pca.png
│   │   │   ├── local_scale1_385.0_390.0_pca.png
│   │   │   ├── local_scale1_387.5_392.5_pca.png
│   │   │   ├── local_scale1_390.0_395.0_pca.png
│   │   │   ├── local_scale1_392.5_397.5_pca.png
│   │   │   ├── local_scale1_395.0_400.0_pca.png
│   │   │   ├── local_scale1_397.5_402.5_pca.png
│   │   │   ├── local_scale1_40.0_45.0_pca.png
│   │   │   ├── local_scale1_400.0_405.0_pca.png
│   │   │   ├── local_scale1_402.5_407.5_pca.png
│   │   │   ├── local_scale1_405.0_410.0_pca.png
│   │   │   ├── local_scale1_407.5_412.5_pca.png
│   │   │   ├── local_scale1_410.0_415.0_pca.png
│   │   │   ├── local_scale1_412.5_417.5_pca.png
│   │   │   ├── local_scale1_415.0_420.0_pca.png
│   │   │   ├── local_scale1_417.5_422.5_pca.png
│   │   │   ├── local_scale1_42.5_47.5_pca.png
│   │   │   ├── local_scale1_420.0_425.0_pca.png
│   │   │   ├── local_scale1_422.5_427.5_pca.png
│   │   │   ├── local_scale1_45.0_50.0_pca.png
│   │   │   ├── local_scale1_47.5_52.5_pca.png
│   │   │   ├── local_scale1_5.0_10.0_pca.png
│   │   │   ├── local_scale1_50.0_55.0_pca.png
│   │   │   ├── local_scale1_50.0_70.0_pca.png
│   │   │   ├── local_scale1_52.5_57.5_pca.png
│   │   │   ├── local_scale1_55.0_60.0_pca.png
│   │   │   ├── local_scale1_57.5_62.5_pca.png
│   │   │   ├── local_scale1_60.0_65.0_pca.png
│   │   │   ├── local_scale1_62.5_67.5_pca.png
│   │   │   ├── local_scale1_65.0_70.0_pca.png
│   │   │   ├── local_scale1_67.5_72.5_pca.png
│   │   │   ├── local_scale1_7.5_12.5_pca.png
│   │   │   ├── local_scale1_75.0_80.0_pca.png
│   │   │   ├── local_scale1_77.5_82.5_pca.png
│   │   │   ├── local_scale1_80.0_85.0_pca.png
│   │   │   ├── local_scale1_82.5_87.5_pca.png
│   │   │   ├── local_scale1_85.0_90.0_pca.png
│   │   │   ├── local_scale1_87.5_92.5_pca.png
│   │   │   ├── local_scale1_90.0_95.0_pca.png
│   │   │   ├── local_scale1_92.5_97.5_pca.png
│   │   │   ├── local_scale1_95.0_100.0_pca.png
│   │   │   └── local_scale1_97.5_102.5_pca.png
│   │   ├── reports
│   │   │   ├── all_windows_metrics.csv
│   │   │   └── local_analysis_summary.md
│   │   ├── trajectories
│   │   └── umap
│   │       ├── local_scale1_0.0_5.0_umap.png
│   │       ├── local_scale1_10.0_15.0_umap.png
│   │       ├── local_scale1_100.0_105.0_umap.png
│   │       ├── local_scale1_102.5_107.5_umap.png
│   │       ├── local_scale1_105.0_110.0_umap.png
│   │       ├── local_scale1_107.5_112.5_umap.png
│   │       ├── local_scale1_110.0_115.0_umap.png
│   │       ├── local_scale1_112.5_117.5_umap.png
│   │       ├── local_scale1_115.0_120.0_umap.png
│   │       ├── local_scale1_117.5_122.5_umap.png
│   │       ├── local_scale1_12.5_17.5_umap.png
│   │       ├── local_scale1_120.0_125.0_umap.png
│   │       ├── local_scale1_122.5_127.5_umap.png
│   │       ├── local_scale1_125.0_130.0_umap.png
│   │       ├── local_scale1_127.5_132.5_umap.png
│   │       ├── local_scale1_130.0_135.0_umap.png
│   │       ├── local_scale1_132.5_137.5_umap.png
│   │       ├── local_scale1_135.0_140.0_umap.png
│   │       ├── local_scale1_137.5_142.5_umap.png
│   │       ├── local_scale1_140.0_145.0_umap.png
│   │       ├── local_scale1_142.5_147.5_umap.png
│   │       ├── local_scale1_145.0_150.0_umap.png
│   │       ├── local_scale1_147.5_152.5_umap.png
│   │       ├── local_scale1_15.0_20.0_umap.png
│   │       ├── local_scale1_150.0_155.0_umap.png
│   │       ├── local_scale1_152.5_157.5_umap.png
│   │       ├── local_scale1_155.0_160.0_umap.png
│   │       ├── local_scale1_157.5_162.5_umap.png
│   │       ├── local_scale1_160.0_165.0_umap.png
│   │       ├── local_scale1_162.5_167.5_umap.png
│   │       ├── local_scale1_165.0_170.0_umap.png
│   │       ├── local_scale1_167.5_172.5_umap.png
│   │       ├── local_scale1_17.5_22.5_umap.png
│   │       ├── local_scale1_170.0_175.0_umap.png
│   │       ├── local_scale1_172.5_177.5_umap.png
│   │       ├── local_scale1_177.5_182.5_umap.png
│   │       ├── local_scale1_180.0_185.0_umap.png
│   │       ├── local_scale1_182.5_187.5_umap.png
│   │       ├── local_scale1_185.0_190.0_umap.png
│   │       ├── local_scale1_187.5_192.5_umap.png
│   │       ├── local_scale1_190.0_195.0_umap.png
│   │       ├── local_scale1_192.5_197.5_umap.png
│   │       ├── local_scale1_195.0_200.0_umap.png
│   │       ├── local_scale1_197.5_202.5_umap.png
│   │       ├── local_scale1_2.5_7.5_umap.png
│   │       ├── local_scale1_20.0_25.0_umap.png
│   │       ├── local_scale1_200.0_205.0_umap.png
│   │       ├── local_scale1_202.5_207.5_umap.png
│   │       ├── local_scale1_205.0_210.0_umap.png
│   │       ├── local_scale1_207.5_212.5_umap.png
│   │       ├── local_scale1_210.0_215.0_umap.png
│   │       ├── local_scale1_212.5_217.5_umap.png
│   │       ├── local_scale1_215.0_220.0_umap.png
│   │       ├── local_scale1_217.5_222.5_umap.png
│   │       ├── local_scale1_22.5_27.5_umap.png
│   │       ├── local_scale1_220.0_225.0_umap.png
│   │       ├── local_scale1_222.5_227.5_umap.png
│   │       ├── local_scale1_225.0_230.0_umap.png
│   │       ├── local_scale1_227.5_232.5_umap.png
│   │       ├── local_scale1_230.0_235.0_umap.png
│   │       ├── local_scale1_232.5_237.5_umap.png
│   │       ├── local_scale1_235.0_240.0_umap.png
│   │       ├── local_scale1_237.5_242.5_umap.png
│   │       ├── local_scale1_240.0_245.0_umap.png
│   │       ├── local_scale1_242.5_247.5_umap.png
│   │       ├── local_scale1_245.0_250.0_umap.png
│   │       ├── local_scale1_247.5_252.5_umap.png
│   │       ├── local_scale1_25.0_30.0_umap.png
│   │       ├── local_scale1_250.0_255.0_umap.png
│   │       ├── local_scale1_252.5_257.5_umap.png
│   │       ├── local_scale1_255.0_260.0_umap.png
│   │       ├── local_scale1_257.5_262.5_umap.png
│   │       ├── local_scale1_260.0_265.0_umap.png
│   │       ├── local_scale1_262.5_267.5_umap.png
│   │       ├── local_scale1_265.0_270.0_umap.png
│   │       ├── local_scale1_267.5_272.5_umap.png
│   │       ├── local_scale1_27.5_32.5_umap.png
│   │       ├── local_scale1_270.0_275.0_umap.png
│   │       ├── local_scale1_272.5_277.5_umap.png
│   │       ├── local_scale1_275.0_280.0_umap.png
│   │       ├── local_scale1_277.5_282.5_umap.png
│   │       ├── local_scale1_280.0_285.0_umap.png
│   │       ├── local_scale1_282.5_287.5_umap.png
│   │       ├── local_scale1_285.0_290.0_umap.png
│   │       ├── local_scale1_287.5_292.5_umap.png
│   │       ├── local_scale1_290.0_295.0_umap.png
│   │       ├── local_scale1_292.5_297.5_umap.png
│   │       ├── local_scale1_295.0_300.0_umap.png
│   │       ├── local_scale1_297.5_302.5_umap.png
│   │       ├── local_scale1_30.0_35.0_umap.png
│   │       ├── local_scale1_300.0_305.0_umap.png
│   │       ├── local_scale1_302.5_307.5_umap.png
│   │       ├── local_scale1_305.0_310.0_umap.png
│   │       ├── local_scale1_307.5_312.5_umap.png
│   │       ├── local_scale1_310.0_315.0_umap.png
│   │       ├── local_scale1_312.5_317.5_umap.png
│   │       ├── local_scale1_315.0_320.0_umap.png
│   │       ├── local_scale1_317.5_322.5_umap.png
│   │       ├── local_scale1_32.5_37.5_umap.png
│   │       ├── local_scale1_320.0_325.0_umap.png
│   │       ├── local_scale1_327.5_332.5_umap.png
│   │       ├── local_scale1_330.0_335.0_umap.png
│   │       ├── local_scale1_332.5_337.5_umap.png
│   │       ├── local_scale1_335.0_340.0_umap.png
│   │       ├── local_scale1_337.5_342.5_umap.png
│   │       ├── local_scale1_340.0_345.0_umap.png
│   │       ├── local_scale1_342.5_347.5_umap.png
│   │       ├── local_scale1_345.0_350.0_umap.png
│   │       ├── local_scale1_347.5_352.5_umap.png
│   │       ├── local_scale1_35.0_40.0_umap.png
│   │       ├── local_scale1_350.0_355.0_umap.png
│   │       ├── local_scale1_352.5_357.5_umap.png
│   │       ├── local_scale1_355.0_360.0_umap.png
│   │       ├── local_scale1_357.5_362.5_umap.png
│   │       ├── local_scale1_360.0_365.0_umap.png
│   │       ├── local_scale1_362.5_367.5_umap.png
│   │       ├── local_scale1_365.0_370.0_umap.png
│   │       ├── local_scale1_367.5_372.5_umap.png
│   │       ├── local_scale1_37.5_42.5_umap.png
│   │       ├── local_scale1_370.0_375.0_umap.png
│   │       ├── local_scale1_372.5_377.5_umap.png
│   │       ├── local_scale1_375.0_380.0_umap.png
│   │       ├── local_scale1_377.5_382.5_umap.png
│   │       ├── local_scale1_380.0_385.0_umap.png
│   │       ├── local_scale1_382.5_387.5_umap.png
│   │       ├── local_scale1_385.0_390.0_umap.png
│   │       ├── local_scale1_387.5_392.5_umap.png
│   │       ├── local_scale1_390.0_395.0_umap.png
│   │       ├── local_scale1_392.5_397.5_umap.png
│   │       ├── local_scale1_395.0_400.0_umap.png
│   │       ├── local_scale1_397.5_402.5_umap.png
│   │       ├── local_scale1_40.0_45.0_umap.png
│   │       ├── local_scale1_400.0_405.0_umap.png
│   │       ├── local_scale1_402.5_407.5_umap.png
│   │       ├── local_scale1_405.0_410.0_umap.png
│   │       ├── local_scale1_407.5_412.5_umap.png
│   │       ├── local_scale1_410.0_415.0_umap.png
│   │       ├── local_scale1_412.5_417.5_umap.png
│   │       ├── local_scale1_415.0_420.0_umap.png
│   │       ├── local_scale1_417.5_422.5_umap.png
│   │       ├── local_scale1_42.5_47.5_umap.png
│   │       ├── local_scale1_420.0_425.0_umap.png
│   │       ├── local_scale1_422.5_427.5_umap.png
│   │       ├── local_scale1_45.0_50.0_umap.png
│   │       ├── local_scale1_47.5_52.5_umap.png
│   │       ├── local_scale1_5.0_10.0_umap.png
│   │       ├── local_scale1_50.0_55.0_umap.png
│   │       ├── local_scale1_50.0_70.0_umap.png
│   │       ├── local_scale1_52.5_57.5_umap.png
│   │       ├── local_scale1_55.0_60.0_umap.png
│   │       ├── local_scale1_57.5_62.5_umap.png
│   │       ├── local_scale1_60.0_65.0_umap.png
│   │       ├── local_scale1_62.5_67.5_umap.png
│   │       ├── local_scale1_65.0_70.0_umap.png
│   │       ├── local_scale1_67.5_72.5_umap.png
│   │       ├── local_scale1_7.5_12.5_umap.png
│   │       ├── local_scale1_75.0_80.0_umap.png
│   │       ├── local_scale1_77.5_82.5_umap.png
│   │       ├── local_scale1_80.0_85.0_umap.png
│   │       ├── local_scale1_82.5_87.5_umap.png
│   │       ├── local_scale1_85.0_90.0_umap.png
│   │       ├── local_scale1_87.5_92.5_umap.png
│   │       ├── local_scale1_90.0_95.0_umap.png
│   │       ├── local_scale1_92.5_97.5_umap.png
│   │       ├── local_scale1_95.0_100.0_umap.png
│   │       └── local_scale1_97.5_102.5_umap.png
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
│   ├── asr_metrics.csv
│   ├── attribution_metrics.csv
│   ├── audio_audit.csv
│   ├── conversational_metrics.csv
│   ├── diarization_metrics.csv
│   ├── phase13_karen_generalization_report.md
│   └── vad_metrics.csv
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
