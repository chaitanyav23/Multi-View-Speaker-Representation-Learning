# Audio Provenance Audit Report

## Audio Metrics Comparison

|   duration |     lufs |       rms |     peak |   crest_factor |   spectral_centroid |   spectral_rolloff |   dominant_freq | label     | path                                                                               | md5                              | sha256                                                           |
|-----------:|---------:|----------:|---------:|---------------:|--------------------:|-------------------:|----------------:|:----------|:-----------------------------------------------------------------------------------|:---------------------------------|:-----------------------------------------------------------------|
|    175.427 | -19.6382 | 0.100942  | 0.862061 |        8.54013 |             1309.61 |            2593.17 |         401.763 | v2_norm   | experiments/phase6_final_diagnostic/v2/audio_preprocessed/bwc_norm_v2.wav          | c0b66acf8b833b6f8515e524b0c9f6e0 | 13144956af2aae0c087ade23ebec37d8e8e548159bae9b66b0c658f599cd7ae6 |
|    175.427 | -20.2781 | 0.07475   | 0.832977 |       11.1435  |             1710.91 |            3224.11 |         399.255 | v2_dfn    | experiments/phase6_final_diagnostic/v2/audio_preprocessed/bwc_deepfilternet_v2.wav | 539fcf09fbffef4bd9a78b8a0c824732 | 3c268eed2a387c55eff4a6382cc45c84d6f59f9c09665541f156f403461369ff |
|    175.427 | -20.0366 | 0.0891602 | 0.999969 |       11.2154  |             1098.37 |            1896.77 |         471.382 | v2_sb     | experiments/phase6_final_diagnostic/v2/audio_preprocessed/bwc_speechbrain_v2.wav   | f429e1458a00e39e94b560cb5f77c936 | df9b7cbbcfe7c22ff90a8601f48571dd605e1161127911f309150cb206826f5c |
|    175.427 | -20.4906 | 0.0727559 | 0.81366  |       11.1834  |             1732.82 |            3234.9  |         399.255 | orig_norm | experiments/phase6_final_diagnostic/audio_preprocessed/bwc_norm.wav                | 679e47f99a4c45255b9e9a6665e6d321 | 712fc91f575ba90987ddcd493fe68870b9fdec731647b951ddf4bcd5bb6160ea |
|    175.427 | -20.708  | 0.0708425 | 0.795197 |       11.2249  |             1672.4  |            3111.22 |         399.255 | orig_dfn  | experiments/phase6_final_diagnostic/audio_preprocessed/bwc_deepfilternet.wav       | 19f7a37b125f3a2dcc13247e2f17887d | e2735f312b8090314020765ec398aada5f2a3558b0ac417bade59202289bbe0f |
|    175.427 | -19.2017 | 0.0839822 | 0.999969 |       11.9069  |             1598.14 |            2992.26 |         399.255 | orig_sb   | experiments/phase6_final_diagnostic/audio_preprocessed/bwc_speechbrain.wav         | b15537c58ad33b953f838a4f5e70fcd7 | 890c31de4d1b7548241f21fbeb9455b1c8c303f066bcd2c1a2c2a97366836874 |

## Comparison Analysis

### ALERT: bwc_norm.wav and bwc_norm_v2.wav are DIFFERENT.
Original MD5: 679e47f99a4c45255b9e9a6665e6d321
V2 MD5: c0b66acf8b833b6f8515e524b0c9f6e0

