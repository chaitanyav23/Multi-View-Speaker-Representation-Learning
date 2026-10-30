# phase11_dfn_ablation Metrics Tables

## asr_eval_improved_phase11_11A_asr.csv

| wer | substitutions | insertions | deletions | ref_count | hyp_count | file |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3002 | 78 | 32 | 44 | 513 | 501 | bwc |


## asr_eval_improved_phase11_11B_asr.csv

| wer | substitutions | insertions | deletions | ref_count | hyp_count | file |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4113 | 99 | 31 | 81 | 513 | 463 | bwc |


## asr_eval_improved_phase11_11C_asr.csv

| wer | substitutions | insertions | deletions | ref_count | hyp_count | file |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3762 | 84 | 18 | 91 | 513 | 440 | bwc |


## audio_audit.csv

| file | md5 | sha256 | duration | sample_rate | LUFS | RMS | Peak | Spectral_Centroid_Mean | Spectral_Rolloff_Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bwc_normalized.wav | c0b66acf8b833b6f8515e524b0c9f6e0 | 13144956af2aae0c087ade23ebec37d8e8e548159bae9b66b0c658f599cd7ae6 | 175.42675 | 16000 | -19.63822965729515 | 0.10094231367111206 | 0.862060546875 | 1309.6061643083788 | 2593.165807951851 |
| bwc_deepfilternet.wav | 539fcf09fbffef4bd9a78b8a0c824732 | 3c268eed2a387c55eff4a6382cc45c84d6f59f9c09665541f156f403461369ff | 175.42675 | 16000 | -20.278062230108635 | 0.07475003600120544 | 0.832977294921875 | 1710.9147126165026 | 3224.114592832391 |


## conversational_eval_phase11_11A_conv.csv

| file | WER | Subs | Dels | Ins | DER | Miss | FA | Conf | SA-WER | cpWER | WDER | deltaCP | deltaSA | JER | IER | 5-Dup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bwc | 0.3244 | 84 | 46 | 27 | 0.4703 | 0.2905 | 0.08 | 0.0998 | 0.438 | 0.5032 | 0.1136 | 0.1788 | 0.1136 | 0.1407 | 0.08 | 3 |


## conversational_eval_phase11_11B_conv.csv

| file | WER | Subs | Dels | Ins | DER | Miss | FA | Conf | SA-WER | cpWER | WDER | deltaCP | deltaSA | JER | IER | 5-Dup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bwc | 0.4215 | 86 | 87 | 31 | 0.6267 | 0.3947 | 0.0666 | 0.1654 | 0.5785 | 0.6528 | 0.157 | 0.2313 | 0.157 | 0.2733 | 0.0666 | 3 |


## conversational_eval_phase11_11C_conv.csv

| file | WER | Subs | Dels | Ins | DER | Miss | FA | Conf | SA-WER | cpWER | WDER | deltaCP | deltaSA | JER | IER | 5-Dup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bwc | 0.3781 | 73 | 93 | 17 | 0.5318 | 0.3691 | 0.0685 | 0.0942 | 0.4793 | 0.5301 | 0.1012 | 0.152 | 0.1012 | 0.1492 | 0.0685 | 4 |


## diarization_eval_improved_phase11_11A_diar.csv

| file | der | miss | fa | conf |
| --- | --- | --- | --- | --- |
| bwc | 0.4679 | 0.2878 | 0.0766 | 0.1035 |


## diarization_eval_improved_phase11_11B_diar.csv

| file | der | miss | fa | conf |
| --- | --- | --- | --- | --- |
| bwc | 0.6228 | 0.391 | 0.0633 | 0.1685 |


## diarization_eval_improved_phase11_11C_diar.csv

| file | der | miss | fa | conf |
| --- | --- | --- | --- | --- |
| bwc | 0.5288 | 0.3657 | 0.0652 | 0.0979 |


## vad_metrics.csv

| Pipeline | Accuracy | Precision | Recall | F1 | Miss Rate | False Alarm Rate |
| --- | --- | --- | --- | --- | --- | --- |
| 11A | 0.7466666666666667 | 0.851221317542561 | 0.7666666666666667 | 0.8067344791301297 | 0.23333333333333334 | 0.29777777777777775 |
| 11B | 0.717816091954023 | 0.864862083161795 | 0.70025 | 0.773899428992448 | 0.29975 | 0.24314814814814814 |


