# phase6_final_diagnostic Metrics Tables

## asr_metrics.csv

| enhancer | file | wer | ref_count | subs | dels | ins |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | bwc | 0.4074 | 513 | 103 | 71 | 35 |
| baseline | podcast | 0.0558 | 2564 | 33 | 76 | 34 |
| deepfilternet | bwc | 0.3996 | 513 | 108 | 69 | 28 |
| speechbrain | bwc | 0.425 | 513 | 119 | 81 | 18 |


## diarization_metrics.csv

| enhancer | file | der | miss | fa | conf |
| --- | --- | --- | --- | --- | --- |
| baseline | bwc | 0.6991 | 0.3932 | 0.0573 | 0.2486 |
| baseline | podcast | 0.1037 | 0.0925 | 0.0045 | 0.0067 |
| deepfilternet | bwc | 0.7041 | 0.3947 | 0.0625 | 0.2469 |
| speechbrain | bwc | 0.6898 | 0.4005 | 0.0609 | 0.2284 |


## final_comparison.csv

| enhancer | avg_wer | avg_der |
| --- | --- | --- |
| baseline | 0.2316 | 0.4014 |
| deepfilternet | 0.3996 | 0.7041 |
| speechbrain | 0.425 | 0.6898 |


## vad_metrics.csv

| enhancer | file | precision | recall | f1 |
| --- | --- | --- | --- | --- |
| baseline | bwc | 0.8666 | 0.6975 | 0.7729 |
| baseline | podcast | 0.989 | 0.9706 | 0.9797 |
| deepfilternet | bwc | 0.8674 | 0.7008 | 0.7752 |
| speechbrain | bwc | 0.8753 | 0.6662 | 0.7566 |


