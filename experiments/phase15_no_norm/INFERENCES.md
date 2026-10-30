# Phase 15: Inferences

## Normalization is Not a "Silver Bullet"
One of the most significant inferences from Phase 15 is that the pipeline remains highly functional without loudness normalization. A WER of 14.35% on raw audio is competitive. This suggests that the current generation of Transformer-based models (Whisper large-v3, TitaNet) are surprisingly resilient to variations in absolute signal level.

## VAD as the Primary Gatekeeper
The high VAD Recall (94%) despite no normalization confirms that Pyannote 3.1 is an excellent front-end. However, the high False Alarm rate (68.9%) suggests that VAD becomes less "selective" on non-normalized audio. This leads to the inference that normalization primarily serves to stabilize the VAD stage rather than the ASR stage.

## MSDD Sensitivity to Signal Consistency
The DER of 40.9% and Confusion of 17.2% are relatively high. We can infer that while Whisper (ASR) is robust to loudness variations, MSDD (Diarization) is much more sensitive. Embedding extraction likely benefits from a more consistent dynamic range, which normalization provides. Removing normalization seems to have a disproportionately negative impact on speaker identification compared to word recognition.

## Dataset Variance
The dramatic improvement over the BWC Phase 11A baseline (WER 14.3% vs 30.0%) suggests that the `karen.wav` dataset is significantly "cleaner" or easier for the models than the BWC dataset. This implies that future benchmarking must always account for dataset-specific characteristics, as a "hardened" pipeline on one dataset may perform poorly on another.

## Deletion Rate as a Proxy for SNR
The high number of deletions (154) in ASR can be used as a proxy to infer that certain parts of the audio have a poor Signal-to-Noise Ratio (SNR) that is exacerbated by the lack of normalization. In normalized audio, these segments might have been boosted enough for Whisper to "catch" them.
