# Future Direction: Phase 14 and Beyond

The results from Phase 13 provide a clear roadmap for the next stage of pipeline development. The focus will transition from "Model Integration" to "Clustering Stability and Precision."

## 1. Speaker Clustering Optimization (Phase 14 Priority)
To address the over-clustering observed in Phase 13, the following technical strategies should be explored:
-   **Tuning HAC Thresholds:** Investigate the `max_rp_threshold` and `sparse_search_volume` parameters in the NeMo MSDD clustering configuration.
-   **Speaker Counting Heuristics:** Implement a post-processing step to merge speaker profiles with high embedding similarity (Cosine Similarity > 0.85).
-   **Constrained Clustering:** Test the impact of providing `oracle_num_speakers` to the MSDD model to establish a "Best Case" baseline for diarization.

## 2. VAD Precision Hardening
The high False Alarm rate (15% in DER) suggests the VAD is capturing too much noise.
-   **Parameter Sweeps:** Conduct a sweep of the `merge_gap` and `min_duration` parameters in the Pyannote VAD post-processing.
-   **Secondary VAD Filter:** Integrate a lightweight secondary classifier (e.g., Silero VAD) to "double-check" the Pyannote segments before they are passed to the diarizer.

## 3. Embedding Analysis Diagnostics
To understand *why* over-clustering occurs:
-   **Visualization:** Use UMAP or t-SNE to visualize the distribution of extracted TitaNet embeddings. This will help determine if speakers are forming discrete clusters or if there is significant overlap/spread.
-   **Centroid Analysis:** Calculate the intra-speaker vs. inter-speaker distance to quantify embedding stability.

## 4. Addressing ASR Deletions
To reduce the 156 word deletions:
-   **Contextual Overlap:** Increase the "padding" around VAD segments (e.g., adding 100ms to the start and end) to ensure no initial or trailing phonemes are clipped.
-   **WhisperX Decoding Params:** Experiment with lower `beam_size` or adjusted `no_speech_threshold` to encourage the model to transcribe short bursts of speech.

## 5. Generalization to more Chaotic Data
The next logical benchmark should involve:
-   **High Overlap Recordings:** Test on data with >10% overlap to truly stress the MSDD multi-scale decoder.
-   **Varying Audio Qualities:** Introduce recordings with non-stationary noise (e.g., street noise, music) to further harden the VAD and ASR layers.
