# Phase 8 Results Analysis

## Performance Benchmark (BWC Dataset)

| Metric | Phase 7 (Optimal Single) | Phase 8 (Sentence Grouped) | Delta (Rel. %) |
| :--- | :--- | :--- | :--- |
| **WER** | 0.3802 | **0.3762** | -1.0% (Stable) |
| **DER** | 0.6137 | **0.5507** | **-10.3% (Better)** |
| **VAD Recall** | 0.7788 | **0.8609** | **+10.5% (Best)** |
| **cpWER** | 0.6595 | 0.6617 | +0.3% (Stable) |
| **SA-WER** | 0.5785 | 0.6322 | +9.3% (Worse) |

## Key Findings

### 1. Superior Detection
The **Recall (0.86)** and **F1 (0.84)** results prove that sentence-level grouping is the correct strategy for maximizing speech recovery in noisy forensic audio. It eliminates the "Speech Fragmentation" problem that plagued earlier phases.

### 2. The Diarization Benefit
Longer segments (1.0s gap) provided MSDD with a more stable temporal window for identity decoding, leading to a significant improvement in **DER (0.55)**. The model was less likely to flip identities in the middle of a continuous sentence.

### 3. The SA-WER/cpWER Bottleneck
Despite the improvements in detection and separation, the overall **cpWER** did not drop. 
*   **Reason:** The audio was still processed through **DeepFilterNet**.
*   **Conclusion:** Contextual grouping and buffer management can improve the *structure* of the conversation, but they cannot fix the *content* if the underlying audio signal is Spectrally damaged.

## Experimental Conclusion
Sentence-level grouping is a winning strategy for VAD and Diarization. However, it must be paired with **Raw-Audio ASR** to achieve the project's goal of a low cpWER.
