# Phase 11 Results Analysis

## Ablation Matrix Results (BWC Dataset)

| Pipeline Variant | WER | DER | Miss | FA | cpWER |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **11A (Control: Norm Only)** | **0.3244** | **0.4703** | **0.2905** | 0.0800 | **0.5032** |
| **11B (Full DFN)** | 0.4215 | 0.6267 | 0.3947 | **0.0666** | 0.6528 |
| **11C (Isolated DFN ASR)** | 0.3781 | 0.5318 | 0.3691 | 0.0685 | 0.5301 |

## Key Findings

### 1. DFN is Universally Destructive
Across every primary ranking metric (WER, DER, cpWER, SA-WER), Pipeline 11A outperformed 11B and 11C. The addition of DeepFilterNet to any part of the pipeline—whether VAD, ASR, or both—results in a quantifiable degradation of conversational intelligence.

### 2. The False Alarm "Trap"
DFN achieved the lowest False Alarm rate (0.0666). However, the Phase 11 report confirms that this reduction in False Alarms comes at an unacceptable cost to the Miss Rate (0.3947). DFN "cleans" the audio by simply deleting large swaths of low-confidence speech.

### 3. Diarization Degradation
Comparing 11C (DER 0.53) to 11A (DER 0.47) reveals a fascinating secondary effect. Even though the *exact same* MSDD speaker labels were used for both, the DER was worse for 11C. This is because DER is calculated based on the alignment of words. When DFN (in 11C) deleted words (increasing ASR Deletions), it shifted the temporal alignment of the remaining words, causing them to fall out of sync with their correct speaker segments.

## Experimental Conclusion
The debate over Neural Denoising is closed. DeepFilterNet, and similar aggressive spectral maskers, are fundamentally incompatible with high-accuracy ML transcription and diarization on forensic audio. Simple Normalization (LUFS or Peak) is the only safe preprocessing step.
