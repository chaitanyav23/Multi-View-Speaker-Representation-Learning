# Stage 5B.1: Projected DER Methodology & Nearest-Neighbor Propagation Audit

## 1. Executive Summary & Mandatory Classification

In the original Stage 5B.1 experimental execution, downstream diarization impact was evaluated using the function `project_vad_to_diarization()`.

### Crucial Methodological Distinction:
This function was **NOT a full diarization pipeline rerun**. It did not re-extract TitaNet or WeSpeaker embeddings, nor did it execute spectral clustering or NeMo NME-SC affinity fusion on the updated speech segments.

Instead, the function performed:
1. Loading the frozen Condition J (M1 baseline) RTTM.
2. Labeling previously unvoiced frames by **nearest-neighbor temporal propagation** from the nearest existing baseline speaker assignment.
3. Applying the newly optimized VAD mask to select the active speech segments.

Under experimental integrity guidelines, these downstream metrics must NOT be described as end-to-end diarization results. They must be explicitly designated as:

> **VAD-mask-projected DER under frozen baseline speaker assignments.**

---

## 2. Quantitative Audit of Nearest-Neighbor Propagation

To quantify the degree to which downstream hypothesis RTTMs depend on nearest-neighbor label propagation, we instrumented `project_vad_to_diarization_audited()` to measure the duration and percentage of hypothesis frames that received their identity from unvoiced baseline gaps.

### 2.1 Per-Recording Nearest-Neighbor Propagation in C50 (Raw Mono VAD)

| Recording ID | Total Audio (s) | C50 Hypothesis Speech (s) | NN-Propagated Duration (s) | NN-Propagated % of Hypothesis | Baseline Speech Dur (s) |
|---|---|---|---|---|---|
| **R01_sara_sharif** | 108.576 | 84.05s | **16.87s** | **20.07%** | 67.18s |
| **R02_dennis_skinner** | 176.320 | 159.50s | **28.51s** | **17.87%** | 130.99s |
| **R03_disabled_parking** | 206.272 | 185.81s | 8.67s | 4.67% | 177.14s |
| **R04_piers_robinson** | 204.608 | 190.14s | 0.03s | 0.02% | 190.11s |
| **R05_southwest_pilot** | 162.496 | 123.89s | **24.71s** | **19.95%** | 99.18s |
| **R06_rich_karen** | 652.224 | 508.08s | **45.57s** | 8.97% | 462.51s |
| **R07_bwc_cheap_shot** | 175.488 | 121.92s | 11.98s | 9.83% | 109.94s |
| **R08_uk_parliament** | 215.392 | 178.33s | 3.38s | 1.90% | 174.95s |
| **MACRO AVERAGE** | — | — | **17.47s** | **10.58%** | — |

---

## 3. Methodological Assumptions and Theoretical Implications

### 3.1 What the Projected DER Measures
The projected DER measures the **isolated acoustic leverage of the VAD boundary mask** under the assumption of local speaker continuity:
- If a speaker pause was artificially created by DFN speech suppression (e.g., word-final consonants or quiet syllables in R01), nearest-neighbor propagation assigns the recovered turn fragment to the speaker who was speaking immediately before or after it.
- Because conversational speech exhibits strong temporal locality (a single speaker typically finishes their turn before another begins), this assumption is frequently valid for short gaps ($< 0.5$s).
- This explains why projected DER on R01 dropped dramatically from $38.06\%$ to $25.54\%$ ($-12.52\%$): the recovered $16.87$ seconds of speech were indeed predominantly spoken by the nearest speaker.

### 3.2 What the Projected DER Misses (Limitations)
1. **Turn-Boundary Speaker Changes in Recovered Turns**:
   If an entire short turn by an *alternating* speaker was previously missed by baseline VAD, nearest-neighbor propagation will misattribute that turn to the preceding speaker, turning a potential Miss into Speaker Confusion.
2. **Cluster Geometry Invariance**:
   Projected DER does not test whether the embeddings extracted from newly detected frames would perturb the affinity matrix, change the eigengap, or cause NeMo NME-SC to estimate a different speaker count $\hat{K}$.
3. **Upper Bound Nature**:
   The projected DER should be regarded as an **optimistic projection / upper bound** of VAD-driven improvement. A complete rerun of embedding extraction and clustering across the newly detected speech frames may introduce slightly higher confusion if the recovered frames contain low-energy speech or acoustic noise.

---

## 4. Formal Protocol Mandate

1. All downstream DER metrics in Stage 5B.1 documentation, tables, and discussions must be strictly prefixed or described as:
   **"VAD-mask-projected DER under frozen baseline speaker assignments"**.
2. Full end-to-end diarization verification (re-extracting TitaNet and WeSpeaker embeddings on optimized VAD segments and clustering them with calibrated NME-SC) is formally deferred to **Stage 5B.4 (Integrated Upstream Diarization Pipeline)**.
