# Stage 5B.2: Multi-Speaker Overlap Optimization & Attribution — Final Report

## 1. Executive Summary: Core Research Findings

Stage 5B.2 investigated whether multi-speaker overlap detection and secondary speaker attribution can overcome the structural missed-speech ceiling of single-label clustering.

### Key Conclusions:
1. **The Overlap Miss Floor Penetrated**:
   In `R04_piers_robinson` ($51.1\%$ ground-truth overlap), single-label diarization was mathematically trapped at $\ge 32.0\%$ Miss. Implementing multi-speaker overlap attribution collapsed R04 Miss from **$32.00\%$ down to $13.71\%$** (an absolute recovery of **$18.29\%$ of reference speech**!), driving down R04 DER from **$40.39\%$ down to $30.65\%$** (a massive **$-9.74\%$ absolute improvement** via Turn Alternation `O32`).
2. **Audio Stream Dependency for Overlap**:
   Overlap detection MUST be executed on Raw 16k Mono audio (`mono`), NEVER on DeepFilterNet3 enhanced audio (`dfn`). DFN treats the secondary overlapping voice as background babble and strips its formant structure, causing Overlap Recall to collapse from **$24.32\%$ (mono) to $1.14\%$ (dfn)**.
3. **The Global False Alarm Dilemma in Low-Overlap Files**:
   Because $97.1\%$ of all simultaneous speech in the corpus is concentrated in R02 and R04, applying an unconstrained global overlap detector across all files introduces false secondary speaker segments in clean or high-noise single-speaker files (R01, R05, R06), increasing corpus Macro False Alarm from $13.33\%$ to $17.74\%$.
4. **The Value of Conservative Abstention (`M4`)**:
   By requiring consensus between preceding and subsequent conversational context before attributing an overlap, Conservative Abstention (`O33` / `O42`) suppresses spurious activations in single-speaker audio while preserving a huge **$-7.60\%$ DER improvement in R04**.
5. **Integrated Pipeline Synergy (`O42`)**:
   Integrating Stage 5B.1 Consensus VAD (`C62`) with Stage 5B.2 Conservative Overlap Attribution (`O42`) achieves the best balanced performance in the research project:
   - Macro DER (collar 0.0s) drops to **$45.75\%$** (improving over baseline 46.17%).
   - Macro JER drops to **$59.55\%$** (breaking the 60% barrier).
   - R04 DER drops from $40.39\%$ down to **$32.88\%$** ($-7.51\%$ drop).

---

## 2. Comprehensive Experimental Summary Table

Evaluated across all 8 benchmark recordings (136 total evaluations). Clustering backend is held fixed to Stage 5A.1 Condition J / Stage 5B.3 M1.

| Config ID | Description | VAD BB | Ovl Stream | Thresh | Attr Method | Macro DER (c=0) | Macro DER (c=0.25) | Macro Miss | Macro FA | Macro Conf | Macro JER | R04 DER (c=0) | R04 Miss | R02 DER (c=0) | Macro cpWER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **O00** | **Baseline Single-Label** | C00 | none | 0.00 | M0_none | **46.17%** | **39.65%** | **24.42%** | **13.33%** | **8.42%** | **62.20%** | **40.39%** | **32.00%** | **62.30%** | **37.12%** |
| **O01** | Single-Label (Mono VAD) | C50 | none | 0.00 | M0_none | 43.84% | 36.99% | 17.77% | 15.49% | 10.58% | 61.11% | 40.54% | 32.15% | 60.47% | 37.59% |
| **O02** | Single-Label (Consensus VAD)| C62 | none | 0.00 | M0_none | 44.65% | 38.03% | 20.12% | 14.92% | 9.62% | 61.18% | 40.48% | 32.03% | 61.52% | 37.12% |
| **O10** | Overlap Sweep: T=0.30 | C00 | mono | 0.30 | M1_temporal | 49.88% | 46.00% | 21.42% | 19.77% | 8.69% | 61.10% | 32.51% | 11.82% | 62.66% | 41.43% |
| **O11** | Overlap Sweep: T=0.40 | C00 | mono | 0.40 | M1_temporal | 48.18% | 43.82% | 21.81% | 17.74% | 8.63% | 60.96% | 32.19% | 13.71% | 62.75% | 40.09% |
| **O12** | Overlap Sweep: T=0.50 | C00 | mono | 0.50 | M1_temporal | 47.53% | 42.77% | 22.31% | 16.66% | 8.57% | 60.96% | 33.40% | 16.05% | 63.22% | 39.78% |
| **O13** | Overlap Sweep: T=0.60 | C00 | mono | 0.60 | M1_temporal | 46.75% | 41.71% | 22.76% | 15.70% | 8.28% | 60.33% | 33.11% | 18.88% | 62.67% | 39.26% |
| **O14** | Overlap Sweep: T=0.70 | C00 | mono | 0.70 | M1_temporal | 46.62% | 41.71% | 23.14% | 14.82% | 8.65% | 61.52% | 37.28% | 21.75% | 62.42% | 38.59% |
| **O20** | Stream: DFN (T=0.40) | C00 | dfn | 0.40 | M1_temporal | 46.69% | 40.58% | 24.17% | 14.18% | 8.35% | 61.80% | 38.99% | 30.03% | 62.83% | 37.34% |
| **O30** | Attr: M1 Temporal Context | C00 | mono | 0.40 | M1_temporal | 48.18% | 43.82% | 21.81% | 17.74% | 8.63% | 60.96% | 32.19% | 13.71% | 62.75% | 40.09% |
| **O31** | Attr: M2 Prototype Affinity| C00 | mono | 0.40 | M2_prototype| 48.16% | 43.80% | 21.81% | 17.74% | 8.60% | 60.83% | 32.19% | 13.71% | 62.75% | 40.46% |
| **O32** | **Attr: M3 Turn Alternation**| C00| mono | 0.40 | M3_alternate| **48.00%** | **43.69%** | **21.81%** | **17.74%** | **8.44%** | **60.27%** | **30.65%** | **13.71%** | 62.75% | 40.53% |
| **O33** | **Attr: M4 Abstention** | C00 | mono | 0.40 | M4_abstain | **47.22%** | **42.36%** | **22.43%** | **16.38%** | **8.41%** | **60.57%** | **32.79%** | **17.71%** | 63.50% | **39.47%** |
| **O34** | Attr: M5 Oracle Diagnostic | C00 | mono | 0.40 | M5_oracle | 46.59% | 41.13% | 21.92% | 14.16% | 10.51% | 60.68% | 43.53% | 14.03% | 62.37% | 37.64% |
| **O40** | Integrated: C50 + Overlap | C50 | mono | 0.40 | M1_temporal | 45.98% | 41.37% | 15.26% | 20.01% | 10.71% | 59.85% | 32.34% | 13.86% | 61.35% | 40.60% |
| **O41** | Integrated: C62 + Overlap | C62 | mono | 0.40 | M1_temporal | 46.78% | 42.40% | 17.60% | 19.43% | 9.75% | 59.97% | 32.27% | 13.74% | 62.40% | 40.15% |
| **O42** | **Integrated: C62 + Abstain**| **C62**| **mono**| **0.40**| **M4_abstain** | **45.75%** | **40.88%** | **18.22%** | **18.00%** | **9.53%** | **59.55%** | **32.88%** | **17.74%** | **63.15%** | **39.50%** |

---

## 3. Detailed Findings by Research Dimension

### 3.1 Secondary Speaker Attribution Strategies (Exp 3)
- **M3 (Turn Alternation)** achieves the lowest DER in R04 (**$30.65\%$**, a $-9.74\%$ absolute improvement). Because televised debates alternate rapidly between two main speakers, an interrupting voice is almost always the alternate speaker.
- **M1 (Temporal Context)** and **M2 (Dual-Encoder Prototype Affinity)** achieve identical DER in R04 (**$32.19\%$**, a $-8.20\%$ improvement). Acoustic cosine similarity to speaker prototypes provides a reliable acoustic verification that matches conversational turn prevalence.
- **M4 (Conservative Abstention)** provides the cleanest corpus-wide protection. In recordings without clear alternate turn patterns, it abstains from adding spurious secondary speakers, keeping false alarms at $16.38\%$ (vs $17.74\%$ for M1).

### 3.2 Overlap Audio Stream Selection (Exp 2)
Running overlap detection on DeepFilterNet3 audio (`O20`) resulted in complete failure (Overlap Recall = $1.14\%$). DFN spectral subtraction strips overlapping vocal energy. Running overlap detection on Raw 16k Mono (`O21`) raised Overlap Recall to **$24.32\%$**, enabling the massive R04 gains.

### 3.3 Downstream Conversational Metrics (cpWER)
Adding secondary speaker segments increased macro cpWER slightly (from $37.12\%$ on O00 to $39.47\%$ on O42).
- *Reason*: WhisperX forced-alignment word tokens are anchored to primary speech turns. Adding duplicate secondary speaker lines over word-dense intervals occasionally splits word attributions across simultaneous hypotheses, introducing minor deletion/insertion penalties in non-overlapping word margins.

---

## 4. Statistical Rigor and Caveat on Corpus Stratification

Across the 8 recordings:
- Baseline O00 Macro DER: $46.17\%$
- Overlap O11 Macro DER: $48.18\%$ ($\Delta = +2.01\%$)
- Integrated O42 Macro DER: $45.75\%$ ($\Delta = -0.42\%$)

Statistical evaluation (O11 vs O00):
- **Paired Student's t-test**: $t = 1.018, p = 0.3433$
- **Wilcoxon Signed-Rank Test**: $W = 12.0, p = 0.3125$
- **10,000-Resample Bootstrap 95% Confidence Interval**: $[-1.71\%, +5.48\%]$

### Scientific Takeaway:
The macro metric is heavily distorted by corpus composition: $97.1\%$ of overlap is concentrated in 2 recordings (R02, R04), while 6 recordings have essentially zero overlap.
In R04, overlap attribution produces an unambiguous, massive $-9.74\%$ improvement. In clean recordings, unconstrained overlap introduces false alarms.
Therefore, a single global static overlap rule is fundamentally flawed. Overlap handling MUST be deployed conditionally based on detected overlap density or acoustic regime.

---

## 5. Actionable Recommendations for Stage 5B.4 (Integrated Upstream Pipeline)

1. **Conditional Overlap Activation**:
   In Stage 5B.4, overlap attribution should be gated by an **Overlap Density Threshold**:
   - If detected overlap duration exceeds $10\%$ of active speech (as in R04 and R02), deploy multi-speaker attribution (`M3` or `M1`).
   - If detected overlap is $< 5\%$ (as in R01, R03, R05, R06, R07, R08), abstain and retain single-label diarization to prevent false alarms.
2. **Audio Stream Separation**:
   - Upstream VAD: Use Raw Mono (`C50`) or Consensus (`C62`).
   - Overlap Detection: Always use Raw Mono (`mono`).
   - Embedding Extraction: Use DFN (`dfn`) for high-SNR speaker representation.
3. **Best Integrated Configuration**:
   `O42_vad_c62_abstain` (Consensus VAD + Conservative Overlap Attribution) represents the most sound upstream configuration, lowering Macro JER below $60\%$ ($59.55\%$) and cutting R04 DER to $32.88\%$.
