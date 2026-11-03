# Stage 5B.2: Forensic Failure Analysis & Acoustic Stratification

## 1. Executive Summary

This forensic failure analysis investigates the precise mechanisms that determine the success or failure of overlap handling across heterogeneous acoustic domains.

### Three Primary Failure Modes Identified:
1. **The R02 vs R04 Asymmetry (Dialogue vs Cross-Talk Chaos)**: Overlap attribution dropped DER in R04 by $-9.74\%$, but had minimal impact in R02 ($-0.17\%$).
2. **DFN Spectral Suppression of Secondary Speakers**: DeepFilterNet3 treats secondary overlapping voices as acoustic interference and filters them out, rendering overlap detection on DFN audio ineffective (Recall = $1.1\%$).
3. **Spurious Overlap False Alarms in Low-Overlap Audio**: In recordings without overlapping speech (R01, R05, R08), background acoustic transients trigger false secondary speaker lines, inflating False Alarm rates.

---

## 2. Failure Mode 1: Why R04 Succeeded and R02 Failed

Both R04 and R02 have near-50% ground-truth overlap ($51.08\%$ in R04, $45.68\%$ in R02). Yet their response to overlap attribution was radically different:
- **R04**: Miss dropped from $32.00\%$ to $13.71\%$; DER improved from $40.39\%$ to **$30.65\%$** ($-9.74\%$).
- **R02**: Miss dropped marginally from $49.96\%$ to $48.78\%$; DER remained at **$62.75\%$** (baseline $62.30\%$).

### Acoustic Diagnosis:
1. **Speaker Cardinality in Overlap Regions**:
   - In R04 (`piers_robinson`), overlap is almost exclusively **two-party structured dialogue**: Piers Robinson and his guest talking over each other. When one interrupts, the secondary speaker is unambiguously the other primary interlocutor.
   - In R02 (`dennis_skinner`), overlap is **multi-party shouting**: Dennis Skinner is speaking from the dispatch box while 3 to 4 different MPs in the chamber heckle simultaneously.
2. **Signal-to-Noise Ratio of Secondary Speakers**:
   - In R04, both speakers wear broadcast lapel microphones; both voices have comparable amplitude ($>-20$ dBFS). The Pyannote overlap detector achieves **$38.2\%$ activation** with high confidence.
   - In R02, Dennis Skinner speaks into the primary microphone, while the hecklers are distant room voices ($<-35$ dBFS, high reverberation). The PyanNet overlap detector achieves only **$11.8\%$ recall** because the distant heckles do not trigger the neural activation threshold.
3. **Primary Error Source**:
   - In R02, the baseline $49.96\%$ Miss was NOT solely caused by overlap; it was primarily caused by **DFN speech suppression of Dennis Skinner's own quiet phrases**! In Stage 5B.1, switching to raw mono VAD (C50) dropped Miss from $49.96\%$ to $40.89\%$. Overlap was secondary to acoustic attenuation.

---

## 3. Failure Mode 2: DFN Spectral Stripping of Secondary Speakers

In Experiment 2, we directly compared running overlap detection on DeepFilterNet3 (`dfn`) vs Raw 16k Mono (`mono`):

| Overlap Audio Stream | Overlap Precision | Overlap Recall | Overlap F1 | R04 Miss Rate | R04 DER (c=0) |
|---|---|---|---|---|---|
| **DFN Stream (`O20`)** | 20.17% | **1.14%** | **0.0213** | 30.03% | 38.99% |
| **Mono Stream (`O21`)** | 22.27% | **24.32%** | **0.1324** | **13.71%** | **32.19%** |

### The Mechanism:
DeepFilterNet3 is an aggressive neural speech enhancement model trained to suppress non-stationary noise. When two speakers speak simultaneously:
- DFN locks onto the louder, primary speaker's pitch tracks and spectral envelope.
- DFN interprets the secondary speaker's overlapping voice as competing acoustic babble/noise and applies strong spectral attenuation.
- By the time the audio reaches the PyanNet overlap detector, the secondary speaker's formant structure has been stripped, causing Overlap Recall to collapse to **$1.14\%$**!
- **Mandate**: **Overlap detection MUST ALWAYS be executed on the Raw 16k Mono stream**, NEVER on aggressively denoised audio.

---

## 4. Failure Mode 3: The Global False Alarm Penalty in Clean Audio

In single-label diarization, when an overlap detector fires, it inserts a duplicate speaker segment:
$$\text{Hypothesis}(t) = \{S_{\text{primary}}, S_{\text{secondary}}\}$$

- If $t$ is a true overlap frame ($|Y(t)| \ge 2$), adding $S_{\text{secondary}}$ eliminates a Miss error ($\Delta \text{DER} = -1.0$).
- If $t$ is a single-speaker frame ($|Y(t)| = 1$), adding $S_{\text{secondary}}$ introduces a pure False Alarm error ($\Delta \text{DER} = +1.0$)!

In our corpus:
- R01, R05, R08 have **0.0 seconds of overlap** in ground truth.
- In R05 (cockpit noise), engine rumble and radio distortion triggered spurious overlap activations, adding $7.45$ seconds of false overlap, raising False Alarm from $30.13\%$ to $34.72\%$.
- In R06 (police bodycam), loud shouting and vehicle sirens added $36.10$ seconds of false overlap, raising False Alarm from $41.34\%$ to $46.88\%$.

### The Solution for Stage 5B.4 Integration:
Global, static overlap attribution is suboptimal across heterogeneous datasets.
To achieve optimal performance across the entire corpus:
1. **Confidence Gating & Conservative Abstention (`M4`)**: Require temporal consensus before attributing secondary speakers.
2. **Acoustic Overlap-Density Condition**: Only deploy overlap attribution when detected overlap density exceeds a minimum threshold (e.g., in debates, interviews, and cross-talk dialogues), while defaulting to single-label assignment in high-noise bodycam/ATC audio.
