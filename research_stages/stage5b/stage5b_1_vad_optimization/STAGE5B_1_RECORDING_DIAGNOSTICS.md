# Stage 5B.1: Recording-by-Recording VAD Diagnostics & Acoustic Stratification

## 1. Executive Summary

This diagnostic report provides a granular forensic analysis of Voice Activity Detection (VAD) behavior across all 8 canonical recordings in the research corpus.

The 8 recordings represent four distinct acoustic regimes:
1. **Clean Multi-party Dialogue** (R01 Sara Sharif, R08 UK Parliament): Low background noise, soft speech turns, high vulnerability to speech attenuation.
2. **Severe Noise Bodycam / Cockpit** (R05 Southwest Pilot, R06 Rich Karen): Heavy acoustic rumble, engine/ATC interference, high false-alarm rates.
3. **High Overlap Dialogue & Debate** (R02 Dennis Skinner, R04 Piers Robinson): Severe simultaneous speech ($34.6\%$ and $51.1\%$), where VAD performance hits a hard ceiling imposed by single-label clustering.
4. **Intermediate BWC / Police Field Audio** (R03 Disabled Parking, R07 Cheap Shot): Transient outdoor noise, mixed conversational turns, moderate false alarm and miss rates.

---

## 2. Granular Analysis by Acoustic Regime

### 2.1 Clean Multi-Party Dialogue: R01 (`sara_sharif`) & R08 (`uk_parliament`)

#### R01 Sara Sharif (4.8% Overlap, High Turn Dynamics)
- **The Problem**: In baseline Condition C00 (DFN Path A), Missed Detection is **27.87%**, resulting in a DER of **38.06%**. DeepFilterNet3 denoiser attenuates soft turn-initial vowels and quiet conversational responses, causing Pyannote VAD to miss over 25 seconds of speech.
- **The Breakthrough with Raw Mono Audio (C50)**:
  - On the uncorrupted 16k mono stream, VAD Recall jumps from **72.4% to 92.1%**.
  - Missed speech collapses from **$27.87\%$ down to $10.37\%$** (a **$17.50\%$ absolute drop**).
  - Downstream DER drops from **$38.06\%$ down to $25.54\%$** (a massive **$12.52\%$ absolute improvement**).
  - False alarm remains virtually negligible ($0.71\%$).
- **Impact of High Threshold (C25 $T=0.80$)**: Disastrous. Missed speech surges to $35.72\%$, blowing DER out to $44.41\%$.

#### R08 UK Parliament (1.8% Overlap, High Reverberation / Microphony)
- **Baseline**: Already strong performance (DER = 18.59%, Miss = 13.03%, FA = 5.13%).
- **Raw Mono (C50)**: DER improves slightly to **17.39%** (Miss = 12.82%, FA = 4.24%).
- **Combined Consensus + Hysteresis (C62)**: DER improves to **16.28%** (Miss drops to 10.73%), achieving the lowest error rate in the entire evaluation.

---

### 2.2 Severe Noise Bodycam: R05 (`southwest_pilot`) & R06 (`rich_karen`)

#### R05 Southwest Pilot (Cockpit Noise, Radio Chatter)
- **The Problem**: Baseline False Alarm is catastrophic: **$30.13\%$** of reference speech duration is falsely triggered silence/noise, driving DER to **$58.76\%$**.
- **Threshold Sweeps (Experiment 2)**:
  - Elevating threshold from $T=0.481$ to $T=0.80$ ($C25$) successfully suppresses false alarms:
    - False Alarm drops from **$30.13\%$ down to $21.72\%$** ($-8.41\%$ reduction).
    - DER drops from **$58.76\%$ down to $53.96\%$** ($-4.80\%$ improvement).
  - However, on the raw mono stream ($C50$), without DFN filtering, FA explodes to $41.41\%$, worsening DER to $60.88\%$.
- **Finding**: Cockpit noise requires both DFN noise suppression and an elevated speech probability threshold to prevent spurious acoustic triggering.

#### R06 Rich Karen (Bodycam Wind, Shouting, Vehicle Rumble)
- **The Problem**: The worst baseline DER in the corpus: **$72.58\%$**, with a False Alarm rate of **$41.34\%$** and Confusion of $19.94\%$.
- **Threshold Sweeps**:
  - Setting $T=0.80$ reduces False Alarm from **$41.34\%$ down to $35.92\%$** ($-5.42\%$).
  - DER drops from **$72.58\%$ down to $70.30\%$** ($-2.28\%$ improvement).
  - Raw mono ($C50$) increases FA to $44.40\%$ (DER = $72.79\%$).
- **Finding**: Heavy bodycam environmental artifacts cannot be solved by single-stream VAD alone; even with $T=0.80$, over $35\%$ false alarms persist due to high-energy non-speech acoustic events mimicking vocal formats.

---

### 2.3 Overlap-Dominated Recordings: R02 (`dennis_skinner`) & R04 (`piers_robinson`)

#### R04 Piers Robinson (51.1% Overlap, Continuous Broadcast Speech)
- **Acoustic Reality**: Reference speech is active across **$99.8\%$** of the recording ($58.7$ seconds out of $60.1$ seconds).
- **VAD Invariance**:
  - Baseline VAD Recall is **$99.8\%$**; VAD F1 is **$0.985$**.
  - Across all 26 configurations tested, DER remains locked between **$40.39\%$ and $40.67\%$**.
  - Missed speech is locked at **$32.00\%$**; False Alarm is negligible ($1.7\%$).
- **Diagnosis**: The $32.00\%$ miss is **100% caused by overlap omission**. In single-label clustering, when two people talk at the same time, the frame can only be assigned to one speaker, mechanically omitting the secondary speaker. **Upstream VAD optimization has zero leverage on R04**. Only Stage 5B.2 (Overlap Detection & Attribution) can unlock performance here.

#### R02 Dennis Skinner (34.6% Overlap, Parliamentary Cross-talk)
- **Baseline**: DER is **$62.30\%$**, with Missed Speech at **$49.96\%$**.
- **Mono Stream Recovery**:
  - Switching to raw mono ($C50$) recovers heavily attenuated speech: Missed Speech drops from **$49.96\%$ down to $40.89\%$** ($-9.07\%$ drop).
  - Downstream DER drops from **$62.30\%$ to $60.47\%$**.
- **Remaining Ceiling**: The remaining $\sim 35\%$ miss is purely structural overlap. Like R04, further progress requires multi-label assignment.

---

### 2.4 Intermediate BWC Audio: R03 (`disabled_parking`) & R07 (`cheap_shot`)

#### R03 Disabled Parking (4.6% Overlap, Outdoor Urban Noise)
- **Baseline**: DER = **33.52%** (Miss = 12.60%, FA = 12.27%, Conf = 8.65%).
- **Mono Stream (C50)**: DER improves to **32.84%** (Miss drops to 9.14%).
- **High Threshold (C25)**: DER worsens to **36.37%** (Miss surges to 18.63%).

#### R07 Cheap Shot (2.5% Overlap, Radio / Field Altercation)
- **Baseline**: DER = **45.18%** (Miss = 21.51%, FA = 14.12%, Conf = 9.54%).
- **Mono Stream (C50)**:
  - Speech recovery drops Miss from **$21.51\%$ down to $15.59\%$**.
  - DER drops from **$45.18\%$ down to $40.26\%$** (a **$4.92\%$ absolute improvement**).
- **Consensus + Hysteresis (C62)**: DER drops to **41.98%** (Miss drops to 16.69%).

---

## 3. Summary Matrix of Diagnostic Findings

| Recording ID | Primary Error Mode | Best VAD Strategy | Best DER achieved | Baseline DER | Absolute $\Delta$ |
|---|---|---|---|---|---|
| **R01** | Speech Attenuation (DFN) | Raw Mono ($C50$) | **25.54%** | 38.06% | **-12.52%** |
| **R02** | Overlap + DFN Attenuation | Raw Mono ($C50$) | **60.47%** | 62.30% | **-1.83%** |
| **R03** | Mixed Urban FA / Miss | Raw Mono ($C50$) | **32.84%** | 33.52% | **-0.68%** |
| **R04** | Overlap Dominated (51%) | Invariant to VAD | **40.39%** | 40.42% | -0.03% |
| **R05** | Extreme Cockpit FA | High Threshold ($T=0.80, C25$) | **53.96%** | 58.76% | **-4.80%** |
| **R06** | Extreme Bodycam FA | High Threshold ($T=0.80, C25$) | **70.30%** | 72.58% | **-2.28%** |
| **R07** | Mixed BWC / Field Miss | Raw Mono ($C50$) | **40.26%** | 45.18% | **-4.92%** |
| **R08** | Clean Reverberant Speech | Consensus + Hysteresis ($C62$) | **16.28%** | 18.59% | **-2.31%** |

### Strategic Insight:
There is no single static VAD parameter set that simultaneously minimizes error across both R01/R07 (where DFN suppresses speech) and R05/R06 (where noise induces false alarms).
However, **Stream Selection** is overwhelmingly the highest-leverage dimension:
- Globally, the Raw 16k Mono stream ($C50$) produces the single largest Macro DER reduction (**$46.17\% \rightarrow 43.84\%$**, a **$-2.33\%$** absolute drop across all 8 recordings).
- For downstream integration in Stage 5B.4, an acoustic SNR-conditioned or multi-stream consensus strategy offers the cleanest compromise.
