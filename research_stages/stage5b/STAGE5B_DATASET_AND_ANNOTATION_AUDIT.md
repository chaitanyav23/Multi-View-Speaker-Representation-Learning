# Stage 5B.0 — Dataset & Ground Truth Annotation Audit

**Date**: 2026-11-03  
**Stage**: Stage 5B.0  
**Status**: COMPLETE AUDIT  
**Data Scope**: 8 Canonical Evaluation Recordings (`R01` – `R08`)  

---

## 1. Corpus Overview & Acoustic Taxonomy

The evaluation corpus comprises 8 heterogeneous recordings spanning three primary acoustic domains:
1. **Body-Worn Camera (BWC) Audio**: Law enforcement field audio (Axon Body 2/3), characterized by extreme non-stationary acoustic noise, wind turbulence, sirens, engine rumble, physical clothing rubbing, and highly imbalanced speaker participation.
2. **Broadcast & Media Cross-Talk**: High-energy television talk shows and parliamentary debates featuring rapid turn-taking, competitive interruptions, heckling, and sustained overlapping speech.
3. **Controlled Studio Dialogue**: Near-field microphone podcast and clean conversational exchanges with minimal background noise and orderly turn-taking.

---

## 2. Granular Per-Recording Ground Truth Audit

The table below presents exact empirical measurements derived directly from the canonical uniform JSON annotations (`research_stages/stage1_data_prep/annotations/uniform_json/{rec_id}.json`).

| Rec ID | Domain / Description | Total Audio (s) | Total Speech (s) | Speech % | Overlap Duration (s) | Overlap % of Speech | GT Spk ($k^*$) | Max Spk Share (%) | Min Spk Share (%) | Imbalance Ratio ($\frac{\text{Max}}{\text{Min}}$) | GT Segments |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R01** | Clean synthetic dialogue | 16.03 | 15.35 | 95.8% | 0.00 | 0.0% | 2 | 53.1% | 46.9% | 1.13 : 1 | 8 |
| **R02** | Parliamentary debate shouting | 185.00 | 162.00 | 87.6% | 56.00 | **34.6%** | 3 | 47.7% | 18.3% | 2.60 : 1 | 24 |
| **R03** | Studio podcast interview | 190.00 | 188.00 | 98.9% | 2.00 | 1.1% | 2 | 77.4% | 22.6% | 3.42 : 1 | 18 |
| **R04** | Heated talk-show debate | 190.00 | 186.00 | 97.9% | 95.00 | **51.1%** | 4 | 30.2% | 19.2% | 1.57 : 1 | 42 |
| **R05** | BWC traffic stop (engine/siren) | 100.00 | 75.00 | 75.0% | 0.00 | 0.0% | 2 | 57.3% | 42.7% | 1.34 : 1 | 14 |
| **R06** | BWC domestic disturbance | 385.00 | 371.00 | 96.4% | 2.00 | 0.5% | 9 | 31.6% | 1.1% | **29.75 : 1** | 68 |
| **R07** | BWC cheap shot altercation | 124.00 | 123.00 | 99.2% | 1.00 | 0.8% | 3 | **87.0%** | **0.8%** | **107.0 : 1** | 22 |
| **R08** | Synthetic noisy dialogue | 20.00 | 19.38 | 96.9% | 0.00 | 0.0% | 2 | 84.5% | 15.5% | 5.46 : 1 | 6 |
| **Total / Mean** | **Full Benchmark Suite** | **1210.03 s (20.17 min)** | **1139.73 s** | **94.2%** | **156.00 s** | **13.7%** | **3.38** (avg) | **58.6%** | **20.9%** | **18.9 : 1** (avg) | **202** |

---

## 3. In-Depth Recording Characterization

### R01: Clean Studio Dialogue (`R01_clean_dialogue_01.wav`)
- **Acoustics**: Studio SNR (>30 dB), zero overlap, perfectly alternating turns between 2 speakers.
- **Diarization Profile**: Baseline DER = 34.50%. The error is entirely driven by VAD boundary imprecision (Miss: 27.91%, FA: 6.59%, SpkErr: 0.00%). The representation fusion perfectly clusters the 2 speakers with 0.0% speaker confusion.

### R02: Parliamentary Debate (`R02_parliament_debate_01.wav`)
- **Acoustics**: Reverberant hall acoustics with constant background murmuring and aggressive interruptions.
- **Overlap Profile**: 56.0 seconds of speech (34.6%) contains two or more speakers speaking simultaneously.
- **Diarization Profile**: Baseline DER = 62.90% (Miss: 49.33%, FA: 6.09%, SpkErr: 7.48%). Because the baseline pipeline assigns only one speaker label per time step, all overlapping speech is treated as a missed detection for the secondary speaker. This single structural limitation accounts for almost the entirety of the 49.33% miss rate.

### R03: Studio Interview Podcast (`R03_interview_podcast_01.wav`)
- **Acoustics**: Professional cardioid microphone capture, high SNR. Orderly dialogue with 2 speakers.
- **Diarization Profile**: Best performing recording in the benchmark (DER = 15.65%, Miss: 7.00%, FA: 4.88%, SpkErr: 3.77%). Speaker clustering is clean ($k=2$).

### R04: Talk Show Argument (`R04_talk_show_argument_01.wav`)
- **Acoustics**: Heated 4-speaker televised debate. Extreme conversational contention.
- **Overlap Profile**: 95.0 seconds of speech (51.1%) is simultaneous multi-speaker cross-talk. Over half of the speech in this recording involves simultaneous speakers.
- **Diarization Profile**: Baseline DER = 49.19% (Miss: 28.52%, FA: 4.70%, SpkErr: 15.97%). Overlap truncation causes the 28.52% miss rate, while mixed-speaker subsegments contaminate embeddings, driving speaker confusion to 15.97%.

### R05: BWC Traffic Stop (`R05_bodycam_traffic_stop_01.wav`)
- **Acoustics**: Real police body-worn camera. Severe low-frequency patrol car engine rumble, siren chirps, and passing highway traffic.
- **Diarization Profile**: Baseline DER = 41.52% (Miss: 3.73%, FA: 37.79%, SpkErr: 0.00%). Speaker clustering is flawless (0.00% SpkErr for the 2 speakers), but VAD false alarms reach 37.79% because non-speech acoustic rumblings trigger the Pyannote speech detector.

### R06: BWC Domestic Disturbance (`R06_bodycam_domestic_call_01.wav`)
- **Acoustics**: Real BWC recording inside an echoing residential apartment during a chaotic domestic call. Contains 9 distinct ground-truth speakers.
- **Speaker Distribution**: Highly skewed distribution:
  - Dominant speakers: Officer 1 (119s), Civilian 1 (105s), Officer 2 (48s), Civilian 2 (32s), Officer 3 (24s), Officer 4 (17s), Dispatcher (16s).
  - Fleeting speakers: Bystander 1 (6s), Bystander 2 (4s).
- **Diarization Profile**: Baseline DER = 51.52% (Miss: 7.62%, FA: 41.33%, SpkErr: 2.57%). Under-clustering occurs: NME-SC selects $k=6$ (discarding the fleeting speakers into majority clusters), while environmental acoustics generate 41.33% false alarms.

### R07: BWC Cheap Shot Altercation (`R07_bwc_cheap_shot.wav`)
- **Acoustics**: Real BWC recording of an outdoor physical altercation. Severe wind noise, clothing friction, yelling, and heavy clipping.
- **Speaker Distribution**: Pathological imbalance:
  - Speaker 1: 107.0s (87.0% of speech)
  - Speaker 2: 15.0s (12.2% of speech, fractured into brief 1–2s interjections)
  - Speaker 3: 1.0s (0.8% of speech, single background command)
  - Imbalance ratio: 107.0 : 1!
- **Diarization Profile**: Baseline DER = 64.95% (Miss: 8.52%, FA: 4.09%, SpkErr: 52.34%). NeMo NME-SC selects $k=1$, collapsing all 3 speakers into a single speaker cluster!

### R08: Synthetic Noisy Two-Speaker (`R08_synthetic_noisy_two_speaker.wav`)
- **Acoustics**: Short 20-second synthetic conversation with injected stationary Gaussian noise.
- **Diarization Profile**: Baseline DER = 51.35% (Miss: 42.57%, FA: 8.78%, SpkErr: 0.00%). VAD over-prunes soft onset speech, resulting in 42.57% miss rate, but clustering is completely error-free (0.00% SpkErr).

---

## 4. Key Takeaways for Stage 5B System Design

1. **Overlap is concentrated in specific acoustic domains**:
   - Overlap is not uniformly distributed: R02 (34.6%) and R04 (51.1%) account for 96.8% (151s out of 156s) of all overlap across the entire dataset!
   - Implementing overlap attribution will directly slash Miss rates in R02 and R04 without affecting single-speaker BWC recordings.

2. **VAD false alarms are concentrated in BWC domains**:
   - R05 (37.79% FA) and R06 (41.33% FA) account for the vast majority of macro false alarm error.
   - Robust non-speech suppression (energy gating, SNR conditioning) will selectively heal R05 and R06.

3. **Speaker confusion is localized to count-collapse sessions**:
   - Excluding R07 (52.34% SpkErr) and R04 (15.97% SpkErr), the remaining 6 recordings have an average speaker error of only 2.30%!
   - Resolving the $k=1$ collapse in R07 will immediately reduce overall macro speaker error by over 6.5 percentage points.
