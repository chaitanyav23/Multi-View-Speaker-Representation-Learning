# Stage 5B.1: Short Conversational Turn Analysis & Turn-Filtering Audit

## 1. Executive Summary

In speech processing pipelines, a common heuristic is to set a high minimum speech duration threshold (such as `min_duration_on = 0.50s`) to suppress transient noise spikes. This experiment (Experiment 1) investigates the trade-off between suppressing false alarms and retaining genuine brief conversational turns (e.g., backchannels, quick interjections, single-word affirmations like "yeah", "right", "mate").

### Key Empirical Findings:
1. **The Cost in Conversational Speech (R01 Sara Sharif)**:
   - In rapid multi-party dialogue (R01), increasing `min_duration_on` from `0.05s` to `0.50s` increases Missed Speech from **$27.78\%$ to $29.59\%$** ($+1.81\%$ speech lost).
   - Downstream Diarization Error Rate (DER) worsens from **$38.06\%$ to $39.20\%$** ($+1.14\%$ absolute degradation).
2. **The Noise-Suppression Benefit in Extreme BWC (R05 Southwest Pilot)**:
   - In high-noise cockpit/BWC recordings (R05), filtering out bursts $< 0.50s$ reduces False Alarms from **$30.13\%$ down to $26.56\%$** ($-3.57\%$ FA reduction).
   - Downstream DER improves from **$58.69\%$ to $56.90\%$** ($-1.79\%$ absolute improvement).
3. **Macro Trade-off**:
   - Across all 8 recordings, macro DER changes only modestly: $46.16\%$ (0.05s) $\rightarrow$ $46.17\%$ (0.10s) $\rightarrow$ $46.14\%$ (0.20s) $\rightarrow$ $46.11\%$ (0.30s) $\rightarrow$ $45.93\%$ (0.50s).
   - However, the macro metric masks a critical failure mode: aggressive filtering systematically destroys short conversational turns in low-noise interactions while acting as a crude band-aid for noise in bodycam recordings.
4. **Architectural Recommendation**:
   - **Do NOT use `min_duration_on = 0.50s`** as a global filter.
   - Maintain `min_duration_on = 0.10s` (the Pyannote default).
   - Address environmental noise false alarms through **dual-threshold hysteresis** ($T_{\text{onset}}=0.60/0.70, T_{\text{offset}}=0.35/0.40$) and consensus stream combination rather than destructive temporal pruning.

---

## 2. Experimental Setup & Grid

We held all parameters fixed to the Stage 5B.3 M1 baseline ($T_{\text{onset}}=0.481, T_{\text{offset}}=0.481, \text{min\_duration\_off}=0.15s$, DFN Path A audio) and swept `min_duration_on` across 5 conditions:
- **C10**: `min_duration_on = 0.05s` (50 ms, highly sensitive)
- **C00**: `min_duration_on = 0.10s` (100 ms, Pyannote default baseline)
- **C11**: `min_duration_on = 0.20s` (200 ms)
- **C12**: `min_duration_on = 0.30s` (300 ms)
- **C13**: `min_duration_on = 0.50s` (500 ms, legacy Phase 20 heuristic)

---

## 3. Detailed Results Across Turn Lengths

### 3.1 Per-Recording Downstream DER ($c=0.0s$)

| Recording ID | Nature of Recording | C10 (0.05s) | C00 (0.10s) | C11 (0.20s) | C12 (0.30s) | C13 (0.50s) | Trend ($0.05s \rightarrow 0.50s$) |
|---|---|---|---|---|---|---|---|
| **R01** | Rapid Multi-party Dialogue | **38.06%** | 38.06% | 38.41% | 38.80% | 39.20% | **Worsens (+1.14%)** |
| **R02** | High Overlap Interruption | **62.27%** | 62.30% | 62.34% | 62.40% | 62.43% | **Worsens (+0.16%)** |
| **R03** | BWC Interaction | 33.52% | 33.52% | 33.52% | 33.52% | 33.52% | Invariant |
| **R04** | Formal Broadcast / Debate | 40.39% | 40.42% | 40.42% | 40.42% | 40.39% | Invariant |
| **R05** | High-Noise Cockpit / ATC | 58.69% | 58.76% | 58.60% | 57.80% | **56.90%** | **Improves (-1.79%)** |
| **R06** | Severe Noise BWC | 72.58% | 72.58% | 72.48% | 72.33% | **71.96%** | **Improves (-0.62%)** |
| **R07** | High-Stress BWC Altercation | 45.18% | 45.18% | 45.18% | 45.18% | **44.61%** | **Improves (-0.57%)** |
| **R08** | Formal Parliamentary Speech | 18.59% | 18.59% | 18.59% | 18.59% | **18.46%** | Invariant / Slight gain |
| **MACRO** | **All 8 Recordings** | **46.16%** | **46.17%** | **46.14%** | **46.11%** | **45.93%** | **-0.24% Net Macro** |

---

## 4. Error Decomposition: Miss vs False Alarm

### 4.1 Missed Speech Rate ($T_{\text{Miss}} / T_{\text{ref}}$)
As `min_duration_on` increases from 0.05s to 0.50s:
- **R01**: Miss rises from **27.78%** to **29.59%** ($+1.81\%$). In R01, over 1.7 seconds of genuine short turns are pruned away.
- **R02**: Miss rises from **49.94%** to **50.39%** ($+0.45\%$).
- **R05**: Miss rises from **27.03%** to **29.74%** ($+2.71\%$).
- **Macro Miss**: Rises from **$24.40\%$** to **$25.13\%$** ($+0.73\%$ macro speech lost).

### 4.2 False Alarm Rate ($T_{\text{FA}} / T_{\text{ref}}$)
As `min_duration_on` increases from 0.05s to 0.50s:
- **R01**: FA remains at **0.49%** (clean studio environment; no transient noise to filter).
- **R05**: FA drops from **30.13%** to **26.56%** ($-3.57\%$).
- **R06**: FA drops from **41.34%** to **40.55%** ($-0.79\%$).
- **R07**: FA drops from **14.12%** to **12.98%** ($-1.14\%$).
- **Macro FA**: Drops from **$13.33\%$** to **$12.60\%$** ($-0.73\%$ macro FA reduction).

---

## 5. Why the 0.50s Legacy Filter is Methodologically Flawed

1. **Symmetric Macro Illusion**:
   At the macro level, the $+0.73\%$ increase in Miss is canceled by the $-0.73\%$ decrease in False Alarm, producing a net change of nearly zero ($-0.24\%$). This creates the false impression that a 0.50s filter is beneficial or harmless.
2. **Harm to Dialogue Attribution**:
   In dialogue and conversational intelligence, short turns carry disproportionate pragmatic significance (confirmations, interjections, objections). Deleting speech turns $< 500$ ms guarantees that conversational turns such as "No", "Yes", "Stop", and "Wait" will never be attributed to the speaker, inflating conversational word error rate (cpWER).
3. **Failure to Address the Root Cause in Noise**:
   Even with a 0.50s filter, R05 still suffers from **$26.56\%$** false alarm rate, and R06 still suffers from **$40.55\%$** false alarm rate. Noise bursts in bodycam audio frequently exceed 500 ms (e.g. sirens, engine hums, road rumble). Temporal length filtering is fundamentally incapable of distinguishing sustained noise from sustained speech.

### Conclusion:
`min_duration_on = 0.10s` (or 0.05s) should be retained as the universal minimum turn duration. Noise robustness must be handled through acoustic probability modeling (hysteresis and multi-stream consensus), not temporal truncation.
