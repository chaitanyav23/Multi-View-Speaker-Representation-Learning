# Stage 5B.1 — Voice Activity Detection (VAD) Optimization: Experiment Protocol

**Date**: 2026-11-03  
**Stage**: Stage 5B.1  
**Status**: COMPLETE & VERIFIED  
**Working Directory**: `research_stages/stage5b/stage5b_1_vad_optimization/`  
**Git Baseline Commit**: `7aaca23`  
**Frozen Baseline Commit**: `ddf089de986f9e7cf026aa507324f3fccaf0cd8f`  

---

## 1. Research Objectives & Scope

Voice Activity Detection (VAD) is the primary upstream component in the speaker diarization pipeline. In Stages 1 through 5A.1.2 and verified in Stage 5B.3, upstream VAD errors:
$$\text{Upstream VAD Error} = \text{Missed Detection (24.41\%)} + \text{False Alarms (13.33\%)} = \mathbf{37.74\%}$$
account for **$81.7\%$ of all diarization error** across the 8-recording benchmark.

The objective of Stage 5B.1 is to conduct a controlled, hypothesis-driven investigation to determine:
1. Whether VAD false alarms can be suppressed in noisy BWC environments (R05, R06) without severely inflating missed speech.
2. Whether genuine short conversational turns can be recovered in dialogue (R01, R08).
3. The impact of the input audio stream: comparing DeepFilterNet3-enhanced audio (`dfn`), raw unsuppressed audio (`16k_mono`), normalized audio (`pathB`), and linear consensus.
4. Whether upstream VAD improvements translate into downstream improvements in NIST Diarization Error Rate (DER), Jaccard Error Rate (JER), and conversational attribution (cpWER).

---

## 2. Experimental Boundaries & Control Standards

- **Strict Component Isolation**: To guarantee that changes in downstream metrics are solely attributable to VAD modifications, the clustering backend is strictly fixed to the established **Stage 5A.1 Condition J / Stage 5B.3 M1 Baseline** (Equal-Weight Affinity Fusion: $S_{\text{fused}} = 0.5 S_{\text{TitaNet}}^{\text{zscore}} + 0.5 S_{\text{ResNet}}^{\text{zscore}}$ with NeMo NME-SC clustering).
- **Zero Retraining or Re-Inference**: Frozen neural network checkpoints (Pyannote PyanNet, TitaNet-Large, ResNet293-LM) are maintained without modification.
- **Cache Reuse**: Continuous frame-level speech activity posteriors $p_{\text{speech}}(t)$ (stride 16.875ms) are loaded directly from the validated caches in `research_stages/stage2_baseline/posteriors/`, eliminating stochastic variance.
- **Evaluation Dual-Reporting**: All VAD metrics are reported under both:
  1. Standard silence-normalized false alarm rate: $FA_{\text{sil}} = \frac{FP}{TN + FP}$.
  2. NIST/DER-aligned reference-speech-normalized false alarm rate: $FA_{\text{ref}} = \frac{FP}{\text{Total Reference Speech}}$.

---

## 3. Systematic Experimental Matrix (26 Evaluated Configurations)

| Exp ID | Configuration Group | Audio Stream | Onset Threshold | Offset Threshold | Min Speech Dur ($s$) | Gap Fill ($s$) | Core Research Hypothesis |
|---|---|---|---|---|---|---|---|
| **C00** | Baseline Reproduction | `dfn` (Path A) | 0.481 | 0.481 | 0.10 | 0.15 | Exact reproduction of Stage 2 / Stage 5A baseline |
| **C10** | Short-Turn Sweep | `dfn` | 0.481 | 0.481 | **0.05** | 0.15 | Minimal filtering retains transient speech |
| **C11** | Short-Turn Sweep | `dfn` | 0.481 | 0.481 | **0.20** | 0.15 | Conservative turn duration floor |
| **C12** | Short-Turn Sweep | `dfn` | 0.481 | 0.481 | **0.30** | 0.15 | Intermediate turn duration floor |
| **C13** | Short-Turn Sweep | `dfn` | 0.481 | 0.481 | **0.50** | 0.15 | Legacy Phase 20 minimum turn duration rule |
| **C20** | Single Threshold | `dfn` | **0.300** | **0.300** | 0.10 | 0.15 | High sensitivity / low threshold |
| **C21** | Single Threshold | `dfn` | **0.400** | **0.400** | 0.10 | 0.15 | Moderately low threshold |
| **C22** | Single Threshold | `dfn` | **0.550** | **0.550** | 0.10 | 0.15 | Moderate noise suppression |
| **C23** | Single Threshold | `dfn` | **0.600** | **0.600** | 0.10 | 0.15 | Strong noise suppression |
| **C24** | Single Threshold | `dfn` | **0.700** | **0.700** | 0.10 | 0.15 | Aggressive noise suppression |
| **C25** | Single Threshold | `dfn` | **0.800** | **0.800** | 0.10 | 0.15 | Extreme noise suppression |
| **C30** | Hysteresis | `dfn` | **0.550** | **0.400** | 0.10 | 0.15 | Asymmetric thresholding prevents chatter |
| **C31** | Hysteresis | `dfn` | **0.600** | **0.350** | 0.10 | 0.15 | High onset with relaxed tail hold |
| **C32** | Hysteresis | `dfn` | **0.650** | **0.350** | 0.10 | 0.15 | Higher onset with relaxed tail hold |
| **C33** | Hysteresis | `dfn` | **0.700** | **0.400** | 0.10 | 0.15 | Heavy noise resistance with tail hold |
| **C34** | Hysteresis | `dfn` | **0.700** | **0.300** | 0.10 | 0.15 | Extreme hysteresis spread (0.40 margin) |
| **C40** | Gap Filling | `dfn` | 0.481 | 0.481 | 0.10 | **0.05** | Tight gap filling (avoids bridge noise) |
| **C41** | Gap Filling | `dfn` | 0.481 | 0.481 | 0.10 | **0.30** | Moderate inter-word gap bridging |
| **C42** | Gap Filling | `dfn` | 0.481 | 0.481 | 0.10 | **0.50** | Aggressive inter-pause gap bridging |
| **C50** | Audio Stream | **`mono`** | 0.481 | 0.481 | 0.10 | 0.15 | Raw unsuppressed audio preserves weak phonemes |
| **C51** | Audio Stream | **`normalized`**| 0.481 | 0.481 | 0.10 | 0.15 | Normalized audio without DFN filtering |
| **C52** | Consensus | **`consensus`** | 0.481 | 0.481 | 0.10 | 0.15 | Linear consensus: $0.5 p_{\text{dfn}} + 0.5 p_{\text{mono}}$ |
| **C60** | Combined | `dfn` | 0.600 | 0.350 | 0.10 | **0.30** | Hysteresis + Gap Bridging |
| **C61** | Combined | `consensus` | 0.600 | 0.350 | 0.10 | 0.15 | Consensus Stream + Hysteresis |
| **C62** | Combined | `consensus` | **0.600** | **0.350** | 0.10 | **0.30** | Consensus + Hysteresis + Gap Bridging |
| **C63** | Combined | `dfn` | **0.700** | **0.400** | **0.15** | 0.15 | Dedicated Anti-Noise Bodycam Profile |

---

## 4. Downstream Diarization Mapping & RTTM Emission Protocol

To ensure 100% rigorous downstream evaluation:
1. Every candidate VAD configuration produces a binary speech activity timeline $V(t) \in \{0, 1\}$ at 10ms resolution.
2. For all frames where $V(t) = 1$, the speaker identity is inherited from the frozen M1 cluster model (subsegments from `subsegments_scale4` in `base_ts`).
3. For newly recovered speech frames outside baseline subsegments, the speaker identity is assigned from the temporally nearest cluster.
4. Contiguous frames with identical speaker labels are serialized to standardized RTTM files under `research_stages/stage5b/stage5b_1_vad_optimization/predictions/{config_id}/`.
5. Standard NIST Diarization Error Rate is computed via `pyannote.metrics` at:
   - Collar $c = 0.0$ seconds (with overlap evaluation).
   - Collar $c = 0.25$ seconds (no-overlap evaluation).
