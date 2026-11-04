# Stage 5B.2 Corrected Rerun Protocol: Focused Rerun Matrix & Rationale

## 1. Rerun Scope & Execution Boundaries

This protocol details the focused rerun matrix for Stage 5B.2 Corrective Patch v2.
All runs were executed using existing cached intermediate representations:
- Cached overlap curves: `overlap_cache/{rec_id}_{stream}_overlap.npz`
- Cached VAD RTTMs: Stage 5B.1 `predictions/C00_baseline/`, `C50_stream_mono/`, `C62_comb_consensus_hyst_gap/`
- Cached embeddings: Stage 3 `cache/dfn/{rec_id}/` (Scale 0 TitaNet and WeSpeaker)
- Ground-truth RTTMs and WhisperX alignments.

No expensive model retraining or inference was performed.

---

## 2. Minimal Rerun Matrix & Justification

| Group | Configuration ID | Rationale for Rerun | Affected Module |
|---|---|---|---|
| **Baselines** | `O00`, `O01`, `O02` | Matched single-label zero-point references evaluated under strict error handling (safe metric computation, no zero fallbacks). | Correction F |
| **Threshold Sweeps** | `O10-R2` through `O14-R2` | Evaluates sensitivity to detection threshold $\theta \in [0.30, 0.70]$ under corrected frame-bin support ($[t_{\text{mid}} - S/2, t_{\text{mid}} + S/2)$) and M1 proximity tie-breaking. | Correction A, D, F |
| **Stream Comparison** | `O20-R2`, `O21-R2` | DFN vs Raw Mono audio comparison under frame-bin support. | Correction D, F |
| **Attribution Strategies** | `O30-R2` (M1) | Local context attribution with distance tie-break. | Correction A |
| | `O31-R2`, `O31-T-R2`, `O31-W-R2`, `O31-NF-R2` (M2) | Prototype affinity with overlap purity filtering ($f_{\text{ovl}} \ge 0.40$) and prototype purity ($P_k \ge 0.80$). | Correction C |
| | `O32-R2` (M3) | Contiguous turn alternation with gap termination and boundary distance calculation. | Correction B |
| | `O33-R2` (M4) | Conservative bidirectional turn agreement with gap termination. | Correction B |
| **Oracle Diagnostics** | `O34-R2` (M5B) | Predicted overlap + Hungarian multi-label oracle attribution. | Correction E |
| | `O35-R2` (M5C) | Full oracle overlap + Hungarian multi-label oracle attribution. | Correction E |
| **Integrated Pipelines** | `O40-R2`, `O41-R2`, `O42-R2` | End-to-end integration of Stage 5B.1 VAD backbones with verified overlap and attribution. | Correction A, B, D, F |

Total runs executed: **22 configurations $\times$ 8 recordings = 176 individual evaluations**.
Total execution runtime: **224.28 seconds**. Exit code: **0**.
