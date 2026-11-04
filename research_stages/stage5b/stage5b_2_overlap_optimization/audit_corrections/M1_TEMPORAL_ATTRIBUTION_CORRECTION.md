# Audit D: Correction of M1 Temporal-Context Attribution

## 1. Executive Summary & Defect Identification

The original Stage 5B.2 implementation of M1 (Local Temporal Context Attribution) contained an undocumented, arbitrary fallback:
```python
else:
    # If no alternate speaker in window, fallback to most prevalent alternate globally
    global_alts = [s for s in all_speakers if s != primary_spk]
    sec_spk = global_alts[0] if global_alts else None
```
When no alternate speaker was present in the local temporal window, `global_alts[0]` selected the first speaker in Python's arbitrary set/list order of all speakers in the recording.
This meant that in single-speaker files or isolated speech bursts, M1 introduced spurious secondary speakers without any local conversational evidence.

---

## 2. Corrected Specification for M1-R1

The corrected implementation `M1_temporal_context_corrected` enforces strict temporal locality:

1. **Context Window Definition**:
   For any detected overlap segment with midpoint $t_{\text{mid}}$, the local context interval is bounded by:
   $$[t_{\text{mid}} - W_{\text{ctx}}, \; t_{\text{mid}} + W_{\text{ctx}}]$$
   where $W_{\text{ctx}} = 3.0\text{ s}$.

2. **Candidate Exclusion**:
   The primary speaker $k_{\text{primary}}$ active at $t_{\text{mid}}$ is strictly excluded:
   $$\mathcal{K}_{\text{alt}} = \{s \in \mathcal{K}_{\text{hyp}} \mid s \neq k_{\text{primary}}\}$$

3. **Temporal Evidence Calculation**:
   For each candidate $s \in \mathcal{K}_{\text{alt}}$, the total duration of activity within the context window is accumulated:
   $$D_{\text{ctx}}(s) = \int_{t_{\text{mid}} - W_{\text{ctx}}}^{t_{\text{mid}} + W_{\text{ctx}}} \mathbb{1}[\text{Hyp}(t) = s]\,dt$$

4. **Minimum Evidence Threshold**:
   A candidate is eligible only if its activity meets or exceeds a minimum evidence threshold:
   $$D_{\text{ctx}}(s) \ge \tau_{\text{evidence}} = 0.20\text{ s}$$

5. **Selection & Tie-Breaking**:
   Among eligible candidates, select the one with the maximum temporal duration:
   $$s^* = \arg\max_{s \in \mathcal{K}_{\text{alt}}, D_{\text{ctx}}(s) \ge \tau_{\text{evidence}}} D_{\text{ctx}}(s)$$
   If multiple candidates have identical duration, tie-breaking selects the candidate whose center of mass is temporally closest to $t_{\text{mid}}$.

6. **Strict Abstention (No Arbitrary Fallback)**:
   If no candidate satisfies $D_{\text{ctx}}(s) \ge \tau_{\text{evidence}}$:
   $$s^* = \text{None} \quad (\text{ABSTAIN})$$
   Under no circumstances is a global speaker list indexed or arbitrary fallback invoked.
