# Audit E: Correction of M3 Turn Alternation & M4 Conservative Abstention

## 1. Executive Summary & Defect Identification

The documented Stage 5B.2 protocol stated that M3 and M4 examine "immediately preceding and subsequent conversational turns."
However, inspection of `run_stage5b_2_experiments.py` revealed:
```python
prev_spks = [s for s in frame_spk[:mid_idx] if s is not None and s != primary_spk]
next_spks = [s for s in frame_spk[mid_idx:] if s is not None and s != primary_spk]
```
By slicing `frame_spk[:mid_idx]` and `frame_spk[mid_idx:]`, the code searched backwards all the way to frame 0 ($t = 0.0\text{ s}$) and forwards to the very end of the file. In a 10-minute recording, a speaker active 8 minutes earlier or later was selected as an "adjacent" conversational turn!
Furthermore, when no candidate was found, both methods fell back to `global_alts[0]`.

---

## 2. Corrected Specification for Local Turn Attribution

The corrected implementation enforces rigorous local conversational turn boundaries:

### 2.1 Local Context Boundaries & Constraints
For an overlap interval with start $t_{\text{start}}$, end $t_{\text{end}}$, and midpoint $t_{\text{mid}}$:
- **Maximum Context Gap ($\Delta t_{\text{max}}$)**: Set to **$5.0\text{ s}$**. Context searches backwards to $\max(0.0, t_{\text{start}} - 5.0)$ and forwards to $\min(T_{\text{rec}}, t_{\text{end}} + 5.0)$.
- **Minimum Turn Duration ($\tau_{\text{turn}}$)**: A candidate speaker turn must have contiguous duration $\ge \mathbf{0.20\text{ s}}$ to qualify as a valid turn (rejecting transient noise spikes).
- **Primary Exclusion**: The primary speaker $k_{\text{primary}}$ at $t_{\text{mid}}$ is excluded.

### 2.2 Local Preceding Turn ($P$)
Scan backwards from $t_{\text{start}}$ within the window $[t_{\text{start}} - 5.0, \; t_{\text{start}}]$.
Find the most recent contiguous turn of an alternate speaker $s \neq k_{\text{primary}}$.
If found and duration $\ge 0.20\text{ s}$, record $P = s$ and its temporal distance $\delta_P = t_{\text{start}} - t_{\text{end}}(P)$.
Otherwise, $P = \text{None}$.

### 2.3 Local Subsequent Turn ($N$)
Scan forwards from $t_{\text{end}}$ within the window $[t_{\text{end}}, \; t_{\text{end}} + 5.0]$.
Find the earliest contiguous turn of an alternate speaker $s \neq k_{\text{primary}}$.
If found and duration $\ge 0.20\text{ s}$, record $N = s$ and its temporal distance $\delta_N = t_{\text{start}}(N) - t_{\text{end}}$.
Otherwise, $N = \text{None}$.

---

## 3. Decision Rules for M3-R1 and M4-R1

### 3.1 M3-R1 (Local Turn Alternation)
In rapid conversational exchanges (such as debates), the interrupting voice is overwhelmingly the immediately alternating dialogue partner:
1. If $P \neq \text{None}$: select $s^* = P$.
2. Else if $N \neq \text{None}$: select $s^* = N$.
3. Else: **ABSTAIN** ($s^* = \text{None}$).

### 3.2 M4-R1 (Conservative Bidirectional Abstention)
To protect clean and single-speaker files from spurious false alarms, M4 requires bidirectional confirmation:
1. Both $P$ and $N$ must exist ($P \neq \text{None}$ and $N \neq \text{None}$).
2. Preceding and subsequent alternate speakers must strictly agree: $P == N$.
3. If $P \neq N$, or if either is missing: **ABSTAIN** ($s^* = \text{None}$).

---

## 4. Offline vs Online Distinction

- **Online / Causal Mode**: An online diarization system operates with zero future lookahead. Only preceding context $P$ is accessible.
- **Offline Mode**: Offline diarization processes the completed recording. M3 with subsequent fallback ($N$) and M4 (requiring $P == N$) are **strictly offline bidirectional attribution methods**.
- M4 cannot be deployed in streaming or real-time pipelines without a minimum latency buffer equal to $\Delta t_{\text{max}} = 5.0\text{ s}$.
