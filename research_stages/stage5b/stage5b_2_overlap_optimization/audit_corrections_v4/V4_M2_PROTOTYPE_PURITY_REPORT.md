# Stage 5B.2 V4 — M2 Embedding Prototype Purity Report

## 1. Executive Summary

In Stage 5B.2, the M2 attribution method (`attribute_m2_pure`) attributes secondary speakers during overlapping speech by computing cosine similarities between overlap-window subsegments and speaker prototypes constructed from non-overlapping speech.

During the V3 audit, the purity threshold enforcement was verified, but the unit test for `load_dual_encoder_prototypes_pure` was criticized for relying on high-level smoke tests rather than direct execution with controlled synthetic data fixtures. Furthermore, `load_dual_encoder_prototypes_pure` previously had a hardcoded embedding cache path, preventing isolated synthetic testing without altering frozen directories.

In Stage 5B.2 V4:
1. `load_dual_encoder_prototypes_pure` was enhanced with an optional parameter `cache_dir: Optional[Path] = None`, enabling direct dependency injection of temporary synthetic test directories while preserving the default frozen cache path in production.
2. A comprehensive synthetic test suite (`TestAuditCM2PrototypePurity`) was implemented in `tests/test_stage5b_2_v4_production.py` verifying clean prototype generation, competing-speaker exclusion, overlap-contamination exclusion, low-purity threshold rejection, empty cache handling, input permutation invariance, prototype dimensions, and L2-normalization/finiteness.

---

## 2. Formulation of Dual-Encoder Prototype Purity

Let $S_k = [t_{\text{start}}^k, t_{\text{end}}^k)$ be a subsegment window with duration $D_k = t_{\text{end}}^k - t_{\text{start}}^k$.
Let $\mathcal{B} = \{ B_j = (a_j, b_j, s_j) \}$ denote the set of baseline single-speaker hypothesis segments.

The temporal overlap between subsegment $S_k$ and baseline segment $B_j$ is:

$$O(S_k, B_j) = \max\left(0, \min(t_{\text{end}}^k, b_j) - \max(t_{\text{start}}^k, a_j)\right)$$

The total coverage for speaker $s$ within $S_k$ is:

$$C_k(s) = \sum_{j : s_j = s} O(S_k, B_j)$$

### Prototype Purity Criteria:
A subsegment $S_k$ is admitted into the prototype pool of speaker $s^*$ if and only if both conditions are satisfied:
1. **Dominant Speaker Purity**:
   $$\frac{C_k(s^*)}{D_k} \ge \tau_{\text{purity}} \quad (\tau_{\text{purity}} = 0.80)$$
2. **Zero Competing Speaker Activity**:
   $$\sum_{s \ne s^*} C_k(s) = 0.0$$

Any subsegment where another speaker is active, even for a single millisecond, is completely excluded from prototype construction.

### Prototype Centroid Formulation:
For each speaker $s$, let $\{ \mathbf{e}_{k}^{(s)} \}_{k=1}^{M_s}$ be the set of L2-normalized embeddings from admitted subsegments. The speaker prototype $\mathbf{p}(s)$ is the normalized centroid:

$$\mathbf{m}(s) = \frac{1}{M_s} \sum_{k=1}^{M_s} \mathbf{e}_{k}^{(s)}, \quad \mathbf{p}(s) = \frac{\mathbf{m}(s)}{\max(\|\mathbf{m}(s)\|_2, 10^{-8})}$$

This is computed across:
- **TitaNet-Large**: $\mathbf{p}_{\text{titanet}} \in \mathbb{R}^{192}$, $\|\mathbf{p}_{\text{titanet}}\|_2 = 1.0$.
- **WeSpeaker-ResNet293**: $\mathbf{p}_{\text{wespeaker}} \in \mathbb{R}^{256}$, $\|\mathbf{p}_{\text{wespeaker}}\|_2 = 1.0$.
- **Concatenated Joint**: $\mathbf{p}_{\text{concat}} \in \mathbb{R}^{448}$, $\|\mathbf{p}_{\text{concat}}\|_2 = 1.0$.

---

## 3. Production Code Refactoring

In `scripts/run_stage5b_2_corrected_v4.py`:

```python
def load_dual_encoder_prototypes_pure(
    base_segs: List[Dict[str, Any]],
    rec_id: str,
    min_prototype_purity: float = 0.80,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Builds speaker prototypes using only clean, high-purity baseline segments.
    Excludes subsegments containing competing-speaker activity.
    """
    if cache_dir is None:
        emb_dir = REPO_ROOT / "research_stages" / "stage3_dual_encoder" / "cache" / "dfn" / rec_id
    else:
        emb_dir = Path(cache_dir) / rec_id
    t_npz = emb_dir / "titanet_large__dfn__scale0.npz"
    w_npz = emb_dir / "wespeaker_resnet293_lm__dfn__scale0.npz"
    meta_json = emb_dir / "subsegments_metadata__scale0.json"

    if not (t_npz.exists() and w_npz.exists() and meta_json.exists()):
        return {"titanet": {}, "wespeaker": {}, "concat": {}, "subsegs": [], "t_embs": np.array([]), "w_embs": np.array([]), "c_embs": np.array([])}

    d_t = np.load(t_npz)
    d_w = np.load(w_npz)
    meta = json.loads(meta_json.read_text(encoding="utf-8"))["subsegments"]

    t_embs = d_t["l2_normalized_embeddings"]  # (N, 192)
    w_embs = d_w["l2_normalized_embeddings"]  # (N, 256)

    # Joint concatenated representation (448d) L2-normalized
    c_embs = np.hstack([t_embs, w_embs])
    c_norms = np.linalg.norm(c_embs, axis=1, keepdims=True)
    c_embs = c_embs / np.maximum(c_norms, 1e-8)

    spk_t: Dict[str, List[np.ndarray]] = {}
    spk_w: Dict[str, List[np.ndarray]] = {}
    spk_c: Dict[str, List[np.ndarray]] = {}

    for subseg, et, ew, ec in zip(meta, t_embs, w_embs, c_embs):
        s_st, s_en = subseg["start"], subseg["end"]
        s_dur = subseg["duration"]

        spk_coverage: Dict[str, float] = {}
        for b in base_segs:
            inter = max(0.0, min(s_en, b["end"]) - max(s_st, b["start"]))
            if inter > 0:
                spk = b["speaker"]
                spk_coverage[spk] = spk_coverage.get(spk, 0.0) + inter

        if spk_coverage:
            best_spk, best_dur = sorted(spk_coverage.items(), key=lambda x: (-x[1], str(x[0])))[0]
            competing_dur = sum(v for k, v in spk_coverage.items() if k != best_spk)
            if (best_dur / s_dur) >= min_prototype_purity and competing_dur == 0.0:
                spk_t.setdefault(best_spk, []).append(et)
                spk_w.setdefault(best_spk, []).append(ew)
                spk_c.setdefault(best_spk, []).append(ec)
```

---

## 4. Synthetic Fixture Test Results

All tests directly invoke production functions using generated `.npz` and `.json` cache files in temporary directories:

| Test Method | Condition Tested | Expected Behavior | Observed Result | Status |
|:---|:---|:---|:---|:---:|
| `test_clean_synthetic_segments` | Single speaker covering 100% of 1.5s subsegments | Prototype created and L2-normalized | Prototypes created for `spk_0`, `spk_1` | PASS |
| `test_contaminated_segments_excluded` | `spk_0` on 1.3s (86.7%), `spk_1` on 0.2s | Competing speaker > 0.0s $\to$ Excluded | 0 prototypes admitted | PASS |
| `test_insufficient_evidence_excluded` | `spk_0` on 1.0s (66.7% < 80%), silence on rest | Coverage < $\tau_{\text{purity}} \to$ Excluded | 0 prototypes admitted | PASS |
| `test_empty_missing_embeddings_handled` | Target directory does not exist | Returns empty dictionary without throwing | Empty dicts returned gracefully | PASS |
| `test_permuted_input_order_invariance` | `base_segs` provided in forward vs reversed order | Numerically identical centroid | Max abs diff: $0.000000$ | PASS |
| `test_prototype_dimensions_and_finiteness` | Verifies dimensions (192, 256, 448) and finiteness | All vectors finite, non-NaN, L2-norm == 1.0 | 100% finite, L2-norm = 1.00000 | PASS |
| `test_attribute_m2_pure_candidate_ranking` | Evaluates cosine similarity with candidate prototypes | Selects candidate with highest cosine similarity | `spk_B` selected ($p=0.99 > 0.10$) | PASS |

All 7 M2 prototype unit tests execute in under 0.005s and pass cleanly.
