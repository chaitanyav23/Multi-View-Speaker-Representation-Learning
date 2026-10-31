# Stage 5A.1 Final Independent Audit Summary and Research Handoff

**Audit Date**: October 31, 2026  
**Auditor**: Independent Senior Machine Learning Auditor & Speech Processing Researcher  
**Status**: Formal Audit Complete — CONDITIONAL PASS (Freeze Approved with Qualifications)  
**Deliverables Directory**: [`research_stages/stage5a_1_final_audit/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1_final_audit/)

---

## 1. Direct Answers to the 10 Core Audit Questions

### 1. Are the Stage 5A.1 scripts methodologically sound?
**Yes.** The primary evaluation pipeline ([`run_stage5a_1_controlled_eval.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/stage5a_1/scripts/run_stage5a_1_controlled_eval.py)) adheres to strict experimental standards. It applies rigorous boundary-containment and overlap filtering to clean speaker turns, uses genuine Leave-One-Recording-Out (LORO) cross-validation with zero test feature leakage, and employs fixed-seed determinism.

### 2. Are the reported verification results numerically correct?
**Yes, with one reporting clarification.** Recomputation across all 9,489 pairwise trials verifies the exact metrics. The reported macro verification values (TitaNet: 0.8238, ResNet: 0.8230, Equal Fusion: 0.8450, Gate: 0.8458) were computed across the **6 eligible recordings** (R01, R03, R05, R06, R07, R08), but mistakenly labeled in report prose as "Macro (5 Eligible)". When strictly restricted to the 5 balanced sessions (excluding R01's 15 trials), the macro numbers are: TitaNet: 0.8172, ResNet: 0.8304, Equal Fusion: 0.8426, Gate: 0.8436. Under both definitions, multi-view fusion unambiguously outperforms both single encoders.

### 3. Are the normalization controls sufficient to eliminate the original confound?
**Yes, decisively.** The suspected normalization confound is disproven both mathematically and experimentally. In NeMo NME-SC spectral clustering (`mask_method='binary'`), graph binarization depends strictly on row-wise argsort rank. Because off-diagonal z-score normalization is a strictly increasing monotonic affine transformation on off-diagonals, single-encoder clustering is **100% invariant** to normalization ($\Delta_{\text{DER}} = 0.0000$, $\Delta_{\text{cpWER}} = 0.0000$ across all recordings). The Stage 5A baseline comparisons were therefore completely fair.

### 4. Does the complementarity result survive independent reconstruction?
**Yes.** Across the 9,489 verification pairs evaluated at frozen LORO decision thresholds:
- Both encoders correct: 5,910 (62.28%)
- Both encoders wrong (shared errors): 1,502 (15.83%)
- TitaNet wrong, ResNet correct: 1,035 (10.91%)
- ResNet wrong, TitaNet correct: 1,042 (10.98%)
- Complementary error fraction: **58.03%** of all failing trials ($2,077 / 3,579$)
- Error Jaccard similarity: **0.4197**
The encoders demonstrate substantial, balanced error complementarity.

### 5. Does reliability-aware gating genuinely outperform equal fusion?
**Only marginally in verification, and not statistically downstream.** In verification, the LORO linear gate achieves a tiny gain (+0.0008 ROC-AUC on 6-rec; +0.0010 on 5-rec) and lowers HTER by -0.79%. Downstream, it reduces speaker confusion from 8.43% to 8.18% ($\Delta_{\text{DER}} = -0.25\%$). However, paired bootstrap hypothesis testing ($B = 10,000$) reveals that this difference is within statistical uncertainty (95% CI includes zero: $[-0.76\%, 0.00\%]$). Equal-weight affinity fusion is simpler, training-free, and equally effective.

### 6. Is the downstream improvement practically meaningful?
**Yes in transcription, but heavily constrained in DER.** Equal fusion achieves a statistically significant **-1.02% reduction in cpWER** over TitaNet (95% CI: $[-2.07\%, -0.10\%]$, $p < 0.05$). However, total DER improves by only $\sim 0.05\% - 0.30\%$ because total DER is dominated by upstream VAD false alarms (13.33%) and missed speech (24.41%), which create an irreducible **37.74% DER error overhead**. Affinity fusion operates exclusively on the remaining $\sim 8.2\%$ speaker confusion.

### 7. Are the statistical claims appropriately qualified?
**Now fully qualified.** Paired non-parametric bootstrap testing ($B=10,000$) clustered at the recording level confirms that cpWER gains over TitaNet are statistically significant, while DER differences and gate-vs-equal differences are not. All claims of "definitive gate superiority" have been retired.

### 8. Is the evidence sufficient to freeze representation fusion?
**Yes.** The representation-fusion baseline has been rigorously validated, mathematically proven invariant to normalization confounds, checked for zero leakage, and verified with exact numerical metrics. It is ready for formal freeze.

### 9. What exact claims should be retained in the research paper?
1. Pretrained TitaNet-Large (DFN) and WeSpeaker ResNet293-LM (Mono) exhibit **58.03% complementary error trials** (Error Jaccard = 0.4197) on noisy BWC audio.
2. Training-free equal affinity fusion improves verification macro ROC-AUC by **+0.0212** over TitaNet and **+0.0220** over ResNet on the 6-recording macro (+0.0254 and +0.0122 on 5-rec).
3. Equal affinity fusion significantly reduces downstream cpWER by **-1.02%** over TitaNet ($p < 0.05$).
4. Single-encoder spectral clustering is mathematically invariant to monotonic affinity normalization, validating the fairness of the baseline comparisons.

### 10. What limitations must accompany those claims?
1. **Domain Scope**: The findings are established on an 8-recording conversational BWC dataset; universal generalization across acoustic domains cannot be claimed.
2. **Upstream VAD Ceiling**: Diarization error is dominated by 37.74% VAD/miss overhead. Affinity fusion operates strictly on speaker confusion ($\sim 8.2\%$) and cannot resolve upstream detection failures.
3. **Speaker Count Estimation**: High acoustic distortion can induce single-speaker clustering collapse (as observed on `R07_bwc_cheap_shot`). Representation fusion does not solve speaker-count estimation.
4. **Gate Equivalence**: The reliability gate requires regression training but yields downstream gains that are statistically indistinguishable from training-free equal fusion.

---

## 2. Directory and Artifact Structure

All final audit deliverables are stored and verified in:

```
research_stages/stage5a_1_final_audit/
├── STAGE5A_1_FINAL_AUDIT_REPORT.md      # Comprehensive 18-section audit report
├── STAGE5A_1_CORRECTED_RESULTS.md       # Reconciled benchmark & verification tables
├── STAGE5A_1_CLAIM_AUDIT.csv            # 12 audited claims with source code citations
├── STAGE5A_1_ISSUES_REGISTER.md         # Categorized issues (ISS-01 through ISS-06)
├── STAGE5A_1_FREEZE_RECORD.md           # Formal cryptographic freeze document
├── STAGE5A_1_FINAL_AUDIT_SUMMARY.md     # This handoff executive summary
├── integrity/
│   └── pre_audit_baseline_inventory.json # Hashes of 284 files before audit
├── reconciliation/
│   ├── turn_pair_reconciliation.json    # Exact turn-by-turn diff (254 vs 250 turns)
│   └── recomputed_audit_metrics.json    # Recomputed verification & diarization metrics
└── scripts/
    ├── audit_turn_discrepancy.py        # Turn discrepancy detection script
    ├── verify_all_results.py            # Comprehensive metric verification script
    └── generate_claim_audit_csv.py      # Claim audit table generator
```

---

## 3. Stop Condition and Next Step

- **Stage 5A / 5A.1 Boundary**: Execution has stopped at the Stage 5A.1 boundary.
- **Stage 5B Prohibition**: No Stage 5B scripts, neural retraining, VAD tuning, or overlap modules have been implemented.
- **Awaiting Approval**: We are paused and waiting for explicit user instructions before initiating Stage 5B.
