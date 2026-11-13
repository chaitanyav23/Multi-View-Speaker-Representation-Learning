# Stage 5B.4 — R01 C62 Consensus VAD Mono-Input Provenance Determination

## Executive Summary
This document provides the formal determination regarding the provenance and eligibility of the 16k mono audio stream (`R01_sara_sharif_16k_mono.wav`) used in the consensus Voice Activity Detection (`C62`) condition for the Stage 5B.4 authorized R01 controlled pilot ($E20$ vs $E21$).

Under the strict zero-backfill provenance policy established across Stages 5B.4-A.1 through A.1.3.2.2, no hash may be retroactively generated or accepted without an authoritative historical commit baseline.

This investigation confirms:
1. `research_stages/stage1_data_prep/reports/audio_validation_report.json` and `audio_validation_report.csv` were committed in the repository's foundational commit `0a6da21` on Oct 31, 2026.
2. In both artifacts, `research_stages/stage1_data_prep/audio/R01_sara_sharif/R01_sara_sharif_16k_mono.wav` is explicitly recorded with SHA-256 hash `96f9af8d250ed8e26749433e2da6e1928a6d78eb6caad002cc1587fffcc0e45d`.
3. The physical file currently on disk matches this SHA-256 hash bit-for-bit (`96f9af8d...`).
4. Therefore, `C62` consensus VAD audio inputs are fully grounded in immutable historical git artifacts, satisfying the zero-backfill provenance policy. `C62` is formally eligible for the authorized R01 pilot.

---

## Formal Responses to Key Evaluation Questions

### Question 1: Specification of C62 in Stage 5B.4 Frozen Protocol
**Question**: Does the Stage 5B.4 frozen protocol specify C62 using the Stage 1 16k mono audio, or does it mandate an exact cryptographic hash for the mono audio input?
**Finding**:
In `STAGE5B_4_PROPOSED_EXPERIMENT_PROTOCOL.md` and `condition_registry.py`:
- Condition C62 is formally defined as the linear consensus combination of the DFN VAD posterior and the 16k mono VAD posterior:
  $$P_{\text{speech}}(t) = 0.5 \cdot P_{\text{DFN}}(t) + 0.5 \cdot P_{\text{mono}}(t)$$
  with onset threshold $\theta_{\text{onset}} = 0.600$ and offset threshold $\theta_{\text{offset}} = 0.350$.
- In the Stage 5B.4 pipeline architecture, VAD posteriors are derived from the Stage 1 audio inputs or loaded from Stage 2 audited posteriors (`research_stages/stage2_baseline/posteriors/`).
- The protocol requires that all input representations maintain unbroken cryptographic provenance back to Stage 1. While `recording_manifest.json` included `mono_wav` path but omitted the `mono_wav_sha256` key (which was included for `dfn_wav_sha256`), the Stage 1 data preparation pipeline recorded full cryptographic validation of all audio variants in companion audit artifacts.

### Question 2: Historical Canonical SHA-256 in Immutable Artifacts
**Question**: Did Stage 1, Stage 5B.1, or Stage 5B.2 establish a canonical SHA-256 for `R01_sara_sharif_16k_mono.wav` in an existing immutable artifact?
**Finding**:
**Yes.**
In Git commit `0a6da21` (dated 2026-10-31 01:45:11 +0000, "added research stages"):
1. `research_stages/stage1_data_prep/reports/audio_validation_report.json` records:
   ```json
   {
     "wav_path": "research_stages/stage1_data_prep/audio/R01_sara_sharif/R01_sara_sharif_16k_mono.wav",
     "sha256": "96f9af8d250ed8e26749433e2da6e1928a6d78eb6caad002cc1587fffcc0e45d",
     "sample_rate": 16000,
     "channels": 1,
     "sample_width_bytes": 2,
     "num_samples": 1615366,
     "duration_sec": 100.9604,
     "source_duration_sec": 100.9894,
     "duration_diff_sec": 0.02901,
     "nan_count": 0,
     "inf_count": 0,
     "rec_id": "R01_sara_sharif",
     "audio_variant": "16k_mono"
   }
   ```
2. `research_stages/stage1_data_prep/reports/audio_validation_report.csv` records at line 2:
   ```csv
   research_stages/stage1_data_prep/audio/R01_sara_sharif/R01_sara_sharif_16k_mono.wav,96f9af8d250ed8e26749433e2da6e1928a6d78eb6caad002cc1587fffcc0e45d,16000,1,2,1615366,100.9604,100.9894,0.02901,0,0,0.996948,0,0.0,-23.82,True,R01_sara_sharif,16k_mono
   ```
These artifacts have remained frozen in Git history since Oct 31, 2026.

### Question 3: Physical File Bit-for-Bit Verification
**Question**: Does the physical file currently at `research_stages/stage1_data_prep/audio/R01_sara_sharif/R01_sara_sharif_16k_mono.wav` match that canonical hash bit-for-bit?
**Finding**:
**Yes.**
Independent verification on the local filesystem:
```bash
$ sha256sum research_stages/stage1_data_prep/audio/R01_sara_sharif/R01_sara_sharif_16k_mono.wav
96f9af8d250ed8e26749433e2da6e1928a6d78eb6caad002cc1587fffcc0e45d  research_stages/stage1_data_prep/audio/R01_sara_sharif/R01_sara_sharif_16k_mono.wav
```
- File size: 3,230,810 bytes
- Sample rate: 16,000 Hz, 1 channel, 16-bit PCM
- Duration: 100.9604 seconds
- Current sha256: `96f9af8d250ed8e26749433e2da6e1928a6d78eb6caad002cc1587fffcc0e45d`
- Historical sha256: `96f9af8d250ed8e26749433e2da6e1928a6d78eb6caad002cc1587fffcc0e45d`
- Status: **100% Bit-for-bit identical.**

### Question 4: Consensus VAD Eligibility Determination
**Question**: Based on (1)-(3), is C62 consensus VAD eligible for the authorized R01 pilot under the zero-backfill policy?
**Determination**:
**ELIGIBLE.**
- Establishing provenance from an existing, frozen report committed in `0a6da21` strictly complies with the zero-backfill policy: no new manifests were created retroactively, no historical files were altered, and no unverifiable assumptions were made.
- The C62 condition combines verified DFN audio (`90520b89...`) and verified 16k mono audio (`96f9af8d...`) using the bit-exact cached posteriors in `research_stages/stage2_baseline/posteriors/`.
- Both inputs have proven, immutable provenance dating to repository inception.
- The pipeline may proceed to evaluate Condition C62 for E20 and E21.
