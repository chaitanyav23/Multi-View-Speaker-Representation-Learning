# ConvoFuse — Final Master Implementation & Empirical Evaluation Report

**Project**: `CS787_Project`  
**Framework**: `ConvoFuse`  
**Date**: 2026-11-13  
**Hardware Platform**: NVIDIA GeForce RTX 4090 (24.5 GB VRAM, CUDA Driver 595.71.05)  

---

## 1. Executive Summary

`ConvoFuse` upgrades the original Phase20 video transcription and confidence analytics application into a unified, end-to-end conversational intelligence platform incorporating all validated research contributions from Stages 1 through 6.1:

1. **Task-Specific Dual-Path Audio Processing**: Routes DeepFilterNet3-conditioned 16 kHz mono audio (`dfn.wav`) to `TitaNet-Large` (192-D), unenhanced 16 kHz mono audio (`16k_mono.wav`) to `WeSpeaker ResNet293-LM` (256-D) and overlap posterior detection, and EBU R128 loudness-normalized 16 kHz mono audio (`normalized.wav`) to `Faster-Whisper Large-v3` ASR and `Wav2Vec2` CTC forced alignment.
2. **Multi-View Speaker Affinity Fusion & Calibrated NME-SC**: Combines off-diagonal Z-score normalized cosine affinities (`0.5 TitaNet + 0.5 WeSpeaker`, Stage 5A.1 Condition J / Stage 5B.4 E20) and clusters via calibrated `NME-SC` (`max_RP_threshold = 0.15`, `sparse_search = True`, `max_num_speakers = 8`) without oracle speaker count.
3. **Temporal Word-to-Speaker Association & Overlap Metadata Exposure**: Associates aligned ASR words with diarization intervals (`max_overlap_speaker` + `0.5s` midpoint collar), preserves unassigned words strictly as `UNK`, and attaches multi-speaker RTTM overlap intervals and Stage 5B.2 frame-level overlap posteriors (`frame_overlap_prob_mean`, `frame_overlap_prob_max`) to every word.
4. **Ground-Truth-Isolated Generative Refinement (`DiarizationLM-Gemma-4-E4B-v1` + LG-TPST)**: Executes local `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (`Q4_K_M` via Ollama CUDA) with strict ground-truth isolation, `UNK` invariance, speaker inventory validation, simultaneous overlap protection, and Locality-Gated Transcript-Preserving Speaker Transfer (LG-TPST).
5. **Complete Chronological Conversational Transcript Reconstruction**: Produces human-readable `[HH:MM:SS.mmm – HH:MM:SS.mmm] SPEAKER_XX: ...` transcripts, speaker summary indices, and 7 structured artifacts per recording, accessible via FastAPI REST endpoints (`/api/v1/convofuse/*`), the Streamlit dashboard, and the CLI.

---

## 2. Empirical Results Across All 8 Benchmark Recordings (`R01`–`R08`)

Executed sequentially via [`convofuse/cli.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/convofuse/cli.py) into [`convofuse_outputs/r01_r08_benchmark/`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/convofuse_outputs/r01_r08_benchmark/CORPUS_SUMMARY.json) (`total_corpus_sec = 88.37s`):

| Recording ID | Duration (s) | Words | Turns (Base $\rightarrow$ Ref) | Speakers | TS-WDER Base $\rightarrow$ Ref ($\Delta$ pp) | Lex-WDER Base $\rightarrow$ Ref ($\Delta$ pp) | cpWER Base $\rightarrow$ Ref ($\Delta$ pp) | Accepted | Rejected | Abstained | Integrity & SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R01_sara_sharif`** (E20) | 100.96 | 266 | 11 $\rightarrow$ 13 | 3 | 20.08% $\rightarrow$ 20.08% (`0.00`) | 19.38% $\rightarrow$ 19.38% (`0.00`) | 52.30% $\rightarrow$ 52.30% (`0.00`) | 1 | 2 | 0 | Verified |
| **`R02_dennis_skinner`** (Cond J) | 176.26 | 307 | 27 $\rightarrow$ 26 | 3 | 37.25% $\rightarrow$ 37.25% (`0.00`) | 18.79% $\rightarrow$ **18.46%** (`-0.33`) | 43.00% $\rightarrow$ **42.67%** (`-0.33`) | 1 | 7 | 1 | Verified |
| **`R03_disabled_parking`** (Cond J) | 206.22 | 641 | 59 $\rightarrow$ 59 | 3 | 8.55% $\rightarrow$ 9.25% (`+0.70`) | 4.83% $\rightarrow$ 5.48% (`+0.65`) | 19.60% $\rightarrow$ 20.52% (`+0.92`) | 11 | 6 | 3 | Verified |
| **`R04_piers_robinson`** (Cond J) | 204.54 | 689 | 59 $\rightarrow$ 61 | 3 | 26.59% $\rightarrow$ 26.59% (`0.00`) | 11.69% $\rightarrow$ **10.88%** (`-0.81`) | 38.57% $\rightarrow$ **37.75%** (`-0.82`) | 22 | 17 | 20 | Verified |
| **`R05_southwest_pilot`** (Cond J) | 162.45 | 372 | 46 $\rightarrow$ 47 | 3 | 2.51% $\rightarrow$ 2.51% (`0.00`) | 8.68% $\rightarrow$ **8.12%** (`-0.56`) | 31.25% $\rightarrow$ **30.21%** (`-1.04`) | 9 | 0 | 10 | Verified |
| **`R06_rich_karen`** (Cond J) | 652.18 | 1719 | 183 $\rightarrow$ 184 | 8 | 21.19% $\rightarrow$ 22.32% (`+1.13`) | 25.33% $\rightarrow$ 25.76% (`+0.43`) | 49.81% $\rightarrow$ **49.42%** (`-0.39`) | 47 | 11 | 35 | Verified |
| **`R07_bwc_cheap_shot`** (Cond J) | 175.43 | 478 | 16 $\rightarrow$ 16 | 1 | 12.53% $\rightarrow$ 12.53% (`0.00`) | 15.93% $\rightarrow$ 15.93% (`0.00`) | 52.99% $\rightarrow$ 52.99% (`0.00`) | 0 | 5 | 0 | Verified |
| **`R08_uk_parliament`** (Cond J) | 215.34 | 468 | 13 $\rightarrow$ 14 | 2 | 0.22% $\rightarrow$ 0.22% (`0.00`) | 1.72% $\rightarrow$ 1.72% (`0.00`) | 10.32% $\rightarrow$ **10.11%** (`-0.21`) | 1 | 0 | 0 | Verified |
| **Macro Average (R01–R08)** | **236.67** | **617.5** | **51.8 $\rightarrow$ 52.5** | — | **16.11% $\rightarrow$ 16.34%** (`+0.23`) | **13.29% $\rightarrow$ 13.22%** (`-0.08`) | **37.23% $\rightarrow$ 37.00%** (`-0.23`) | **92** | **48** | **69** | **100% Verified** |

### Key Empirical Takeaways
1. **Elimination of R01 `UNK` Degradation**: By enforcing strict `UNK` invariance (`REJECTED_UNKNOWN_SPEAKER_MAPPING`), ConvoFuse prevents `DiarizationLM-Gemma` from reassigning pre-VAD `UNK` words (`"Yes sir."` at `2.052s–2.377s` on R01), completely eliminating the `+0.70 pp` cpWER degradation observed in the raw Stage 6.1 benchmark (`52.30%` preserved vs `53.00%` in Stage 6.1) while cleanly recovering the turn boundary at `[00:00:09.409 – 00:00:09.850] SPEAKER_01: Alright.`
2. **Corpus-Wide cpWER & Lexical WDER Improvement**: Across all 8 recordings, ConvoFuse LG-TPST improves `cpWER` on **5 of 8 recordings** (`R02`: `-0.33 pp`, `R04`: `-0.82 pp`, `R05`: `-1.04 pp`, `R06`: `-0.39 pp`, `R08`: `-0.21 pp`), holds `cpWER` neutral on **2 of 8 recordings** (`R01`, `R07`), and degrades `cpWER` on only **1 recording** (`R03`: `+0.92 pp`), yielding a net macro `cpWER` reduction of **`-0.23 pp`** (`37.23% -> 37.00%`) and a net macro Lexical WDER reduction of **`-0.08 pp`** (`13.29% -> 13.22%`).
3. **100% Transcript & Timestamp Immutability**: Across all 4,940 words in `R01`–`R08`, 0 words were added, deleted, reordered, or textually modified, and 0 timestamps were altered.

---

## 3. Test Suite Verification Summary

1. **Research + ConvoFuse Core Test Suite** (`.venv_phase20`):
   - Command: `PYTHONPATH=. pytest convofuse/tests/ research_stages/stage6_generative_refinement/tests/ research_stages/stage5b/stage5b_4_fresh_diarization/production_integration/tests/ research_stages/stage5b/stage5b_4_fresh_diarization/tests/ -v`
   - Result: **82 collected, 82 passed, 0 failed, 0 skipped, 0 warnings in 8.29s**.
2. **Copied Phase20 Application + ConvoFuse API Test Suite in Repository B** (`cs698b_phase20`):
   - Command: `cd /home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/app && /home/priyanka/anaconda3/envs/cs698b_phase20/bin/pytest tests/ -v`
   - Result: **128 collected, 127 passed, 1 skipped, 0 failed, 1 warning in 14.42s**.
3. **Restored Original Phase20 Application Test Suite in Repository A** (`cs698b_phase20`):
   - Command: `cd /home/priyanka/chaitanya/Video-Transcription-with-Confidence-Analytics/app && /home/priyanka/anaconda3/envs/cs698b_phase20/bin/pytest tests/ -v`
   - Result: **124 collected, 123 passed, 1 skipped, 0 failed, 1 warning in 7.70s** (0 ConvoFuse files or references remain in Repository A).
