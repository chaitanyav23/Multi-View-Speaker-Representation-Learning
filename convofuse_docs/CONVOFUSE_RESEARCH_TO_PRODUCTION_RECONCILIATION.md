# ConvoFuse — Research-to-Deliverable Reconciliation Report

**Project**: `Multi-View-Speaker-Representation-Learning`  
**Framework**: `ConvoFuse`  
**Date**: 2026-11-13  

---

## 1. Repository Discovery & Deliverable Scope

The `ConvoFuse` unified conversational intelligence framework resides exclusively in `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning`:

1. **Frozen Research Stages (`research_audit/`, `research_stages/`)**:
   - Hosts the frozen Stage 1–5B.4 research audit (`research_audit/`, `research_stages/stage1_data_prep/`, `stage2_baseline/`, `stage3_dual_encoder/`, `stage4_1/`, `stage5a_1/`, `stage5a_1_final_audit/`, `stage5a_1_2_freeze_closure/`, `stage5b/`) and Stage 6 / Stage 6.1 generative refinement studies (`research_stages/stage6_generative_refinement/`).
2. **Primary CLI-First Research Deliverable (`convofuse/`)**:
   - Hosts the unified `convofuse/` Python package (`contracts.py`, `audio_diarization_adapter.py`, `generative_refiner.py`, `transcript_reconstructor.py`, `evaluator.py`, `pipeline.py`, `cli.py`), executed via `python -m convofuse.cli`.
   - Generates canonical per-recording chronological transcripts (`<rec_id>__transcript.txt` and `<rec_id>__transcript.json`) and structured evaluation artifacts in `convofuse_outputs/final_release/`.
3. **Optional Demonstration UI (`app/`)**:
   - Contains a copied optional FastAPI/Streamlit interface (`app/README.md` explicitly marks it as an optional, non-deliverable demonstration interface). The separate `/home/priyanka/chaitanya/Video-Transcription-with-Confidence-Analytics` repository remains completely untouched.

---

## 2. Authoritative Reconciliation of Research Decisions

### A. Canonical Audio Extraction & Sample-Rate Policy
- **Validated Finding (`STAGE1_DATA_PREPARATION_REPORT.md`)**: All recordings (`R01`–`R08`, `1,893.38s` total duration) are extracted to canonical 16 kHz, 16-bit mono PCM WAV (`pcm_s16le`, `-ar 16000 -ac 1`) with `-af aresample=async=1:first_pts=0` to preserve exact temporal origin `t = 0.000s` for alignment with reference annotations and multi-model streams.
- **ConvoFuse Rule**: `CanonicalAudioProcessor` (`convofuse/audio_diarization_adapter.py`) enforces 16 kHz mono PCM extraction, verifies finite samples (`np.isfinite`), records sample count and duration, and computes SHA-256 digests before downstream inference.

### B. Task-Specific Dual-Path Audio Conditioning
- **Validated Finding (`STAGE1_DATA_PREPARATION_REPORT.md`, `STAGE3_EXTRACTION_REPORT.md`, `STAGE4_1_FINAL_AUDIT_REPORT.md`, `STAGE5A_1_2_FINAL_FREEZE_RECORD.md`)**:
  - **TitaNet-Large (192-D)** benefits from DeepFilterNet3 (DFN) speech enhancement on noisy BWC audio (`dfn.wav`), avoiding the single-speaker clustering collapse observed on `R01_sara_sharif` raw audio (`46.22%` Macro DER and `38.07%` Macro cpWER on DFN).
  - **WeSpeaker ResNet293-LM (256-D)** is trained with heavy additive noise augmentation and suffers spectral distortion when fed DFN-filtered audio (`47.60%` Macro DER / `39.96%` Macro cpWER on DFN vs **`45.98%` Macro DER / `36.44%` Macro cpWER** on unenhanced `16k_mono.wav`, a statistically significant **`-1.63 pp` Macro DER improvement**, `95% CI [-2.66%, -0.68%]`, winning 7 of 8 recordings).
  - **Faster-Whisper Large-v3 ASR & Wav2Vec2 CTC Alignment** operate on EBU R128 loudness-normalized audio (`loudnorm=I=-20:TP=-1.5:LRA=11`, `normalized.wav`).
  - **Pyannote Overlap Posteriors** are extracted from raw `16k_mono.wav` (with DFN overlap posteriors cached separately in Stage 5B.2).
- **ConvoFuse Rule**: `TaskSpecificAudioPaths` explicitly routes DFN-conditioned audio to TitaNet-Large, unenhanced 16 kHz mono audio to WeSpeaker ResNet293-LM and overlap detection, and EBU R128 loudness-normalized audio to ASR and CTC forced alignment.

### C. Speaker Embedding Models & Complementary Views
- **Validated Finding (`STAGE3_EXTRACTION_REPORT.md`, `STAGE4_1_FINAL_AUDIT_REPORT.md`, `STAGE5A_1_2_FINAL_FREEZE_RECORD.md`)**:
  - **View 1**: `nvidia/speakerverification_en_titanet_large` (192-D L2-normalized embeddings across 5 multiscale granularities `[1.5s, 1.2s, 1.0s, 0.8s, 0.5s]`, `15,994` total subsegments).
  - **View 2**: `WeSpeaker ResNet293-LM` (256-D L2-normalized embeddings across the identical `15,994` multiscale subsegments).
  - Across `9,489` clean within-recording turn verification pairs (`250` turns), `58.03%` (`2,077 / 3,579`) of unique verification errors are strictly non-overlapping between the two encoders (Error Jaccard = `0.4197`).

### D. Affinity Normalization, Multiscale Fusion & Spectral Clustering
- **Validated Finding (`STAGE5A_1_CORRECTED_RESULTS.md`, `STAGE5A_1_2_FINAL_FREEZE_RECORD.md`, `R01_POST_RUN_AUDIT_ADDENDUM.md`)**:
  - Raw cosine affinity matrices from TitaNet (`mean = 0.0887, std = 0.1610`) and WeSpeaker (`mean = 0.2341, std = 0.1492`) have incompatible off-diagonal score distributions. Unnormalized fusion causes severe under-clustering on multi-speaker recordings (collapsing `R06` from 9 speakers to 2 and jumping `R06` DER to `62.23%`).
  - **Off-Diagonal Z-Score Normalization + Logistic Sigmoid Mapping (`tau = 1.0`)** followed by **Equal-Weight Fusion** (`0.5 * S_titanet_norm + 0.5 * S_wespeaker_norm`, Condition J) or **Linear Clipped LORO Reliability-Gated Fusion** (`w_t in [0.35, 0.65]`, Condition K) resolves scale mismatch:
    - **Verification**: Condition J reaches `0.8450` 6-recording Macro ROC-AUC (`+0.0212` over TitaNet, `+0.0220` over ResNet293) and Condition K reaches `0.8458` 6-recording Macro ROC-AUC and `31.90%` Macro HTER (`27.62%` Pooled OOF HTER).
    - **Downstream Diarization**: Condition J achieves the **lowest speaker count MAE (`1.250`)**, recovers `8/9` speakers on `R06`, achieves the **highest Macro-SACC (`50.25%`)**, and reduces **Macro cpWER to `37.05%`** (`-1.02 pp` over TitaNet, `95% CI [-2.07%, -0.10%]`, $p < 0.05$). Condition K achieves the **lowest 8-recording Macro DER (`45.92%`)** and **lowest speaker confusion (`8.18%`)**.
  - **Clustering**: Calibrated Normalized Maximum Eigengap Spectral Clustering (`NME-SC`, `rp_threshold = 0.15`, `sparse_search = True`, `max_num_speakers = 8`) estimates speaker count automatically without ground-truth supervision.

### E. VAD Modernization & Overlap Attribution Findings
- **Validated Finding (`STAGE5B_CONSOLIDATED_RESULTS_REPORT.md`, `R01_POST_RUN_AUDIT_ADDENDUM.md`)**:
  - Stage 5B.1 tuned 2-of-3 Consensus VAD (`C62`) reduces missed speech on `R01_sara_sharif` (`18.13%` Miss, `1.35%` FA in fresh diarization) compared to legacy MarbleNet VAD (`26.25%` Miss, `9.29%` FA).
  - Stage 5B.4 fresh diarization pilot (`E20` = C62 VAD + Condition J Equal Fusion + M0 single-speaker vs `E21` = C62 + Condition J + M4-DFN overlap attribution) reduced `R01_sara_sharif` fresh diarization `DER (c=0.0s)` from `38.06%` (Condition J) to **`32.91%`** (`E20`) and **`32.58%`** (`E21`, `-0.33 pp`), while confirming that second-speaker overlap attribution leaves `cpWER` (`52.30%`), `Lexical WDER` (`19.38%`), and `Timestamp WDER` (`20.08%`) completely unchanged because single-stream ASR emits one lexical sequence per timestamp interval.
- **ConvoFuse Rule**: ConvoFuse uses `M0` (`E20` on R01 / `Condition J` across R02–R08) as the primary word-attribution diarization input while attaching explicit multi-speaker RTTM overlap intervals and Stage 5B.2 frame-level overlap posterior statistics (`frame_overlap_prob_mean`, `frame_overlap_prob_max`) to every word and turn without fabricating a second lexical stream.

### F. Generative Speaker Refinement (`DiarizationLM-Gemma-4-E4B-v1` + `LG-TPST`)
- **Validated Finding (`STAGE6_1_BENCHMARK_REPORT.md` & ConvoFuse Remediation)**:
  - Local `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (Q4_K_M GGUF via Ollama on CUDA) executes at ~145 tokens/sec with zero API cost.
  - **Defect Fixed in ConvoFuse**: In the initial Stage 6.1 prototype (`diarizationlm_adapter.py`), `UNK` was mapped to `<speaker:1>` (`spk_to_tag["UNK"] = "1"`), which caused `DiarizationLM-Gemma` to reassign two `UNK` words (`"Yes sir."` at `2.052s–2.377s` on R01) to `speaker_1`, degrading `cpWER` by `+0.70 pp` (`52.30% -> 53.00%`). Furthermore, `extract_text_and_spk(..., skip_meaningless_speaker=True)` silently fell back to `previous_spk` on invalid tags.
  - **ConvoFuse Remediation**: `SpeakerIdentityMapper` (`convofuse/contracts.py`) and `ConvoFuseGenerativeRefiner` (`convofuse/generative_refiner.py`) isolate `UNK` (`<speaker:0>`), strictly forbid mapping `UNK` to any valid speaker (`REJECTED_UNKNOWN_SPEAKER_MAPPING` / `ABSTAINED_PRESERVE_UNK`), and use `extract_text_and_spk_strict` to explicitly reject missing, malformed, or out-of-inventory tags. As a result, the `+0.70 pp` R01 cpWER degradation is completely eliminated (`cpWER = 52.30%`, `Δ = 0.00 pp`), while corpus-wide macro `cpWER` across R01–R08 improves from `37.23%` to `37.00%` (`-0.23 pp`, 5 improved / 2 unchanged / 1 degraded) and macro Lexical WDER improves from `13.29%` to `13.22%` (`-0.08 pp`, 3 improved / 3 unchanged / 2 degraded).
