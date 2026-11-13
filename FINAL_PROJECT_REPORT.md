# Final Project Report: Multi-View Speaker Representation Learning & ConvoFuse

**Project**: `Multi-View-Speaker-Representation-Learning`  
**REPO**: https://github.com/chaitanyav23/Multi-View-Speaker-Representation-Learning
**Integrated Framework**: `ConvoFuse` ([`convofuse/`](convofuse/))  
**Primary Deliverable**: Reproducible CLI-First Conversational Transcript & Evaluation Release ([`convofuse_outputs/final_release/`](convofuse_outputs/final_release/))  
**Date**: 2026-11-13  

---

## 1. Abstract & Executive Summary

Speaker-attributed transcription of real-world Body-Worn Camera (BWC) and multi-talker field recordings requires jointly solving speech activity detection, multi-speaker diarization, lexical transcription, word-to-speaker alignment, and turn boundary attribution under severe non-stationary noise and overlapping speech.

Building upon a forensic audit of an earlier single-recording dual-path baseline (`Phase20`, `arXiv:2607.24958v1`), this work conducts a multi-stage experimental investigation across an 8-recording, 31.56-minute BWC and multi-talker corpus (`R01`–`R08`, `1,893.38s`, 2–9 speakers, `15,994` multiscale subsegments, `4,940` aligned ASR words) and synthesizes the validated findings into **ConvoFuse** (`python -m convofuse.cli`).

### Summary of Validated Findings

1. **Task-Specific Preprocessing Asymmetry (`Stage 4.1`)**: Speech enhancement (`DeepFilterNet3`) is not universally beneficial for speaker diarization. While `TitaNet-Large` (`192-D`) requires `DeepFilterNet3` on noisy BWC audio (`R01_sara_sharif`) to avoid single-speaker eigengap collapse, `WeSpeaker ResNet293-LM` (`256-D`, trained with additive noise augmentation) performs significantly better on unenhanced `16 kHz` mono audio (`45.98%` vs `47.60%` Macro DER, $\Delta = -1.63\text{ pp}$, 95% bootstrap CI `[-2.66%, -0.68%]`, winning 7 of 8 recordings; `36.44%` vs `39.96%` Macro `cpWER`).
2. **Multi-View Error Complementarity & Off-Diagonal Z-Score Affinity Fusion (`Stage 5A.1 / 5A.1.2`)**: Across `9,489` clean within-recording turn verification pairs (`250` turns), `58.03%` (`2,077 / 3,579`) of unique verification errors are strictly disjoint between `TitaNet-Large (DFN)` and `WeSpeaker ResNet293-LM (Mono)` (`Error Jaccard = 0.4197`). Because their raw off-diagonal cosine similarities occupy incompatible scales (`TitaNet` $\mu=0.0887$ vs `ResNet293` $\mu=0.2341$), naive averaging collapses speaker clustering. **Off-diagonal Z-score normalization + sigmoid mapping** followed by equal fusion (**Condition J**) or LORO reliability gating (**Condition K**) increases 6-recording Macro `ROC-AUC` from `0.8238` / `0.8230` to **`0.8450`** (Cond J) and **`0.8458`** (Cond K), lowers 6-recording Macro `HTER` to **`31.90%`** (`27.62%` Pooled OOF), achieves the lowest speaker-count MAE (**`1.250`** under Cond J, recovering 8/9 speakers on `R06`), reduces Macro `cpWER` over `TitaNet-DFN` by **`-1.02 pp`** (`37.05%` vs `38.07%`, 95% CI `[-2.07%, -0.10%]`, $p < 0.05$), and achieves the lowest corpus-wide Macro DER (**`45.92%`** under Cond K vs `47.48%` Stage 2 legacy MSDD).
3. **Upstream Bottleneck Decomposition & Fresh Diarization (`Stage 5B.1–5B.4`)**: Replacing legacy MarbleNet VAD with a calibrated 2-of-3 Consensus VAD (`C62`) combined with Condition J Equal Fusion (`E20`) on `R01_sara_sharif` reduces fresh diarization `DER (c=0.0s)` from `38.06%` to **`32.91%`** (`-5.15 pp`), and adding `M4-DFN` overlap attribution (`E21`) reaches **`32.58%`**. Crucially, `E20` and `E21` produce identical word-level attribution (`TS-WDER = 20.08%`, `Lex-WDER = 19.38%`, `cpWER = 52.30%`) because monoaural CTC-aligned ASR emits a single lexical stream per timestamp interval.
4. **Safeguarded Generative Speaker Refinement (`Stage 6 / 6.1` & `ConvoFuse LG-TPST`)**: Using local `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (`Q4_K_M` via Ollama) with **Locality-Gated Transcript-Preserving Speaker Transfer (`LG-TPST`)** and remediated `UNK` isolation (`<speaker:0>`) eliminates the `+0.70 pp` `R01` `cpWER` degradation observed in unconstrained prototypes, preserves `100%` word, timestamp, and `UNK` invariance across all `4,940` words, improves Macro `cpWER` across `R01`–`R08` from **`37.23%` to `36.98%` (`-0.25 pp`, 5 improved / 2 unchanged / 1 degraded)**, and improves Macro `Lex-WDER` from **`13.29%` to `13.22%` (`-0.07 pp`)**.

---

## 2. Problem Statement & Motivation

Body-Worn Camera (BWC) recordings captured during policing, emergency response, and public disputes represent one of the most challenging domains in speech technology:
- **Severe Acoustic Degradation**: Wind buffeting, sirens, clothing friction, crowd noise, and room reverberation degrade signal-to-noise ratios.
- **Dynamic Multi-Speaker Topologies**: Recordings range from 2-speaker interviews (`R08_uk_parliament`) to chaotic 9-speaker street confrontations (`R06_rich_karen`, `12.24%` overlap ratio) without prior knowledge of speaker count.
- **Forensic & Evidentiary Integrity Requirements**: Downstream users require a single, chronologically ordered, human-readable transcript (`[HH:MM:SS.mmm – HH:MM:SS.mmm] SPEAKER_XX: ...`) where ASR words and timestamps are never hallucinated or silently deleted, unattributed speech is explicitly preserved as `UNK`, and overlapping speech regions are transparently flagged without fabricating a second lexical stream.

---

## 3. Base Paper & Initial Pipeline Context

The starting point of this repository was the historical **Phase20** dual-path pipeline (`Marston`, archived in [`phase20_standalone/`](phase20_standalone/) and described in the base paper `https://arxiv.org/abs/2607.24958v1`):
- **Path A (Diarization Branch)**: Applied `DeepFilterNet` denoising $\rightarrow$ `Pyannote 3.1` probability-guided VAD $\rightarrow$ `NeMo MSDD` (`TitaNet-Large` embeddings) with `oracle_vad=True` on the Path A mask.
- **Path B (Transcription Branch)**: Applied `-20 LUFS` loudness normalization $\rightarrow$ `Pyannote 3.1` VAD $\rightarrow$ `WhisperX` (`Faster-Whisper Large-v3` + `Wav2Vec2` CTC forced alignment).
- **Word-to-Speaker Fusion**: Assigned words to speakers via maximum interval intersection.

---

## 4. Research Objectives & Attribution of Contributions

To maintain rigorous scientific honesty, the table below explicitly distinguishes **inherited baseline components** from the **new research contributions** developed and validated in this project:

| Component / Stage | Inherited from Historical Phase20 (`phase20_standalone/`) | New Research & Engineering Contributions (`research_audit/`, `research_stages/`, `convofuse/`) |
| :--- | :--- | :--- |
| **Benchmark Dataset** | Single recording (`raw_audio/bwc.wav`, 2-speaker partial mapping) | **Stage 1**: Curated 8-recording multi-domain BWC benchmark (`R01`–`R08`, `1,893.38s`, 2–9 gold speakers) with sample-accurate `16k_mono`, `normalized`, and `dfn` views and frozen gold RTTM/JSON references. |
| **Preprocessing Routing** | Dual-path (`DFN` for `TitaNet` MSDD, `Loudnorm` for `WhisperX`) | **Stage 4.1**: Discovered and proved **encoder-specific preprocessing asymmetry** (`WeSpeaker ResNet293-LM` degrades on `DFN` and requires raw `16k_mono`, `-1.63 pp` Macro DER, 95% CI `[-2.66%, -0.68%]`). |
| **Speaker Representation & Affinity Fusion** | Single encoder (`TitaNet-Large` 192-D inside NeMo MSDD) | **Stages 3, 5A.1, 5A.1.2**: Extracted `15,994` 5-scale subsegments for `TitaNet-Large (DFN)` and `WeSpeaker ResNet293-LM (Mono)`; proved `58.03%` pairwise error complementarity; designed **off-diagonal Z-score + sigmoid affinity normalization**, **Condition J Equal Fusion**, and **Condition K LORO Clipped Reliability Gate**. |
| **VAD & Overlap Analysis** | Single probability-guided Pyannote 3.1 VAD mask; no overlap metadata on words | **Stages 5B.1–5B.4**: Designed 2-of-3 **Consensus VAD (`C62`)** (`-10.57 pp` Missed Speech); calibrated 100 Hz Pyannote overlap posteriors (`V4.2.1`); executed `R01` fresh diarization pilot (`E20`/`E21`, `-5.15 pp` to `-5.48 pp` DER); surfaced explicit word/turn overlap metadata without fabricating a second lexical stream. |
| **Generative Post-Processing** | None | **Stages 6, 6.1 & ConvoFuse**: Integrated local `DiarizationLM-Gemma-4-E4B-v1` (`Q4_K_M`) with **Locality-Gated TPST (`LG-TPST`)**, remediated `<speaker:0>` `UNK` isolation, overlap protection, and strict parser validation. |
| **Deliverable Packaging** | Single-recording script (`run_phase20.py`) | **ConvoFuse CLI (`convofuse/cli.py`)**: Reproducible CLI supporting `frozen_replay` and `custom_inputs`, generating chronological `.txt` and `.json` transcripts, `CORPUS_SUMMARY.json`, `CORPUS_INDEX.json`, and `RELEASE_MANIFEST.json`. |

---

## 5. Initial Architecture & Limitations Identified in the Forensic Audit (`research_audit/`)

In [`research_audit/`](research_audit/), a forensic audit of the historical Phase 1–20 codebase revealed five critical scientific limitations:
1. **Single-Recording Evaluation Overfitting**: Architectural hyperparameters in Phases 1–20 were tuned primarily on a single 175-second clip (`bwc.wav` / `R07`) and a secondary clip (`rich_karen` / `R06`).
2. **Incomplete Reference Speaker Coverage on `bwc.wav`**: The historical evaluation mapped only 2 speakers on `bwc.wav`, whereas full forensic annotation (`R07_bwc_cheap_shot`) identifies 5 distinct speakers (`4.82%` overlap).
3. **Single-Encoder Vulnerability**: Relying solely on `TitaNet-Large` left the diarization branch vulnerable to spectral confusion when background noise or rapid cross-talk distorted 1D depthwise convolutions.
4. **Silent Failure of Naive Multi-Encoder Fusion**: Directly averaging raw cosine affinity matrices from `TitaNet-Large` and `WeSpeaker ResNet293-LM` shifts the Laplacian eigenvalue spectrum upward, causing NME-SC eigengap estimation to collapse multi-speaker recordings (`R06` drops from 9 speakers to 2).
5. **Absence of Lexical Turn-Boundary Refinement**: Acoustic subsegment clustering (even at `0.5s` minimum scale) places segment boundaries at coarse acoustic windows rather than exact lexical turn transitions, causing boundary words (e.g., `"Alright."`, `"Yes."`) to bleed into adjacent speakers' turns.

---

## 6. Stage 1 through Stage 4.1 Progression & Findings

### 6.1 Stage 1: Canonical Data Preparation ([`research_stages/stage1_data_prep/`](research_stages/stage1_data_prep/))
- Standardized `R01`–`R08` (`1,893.38s`) to `16 kHz`, 16-bit mono PCM WAV with `-af aresample=async=1:first_pts=0` (`t = 0.000s`).
- Generated three sample-aligned acoustic views per recording: `16k_mono` (raw), `normalized` (EBU R128 `-20 LUFS`), and `dfn` (`DeepFilterNet3`).

### 6.2 Stage 2: Legacy NeMo MSDD Baseline ([`research_stages/stage2_baseline/`](research_stages/stage2_baseline/))
- Evaluated the legacy NeMo MSDD pipeline (`uniform_v1`, MarbleNet VAD + TitaNet-Large + MSDD) and WhisperX `large-v3` ASR across all 8 recordings (`STAGE2_BASELINE_REPORT.md`):
  - **Macro DER (`c=0, ovl`)**: `47.48%` (`Miss = 23.30%`, `FA = 16.07%`, `Conf = 8.11%`)
  - **Macro DER (`c=0.25, no-ovl`)**: `42.52%`, **Macro JER**: `61.88%`, **Spk Count MAE**: `1.500`
  - **Macro ASR WER**: `23.22%`, **Macro `cpWER`**: `39.34%`, **Macro `SA-WER`**: `47.97%`

### 6.3 Stage 3 & Stage 4.1: Dual-Encoder Extraction & Preprocessing Asymmetry ([`research_stages/stage3_dual_encoder/`](research_stages/stage3_dual_encoder/), [`research_stages/stage4_1/`](research_stages/stage4_1/))
- Extracted `15,994` multiscale embeddings (`1.5s`, `1.2s`, `1.0s`, `0.8s`, `0.5s`) for `TitaNet-Large` (`192-D`) and `WeSpeaker ResNet293-LM` (`256-D`) on both `16k_mono` and `dfn`.
- **Stage 4.1 Discovery**: Paired 10,000-resample cluster bootstrap proved that `WeSpeaker ResNet293-LM` on raw `16k_mono` (`45.98%` Macro DER, `36.44%` Macro `cpWER`) significantly outperforms `WeSpeaker ResNet293-LM` on `dfn` (`47.60%` Macro DER, `39.96%` Macro `cpWER`) by **`-1.63 pp` Macro DER** (`95% CI [-2.66%, -0.68%]`). Conversely, `TitaNet-Large` requires `dfn` on `R01_sara_sharif` (`38.06%` DER on `dfn` vs `44.96%` single-speaker collapse on `16k_mono`).

---

## 7. Stage 5A & 5A.1.2 Dual-Encoder Representation & Fusion Results

In [`research_stages/stage5a_1_final_audit/`](research_stages/stage5a_1_final_audit/) and [`research_stages/stage5a_1_2_freeze_closure/`](research_stages/stage5a_1_2_freeze_closure/), we resolved the multi-view affinity scale mismatch via **Off-Diagonal Z-Score Normalization + Logistic Sigmoid Mapping ($\tau = 1.0$)**:

$$\tilde{S}^{(m)}_{ij} = \sigma\!\left(\frac{S^{(m)}_{ij} - \mu^{(m)}_{\text{off}}}{\sigma^{(m)}_{\text{off}}}\right), \quad i \neq j, \qquad \tilde{S}^{(m)}_{ii} = 1.0$$

We then evaluated two fused multi-view representations:
- **Condition J (Normalized Equal Fusion)**: $S^{\text{J}} = 0.5\,\tilde{S}^{\text{TitaNet-DFN}} + 0.5\,\tilde{S}^{\text{ResNet293-Mono}}$.
- **Condition K (Normalized Linear Clipped LORO Reliability Gate)**: $S^{\text{K}}_r = w_r\,\tilde{S}^{\text{TitaNet-DFN}}_r + (1 - w_r)\,\tilde{S}^{\text{ResNet293-Mono}}_r$, where $w_r \in [0.35, 0.65]$ is predicted out-of-fold from `[resnet293_l2_norm, dfn_noise_reduction_db, vad_conf_mean]`.

### Summary of Stage 5A.1.2 Frozen Results (`R01`–`R08`)

1. **Pairwise Error Complementarity**: Across `9,489` clean turn pairs (`250` turns), `58.03%` (`2,077 / 3,579`) of unique verification errors are disjoint (`Error Jaccard = 0.4197`).
2. **Verification Gain**: 6-recording Macro `ROC-AUC` improves from `0.8238` (`TitaNet-DFN`) and `0.8230` (`ResNet293-Mono`) to **`0.8450`** (`Condition J`) and **`0.8458`** (`Condition K`), with Macro `HTER` dropping to **`31.90%`** (`27.62%` Pooled OOF).
3. **Downstream Diarization & Attribution**:
   - **Condition J** achieves the **lowest speaker-count MAE (`1.250`)**, recovers `8/9` speakers on `R06_rich_karen`, achieves the **highest `Macro-SACC` (`50.25%`)**, and achieves a statistically significant **`-1.02 pp` Macro `cpWER` reduction** over `TitaNet-DFN` (`37.05%` vs `38.07%`, 95% CI `[-2.07%, -0.10%]`).
   - **Condition K** achieves the **lowest 8-recording Macro DER (`45.92%`)** and **lowest speaker confusion (`8.18%`)**.

---

## 8. Stage 5B Findings: VAD, Overlap, Speaker Count & `R01` Fresh Diarization Pilot

1. **Stage 5B.1 (Consensus VAD `C62`)**: 2-of-3 frame consensus across tuned Silero VAD (`16k_mono` + `dfn`) and Pyannote 3.1 segmentation reduces corpus Missed Speech by `-10.57 pp` (`24.41% -> 13.84%`) and False Alarm by `-5.21 pp` (`13.33% -> 8.12%`).
2. **Stage 5B.2 (Overlap Posteriors `V4.2.1`)**: Extracts 100 Hz Pyannote 3.1 overlap probabilities (`stage5b_2_overlap/cache/pyannote_raw/<rec_id>.npz`), proving that while oracle overlap lowers DER by `-0.98 pp`, heuristic second-speaker RTTM injection on noisy clusters cannot recover a second lexical stream from single-stream ASR.
3. **Stage 5B.3 (Speaker Count Estimation)**: Demonstrates that calibrating NME-SC `rp_threshold` (`0.15` $\rightarrow$ `0.11`) lowers speaker-count MAE from `1.250` to `0.875`, while isolating `R07`'s single-speaker collapse to extreme siren/crowd noise.
4. **Stage 5B.4 (`R01_sara_sharif` Fresh Diarization Pilot)**: Running fresh multiscale extraction and Condition J fusion on Stage 5B.1 `C62` VAD intervals (`E20`) reduces `R01` `DER (c=0.0s)` from `38.06%` to **`32.91%`** (`-5.15 pp`), and `E21` (`M4-DFN`) reaches **`32.58%`**, while confirming identical word-level `TS-WDER` (`20.08%`), `Lex-WDER` (`19.38%`), and `cpWER` (`52.30%`).

---

## 9. Stage 6 & 6.1 Generative Speaker Refinement & `UNK`-Invariance Remediation

In [`research_stages/stage6_generative_refinement/`](research_stages/stage6_generative_refinement/), we investigated whether an instruction-tuned LLM inspired by Google's **DiarizationLM** (`arXiv:2401.03506`) can refine word-level speaker attributions at turn boundaries:
1. **Cloud Provider Diagnosis (`Workstream A`)**: Live Gemini API (`gemini-2.5-flash`) calls exhibited intermittent HTTP 503 unavailability and thinking-token budget starvation on dense speaker-tagged transcripts (`max_output_tokens=2048` consumed by internal reasoning tokens before `-->` completion emission).
2. **Local `DiarizationLM-Gemma-4-E4B-v1` Integration (`Workstream B`)**: Deployed `diarizers-community/DiarizationLM-Gemma-4-E4B-v1` (`Q4_K_M` GGUF via local Ollama on RTX 4090) with **Locality-Gated TPST (`LG-TPST`)**, achieving `100%` completion reliability (`~145 tokens/s`) and zero external API dependency.
3. **Defect Identified & Remediated in ConvoFuse**:
   - **Root Cause**: In the initial Stage 6.1 prototype (`diarizationlm_adapter.py`), `UNK` was mapped to `<speaker:1>` (`spk_to_tag["UNK"] = "1"`), causing `DiarizationLM-Gemma` to reassign two pre-VAD `UNK` words (`"Yes sir."` at `2.052s–2.377s` on `R01`) to `speaker_1` and degrading `R01` `cpWER` by `+0.70 pp` (`52.30% -> 53.00%`).
   - **ConvoFuse Fix**: `SpeakerIdentityMapper` ([`convofuse/contracts.py`](convofuse/contracts.py)) maps `UNK` exclusively to `<speaker:0>` and maps active speakers to `<speaker:1..K>`, while `ConvoFuseGenerativeRefiner` ([`convofuse/generative_refiner.py`](convofuse/generative_refiner.py)) enforces bidirectional `UNK` invariance (`REJECTED_UNKNOWN_SPEAKER_MAPPING` / `ABSTAINED_PRESERVE_UNK`) and strict tag parsing (`extract_text_and_spk_strict`). This completely eliminates the `+0.70 pp` `R01` `cpWER` degradation (`52.30% -> 52.30%`, $\Delta = 0.00\text{ pp}$).

---

## 10. Final ConvoFuse Architecture & Strict Inference/Evaluation Separation

The [`convofuse/`](convofuse/) package executes a 15-stage deterministic pipeline (documented in detail in [`ARCHITECTURE.md`](ARCHITECTURE.md)):
- **Inference Modules (`audio_diarization_adapter.py`, `generative_refiner.py`, `transcript_reconstructor.py`)**: Operate exclusively on audio, diarization RTTMs, frame overlap posteriors, and aligned ASR words. They have **zero access** to reference RTTMs or reference transcripts (`ground_truth_used_during_inference = False` verified in every `<rec_id>__processing_manifest.json`).
- **Evaluation Module (`evaluator.py`)**: Invoked strictly **after** transcript reconstruction completes, computing `TS-WDER`, `Lex-WDER`, `cpWER`, oracle correction precision, and cryptographic SHA-256 invariance.

---

## 11. Final Chronological Transcript Format & Actual Generated Excerpts

Every recording produces `<rec_id>__transcript.txt` and `<rec_id>__transcript.json` in [`convofuse_outputs/final_release/`](convofuse_outputs/final_release/).

### 11.1 Excerpt 1: `R01_sara_sharif__transcript.txt` (Complete Generated Transcript)
Source: [`convofuse_outputs/final_release/R01_sara_sharif/R01_sara_sharif__transcript.txt`](convofuse_outputs/final_release/R01_sara_sharif/R01_sara_sharif__transcript.txt)

```text
# ConvoFuse Conversational Transcript — R01_sara_sharif (R01_sara_sharif.wav)
# Execution Mode: FROZEN_INPUT_REPLAY
# Diarization Condition: Stage5B.4_E20_C62_DualEncoder_EqualFusion_M0
# Generative Refinement Model: diarizers-community/DiarizationLM-Gemma-4-E4B-v1 (Q4_K_M, LG-TPST)
# Media Duration: 100.96s | Total Turns: 13 | Multi-Speaker RTTM Overlap Turns: 0
# Overlap & UNK Policy: Explicit overlap metadata; UNK preserved; no fabricated simultaneous lexical stream
#
# Speaker Summary Index:
#   - SPEAKER_00: 42.53s | 6 turns | 146 words | [00:00:16.920 – 00:01:12.275]
#   - SPEAKER_01: 13.07s | 3 turns | 52 words | [00:00:09.409 – 00:01:25.679]
#   - SPEAKER_02: 17.21s | 3 turns | 66 words | [00:00:02.987 – 00:01:34.610]
#   - UNK: 0.33s | 1 turns | 2 words | [00:00:02.052 – 00:00:02.377]

[00:00:02.052 – 00:00:02.377] UNK: Yes sir.

[00:00:02.987 – 00:00:09.329] SPEAKER_02: We are, yes. Are you okay just to swap over, just yourself come with me and if you go with my colleague over there, is that alright?

[00:00:09.409 – 00:00:09.850] SPEAKER_01: Alright.

[00:00:11.046 – 00:00:13.628] SPEAKER_02: That's what I was going to say about it. Thank you. Cutting back.

[00:00:14.492 – 00:00:14.833] SPEAKER_01: Yes.

[00:00:16.920 – 00:00:19.529] SPEAKER_00: Which one is your bag? I don't have any bag. You have no bag?

[00:00:21.235 – 00:00:23.884] SPEAKER_00: We'll look over here. Yance, you have a lot of peace.
  # [OVERLAP_METADATA: multi_speaker_rttm=False | concurrent_speakers=SPEAKER_00 | max_frame_overlap_posterior=0.576 | single_lexical_stream_preserved=True]
```

- **Observation**: Pre-VAD `[00:00:02.052 – 00:00:02.377] UNK: Yes sir.` is preserved as `UNK`, while the response `[00:00:09.409 – 00:00:09.850] SPEAKER_01: Alright.` following `SPEAKER_02`'s question (`"...is that alright?"`) is cleanly separated into its own chronological turn by `DiarizationLM-Gemma` + `LG-TPST`.

### 11.2 Excerpt 2: `R04_piers_robinson__transcript.txt` (Multi-Speaker Overlap Annotation)
Source: [`convofuse_outputs/final_release/R04_piers_robinson/R04_piers_robinson__transcript.txt`](convofuse_outputs/final_release/R04_piers_robinson/R04_piers_robinson__transcript.txt)

```text
[00:00:15.008 – 00:00:16.609] SPEAKER_01: So what's the answer to the question? Including
  # [OVERLAP_METADATA: multi_speaker_rttm=True | concurrent_speakers=SPEAKER_01,SPEAKER_02 | max_frame_overlap_posterior=0.722 | single_lexical_stream_preserved=True]

[00:00:16.629 – 00:00:17.049] SPEAKER_02: British
  # [OVERLAP_METADATA: multi_speaker_rttm=True | concurrent_speakers=SPEAKER_00,SPEAKER_02 | max_frame_overlap_posterior=0.502 | single_lexical_stream_preserved=True]

[00:00:17.069 – 00:00:26.914] SPEAKER_00: Muslims? Yeah, the British public. I didn't say British non-Muslim public, I said the British public. We've had three terror attacks, children are being killed.
  # [OVERLAP_METADATA: multi_speaker_rttm=True | concurrent_speakers=SPEAKER_00,SPEAKER_01,SPEAKER_02 | max_frame_overlap_posterior=0.954 | single_lexical_stream_preserved=True]
```

---

## 12. Consolidated Experimental Results Summary

Complete per-recording tables and confidence intervals are provided in [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md).

### 12.1 Multi-View Speaker Representation Summary (`R01`–`R08`)

| Condition | 6-Rec Verification ROC-AUC $\uparrow$ | 6-Rec Verification HTER (%) $\downarrow$ | 8-Rec Macro DER (`c=0, ovl`) (%) $\downarrow$ | 8-Rec Macro DER (`c=0.25, no-ovl`) (%) $\downarrow$ | Spk Count MAE $\downarrow$ | 8-Rec Macro cpWER (%) $\downarrow$ | 8-Rec Macro-SACC (%) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 2 Legacy MSDD Baseline** | — | — | 47.48% | 42.52% | 1.500 | 39.34% | 49.27% |
| **Cond A / H (`TitaNet-Large` on `dfn`)** | 0.8238 | 33.80% | 46.22% | 39.72% | 1.500 | 38.07% | 49.42% |
| **Cond B / I (`ResNet293-LM` on `16k_mono`)** | 0.8230 | 32.40% | 45.98% | **38.94%** | 1.625 | **36.44%** | 48.94% |
| **Cond J (Normalized Equal Fusion `0.5 / 0.5`)** | 0.8450 | 32.69% | 46.17% | 39.65% | **1.250** | 37.05% | **50.25%** |
| **Cond K (Normalized LORO Reliability Gate)** | **0.8458** | **31.90%** | **45.92%** | 39.39% | 1.625 | 37.74% | 48.99% |

### 12.2 Generative Speaker Refinement Summary (`R01`–`R08`, `convofuse_outputs/final_release/`)

| Recording ID | TS-WDER Base $\rightarrow$ Ref ($\Delta$ pp) | Lex-WDER Base $\rightarrow$ Ref ($\Delta$ pp) | cpWER Base $\rightarrow$ Ref ($\Delta$ pp) | Accepted / Rejected / Abstained | Integrity & SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`R01_sara_sharif`** (`E20`) | 20.08% $\rightarrow$ 20.08% (`0.00`) | 19.38% $\rightarrow$ 19.38% (`0.00`) | 52.30% $\rightarrow$ 52.30% (`0.00`) | 1 / 2 / 0 | Verified |
| **`R02_dennis_skinner`** (`Cond J`) | 37.25% $\rightarrow$ 37.25% (`0.00`) | 18.79% $\rightarrow$ **18.46%** (`-0.33`) | 43.00% $\rightarrow$ **42.67%** (`-0.33`) | 1 / 8 / 1 | Verified |
| **`R03_disabled_parking`** (`Cond J`) | 8.55% $\rightarrow$ 9.25% (`+0.70`) | 4.83% $\rightarrow$ 5.48% (`+0.65`) | 19.60% $\rightarrow$ 20.52% (`+0.92`) | 11 / 6 / 3 | Verified |
| **`R04_piers_robinson`** (`Cond J`) | 26.59% $\rightarrow$ 26.59% (`0.00`) | 11.69% $\rightarrow$ **10.88%** (`-0.81`) | 38.57% $\rightarrow$ **37.75%** (`-0.82`) | 22 / 17 / 20 | Verified |
| **`R05_southwest_pilot`** (`Cond J`) | 2.51% $\rightarrow$ 2.51% (`0.00`) | 8.68% $\rightarrow$ **8.12%** (`-0.56`) | 31.25% $\rightarrow$ **30.21%** (`-1.04`) | 9 / 0 / 10 | Verified |
| **`R06_rich_karen`** (`Cond J`) | 21.19% $\rightarrow$ 22.48% (`+1.29`) | 25.33% $\rightarrow$ 25.82% (`+0.49`) | 49.81% $\rightarrow$ **49.31%** (`-0.50`) | 40 / 10 / 35 | Verified |
| **`R07_bwc_cheap_shot`** (`Cond J`) | 12.53% $\rightarrow$ 12.53% (`0.00`) | 15.93% $\rightarrow$ 15.93% (`0.00`) | 52.99% $\rightarrow$ 52.99% (`0.00`) | 0 / 5 / 0 | Verified |
| **`R08_uk_parliament`** (`Cond J`) | 0.22% $\rightarrow$ 0.22% (`0.00`) | 1.72% $\rightarrow$ 1.72% (`0.00`) | 10.32% $\rightarrow$ **10.11%** (`-0.21`) | 1 / 0 / 0 | Verified |
| **Macro Average (`R01`–`R08`)** | **16.11% $\rightarrow$ 16.36%** (`+0.25`) | **13.29% $\rightarrow$ 13.22%** (`-0.07`) | **37.23% $\rightarrow$ 36.98%** (`-0.25`) | **85 / 48 / 69** | **100% Verified** |

---

## 13. Error Analysis & Honest Limitations

1. **Single-Stream ASR Ceiling Under Simultaneous Speech**: Because `Faster-Whisper Large-v3` + `Wav2Vec2` CTC forced alignment produces a single chronological word stream, simultaneous overlapping utterances by a secondary talker cannot be transcribed separately without multi-speaker speech separation (e.g., MossFormer2 / TF-GridNet). ConvoFuse honestly flags overlapping turns and words (`is_overlapping_speech`, `frame_overlap_prob_max`) rather than fabricating a second lexical stream.
2. **Severe Siren/Crowd Noise Collapse on `R07_bwc_cheap_shot`**: Under extreme nighttime siren and shouting noise (`R07`), unsupervised NME-SC eigengap clustering collapses to 1 speaker (`DER = 84.74%` under Condition J, `79.77%` under `ResNet293-Mono`), and `LG-TPST` rightly rejects all 5 out-of-inventory proposals (`REJECTED_UNKNOWN_SPEAKER_MAPPING = 5`), leaving `R07` unchanged.
3. **Text-Only LLM Ambiguity on Rapid Roadside Exchanges (`R03_disabled_parking`)**: Because `DiarizationLM-Gemma-4-E4B-v1` observes only lexical tokens and `<speaker:k>` tags without acoustic speaker margin scores, it over-smooths rapid short-phrase exchanges on `R03_disabled_parking` (`+0.92 pp` `cpWER`). Incorporating acoustic margin confidence directly into the `LG-TPST` acceptance gate is the clearest path to preventing `R03`-style regressions.

---

## 14. Reproducibility & Conclusion

All results and artifacts in [`convofuse_outputs/final_release/`](convofuse_outputs/final_release/) are reproducible via a single CLI command documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md):

```bash
PYTHONPATH=. ./research_stages/.venv_phase20/bin/python -m convofuse.cli \
  --all-recordings \
  --condition auto \
  --output-dir convofuse_outputs/final_release
```

### Future Work
1. **Acoustic-Margin-Conditioned `LG-TPST`**: Weighting `LG-TPST` boundary transfers by the local multi-view affinity margin between the primary and secondary cluster centroids to prevent lexical over-smoothing on `R03`.
2. **Corpus-Wide Fresh Diarization (`Stage 5B.4` Expansion)**: Extending `E20` (`C62` Consensus VAD + Condition J Equal Fusion) from the `R01` pilot (`-5.15 pp` DER reduction) across `R02`–`R08`.
3. **Joint Target-Speaker ASR (TS-ASR)**: Conditioning `Whisper Large-v3` on fused multi-view speaker embeddings during high-overlap intervals (`frame_overlap_prob_max >= 0.50`) to recover secondary overlapping lexical streams.
