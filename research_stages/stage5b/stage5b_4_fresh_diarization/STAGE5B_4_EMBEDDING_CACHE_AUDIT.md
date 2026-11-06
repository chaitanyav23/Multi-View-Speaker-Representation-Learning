# Stage 5B.4 — Feature Extraction & Embedding Cache Audit

**Audit Status**: Complete  
**Date**: 2026-11-06  
**Auditor**: ML Research Engineer & Diarization Systems Auditor  
**Repository**: [Multi-View-Speaker-Representation-Learning](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning)  
**Deliverable Path**: `research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_EMBEDDING_CACHE_AUDIT.md`

---

## 1. Executive Summary

Feature representation in this project relies on a dual-encoder architecture designed to capture complementary acoustic and channel characteristics:
1. **NVIDIA TitaNet-Large** operating on DeepFilterNet3 speech-enhanced audio (`dfn_wav`), producing 192-dimensional embeddings.
2. **WeSpeaker VoxCeleb ResNet293-LM** operating on raw 16kHz mono audio (`mono_wav`), producing 256-dimensional embeddings.

This audit evaluates the model implementations, frontends, checkpoints, and crucially, the **embedding cache architecture**. The primary finding is a **major cache invalidation vulnerability**: the existing Stage 3 cache schema is keyed solely on audio condition and scale index, completely omitting VAD configuration ID and subsegment timestamps. If Stage 5B.4 naively accesses or populates this cache, cross-condition data corruption or stale embedding leakage will occur.

---

## 2. Model & Checkpoint Verification

Both pretrained speaker embedding encoders were located and cryptographically verified on local disk:

```
+-------------------------------------------------------------------------------------------------------------------------------+
| Model Identifier     | Embedding Dim | Checkpoint Path & Location                           | Size (Bytes) | SHA-256 Hash    |
+-------------------------------------------------------------------------------------------------------------------------------+
| NVIDIA TitaNet-Large | 192           | ~/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/ | 107,609,008  | 450f6bd279a3f783|
|                      |               | 3c3697a0a46f945574fa407149975a13/                    |              | 48e883359f9b3b34|
|                      |               | diar_msdd_telephonic.nemo                            |              | 687dbd15814fd0f1|
+-------------------------------------------------------------------------------------------------------------------------------+
| WeSpeaker            | 256           | ~/.cache/huggingface/hub/models--Wespeaker--         | 133,998,630  | e7b8447ea84345e2|
| ResNet293-LM (PyTorch|               | wespeaker-voxceleb-resnet293-LM/snapshots/           |              | 6432461b4dda224d|
|                      |               | 6e6bffe5bf3d772a1f143dc6dbfea58a0799ea83/avg_model.pt|              | 1af34a3985a43826|
+-------------------------------------------------------------------------------------------------------------------------------+
| WeSpeaker            | 256           | Same snapshot directory as above:                    | 114,336,285  | dbb1ccc7754caff5|
| ResNet293-LM (ONNX)  |               | voxceleb_resnet293_LM.onnx                           |              | 52ebc46347a51aae|
+-------------------------------------------------------------------------------------------------------------------------------+
```

### Acoustic Frontend Verification:
- **TitaNet-Large**: Built directly into NeMo's `EncDecSpeakerLabelModel`. Consumes raw 16kHz float32 tensors with length tensors. Extracts 80-channel log-mel filterbanks internally via `AudioToMelSpectrogramPreprocessor` matching NeMo's standard training configuration (`window_size=0.025`, `window_stride=0.010`, `n_fft=512`, `n_mels=80`).
- **WeSpeaker ResNet293-LM**: Uses Torchaudio Kaldi-compliant 80-dimensional filterbanks (`torchaudio.compliance.kaldi.fbank`). Parameters: `num_mel_bins=80`, `frame_shift=10.0`, `frame_length=25.0`, `dither=0.0`, `energy_floor=0.0`, followed by cepstral mean normalization (CMN) subtractive centering.
- **Parity Check**: In Stage 3, ONNX runtime parity against native PyTorch was verified to achieve cosine similarity $\ge 0.99999$ and maximum absolute difference $\le 3.5 \times 10^{-6}$.

---

## 3. Vulnerability Audit: The Existing Stage 3 Cache Schema

### 3.1 Inspection of Frozen Stage 3 Storage Structure
In `research_stages/stage3_dual_encoder/cache/`, cached embeddings are organized as:
```
stage3_dual_encoder/cache/
├── dfn/
│   └── {rec_id}/
│       ├── subsegments_metadata__scale0.json
│       ├── subsegments_metadata__scale1.json
│       ├── subsegments_metadata__scale2.json
│       ├── subsegments_metadata__scale3.json
│       ├── subsegments_metadata__scale4.json
│       ├── titanet_large__dfn__scale0.npz
│       └── wespeaker_resnet293_lm__dfn__scale0.npz
├── 16k_mono/
│   └── {rec_id}/...
└── normalized/
    └── {rec_id}/...
```

### 3.2 The Flaw: Missing VAD Identity and Segmentation Hash
Inspection of `subsegments_metadata__scale{k}.json` and `*.npz` reveals:
1. **Filename Keys**: Filenames follow `titanet_large__{cond}__scale{scale_k}.npz`.
2. **Missing Metadata**: The metadata dictionary records segment `[start, end, duration]`, but **records zero information about which VAD configuration produced those segments**, nor does it store an MD5/SHA-256 hash of the parent speech intervals.
3. **Hardcoded C00 Assumption**: Every single file currently in `research_stages/stage3_dual_encoder/cache/` was generated from the Stage 2 NeMo MSDD Path A subsegments (`C00`).

### 3.3 Collision and Contamination Risks in Stage 5B.4
If Stage 5B.4 attempts to cache embeddings for `C50` (Raw Mono VAD) or `C62` (Consensus VAD):
- **Scenario A (Direct Overwrite)**: If written to the existing Stage 3 cache directory, it would **overwrite and destroy frozen Stage 3 artifacts**, violating repository preservation boundaries.
- **Scenario B (Stale Cache Hit)**: If Stage 5B.4 code checks `if cache_file.exists(): load()`, it would mistakenly load the `C00` embeddings for `C50` and `C62`. The experimental conditions would be silently evaluated on the wrong segmentation!
- **Scenario C (Dimension Mismatch Crash)**: When `C50` generates 1,200 base subsegments and `C00` had 1,050 base subsegments, loading `C00` embeddings would cause catastrophic matrix shape mismatches during affinity construction and clustering.

---

## 4. Proposed Solution: Collision-Free, VAD-Aware Cache Schema

Stage 5B.4 must enforce absolute isolation by creating an independent cache under `research_stages/stage5b/stage5b_4_fresh_diarization/cache/`.

### 4.1 Canonical Hash Key Specification
Every VAD configuration produces an ordered sequence of non-overlapping speech intervals:
$$S = \{(s_1, e_1), (s_2, e_2), \dots, (s_N, e_N)\}$$

We define the canonical VAD segmentation hash as:
$$\text{Hash}_{\text{VAD}} = \text{SHA256}\left( \text{rec\_id} \,\|\, \text{vad\_id} \,\|\, \left[\text{round}(s_i, 4), \text{round}(e_i, 4)\right]_{i=1}^N \right)[:16]$$

### 4.2 Proposed File Naming Convention
Embeddings and subsegment manifests must be named using this composite key:
```
research_stages/stage5b/stage5b_4_fresh_diarization/cache/
└── {vad_id}/                     # e.g., C00, C50, C62
    └── {rec_id}/                 # e.g., R01_sara_sharif
        ├── subsegments__scale{k}__{vad_hash}.json
        ├── titanet_large__dfn__scale{k}__{vad_hash}.npz
        └── wespeaker_resnet293_lm__mono__scale{k}__{vad_hash}.npz
```

### 4.3 Data Structures inside `.npz`
Each `.npz` file must store:
- `embeddings`: Float32 array of shape `(N_subsegments, embedding_dim)`.
- `l2_norms`: Float32 array of shape `(N_subsegments,)` recording pre-normalization vector norms.
- `subsegment_starts`: Float64 array of shape `(N_subsegments,)`.
- `subsegment_ends`: Float64 array of shape `(N_subsegments,)`.
- `vad_config_id`: String (e.g., `"C62"`).
- `vad_intervals_sha256`: String (full 64-char hex hash).
- `encoder_name`: String (e.g., `"titanet_large"` or `"wespeaker_resnet293_lm"`).

---

## 5. Audit Verdict on Feature Extraction & Embedding Cache

- **Model Readiness**: **READY (VERIFIED)**. Checkpoints, ONNX models, and frontends exist locally and pass parity.
- **Cache Architecture Readiness**: **NOT READY (DEFECT IDENTIFIED)**. Existing Stage 3 cache cannot be reused or written to without severe data contamination.
- **Remedy**: Implement the VAD-aware isolated cache schema described above within the Stage 5B.4 workspace.
