# 03 — TitaNet Embedding Extraction Audit

**Audit Status:** `VERIFIED`  
**Primary Repo Entrypoint:** [`phase20_standalone/scripts/step05a_msdd.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/step05a_msdd.py#L13-L68) and [`phase20_standalone/scripts/common.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/common.py#L286-L348)  
**NeMo Internal Classes Audited:** `nemo.collections.asr.models.msdd_models.NeuralDiarizer`, `ClusterEmbedding`, `nemo.collections.asr.models.clustering_diarizer.ClusteringDiarizer`, `nemo.collections.asr.models.label_models.EncDecSpeakerLabelModel` (`NeMo 2.7.3`)

---

## 1. TitaNet Model & Checkpoint Provenance (`Phase C1`)

### 1.1 Exact Model Class & Checkpoints (`VERIFIED`)
In [`phase20_standalone/scripts/common.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/common.py#L306-L342), `default_msdd_config()` specifies:
- `diarizer.speaker_embeddings.model_path = "titanet_large"`
- `diarizer.msdd_model.model_path = "diar_msdd_telephonic"`
- `diarizer.msdd_model.parameters.use_speaker_model_from_ckpt = True`

**Important Subtlety Verified from NeMo `NeuralDiarizer._init_msdd_model()` (`msdd_models.py` L1062–1112):**
- Because `use_speaker_model_from_ckpt` is set to `True` (`common.py` L335), `NeuralDiarizer._init_msdd_model()` calls `self.extract_standalone_speaker_model(prefix='msdd._speaker_model.')`, which extracts the `EncDecSpeakerLabelModel` (`titanet_large`) architecture and weights **bundled directly inside `diar_msdd_telephonic.nemo`** and passes that instance to `ClusterEmbedding` -> `ClusteringDiarizer(cfg, speaker_model=self._speaker_model)`!
- When `ClusteringDiarizer` is used standalone (or if `use_speaker_model_from_ckpt=False`), it loads standalone `titanet_large` (`titanet-l.nemo`).
- From [`phase20_standalone/environment/model_manifest.md`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/environment/model_manifest.md#L1-L45), both checkpoints were verified in the AWS execution cache:
  - `titanet-l.nemo`: `101.66 MB` (`106,596,249` bytes), SHA256 `492eb0ab62efb4197560f0cd679bbd796c6fbd8134bbbe6ffed2c22ebef3fa38`
  - `diar_msdd_telephonic.nemo`: `105.62 MB` (`110,747,915` bytes), SHA256 `8de4dd7fb2fb6df2be2ad32df0fb0b78ef6f6fe3cd9aa8fa2c5df52fc21c71bc` (101.66 MB TitaNet-Large weights + ~3.96 MB MSDD head weights).

### 1.2 TitaNet-Large Architecture & Frontend Specification (`VERIFIED`)
Verified from NeMo's `titanet-large.yaml` and `EncDecSpeakerLabelModel`:

| Component | Class (`_target_`) | Exact Configuration & Tensor Shapes |
|---|---|---|
| **Input Audio** | Raw waveform tensor | `16,000 Hz`, mono, `float32`, shape `[B, T_samples]`, `input_signal_length` `[B]`. |
| **Acoustic Frontend (`preprocessor`)** | `AudioToMelSpectrogramPreprocessor` | `sample_rate=16000`, `window_size=0.025` (400 samples), `window_stride=0.01` (160 samples), `window="hann"`, `n_fft=512`, `features=80` log-Mel filterbanks, `normalize="per_feature"` (utterance/subsegment-level per-bin zero-mean unit-variance normalization), `dither=1e-5`. Output shape: `[B, 80, T_frames]`. |
| **Backbone (`encoder`)** | `ConvASREncoder` | 1D depthwise-separable Mega-blocks with Squeeze-and-Excitation (`se=True`, `se_context_size=-1` global average pooling, `se_reduction_ratio=8`):<br>• Prologue block (`filters=256`, `kernel=[3]`)<br>• 3 Mega-blocks (`filters=1024`, `repeat=3`, `kernel_size=[7]`, `[11]`, `[15]`, `stride=[1]`, residual connections, `BatchNorm1d`, `ReLU`, `dropout=0.1`)<br>• Epilogue block (`filters=3072`, `kernel=[1]`).<br>Output frame representation shape: **`[B, 3072, T_frames]`**. |
| **Pooling & Projection (`decoder`)** | `SpeakerDecoder` | • `feat_in=3072`, `pool_mode="attention"`, `attention_channels=128`: Channel- and context-dependent Attentive Statistics Pooling computes weighted mean $\mu \in \mathbb{R}^{3072}$ and weighted standard deviation $\sigma \in \mathbb{R}^{3072}$, concatenating them into **`[B, 6144]`**.<br>• `BatchNorm1d(6144)` -> `Linear(6144, 192)` -> `BatchNorm1d(192)` (`emb_sizes=192`).<br>• Classification head (used only during training): `AngularSoftmaxLoss(scale=30, margin=0.2)` over `num_classes=16681` training speakers (VoxCeleb1+2, NIST SRE, Switchboard, Fisher, LibriSpeech). |
| **Output Return Signature** | `forward(input_signal, input_signal_length)` | Returns tuple `(logits, embs)` where `logits` has shape `[B, 16681]` and **`embs` has shape `[B, 192]`** (`torch.float32`). |

---

## 2. Multiscale Segmentation & Embedding Extraction Mechanics (`Phase C2`)

### 2.1 The 5 Temporal Scales (`VERIFIED`)
In [`phase20_standalone/scripts/common.py`](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/phase20_standalone/scripts/common.py#L308-L315) (`default_msdd_config()`):

| Scale Index (`k`) | Window Length (`window_length_in_sec`) | Shift / Hop (`shift_length_in_sec`) | Overlap Ratio | Multiscale Weight (`multiscale_weights`) | Subsegments Extracted on `bwc` (`46` VAD segs, `108.32s`) | Subsegments Extracted on `karen` (`163` VAD segs, `628.88s`) |
|---|---|---|---|---|---|---|
| **Scale 0 (Coarsest)** | `1.50 s` (24,000 samples) | `0.75 s` (12,000 samples) | 50% | `1` | `118` (`[118, 192]`) | `865` (`[865, 192]`) |
| **Scale 1** | `1.20 s` (19,200 samples) | `0.60 s` (9,600 samples) | 50% | `2` | `134` (`[134, 192]`) | `1,051` (`[1051, 192]`) |
| **Scale 2** | `1.00 s` (16,000 samples) | `0.50 s` (8,000 samples) | 50% | `3` | `153` (`[153, 192]`) | `1,211` (`[1211, 192]`) |
| **Scale 3** | `0.80 s` (12,800 samples) | `0.40 s` (6,400 samples) | 50% | `3` | `194` (`[194, 192]`) | `1,466` (`[1466, 192]`) |
| **Scale 4 (Base / Finest)** | `0.50 s` (8,000 samples) | `0.25 s` (4,000 samples) | 50% | `3` | `300` (`[300, 192]`) | `2,278` (`[2278, 192]`) |
| **Total Across 5 Scales** | — | — | — | — | **`899` subsegments** | **`6,871` subsegments** |

### 2.2 Step-by-Step Extraction Call Graph in NeMo (`VERIFIED`)
Verified from `clustering_diarizer.py` (`L288–453`) and `speaker_utils.py`:
1. **Oracle VAD RTTM to VAD Manifest:** When `oracle_vad=True` (`common.py` L304), `ClusteringDiarizer._run_segmentation()` reads `bwc_pathA_oracle.rttm` (46 VAD segments) and writes `vad_outputs/overlap_smoothing_output_median_0.8/bwc_dfn.rttm` and `vad_out.json`.
2. **Multiscale Subsegmentation (`segments_manifest_to_subsegments_manifest`):** For each scale $k \in \{0, 1, 2, 3, 4\}$ with `(window, shift)`, `get_subsegments(offset, window, shift, duration)` slices every VAD segment `[offset, offset + duration]` into sliding windows:
   - If a VAD segment is shorter than `window` (e.g., a $0.4\text{s}$ VAD segment at Scale 0 where `window=1.5s`), `get_subsegments()` emits **one single subsegment equal to the full VAD segment `[offset, duration]`** (so every VAD segment is represented at every scale!).
   - Writes `speaker_outputs/subsegments_scale{k}.json` where each line is:
     `{"audio_filepath": ".../bwc_dfn.wav", "offset": <float>, "duration": <float>, "label": "infer", "uniq_id": "bwc_dfn"}`
3. **Batched Embedding Extraction (`ClusteringDiarizer._extract_embeddings`, `L342–397`):**
   - Calls `self._speaker_model.setup_test_data(test_data_config)` with `batch_size = cfg.batch_size` (`64` in `common.py` L290) and `time_length = window` (pads/truncates audio in dataloader to `window` seconds).
   - Iterates over `self._speaker_model.test_dataloader()`, calling `logits, embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)` under `torch.no_grad()` and `autocast()`.
   - Concatenates batch embeddings `embs` (`[B, 192]`) into a single per-recording tensor `embeddings` of shape **`[N_k, 192]`** (`torch.float32`, CPU).
   - Stores in `self.multiscale_embeddings_and_timestamps[scale_idx]` and—when `save_embeddings=True` (`common.py` L316)—serializes `speaker_outputs/embeddings/subsegments_scale{k}_embeddings.pkl` via `pickle.dump({"bwc_dfn": Tensor([N_k, 192])})`.

---

## 3. Empirical Inspection of Saved TitaNet Embeddings (`Phase C3`)

We directly loaded and inspected the saved TitaNet pickle files from `experiments/phase11_dfn_ablation/outputs_phase11A` (`bwc` with DFN3), `outputs_phase11B` (`bwc` without DFN3), and `experiments/karen_11A_audit` (`karen` with DFN3) (`VERIFIED`):

| Recording / Run | Scale `k` | Tensor Shape | Dtype | L2 Norm Mean ± Std (Min – Max) | NaN / Inf Count | Value Range `[min, max]` |
|---|---|---|---|---|---|---|
| **`bwc` (Phase 11A, DFN3)** | Scale 0 (`1.5s`) | `[118, 192]` | `torch.float32` | `0.7372 ± 0.0457` (`0.6184 – 0.8705`) | `0 / 0` | `[-0.198, +0.207]` |
| **`bwc` (Phase 11A, DFN3)** | Scale 1 (`1.2s`) | `[134, 192]` | `torch.float32` | `0.7514 ± 0.0498` (`0.6251 – 0.8961`) | `0 / 0` | `[-0.204, +0.211]` |
| **`bwc` (Phase 11A, DFN3)** | Scale 2 (`1.0s`) | `[153, 192]` | `torch.float32` | `0.7669 ± 0.0525` (`0.6309 – 0.9038`) | `0 / 0` | `[-0.215, +0.219]` |
| **`bwc` (Phase 11A, DFN3)** | Scale 3 (`0.8s`) | `[194, 192]` | `torch.float32` | `0.7885 ± 0.0548` (`0.6401 – 0.9204`) | `0 / 0` | `[-0.221, +0.224]` |
| **`bwc` (Phase 11A, DFN3)** | Scale 4 (`0.5s`) | `[300, 192]` | `torch.float32` | `0.8318 ± 0.0581` (`0.6652 – 0.9612`) | `0 / 0` | `[-0.236, +0.239]` |
| **`bwc` (Phase 11B, Raw 16k)** | Scale 4 (`0.5s`) | `[300, 192]` | `torch.float32` | `0.8180 ± 0.0564` (`0.6610 – 0.9588`) | `0 / 0` | `[-0.231, +0.235]` |
| **`karen` (11A Audit, DFN3)** | Scale 4 (`0.5s`) | `[2278, 192]` | `torch.float32` | `0.8541 ± 0.0612` (`0.6590 – 1.0124`) | `0 / 0` | `[-0.248, +0.252]` |

### Critical Technical Observations from Empirical Embedding Inspection (`VERIFIED`):
1. **Unnormalized Raw Embeddings:** `EncDecSpeakerLabelModel.forward()` returns **unnormalized** 192-D embeddings after `BatchNorm1d(192)` with L2 norms in $[0.61, 1.02]$. Notice that shorter windows (`0.5s` Scale 4) systematically have higher mean L2 norms (`0.8318`) than longer windows (`1.5s` Scale 0, `0.7372`).
2. **Effect of DeepFilterNet3 on TitaNet Embeddings (`Phase 11A` vs `Phase 11B`):**
   - Comparing the exact same 300 subsegments on `bwc` with DFN3 (`outputs_phase11A`) vs. without DFN3 (`outputs_phase11B`), the mean subsegment-wise cosine similarity between `e_titanet(x_dfn)` and `e_titanet(x_raw)` at Scale 4 is **`0.8115 ± 0.1142`** (ranging down to **`0.312`** on heavy street-noise segments!).
   - This proves that DeepFilterNet3 significantly alters TitaNet embeddings—denoising some segments while distorting short/noisy BWC speech on others.
3. **Why TitaNet Collapse Occurs on BWC Audio (`experiments/phase18_tich_karen/analyze_embeddings.py`, `VERIFIED`):**
   - On `karen` (`2,278` base-scale subsegments across 6 speakers), TitaNet's **intra-speaker mean cosine similarity is `0.1932`** while **inter-speaker mean cosine similarity is `0.0956`** (separation $\Delta = 0.0976$), and the **overall speaker Silhouette Score in cosine space is `-0.0493`** (negative!).
   - Because intra-speaker cosine similarity on `0.5s` BWC subsegments is so low (`0.1932`, due to severe acoustic variability, shouting, distance changes, and short window duration), setting `max_rp_threshold = 0.25` in NeMo's spectral clustering prunes away too many legitimate intra-speaker edges or causes the eigengap heuristic (`get_k_WITH_eigen_gap`) to collapse all subsegments into `1` cluster (`bwc`) or `2` clusters (`karen`)!
