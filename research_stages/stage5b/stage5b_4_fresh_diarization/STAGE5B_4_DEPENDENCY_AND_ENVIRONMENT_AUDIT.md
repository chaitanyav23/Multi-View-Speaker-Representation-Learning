# Stage 5B.4 — Dependency & Environment Pre-Execution Audit

**Audit Status**: Complete  
**Date**: 2026-11-06  
**Auditor**: ML Research Engineer & Diarization Systems Auditor  
**Repository**: [Multi-View-Speaker-Representation-Learning](file:///home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning)  
**Deliverable Path**: `research_stages/stage5b/stage5b_4_fresh_diarization/STAGE5B_4_DEPENDENCY_AND_ENVIRONMENT_AUDIT.md`

---

## 1. Executive Summary

This audit assesses the hardware, runtime environment, libraries, model checkpoints, and computational budgets required to execute Stage 5B.4 fresh diarization. All required packages, CUDA kernels, and model weights are verified present and functional within the repository's dedicated virtual environment (`research_stages/.venv_phase20/`). The system operates with zero external network access requirements during execution.

---

## 2. Hardware & Host Environment Verification

| Property | Detected System Specification | Audit Status |
| :--- | :--- | :--- |
| **Operating System** | Linux (Ubuntu 6.8.0-52-generic, x86_64) | Compliant |
| **GPU Hardware** | 1 $\times$ NVIDIA GeForce RTX 4090 (24,564 MiB VRAM) | Compliant |
| **NVIDIA Driver** | 570.86.16 | Compliant |
| **CUDA Driver Version** | 12.8 | Compliant |
| **Host System Memory** | 62.7 GiB Physical RAM | Compliant |
| **Host CPU** | AMD Ryzen 9 7900X 12-Core Processor (24 threads) | Compliant |
| **Storage Availability** | > 1.2 TB available on workspace filesystem | Compliant |

---

## 3. Python Virtual Environment & Library Stack

The authoritative execution environment is `/home/priyanka/chaitanya/Multi-View-Speaker-Representation-Learning/research_stages/.venv_phase20/bin/python`.

```
+---------------------------------------------------------------------------------------------------+
| Component / Package     | Verified Version     | Source / Wheel                   | Compatibility |
+---------------------------------------------------------------------------------------------------+
| Python Runtime          | 3.10.20              | CPython (64-bit)                 | Full          |
| PyTorch (`torch`)       | 2.8.0+cu128          | PyTorch CUDA 12.8 binary         | Full          |
| TorchAudio (`torchaudio`)| 2.8.0+cu128         | PyTorch CUDA 12.8 binary         | Full          |
| TorchVision (`torchvision`)| 0.23.0+cu128       | PyTorch CUDA 12.8 binary         | Full          |
| NeMo Toolkit (`nemo_toolkit`)| 2.7.3           | NVIDIA NeMo ASR collection       | Full          |
| Pyannote Audio          | 4.0.4                | HuggingFace / pyannote           | Full          |
| Pyannote Metrics        | 4.1                  | Standard diarization metrics     | Full          |
| WhisperX                | 3.8.5                | Faster-Whisper alignment engine  | Full          |
| ONNX Runtime GPU        | 1.23.2               | CUDA/TensorRT execution provider | Full          |
| JiWER                   | 3.1.0                | cpWER Hungarian alignment        | Full          |
| SciPy / NumPy           | 1.15.2 / 2.2.3       | Array and spectral operations    | Full          |
+---------------------------------------------------------------------------------------------------+
```

### CUDA Acceleration Verification:
- `torch.cuda.is_available()` returns `True`.
- Device name: `NVIDIA GeForce RTX 4090`.
- CuDNN enabled: `torch.backends.cudnn.enabled == True`, version `90800`.
- CuBLAS matrix multiplication verified functional on device `cuda:0`.

---

## 4. Model Checkpoint Inventory & Integrity

All neural model weights required for feature extraction, clustering, and overlap detection exist locally on disk and require zero internet downloads:

```
+-------------------------------------------------------------------------------------------------------------------------+
| Architecture       | Checkpoint Path                                                   | Size (Bytes) | SHA-256 (Trunc) |
+-------------------------------------------------------------------------------------------------------------------------+
| Pyannote PyanNet   | research_stages/.venv_phase20/lib/python3.10/site-packages/       | 17,719,103   | 0b5b3216d60a2d32|
| VAD & Overlap      | whisperx/assets/pytorch_model.bin                                 |              | fcbe620d6d0b83e2|
+-------------------------------------------------------------------------------------------------------------------------+
| WeSpeaker          | ~/.cache/huggingface/hub/models--Wespeaker--wespeaker-voxceleb-   | 133,998,630  | e7b8447ea84345e2|
| ResNet293-LM (PyT) | resnet293-LM/snapshots/6e6bffe5bf3d.../avg_model.pt               |              | 1af34a3985a43826|
+-------------------------------------------------------------------------------------------------------------------------+
| WeSpeaker          | Same snapshot directory as above:                                 | 114,336,285  | dbb1ccc7754caff5|
| ResNet293-LM (ONNX)| voxceleb_resnet293_LM.onnx                                        |              | 52ebc46347a51aae|
+-------------------------------------------------------------------------------------------------------------------------+
| NVIDIA             | ~/.cache/torch/NeMo/NeMo_2.7.3/diar_msdd_telephonic/              | 107,609,008  | 450f6bd279a3f783|
| TitaNet-Large      | 3c3697a0a46f945574fa407149975a13/diar_msdd_telephonic.nemo        |              | 687dbd15814fd0f1|
+-------------------------------------------------------------------------------------------------------------------------+
```

---

## 5. Runtime & Memory Budget Estimation

Based on execution profiles established in Stage 3 and Stage 5A.1 on this exact RTX 4090 system:

### 5.1 Workload Volume:
- Total recordings: 8 (R01–R08).
- Total corpus audio duration: ~2,400 seconds (40.0 minutes).
- Average subsegments per VAD condition: ~5,200 multi-scale slices across scales $0..4$.
- Number of fresh VAD conditions: 3 (`C00`, `C50`, `C62`).
- Number of fresh overlap conditions: 2 (`O02`, `O42-R2`).
- Total evaluated conditions: $3 \times 2 = 6$ (`E00`, `E01`, `E10`, `E11`, `E20`, `E21`).

### 5.2 Step-by-Step Runtime Estimate:
1. **VAD Generation**: Frame-level posteriors are already computed and cached in `stage2_baseline/posteriors/`. Thresholding and hysteresis take $< 2$ seconds total.
2. **Subsegmentation**: Pure CPU slice indexing using `generate_neMo_identical_subsegments` takes $< 5$ seconds for all 8 recordings $\times 3$ conditions.
3. **Dual-Encoder Feature Extraction**:
   - TitaNet-Large on RTX 4090 (batch size 64): ~45 seconds per VAD condition.
   - WeSpeaker ResNet293-LM on RTX 4090: ~60 seconds per VAD condition.
   - Total extraction time: $(45\text{s} + 60\text{s}) \times 3 \approx 5.25\text{ minutes}$.
4. **Affinity Construction & Condition J Fusion**:
   - Matrix inner products and off-diagonal z-scoring on GPU: ~15 seconds total.
5. **NME-SC Clustering**:
   - Unit-infer spectral clustering with eigengap search ($K \in [1, 10]$): ~45 seconds per condition $\times 6$ conditions $\approx 4.5\text{ minutes}$.
6. **Overlap Attribution**:
   - Local turn alternation search (`attribute_m3_m4_corrected`) on CPU: ~15 seconds total.
7. **Downstream Metrics & cpWER**:
   - Diarization metrics via `pyannote.metrics` and cpWER via `jiwer`: ~2.5 minutes total.

**Total Projected Wall-Clock Time**: **12 to 15 minutes**.

### 5.3 Memory & VRAM Budget:
- **Peak GPU VRAM**: TitaNet-Large and WeSpeaker require $< 4.5$ GiB VRAM during batch extraction (well within the 24.5 GiB available on the RTX 4090).
- **Peak Host RAM**: Affinity matrices and audio slices consume $< 6$ GiB RAM (well within the 62.7 GiB available).
- **Disk Footprint**: Embedding cache for 3 conditions $\times 8$ recordings $\times 5$ scales $\times 2$ encoders requires approximately $120$ MB of disk space.

---

## 6. Audit Verdict on Dependencies & Environment

- **Environment Status**: **FULLY COMPLIANT & READY**.
- All dependencies, CUDA acceleration kernels, and neural checkpoints are present and verified.
- The compute system has ample capacity to execute Stage 5B.4 within a ~15-minute runtime window once orchestration scripts are authorized and deployed.
