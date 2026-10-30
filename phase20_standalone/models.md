# External Models

| Model | Source | Version / identifier | License | Download method | Authentication |
|---|---|---|---|---|---|
| Pyannote VAD | Hugging Face | `pyannote/voice-activity-detection` | Upstream model license | `download_models.py` and `step03_vad.py` via `pyannote.audio` | `HF_TOKEN` required; user must have accepted model terms |
| WhisperX ASR | WhisperX / faster-whisper | `large-v3` | Upstream OpenAI Whisper / derivative package licenses | `download_models.py` and `step05b_whisperx.py` via `whisperx.load_model` | Internet access; no token normally required |
| WhisperX alignment | WhisperX / torchaudio ecosystem | English alignment model selected by WhisperX | Upstream model license | `whisperx.load_align_model(language_code="en")` | Internet access |
| TitaNet speaker embeddings | NVIDIA NeMo | `titanet_large` | NVIDIA NeMo model license | `download_models.py`; also resolved by NeMo during `step05a_msdd.py` | Internet access |
| NeMo MSDD | NVIDIA NeMo | `diar_msdd_telephonic` | NVIDIA NeMo model license | Resolved by NeMo during `step05a_msdd.py` | Internet access |
| DeepFilterNet | DeepFilterNet package assets | `DeepFilterNet==0.5.6` / `DeepFilterLib==0.5.6` | Upstream package license | Installed from `requirements.txt`; invoked as `deepFilter` | None |
