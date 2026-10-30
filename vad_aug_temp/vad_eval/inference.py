from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .common import align_probabilities, atomic_save_npy, project_path, read_csv, write_csv
from .config import FRAME_SHIFT_SEC, METADATA_PATH, MODEL_NAME, PREDICTIONS_DIR, PROBABILITIES_DIR


def _speech_index(model: Any, class_count: int) -> int:
    labels = []
    try:
        labels = list(model.cfg.labels)
    except Exception:
        pass
    for index, label in enumerate(labels):
        if str(label).lower() in {"speech", "1"}:
            return index
    return 1 if class_count > 1 else 0


def _to_probabilities(output: Any, model: Any) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for inference") from exc

    if isinstance(output, (tuple, list)) and len(output) == 1:
        output = output[0]
    for attribute in ("y_sequence", "logits", "log_probs", "probabilities"):
        if hasattr(output, attribute):
            value = getattr(output, attribute)
            if value is not None:
                output = value
                break
    if isinstance(output, dict):
        for key in ("logprobs", "log_probs", "logits", "probabilities", "probs"):
            if key in output:
                output = output[key]
                break
    if torch.is_tensor(output):
        output = output.detach().float().cpu().numpy()
    array = np.asarray(output)
    array = np.squeeze(array)
    if array.ndim == 0:
        raise ValueError("NeMo returned a single class instead of frame-level scores")
    if array.ndim == 1:
        if np.all((array >= 0.0) & (array <= 1.0)):
            return array.astype(np.float32)
        return (1.0 / (1.0 + np.exp(-array))).astype(np.float32)
    if array.ndim > 2:
        array = array.reshape(-1, array.shape[-1])
    if array.shape[-1] <= 8:
        logits = array
    elif array.shape[0] <= 8:
        logits = array.T
    else:
        raise ValueError(f"Cannot identify class dimension in NeMo output shape {array.shape}")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return probabilities[:, _speech_index(model, probabilities.shape[-1])].astype(np.float32)


def load_model(model_name: str = MODEL_NAME, device: str = "auto") -> Any:
    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError("Install torch, torchaudio, and nemo_toolkit[asr] before inference") from exc

    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    try:
        model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name=model_name)
    except Exception as exc:
        available = nemo_asr.models.EncDecClassificationModel.list_available_models()
        names = [item.pretrained_model_name for item in available if "vad" in item.pretrained_model_name.lower()]
        raise RuntimeError(f"Could not load {model_name!r}. Available VAD models: {names}") from exc
    model.eval()
    return model.to(selected_device)


def transcribe_one(model: Any, audio_path: Path) -> np.ndarray:
    attempts = (
        {"paths2audio_files": [str(audio_path)], "batch_size": 1, "logprobs": True},
        {"audio": [str(audio_path)], "batch_size": 1, "logprobs": True},
        {"paths2audio_files": [str(audio_path)], "batch_size": 1},
    )
    last_error = None
    for kwargs in attempts:
        try:
            result = model.transcribe(**kwargs)
            return _to_probabilities(result[0] if isinstance(result, list) else result, model)
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(
        "The installed NeMo version did not return frame-level MarbleNet scores via transcribe(). "
        "Pin the versions in requirements.txt or inspect the model's transcribe API."
    ) from last_error


def run_inference(
    metadata_path: Path = METADATA_PATH,
    model_name: str = MODEL_NAME,
    device: str = "auto",
    limit: int | None = None,
    overwrite: bool = False,
) -> None:
    rows = read_csv(metadata_path)
    if limit:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"No metadata rows found in {metadata_path}")
    model = load_model(model_name, device)
    for index, row in enumerate(rows, start=1):
        file_id = row["file_id"]
        probability_path = PROBABILITIES_DIR / f"{file_id}.npy"
        if probability_path.exists() and not overwrite:
            print(f"[{index}/{len(rows)}] skip {file_id}")
            continue
        duration = float(row["duration_sec"])
        frame_count = max(1, int(np.ceil(duration / FRAME_SHIFT_SEC)))
        raw = transcribe_one(model, project_path(row["audio_path"]))
        probabilities = align_probabilities(raw, frame_count)
        atomic_save_npy(probability_path, probabilities)
        predictions = [
            {
                "file_id": file_id,
                "start": f"{i * FRAME_SHIFT_SEC:.2f}",
                "end": f"{min(duration, (i + 1) * FRAME_SHIFT_SEC):.2f}",
                "speech_probability": f"{value:.7f}",
                "pred_label": int(value >= 0.5),
            }
            for i, value in enumerate(probabilities)
        ]
        write_csv(
            PREDICTIONS_DIR / f"{file_id}.csv",
            predictions,
            ["file_id", "start", "end", "speech_probability", "pred_label"],
        )
        print(f"[{index}/{len(rows)}] wrote {file_id}: {probabilities.size} frames")
