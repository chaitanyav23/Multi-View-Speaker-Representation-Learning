"""
scripts/utils/device.py
=======================
GPU / CPU device detection and memory-management helpers.

All pipeline modules call ``get_device()`` to obtain the correct
:class:`torch.device` rather than hardcoding ``cuda`` or ``cpu``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

log = logging.getLogger("bwc_pipeline.device")


# ── Public API ───────────────────────────────────────────────────────────────

def get_device(
    prefer_cuda: bool = True,
    cuda_device_index: int = 0,
    fallback_to_cpu: bool = True,
) -> torch.device:
    """
    Return the best available :class:`torch.device`.

    Parameters
    ----------
    prefer_cuda:
        Attempt to use a CUDA GPU if available.
    cuda_device_index:
        Which GPU to use (0-indexed).
    fallback_to_cpu:
        If ``prefer_cuda`` is True but no GPU is available, fall back to CPU
        instead of raising.

    Returns
    -------
    torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if cuda_device_index >= n_gpus:
            log.warning(
                "Requested GPU %d but only %d GPU(s) found. "
                "Using GPU 0 instead.",
                cuda_device_index,
                n_gpus,
            )
            cuda_device_index = 0

        device = torch.device(f"cuda:{cuda_device_index}")
        log.info(
            "Using GPU: %s  |  VRAM: %.1f GB  |  CUDA %s",
            torch.cuda.get_device_name(cuda_device_index),
            torch.cuda.get_device_properties(cuda_device_index).total_memory
            / 1024 ** 3,
            torch.version.cuda,
        )
        return device

    if prefer_cuda and not torch.cuda.is_available():
        msg = "CUDA requested but torch.cuda.is_available() returned False."
        if fallback_to_cpu:
            log.warning("%s  Falling back to CPU.", msg)
        else:
            raise RuntimeError(msg)

    log.info("Using CPU (torch threads: %d)", torch.get_num_threads())
    return torch.device("cpu")


def log_gpu_memory(device: torch.device, tag: str = "") -> None:
    """
    Log current GPU VRAM usage.  No-op on CPU.
    """
    if device.type != "cuda":
        return
    idx = device.index if device.index is not None else 0
    alloc = torch.cuda.memory_allocated(idx) / 1024 ** 2
    reserved = torch.cuda.memory_reserved(idx) / 1024 ** 2
    log.debug(
        "[%s] GPU VRAM — allocated: %.1f MB  |  reserved: %.1f MB",
        tag or "mem",
        alloc,
        reserved,
    )


def clear_gpu_cache(device: torch.device) -> None:
    """
    Release cached (but not active) GPU tensors.  No-op on CPU.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        log.debug("GPU cache cleared.")


def move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Move *tensor* to *device* with a dtype safety guard.
    Silero VAD requires float32 — this function enforces that.
    """
    return tensor.to(device=device, dtype=torch.float32)
