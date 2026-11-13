"""Isolated NVIDIA NeMo MSDD / TitaNet Diarization Worker CLI.

Executes strictly within the isolated 'cs698b_nemo' environment:
/home/priyanka/anaconda3/envs/cs698b_nemo/bin/python

This script has ZERO imports from the host application to maintain 100%
dependency isolation and runtime independence.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (nemo_worker) %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("nemo_diarization_worker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated NeMo MSDD/TitaNet Diarization Subprocess Worker"
    )
    parser.add_argument("--audio", required=True, type=Path, help="Input 16kHz mono WAV file")
    parser.add_argument(
        "--vad-rttm",
        required=True,
        type=Path,
        help="Speech-only VAD RTTM (oracle intervals with dummy speaker labels)",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output working directory")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to diar_msdd_telephonic.nemo checkpoint",
    )
    parser.add_argument(
        "--max-speakers",
        default=10,
        type=int,
        help="Maximum number of speakers for spectral clustering",
    )
    parser.add_argument("--device", default="cuda", type=str, help="Inference device: 'cuda' or 'cpu'")
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep multiscale embedding pickle files after inference",
    )
    return parser.parse_args()


def parse_rttm_stats(rttm_path: Path) -> Dict[str, Any]:
    """Parse NIST RTTM to extract speaker turns and distinct identities."""
    speakers = set()
    turns = 0
    total_speech_dur = 0.0

    if not rttm_path.exists():
        return {
            "speaker_count": 0,
            "speakers": [],
            "turn_count": 0,
            "total_speech_seconds": 0.0,
        }

    with open(rttm_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                turns += 1
                dur = float(parts[4])
                total_speech_dur += max(0.0, dur)
                spk = parts[7]
                speakers.add(spk)

    return {
        "speaker_count": len(speakers),
        "speakers": sorted(list(speakers)),
        "turn_count": turns,
        "total_speech_seconds": round(total_speech_dur, 3),
    }


def main() -> int:
    args = parse_args()

    # 1. Input Validation
    if not args.audio.exists():
        logger.error("Audio file does not exist: %s", args.audio)
        print(json.dumps({"status": "FAILED", "error": f"Audio file not found: {args.audio}"}))
        return 1

    if not args.vad_rttm.exists():
        logger.error("VAD RTTM file does not exist: %s", args.vad_rttm)
        print(json.dumps({"status": "FAILED", "error": f"VAD RTTM file not found: {args.vad_rttm}"}))
        return 2

    if not args.checkpoint.exists():
        logger.error("NeMo MSDD checkpoint does not exist: %s", args.checkpoint)
        print(json.dumps({"status": "FAILED", "error": f"Checkpoint not found: {args.checkpoint}"}))
        return 3

    args.out_dir.mkdir(parents=True, exist_ok=True)
    file_id = args.audio.stem

    # 2. Check if VAD RTTM contains any speech intervals
    vad_stats = parse_rttm_stats(args.vad_rttm)
    if vad_stats["turn_count"] == 0:
        logger.warning("VAD RTTM contains 0 speech intervals. Short-circuiting without model loading.")
        out_meta = {
            "status": "NO_SPEECH",
            "reason": "VAD RTTM contains 0 speech intervals.",
            "rttm_path": None,
            "speaker_count": 0,
            "speakers": [],
            "turn_count": 0,
            "total_speech_seconds": 0.0,
            "inference_time_seconds": 0.0,
        }
        print(json.dumps(out_meta))
        return 0

    # 3. Create Manifest JSON
    manifest_path = args.out_dir / f"{file_id}_manifest.json"
    manifest_entry = {
        "audio_filepath": str(args.audio.resolve()),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": str(args.vad_rttm.resolve()),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_entry) + "\n")

    # 4. Import NeMo and PyTorch
    t0 = time.time()
    try:
        import torch
        from nemo.collections.asr.models import NeuralDiarizer
        from omegaconf import OmegaConf
    except Exception as exc:
        logger.error("Failed to import NeMo dependencies: %s", exc)
        print(json.dumps({"status": "FAILED", "error": f"ImportError: {exc}"}))
        return 4

    dev = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    logger.info("Initializing NeuralDiarizer on %s (checkpoint: %s)...", dev, args.checkpoint.name)

    # 5. Build Phase 20 canonical configuration
    # Note: diar_msdd_telephonic.nemo embeds the TitaNet-Large speaker model.
    # Passing the verified local checkpoint path to msdd_model restores the full architecture.
    cfg = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": str(manifest_path.resolve()),
            "out_dir": str(args.out_dir.resolve()),
            "oracle_vad": True,
            "collar": 0.25,
            "ignore_overlap": False,
            "vad": {
                "model_path": None,
                "external_vad_manifest": None,
                "parameters": {},
            },
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.2, 1.0, 0.8, 0.5],
                    "shift_length_in_sec": [0.75, 0.6, 0.5, 0.4, 0.25],
                    "multiscale_weights": [1, 2, 3, 3, 3],
                    "save_embeddings": True,
                },
            },
            "msdd_model": {
                "model_path": str(args.checkpoint.resolve()),
                "parameters": {
                    "use_backim_for_multiscale": True,
                    "sigmoid_threshold": [0.70],
                    "use_adaptive_thres": False,
                    "overlap_infer_spk_limit": 5,
                    "diar_window_length": 50,
                    "infer_batch_size": 16,
                    "seq_eval_mode": False,
                    "split_infer": True,
                    "split_batch_size": 1,
                },
            },
            "clustering": {
                "parameters": {
                    "max_num_speakers": args.max_speakers,
                    "oracle_num_speakers": False,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                },
            },
        },
        "batch_size": 16,
        "num_workers": 1,
        "sample_rate": 16000,
        "verbose": False,
        "device": dev,
    })

    t_load = time.time() - t0
    logger.info("Configuration built in %.2fs. Beginning inference...", t_load)

    # 6. Execute NeuralDiarizer Inference
    t_infer_start = time.time()
    try:
        diarizer = NeuralDiarizer(cfg=cfg).to(torch.device(dev))
        diarizer.diarize()
    except Exception as exc:
        logger.error("NeMo diarization execution failed: %s", exc, exc_info=True)
        print(json.dumps({"status": "FAILED", "error": f"DiarizationExecutionError: {exc}"}))
        return 5

    t_infer = time.time() - t_infer_start
    logger.info("Inference completed in %.2fs", t_infer)

    # 7. Locate Expected Generated RTTM for Current Audio
    pred_rttms_dir = args.out_dir / "pred_rttms"
    expected_rttm = pred_rttms_dir / f"{file_id}.rttm"
    if not expected_rttm.is_file():
        logger.error("Expected RTTM for '%s' was not produced at: %s", file_id, expected_rttm)
        print(json.dumps({"status": "FAILED", "error": f"Expected RTTM not found: {expected_rttm}"}))
        return 6

    # Promote strictly the verified output to canonical destination
    dest_rttm = args.out_dir / "predicted.rttm"
    shutil.copy2(expected_rttm, dest_rttm)
    logger.info("Promoted verified output '%s' to: %s", expected_rttm.name, dest_rttm)

    # 8. Clean intermediate files if requested
    if not args.keep_intermediate:
        emb_dir = args.out_dir / "speaker_outputs" / "embeddings"
        if emb_dir.exists():
            for pkl in emb_dir.glob("*.pkl"):
                try:
                    pkl.unlink()
                except Exception:
                    pass

    # 9. Return structured JSON payload on stdout
    stats = parse_rttm_stats(dest_rttm)
    final_status = "SUCCESS" if stats["turn_count"] > 0 else "NO_SPEECH"
    payload = {
        "status": final_status,
        "rttm_path": str(dest_rttm.resolve()),
        "speaker_count": stats["speaker_count"],
        "speakers": stats["speakers"],
        "turn_count": stats["turn_count"],
        "total_speech_seconds": stats["total_speech_seconds"],
        "model_loading_seconds": round(t_load, 3),
        "inference_time_seconds": round(t_infer, 3),
    }

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
