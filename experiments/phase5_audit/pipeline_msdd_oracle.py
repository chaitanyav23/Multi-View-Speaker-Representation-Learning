import json
import torch
import numpy as np
import argparse
import os
import time
from pathlib import Path
from nemo.collections.asr.models import NeuralDiarizer
from omegaconf import OmegaConf
import sys

def main():
    parser = argparse.ArgumentParser(description="Phase 5: NeMo MSDD Diarization with Oracle VAD")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--oracle-rttm-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracle-vad", action="store_true", default=True)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    oracle_rttm_dir = Path(args.oracle_rttm_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    timing_logs = []

    for audio_path in audio_dir.glob("*.wav"):
        print(f"\n>>> Processing: {audio_path.name}")
        
        stem = audio_path.stem
        oracle_rttm_path = oracle_rttm_dir / f"{stem}.rttm"
        
        if not oracle_rttm_path.exists():
            print(f"Skipping {stem}: Oracle RTTM not found at {oracle_rttm_path}")
            continue

        manifest_path = output_dir / f"{stem}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                "audio_filepath": str(audio_path.absolute()),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None,
                "rttm_filepath": str(oracle_rttm_path.absolute())
            }, f)

        config = OmegaConf.create({
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(output_dir / stem),
                "oracle_vad": args.oracle_vad,
                "collar": 0.25,
                "ignore_overlap": False,
                "vad": {
                    "model_path": None,
                    "external_vad_manifest": None,
                    "parameters": {}
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.2, 1.0, 0.8, 0.5],
                        "shift_length_in_sec": [0.75, 0.6, 0.5, 0.4, 0.25],
                        "multiscale_weights": [1, 1, 1, 1, 1],
                        "save_embeddings": True
                    }
                },
                "msdd_model": {
                    "model_path": "diar_msdd_telephonic",
                    "parameters": {
                        "use_backim_for_multiscale": True,
                        "sigmoid_threshold": [0.7],
                        "diar_window_length": 50,
                        "infer_batch_size": 16,
                        "seq_eval_mode": False,
                        "split_infer": True,
                        "split_batch_size": 1
                    }
                },
                "clustering": {
                    "parameters": {
                        "max_num_speakers": 20,
                        "oracle_num_speakers": False,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30
                    }
                }
            },
            "batch_size": 16,
            "num_workers": 4,
            "sample_rate": 16000,
            "verbose": True,
            "device": device
        })
        
        start_time = time.time()
        print(f"Starting MSDD Diarization (Oracle VAD={args.oracle_vad})...")
        msdd_model = NeuralDiarizer(cfg=config).to(torch.device(device))
        msdd_model.diarize()
        end_time = time.time()
        
        runtime = end_time - start_time
        print(f"MSDD Runtime: {runtime:.2f} seconds")
        
        rttm_path = output_dir / stem / "pred_rttms" / f"{stem}.rttm"
        
        diar_list = []
        if rttm_path.exists():
            with open(rttm_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    start = float(parts[3])
                    end = start + float(parts[4])
                    speaker = parts[7]
                    diar_list.append({"start": start, "end": end, "speaker": speaker})
        
        json_out_path = output_dir / f"{stem}_diarization.json"
        with open(json_out_path, 'w') as f:
            json.dump(diar_list, f)
            
        timing_logs.append({
            "file": stem,
            "runtime_sec": round(runtime, 2),
            "oracle_vad": args.oracle_vad,
            "rttm_generated": rttm_path.exists()
        })

    with open(output_dir / "timing_report.json", "w") as f:
        json.dump(timing_logs, f, indent=2)

if __name__ == "__main__":
    main()
