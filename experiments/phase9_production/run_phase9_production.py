"""
experiments/phase9_production/run_phase9_production.py
======================================================
Phase 9: PRODUCTION CANDIDATE PIPELINE

Objective: Assemble the best-performing components discovered across Phases 1-7
into a single optimized pipeline and perform a complete end-to-end evaluation.

This phase focuses exclusively on bwc.wav.

Pipeline architecture:
- Stage 0: Input raw_audio/bwc.wav
- Stage 1: Normalization (LUFS -20, 16kHz, mono) -> audio_normalized/bwc_normalized.wav
- Stage 2: Master Grounding (Pyannote 3.1 VAD with high-recall config) -> grounding_master/bwc_grounding.json
- Stage 3: Grounded Audio (Muted non-speech regions) -> audio_grounded/bwc_grounded.wav
- Stage 4: Hardened ASR (WhisperX Large-v3, manual VAD-grounded slicing, condition_on_previous_text=False) -> outputs_asr/bwc_asr.json
- Stage 5: Alignment (WhisperX alignment on normalized audio) -> outputs_alignment/bwc_aligned.json
- Stage 6: Oracle VAD MSDD (Using normalized audio + Oracle RTTM from Master Grounding) -> outputs_msdd/bwc_diarization.json
- Stage 7: Speaker Attribution (Assign speakers to aligned words) -> outputs_final/bwc_final.json & bwc_final.txt
- Stage 8: Diagnostics Generation.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import whisperx
import yaml
from dotenv import dotenv_values
from nemo.collections.asr.models import NeuralDiarizer
from omegaconf import OmegaConf
from pyannote.audio import Pipeline
from pyannote.audio.utils.hf_hub import AssetFileName, download_from_hf_hub
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_DIR = PROJECT_ROOT / "experiments" / "phase9_production"
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
EXTERNAL_WX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"

AUDIO_NORM_DIR = PHASE_DIR / "audio_normalized"
GROUNDING_DIR = PHASE_DIR / "grounding_master"
AUDIO_GROUNDED_DIR = PHASE_DIR / "audio_grounded"
OUTPUT_ASR_DIR = PHASE_DIR / "outputs_asr"
OUTPUT_ALIGN_DIR = PHASE_DIR / "outputs_alignment"
OUTPUT_MSDD_DIR = PHASE_DIR / "outputs_msdd"
OUTPUT_FINAL_DIR = PHASE_DIR / "outputs_final"
DIAGNOSTICS_DIR = PHASE_DIR / "diagnostics"
REPORTS_DIR = PHASE_DIR / "reports"

ENV_PATH = EXTERNAL_WX_DIR / ".env"
ENV_VALUES = dotenv_values(ENV_PATH)
HF_TOKEN = ENV_VALUES.get("HF_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_command(cmd, check=True):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command Failed: {' '.join(cmd)}")
        print(f"Error: {result.stderr}")
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result

def run_stage1_normalization(input_path: Path, output_path: Path, overwrite: bool) -> Path:
    print("\n--- STAGE 1: Normalization ---")
    if output_path.exists() and not overwrite:
        print(f"Skipping normalization, found {output_path}")
        return output_path
    print(f"Normalizing: {input_path.name}")
    run_command([
        "ffmpeg", "-i", str(input_path),
        "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
        "-ar", "16000", "-ac", "1",
        str(output_path), "-y"
    ])
    return output_path

def load_pyannote_vad_pipeline(hf_token: str):
    model_id = "pyannote/voice-activity-detection"
    config_path = download_from_hf_hub(model_id, AssetFileName.Pipeline, token=hf_token)
    if config_path is None:
        raise ValueError(f"Could not download pipeline config for {model_id}.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    segmentation = config.get("pipeline", {}).get("params", {}).get("segmentation")
    if isinstance(segmentation, str) and "@" in segmentation:
        checkpoint, revision = segmentation.split("@", 1)
        config["pipeline"]["params"]["segmentation"] = {
            "checkpoint": checkpoint,
            "revision": revision,
        }
    return Pipeline.from_pretrained(config, token=hf_token)

def run_stage2_master_grounding(audio_path: Path, output_json: Path, vad_pipeline, overwrite: bool) -> list[dict]:
    print("\n--- STAGE 2: Master Grounding ---")
    if output_json.exists() and not overwrite:
        with open(output_json, "r") as f:
            return json.load(f)["segments"]

    print(f"Running Master VAD Grounding: {audio_path.name}")
    vad = vad_pipeline(str(audio_path))
    info = sf.info(str(audio_path))
    duration = info.duration

    # High-recall configuration
    min_duration = 0.05
    merge_gap = 0.0
    padding = 0.1

    raw_segments = []
    for s in vad.get_timeline().support():
        if (s.end - s.start) < min_duration:
            continue
        raw_segments.append({"start": float(s.start), "end": float(s.end)})

    cleaned = []
    for seg in raw_segments:
        start = max(0.0, seg["start"] - padding)
        end = min(duration, seg["end"] + padding)
        if end <= start:
            continue
        if not cleaned:
            cleaned.append({"start": start, "end": end})
            continue
        if start - cleaned[-1]["end"] <= merge_gap:
            cleaned[-1]["end"] = max(cleaned[-1]["end"], end)
        else:
            cleaned.append({"start": start, "end": end})

    with open(output_json, "w") as f:
        json.dump({"file": audio_path.name, "segments": cleaned}, f, indent=2)
    return cleaned

def run_stage3_grounded_audio(audio_path: Path, segments: list[dict], output_wav: Path, overwrite: bool) -> Path:
    print("\n--- STAGE 3: Grounded Audio (Muting) ---")
    if output_wav.exists() and not overwrite:
        print(f"Skipping grounded audio, found {output_wav}")
        return output_wav

    audio, sr = sf.read(str(audio_path))
    mask = np.zeros_like(audio)
    for seg in segments:
        start_idx = int(seg["start"] * sr)
        end_idx = int(seg["end"] * sr)
        mask[start_idx:end_idx] = 1.0

    muted_audio = audio * mask
    sf.write(str(output_wav), muted_audio, sr)
    print(f"Saved muted audio to {output_wav}")
    return output_wav

def run_stage4_hardened_asr(audio_path: Path, segments: list[dict], output_json: Path, asr_model, overwrite: bool) -> list[dict]:
    print("\n--- STAGE 4: Hardened ASR (Manual VAD Slicing, No Conditioning) ---")
    if output_json.exists() and not overwrite:
        with open(output_json, "r") as f:
            return json.load(f)["segments"]

    audio = whisperx.load_audio(str(audio_path))
    sr = 16000

    transcript_segments = []
    for seg in tqdm(segments, desc="ASR Slicing"):
        start_idx = int(seg["start"] * sr)
        end_idx = int(seg["end"] * sr)
        audio_slice = audio[start_idx:end_idx]
        if len(audio_slice) == 0:
            continue
        
        # Hardened settings: language="en", condition_on_previous_text=False
        res, _ = asr_model.model.transcribe(
            audio_slice,
            language="en",
            task="transcribe",
            condition_on_previous_text=False
        )
        text = "".join([s.text for s in res]).strip()
        if text:
            transcript_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
            })

    with open(output_json, "w") as f:
        json.dump({"segments": transcript_segments}, f, indent=2)
    return transcript_segments

def run_stage5_alignment(audio_path: Path, asr_segments: list[dict], output_json: Path, align_model, align_metadata, overwrite: bool) -> dict:
    print("\n--- STAGE 5: Alignment ---")
    if output_json.exists() and not overwrite:
        with open(output_json, "r") as f:
            return json.load(f)

    audio = whisperx.load_audio(str(audio_path))
    aligned_res = whisperx.align(asr_segments, align_model, align_metadata, audio, DEVICE)
    
    with open(output_json, "w") as f:
        # Save pre-speaker-attribution alignment
        json.dump(aligned_res, f, indent=2, default=str)
    return aligned_res

def write_oracle_rttm(stem: str, vad_segments: list[dict], output_path: Path) -> None:
    with open(output_path, "w") as f:
        for seg in vad_segments:
            duration = seg["end"] - seg["start"]
            if duration <= 0:
                continue
            f.write(f"SPEAKER {stem} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> speech <NA>\n")

def run_stage6_oracle_msdd(audio_path: Path, vad_segments: list[dict], output_dir: Path, diar_json_path: Path, overwrite: bool) -> Path:
    print("\n--- STAGE 6: Oracle VAD MSDD ---")
    if diar_json_path.exists() and not overwrite:
        print(f"Skipping MSDD, found {diar_json_path}")
        return diar_json_path

    oracle_rttm_path = output_dir / f"{audio_path.stem}_oracle.rttm"
    write_oracle_rttm(audio_path.stem, vad_segments, oracle_rttm_path)

    manifest_path = output_dir / f"{audio_path.stem}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "audio_filepath": str(audio_path.absolute()),
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": 2,
            "rttm_filepath": str(oracle_rttm_path.absolute()),
        }, f)

    config = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": str(manifest_path),
            "out_dir": str(output_dir / audio_path.stem),
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
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "use_backim_for_multiscale": True,
                    "sigmoid_threshold": [0.7],
                    "use_adaptive_thres": False,
                    "overlap_infer_spk_limit": 3,
                    "diar_window_length": 50,
                    "infer_batch_size": 16,
                    "seq_eval_mode": False,
                    "split_infer": True,
                    "split_batch_size": 1,
                },
            },
            "clustering": {
                "parameters": {
                    "max_num_speakers": 2,
                    "oracle_num_speakers": True,
                    "max_rp_threshold": 0.25,
                    "sparse_search_volume": 30,
                },
            },
        },
        "batch_size": 16,
        "num_workers": 4,
        "sample_rate": 16000,
        "verbose": True,
        "device": DEVICE,
    })

    print(f"Running MSDD: {audio_path.name}")
    msdd_model = NeuralDiarizer(cfg=config).to(torch.device(DEVICE))
    msdd_model.diarize()

    rttm_path = output_dir / audio_path.stem / "pred_rttms" / f"{audio_path.stem}.rttm"
    diar_list = []
    if rttm_path.exists():
        with open(rttm_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                start = float(parts[3])
                end = start + float(parts[4])
                speaker = parts[7]
                diar_list.append({"start": start, "end": end, "speaker": speaker})

    with open(diar_json_path, "w") as f:
        json.dump(diar_list, f, indent=2)
    return diar_json_path

def run_stage7_speaker_attribution(aligned_res: dict, diar_json_path: Path, out_json: Path, out_txt: Path, overwrite: bool) -> Path:
    print("\n--- STAGE 7: Speaker Attribution ---")
    if out_json.exists() and not overwrite:
        print(f"Skipping speaker attribution, found {out_json}")
        return out_json

    with open(diar_json_path, "r") as f:
        diar_list = json.load(f)
    diar_df = pd.DataFrame(diar_list)

    final_result = whisperx.assign_word_speakers(diar_df, aligned_res)

    with open(out_json, "w") as f:
        json.dump(final_result, f, indent=2, default=str)
    with open(out_txt, "w") as f:
        for s in final_result["segments"]:
            spk = s.get("speaker", "UNKNOWN")
            f.write(f"[{s['start']:06.2f}s] {spk:10}: {s.get('text', '')}\n")
    return out_json

def generate_diagnostics(gt_path: Path, pred_json: Path, output_dir: Path):
    print("\n--- STAGE 8: Diagnostics ---")
    pass # Real diagnostics will be created via a separate script or eval step if needed.
    # Leaving stub here. Evaluation script will handle full metrics.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN required.")

    AUDIO_NORM_DIR.mkdir(parents=True, exist_ok=True)
    GROUNDING_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_GROUNDED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ASR_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MSDD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Models...")
    vad_pipeline = load_pyannote_vad_pipeline(HF_TOKEN).to(torch.device(DEVICE))
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

    raw_audio = RAW_AUDIO_DIR / "bwc.wav"
    if not raw_audio.exists():
        raise FileNotFoundError(f"Missing {raw_audio}")
        
    print(f"\n[PHASE 9] Starting production pipeline for {raw_audio.name}")
    
    norm_audio = AUDIO_NORM_DIR / f"{raw_audio.stem}_normalized.wav"
    run_stage1_normalization(raw_audio, norm_audio, args.overwrite)
    
    grounding_json = GROUNDING_DIR / f"{raw_audio.stem}_grounding.json"
    vad_segments = run_stage2_master_grounding(norm_audio, grounding_json, vad_pipeline, args.overwrite)
    
    grounded_audio = AUDIO_GROUNDED_DIR / f"{raw_audio.stem}_grounded.wav"
    run_stage3_grounded_audio(norm_audio, vad_segments, grounded_audio, args.overwrite)
    
    asr_json = OUTPUT_ASR_DIR / f"{raw_audio.stem}_asr.json"
    asr_segments = run_stage4_hardened_asr(norm_audio, vad_segments, asr_json, asr_model, args.overwrite)
    
    aligned_json = OUTPUT_ALIGN_DIR / f"{raw_audio.stem}_aligned.json"
    aligned_res = run_stage5_alignment(norm_audio, asr_segments, aligned_json, align_model, align_metadata, args.overwrite)
    
    diar_json = OUTPUT_MSDD_DIR / f"{raw_audio.stem}_diarization.json"
    run_stage6_oracle_msdd(norm_audio, vad_segments, OUTPUT_MSDD_DIR, diar_json, args.overwrite)
    
    final_json = OUTPUT_FINAL_DIR / f"{raw_audio.stem}_final.json"
    final_txt = OUTPUT_FINAL_DIR / f"{raw_audio.stem}_final.txt"
    run_stage7_speaker_attribution(aligned_res, diar_json, final_json, final_txt, args.overwrite)
    
    print("\nPhase 9 Production Candidate Pipeline Complete.")

if __name__ == "__main__":
    main()
