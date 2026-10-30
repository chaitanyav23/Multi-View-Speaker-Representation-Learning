"""
experiments/phase8_sentence_grouped/run_phase8_main.py
=====================================================
Phase 8: Sentence-Level Grouped Pipeline.
Focus: Contextual recovery by merging high-sensitivity segments (1.0s gap).
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

PROJECT_ROOT = Path.cwd()
PHASE_DIR = PROJECT_ROOT / "experiments" / "phase8_sentence_grouped"
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
EXTERNAL_WX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"

AUDIO_DIR = PHASE_DIR / "audio"
GROUNDING_DIR = PHASE_DIR / "grounding"
DIAR_DIR = PHASE_DIR / "diarization"
OUTPUT_DIR = PHASE_DIR / "outputs"
ASR_DIR = OUTPUT_DIR / "asr"

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

def normalize_audio(input_path: Path, output_path: Path) -> None:
    if output_path.exists(): return
    print(f"Normalizing: {input_path.name}")
    run_command([
        "ffmpeg", "-i", str(input_path),
        "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
        "-ar", "16000", "-ac", "1",
        str(output_path), "-y"
    ])

def enhance_deepfilternet(input_path: Path, output_path: Path, device: str | None = None) -> None:
    if output_path.exists(): return
    print(f"Enhancing (deepfilternet): {input_path.name}")
    tmp_dir = output_path.parent / "_tmp_deepfilternet"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", "experiments/phase2_enhancement/pipeline_enhancement_deepfilternet.py",
        "--file", str(input_path),
        "--output-dir", str(tmp_dir),
    ]
    if device: cmd.extend(["--device", device])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 and device != "cpu":
        print("DeepFilterNet GPU failed, retrying on CPU...")
        run_command([
            "python3", "experiments/phase2_enhancement/pipeline_enhancement_deepfilternet.py",
            "--file", str(input_path),
            "--output-dir", str(tmp_dir),
            "--device", "cpu",
        ])
    elif res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd, res.stdout, res.stderr)

    candidates = [tmp_dir / input_path.name, tmp_dir / f"{input_path.stem}_DeepFilterNet3.wav"]
    for candidate in candidates:
        if candidate.exists():
            os.replace(candidate, output_path)
            return
    raise FileNotFoundError(f"Enhanced audio not found in {tmp_dir}")

def prepare_conditioned_audio(raw_path: Path, overwrite: bool) -> Path:
    conditioned_path = AUDIO_DIR / f"{raw_path.stem}.wav"
    if conditioned_path.exists() and not overwrite: return conditioned_path

    norm_path = AUDIO_DIR / f"{raw_path.stem}_norm.wav"
    normalize_audio(raw_path, norm_path)
    
    if "podcast" in raw_path.stem.lower():
        print(f"Skipping enhancement for {raw_path.name}")
        if conditioned_path.exists(): os.remove(conditioned_path)
        os.symlink(norm_path, conditioned_path)
        return conditioned_path

    enhance_deepfilternet(norm_path, conditioned_path)
    return conditioned_path

def load_pyannote_vad_pipeline(hf_token: str):
    model_id = "pyannote/voice-activity-detection"
    config_path = download_from_hf_hub(model_id, AssetFileName.Pipeline, token=hf_token)
    with open(config_path, "r") as f: config = yaml.safe_load(f)
    segmentation = config.get("pipeline", {}).get("params", {}).get("segmentation")
    if isinstance(segmentation, str) and "@" in segmentation:
        checkpoint, revision = segmentation.split("@", 1)
        config["pipeline"]["params"]["segmentation"] = {"checkpoint": checkpoint, "revision": revision}
    return Pipeline.from_pretrained(config, token=hf_token)

def run_soft_vad(audio_path: Path, output_json: Path, vad_pipeline, min_duration: float,
                 merge_gap: float, padding: float, overwrite: bool) -> list[dict]:
    if output_json.exists() and not overwrite:
        with open(output_json, "r") as f: return json.load(f)["segments"]

    print(f"Running Soft VAD (Gap Merge: {merge_gap}s): {audio_path.name}")
    vad = vad_pipeline(str(audio_path))
    info = sf.info(str(audio_path))
    duration = info.duration

    raw_segments = []
    for s in vad.get_timeline().support():
        if (s.end - s.start) < min_duration: continue
        raw_segments.append({"start": float(s.start), "end": float(s.end)})

    cleaned = []
    for seg in raw_segments:
        start = max(0.0, seg["start"] - padding)
        end = min(duration, seg["end"] + padding)
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

def write_oracle_rttm(stem: str, vad_segments: list[dict], output_path: Path) -> None:
    with open(output_path, "w") as f:
        for seg in vad_segments:
            f.write(f"SPEAKER {stem} 1 {seg['start']:.3f} {seg['end']-seg['start']:.3f} <NA> <NA> speech <NA>\n")

def run_msdd_fixed2(audio_path: Path, oracle_rttm_path: Path, output_dir: Path, overwrite: bool) -> Path:
    diar_json_path = DIAR_DIR / f"{audio_path.stem}_diarization.json"
    if diar_json_path.exists() and not overwrite: return diar_json_path

    manifest_path = output_dir / f"{audio_path.stem}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "audio_filepath": str(audio_path.absolute()), "offset": 0, "duration": None,
            "label": "infer", "text": "-", "num_speakers": 2,
            "rttm_filepath": str(oracle_rttm_path.absolute()),
        }, f)

    config = OmegaConf.create({
        "diarizer": {
            "manifest_filepath": str(manifest_path),
            "out_dir": str(output_dir / audio_path.stem),
            "oracle_vad": True, "collar": 0.25, "ignore_overlap": False,
            "vad": {"model_path": None, "external_vad_manifest": None, "parameters": {}},
            "speaker_embeddings": {
                "model_path": "titanet_large",
                "parameters": {
                    "window_length_in_sec": [1.5, 1.2, 1.0, 0.8, 0.5],
                    "shift_length_in_sec": [0.75, 0.6, 0.5, 0.4, 0.25],
                    "multiscale_weights": [1, 2, 3, 3, 3], "save_embeddings": True,
                },
            },
            "msdd_model": {
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "use_backim_for_multiscale": True, "sigmoid_threshold": [0.7],
                    "use_adaptive_thres": False, "overlap_infer_spk_limit": 3,
                    "diar_window_length": 50, "infer_batch_size": 16,
                    "seq_eval_mode": False, "split_infer": True, "split_batch_size": 1,
                },
            },
            "clustering": {
                "parameters": {
                    "max_num_speakers": 2, "oracle_num_speakers": True,
                    "max_rp_threshold": 0.25, "sparse_search_volume": 30,
                },
            },
        },
        "batch_size": 16, "num_workers": 4, "sample_rate": 16000, "verbose": True, "device": DEVICE,
    })

    print(f"Running MSDD (Fixed-2): {audio_path.name}")
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

def run_contextual_asr(audio_path: Path, vad_segments: list[dict], buffer_s: float,
                       asr_model, overwrite: bool) -> list[dict]:
    asr_json_path = ASR_DIR / f"{audio_path.stem}_asr.json"
    if asr_json_path.exists() and not overwrite:
        with open(asr_json_path, "r") as f: return json.load(f)["segments"]

    audio = whisperx.load_audio(str(audio_path))
    sr = 16000
    duration = len(audio) / sr

    print(f"Running ASR: {audio_path.name}")

    transcript_segments = []
    prev_text = ""
    for seg in tqdm(vad_segments, desc="ASR"):
        start = max(0.0, seg["start"] - buffer_s)
        end = min(duration, seg["end"] + buffer_s)
        audio_slice = audio[int(start * sr):int(end * sr)]
        
        current_prompt = prev_text if 0 < len(prev_text.split()) < 50 else ""
        
        res, _ = asr_model.model.transcribe(
            audio_slice, language="en", task="transcribe",
            condition_on_previous_text=True, initial_prompt=current_prompt
        )
        text = "".join([s.text for s in res]).strip()
        
        # Log repetition but don't delete text
        if text.lower() == prev_text.lower() and len(text) > 5:
            print(f"Warning: Potential repetition detected at {seg['start']:.2f}s")
            
        transcript_segments.append({"start": seg["start"], "end": seg["end"], "text": text})
        prev_text = text

    with open(asr_json_path, "w") as f:
        json.dump({"segments": transcript_segments}, f, indent=2)
    return transcript_segments

def align_and_merge(audio_path: Path, diar_json: Path, transcript_segments: list[dict],
                    align_model, align_metadata, overwrite: bool) -> Path:
    out_json = OUTPUT_DIR / f"{audio_path.stem}_final.json"
    out_txt = OUTPUT_DIR / f"{audio_path.stem}_final.txt"
    if out_json.exists() and not overwrite: return out_json

    audio = whisperx.load_audio(str(audio_path))
    with open(diar_json, "r") as f: diar_list = json.load(f)
    diar_df = pd.DataFrame(diar_list)

    aligned_res = whisperx.align(transcript_segments, align_model, align_metadata, audio, DEVICE)
    final_result = whisperx.assign_word_speakers(diar_df, aligned_res)

    with open(out_json, "w") as f: json.dump(final_result, f, indent=2, default=str)
    with open(out_txt, "w") as f:
        for s in final_result["segments"]:
            f.write(f"[{s['start']:06.2f}s] {s.get('speaker', 'UNK'):10}: {s['text']}\n")
    return out_json

def main():
    parser = argparse.ArgumentParser(description="Phase 8: Sentence-Level Grouped Pipeline")
    parser.add_argument("--audio-dir", default=str(RAW_AUDIO_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vad-min-duration", type=float, default=0.05)
    parser.add_argument("--vad-merge-gap", type=float, default=1.0) # Group into sentences
    parser.add_argument("--vad-padding", type=float, default=0.1)
    parser.add_argument("--asr-buffer", type=float, default=0.5)
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Error: HF_TOKEN required.")
        return

    print("Loading models...")
    vad_pipeline = load_pyannote_vad_pipeline(HF_TOKEN).to(torch.device(DEVICE))
    asr_model = whisperx.load_model("large-v3", DEVICE, compute_type="float16")
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    GROUNDING_DIR.mkdir(parents=True, exist_ok=True)
    DIAR_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASR_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(list(Path(args.audio_dir).glob("*.wav")))

    for raw_path in audio_files:
        print(f"\n>>> Processing: {raw_path.name}")
        if "podcast" in raw_path.stem.lower():
            print(f"Skipping podcast input: {raw_path.name}")
            continue

        conditioned_path = prepare_conditioned_audio(raw_path, overwrite=args.overwrite)

        vad_json = GROUNDING_DIR / f"{conditioned_path.stem}_soft_vad.json"
        vad_segments = run_soft_vad(
            conditioned_path, vad_json, vad_pipeline,
            min_duration=args.vad_min_duration, merge_gap=args.vad_merge_gap,
            padding=args.vad_padding, overwrite=args.overwrite,
        )

        oracle_rttm = DIAR_DIR / f"{conditioned_path.stem}_soft_vad.rttm"
        write_oracle_rttm(conditioned_path.stem, vad_segments, oracle_rttm)

        diar_json = run_msdd_fixed2(conditioned_path, oracle_rttm, DIAR_DIR, overwrite=args.overwrite)

        transcript_segments = run_contextual_asr(
            conditioned_path, vad_segments, buffer_s=args.asr_buffer,
            asr_model=asr_model, overwrite=args.overwrite,
        )

        align_and_merge(conditioned_path, diar_json, transcript_segments, align_model, align_metadata, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
