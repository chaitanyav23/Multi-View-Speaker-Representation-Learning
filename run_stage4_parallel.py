"""
run_stage4_parallel.py
======================
Executes ASR (Transcription + Alignment) and Diarization in parallel.
Uses segments from vad_outputs_whisperx/ as the grounding for both tasks.

Outputs:
- asr_outputs_wx/*.json (Aligned word-level transcripts)
- diarization_outputs_wx/*.json (Speaker segments)
"""

import json
import time
import sys
import importlib.util
import argparse
import torch
import numpy as np
import whisperx
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

PROJECT_ROOT = Path.cwd()
EXTERNAL_WHISPERX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"

from scripts.utils.logger import get_logger

# Paths
VAD_INPUT_DIR = PROJECT_ROOT / "vad_outputs_whisperx"
RAW_AUDIO_DIR = PROJECT_ROOT / "raw_audio"
ASR_OUTPUT_DIR = PROJECT_ROOT / "asr_outputs_wx"
DIAR_OUTPUT_DIR = PROJECT_ROOT / "diarization_outputs_wx"
LOGS_DIR = PROJECT_ROOT / "logs" / "wx_parallel_pipeline"

# Ensure dirs
ASR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

log = get_logger(name="parallel_pipeline", level="INFO", log_dir=LOGS_DIR, log_filename="pipeline.log")

def _load_module(module_name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing WhisperX module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load the external utils
whisperx_utils = _load_module("whisperx_utils", EXTERNAL_WHISPERX_DIR / "utils.py")

def _json_default(obj):
    """Handle non-serializable objects like Segment, Tensor, or Numpy types."""
    # Handle Pyannote Segment
    if hasattr(obj, "start") and hasattr(obj, "end"):
        return {"start": obj.start, "end": obj.end}
    
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
        
    return str(obj)

def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)

def run_asr_task(audio, asr_model, audio_name, vad_segments, language=None):
    """Task: Transcribe + Align."""
    t0 = time.perf_counter()
    try:
        # 1. Transcribe
        # Use our VAD segments to guide the transcription
        # WhisperX transcribe takes 'chunk_size' but internally uses VAD.
        # We pass our segments if the API supports it, otherwise we align to them.
        transcribe_args = {"batch_size": 16}
        if language:
            transcribe_args["language"] = language
            
        result = asr_model.transcribe(audio, **transcribe_args) 
        
        # 2. Align
        lang_code = result["language"]
        model_a, metadata = whisperx.load_align_model(
            language_code=lang_code, 
            device=whisperx_utils.DEVICE
        )
        aligned_result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            whisperx_utils.DEVICE, 
            return_char_alignments=False
        )
        
        output_path = ASR_OUTPUT_DIR / f"{audio_name}_asr.json"
        _write_json(output_path, aligned_result)
            
        log.info(f"ASR Complete: {audio_name} ({time.perf_counter()-t0:.2f}s)")
        return True
    except Exception as e:
        log.error(f"ASR Failed for {audio_name}: {e}")
        return False

def run_diarization_task(audio, diarize_model, audio_name, vad_segments, min_speakers=None, max_speakers=None):
    """Task: Diarization."""
    t0 = time.perf_counter()
    try:
        # Run diarization
        diarize_segments = diarize_model(
            audio, 
            min_speakers=min_speakers, 
            max_speakers=max_speakers
        )
        
        # Note: If we want to strictly use our VAD segments, we would need 
        # to filter or force the diarizer. For now, we run the diarizer 
        # normally as it's an end-to-end pyannote pipeline.
        
        if hasattr(diarize_segments, "to_dict"):
            segments_list = diarize_segments.to_dict(orient="records")
        else:
            segments_list = diarize_segments

        output_path = DIAR_OUTPUT_DIR / f"{audio_name}_diarization.json"
        _write_json(output_path, {"file": audio_name, "segments": segments_list})
            
        log.info(f"Diarization Complete: {audio_name} ({time.perf_counter()-t0:.2f}s)")
        return True
    except Exception as e:
        log.error(f"Diarization Failed for {audio_name}: {e}")
        return False

def process_file(vad_json_path, asr_model, diarize_model, args):
    audio_name = vad_json_path.stem.replace("_vad", "")
    audio_path = RAW_AUDIO_DIR / f"{audio_name}.wav"
    
    if not audio_path.exists():
        log.error(f"Audio file not found: {audio_path}")
        return

    asr_out = ASR_OUTPUT_DIR / f"{audio_name}_asr.json"
    diar_out = DIAR_OUTPUT_DIR / f"{audio_name}_diarization.json"

    if not args.overwrite and asr_out.exists() and diar_out.exists():
        log.info(f"Skipping {audio_name} (outputs exist).")
        return

    log.info(f"--- Processing {audio_name} ---")
    audio = whisperx.load_audio(str(audio_path))
    
    with open(vad_json_path, "r") as f:
        vad_data = json.load(f)
    vad_segments = vad_data["segments"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_asr = executor.submit(
            run_asr_task, 
            audio, 
            asr_model, 
            audio_name, 
            vad_segments,
            language=args.language
        )
        future_diar = executor.submit(
            run_diarization_task, 
            audio, 
            diarize_model, 
            audio_name, 
            vad_segments,
            min_speakers=args.min_speakers, 
            max_speakers=args.max_speakers
        )
        future_asr.result()
        future_diar.result()

def main():
    parser = argparse.ArgumentParser(description="Parallel ASR and Diarization Pipeline.")
    parser.add_argument("--language", default=None, help="Force language code (e.g., en, es).")
    parser.add_argument("--min-speakers", type=int, default=None, help="Min speakers.")
    parser.add_argument("--max-speakers", type=int, default=None, help="Max speakers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--model-size", default="large-v3", help="Whisper model size.")
    args = parser.parse_args()

    # Load models
    asr_model = whisperx_utils.load_models(model_size=args.model_size)
    diarize_model = whisperx_utils.load_diarization_model()

    vad_files = list(VAD_INPUT_DIR.glob("*.json"))
    for vad_file in tqdm(vad_files):
        process_file(vad_file, asr_model, diarize_model, args)

if __name__ == "__main__":
    main()
