import json
import argparse
import torch
import whisperx
from pathlib import Path
from tqdm import tqdm
import importlib.util
import os

PROJECT_ROOT = Path.cwd()
EXTERNAL_WHISPERX_DIR = PROJECT_ROOT / "external" / "WhisperX-Audio-Intelligence-Platform"

def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

whisperx_utils = _load_module("whisperx_utils", EXTERNAL_WHISPERX_DIR / "utils.py")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--grounding_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    grounding_path = Path(args.grounding_json)
    output_json = Path(args.output_json)
    language = args.language
    
    # 1. Load data
    print(f"Loading audio from {audio_path}...")
    audio = whisperx.load_audio(str(audio_path))
    with open(grounding_path, "r") as f:
        grounding = json.load(f)
    segments = grounding["segments"]

    # 2. Load Whisper Model
    model_size = "large-v3"
    print(f"Loading Whisper {model_size}...")
    model = whisperx.load_model(
        model_size, 
        whisperx_utils.DEVICE, 
        compute_type="float16",
        language=language
    )

    # 3. Transcribe
    print(f"Transcribing {len(segments)} segments...")
    transcription_results = []
    sample_rate = 16000
    
    for i, seg in enumerate(tqdm(segments, desc="Transcribing")):
        start_samp = int(seg["start"] * sample_rate)
        end_samp = int(seg["end"] * sample_rate)
        audio_slice = audio[start_samp:end_samp]
        
        # transcribe
        gen_result, _ = model.model.transcribe(
            audio_slice,
            language=language,
            task="transcribe",
            word_timestamps=False,
            condition_on_previous_text=False
        )
        
        full_text = ""
        for s in gen_result:
            full_text += s.text
            
        transcription_results.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": full_text.strip()
        })

    # 4. Align
    print("Loading Alignment Model...")
    model_a, metadata = whisperx.load_align_model(
        language_code=language, 
        device=whisperx_utils.DEVICE
    )

    print("Aligning...")
    alignment_result = whisperx.align(
        transcription_results, 
        model_a, 
        metadata, 
        audio, 
        whisperx_utils.DEVICE, 
        return_char_alignments=False
    )

    # 5. Extract and flatten words
    all_words = []
    for seg in alignment_result["word_segments"]:
        all_words.append({
            "word": seg.get("word"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "score": seg.get("score")
        })

    # Save
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(all_words, f, indent=2)
    
    print(f"Saved {len(all_words)} words to {output_json}")

if __name__ == "__main__":
    main()

