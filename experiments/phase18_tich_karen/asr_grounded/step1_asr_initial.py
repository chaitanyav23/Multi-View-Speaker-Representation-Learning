import json
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
    BASE_DIR = Path("experiments/phase18_tich_karen/asr_grounded")
    AUDIO_PATH = Path("experiments/phase18_tich_karen/audio/rich_karen_normalized.wav")
    OUTPUT_WORDS = BASE_DIR / "outputs" / "aligned_words_initial.json"
    GROUNDING_PATH = BASE_DIR / "grounding" / "asr_derived_grounding.json"
    
    print(f"Loading audio from {AUDIO_PATH}...")
    audio = whisperx.load_audio(str(AUDIO_PATH))

    # 1. Full Transcription
    model_size = "large-v3"
    print(f"Loading Whisper {model_size} for full transcription...")
    model = whisperx.load_model(model_size, whisperx_utils.DEVICE, compute_type="float16", language="en")
    
    print("Transcribing full audio...")
    result = model.transcribe(audio, batch_size=16)

    # 2. Alignment
    print("Loading Alignment Model...")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=whisperx_utils.DEVICE)

    print("Aligning full transcription...")
    alignment_result = whisperx.align(result["segments"], model_a, metadata, audio, whisperx_utils.DEVICE, return_char_alignments=False)

    # 3. Extract Words
    all_words = []
    for seg in alignment_result["word_segments"]:
        if seg.get("start") is not None:
            all_words.append({
                "word": seg.get("word"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "score": seg.get("score")
            })

    with open(OUTPUT_WORDS, "w") as f:
        json.dump(all_words, f, indent=2)

    # 4. Generate Grounding (Merge words into continuous speech blocks)
    # If gap between words < 1.0s, merge them
    grounding_segments = []
    if all_words:
        curr = {"start": all_words[0]["start"], "end": all_words[0]["end"]}
        for w in all_words[1:]:
            if w["start"] - curr["end"] < 1.0:
                curr["end"] = w["end"]
            else:
                grounding_segments.append(curr)
                curr = {"start": w["start"], "end": w["end"]}
        grounding_segments.append(curr)

    out_data = {
        "file": AUDIO_PATH.name,
        "segments": grounding_segments,
        "total_speech_duration": sum(s["end"] - s["start"] for s in grounding_segments)
    }
    
    with open(GROUNDING_PATH, "w") as f:
        json.dump(out_data, f, indent=2)
    
    print(f"ASR-derived grounding saved to {GROUNDING_PATH}")
    print(f"Identified {len(grounding_segments)} speech segments.")

if __name__ == "__main__":
    main()
