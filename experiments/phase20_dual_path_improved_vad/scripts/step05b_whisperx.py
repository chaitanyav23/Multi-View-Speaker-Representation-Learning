import argparse
from pathlib import Path

import torch
import whisperx
from tqdm import tqdm

from phase19_common import BASE_DIR, load_external_whisperx_utils, read_json, setup_logging, stage_metadata, write_json


def main():
    parser = argparse.ArgumentParser(description="Phase 19 Step 5B: WhisperX large-v3 transcription and alignment.")
    parser.add_argument("--audio", type=Path, default=BASE_DIR / "audio" / "bwc_normalized.wav")
    parser.add_argument("--vad", type=Path, default=BASE_DIR / "grounding" / "pathB_vad.json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "transcription" / "aligned_words.json")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("step05b_whisperx")
    if args.output.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", args.output)
        return
    util = load_external_whisperx_utils()
    device = getattr(util, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"
    audio = whisperx.load_audio(str(args.audio))
    segments = read_json(args.vad)["segments"]
    logger.info("Loading WhisperX %s on %s", args.model, device)
    model = whisperx.load_model(args.model, device, compute_type=compute_type, language="en")
    transcription = []
    sample_rate = 16000
    for seg in tqdm(segments, desc="Transcribing"):
        start = int(seg["start"] * sample_rate)
        end = int(seg["end"] * sample_rate)
        generated, _ = model.model.transcribe(audio[start:end], language="en", task="transcribe", word_timestamps=False, condition_on_previous_text=False)
        text = "".join(s.text for s in generated).strip()
        if text:
            transcription.append({"start": seg["start"], "end": seg["end"], "text": text})
    logger.info("Aligning %d transcription segments", len(transcription))
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned = whisperx.align(transcription, align_model, metadata, audio, device, return_char_alignments=False)
    words = [{"word": w.get("word"), "start": w.get("start"), "end": w.get("end"), "confidence": w.get("score")} for w in aligned.get("word_segments", [])]
    write_json(args.output, words)
    write_json(BASE_DIR / "outputs" / "step05b_whisperx.json", stage_metadata("whisperx", [args.audio, args.vad], [args.output], {"words": len(words), "model": args.model}))
    logger.info("Saved %d aligned words to %s", len(words), args.output)


if __name__ == "__main__":
    main()
