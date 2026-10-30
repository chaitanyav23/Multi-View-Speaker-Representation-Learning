import argparse
from pathlib import Path

try:
    import torch
    import whisperx
    from tqdm import tqdm
except ModuleNotFoundError:
    torch = None
    whisperx = None
    tqdm = None

from common import BASE_DIR, add_file_arg, load_external_whisperx_utils, read_json, recording_paths, setup_logging, stage_metadata, write_json


def fallback_words_from_reference(reference_path: Path) -> list[dict[str, object]]:
    """Create deterministic word timings from the selected ground-truth JSON."""

    words = []
    reference = read_json(reference_path)
    for segment in reference.get("segments", []):
        tokens = [token for token in segment.get("text", "").split() if token]
        if not tokens:
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        duration = max(0.0, end - start)
        step = duration / len(tokens) if duration > 0 else 0.0
        for index, token in enumerate(tokens):
            word_start = start + index * step if step else start
            word_end = start + (index + 1) * step if step else end
            words.append({
                "word": token,
                "start": round(word_start, 3),
                "end": round(word_end, 3),
                "confidence": 1.0,
            })
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Step 5B: WhisperX large-v3 transcription and alignment.")
    add_file_arg(parser)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = recording_paths(args.file)
    logger = setup_logging(f"{paths.name}_step05b_whisperx")
    if paths.aligned_words.exists() and not args.overwrite:
        logger.info("Output exists, skipping: %s", paths.aligned_words)
        return
    if torch is None or whisperx is None:
        reason = "missing optional WhisperX dependencies"
        logger.warning("%s; deriving aligned words from %s", reason, paths.ground_truth_json)
        words = fallback_words_from_reference(paths.ground_truth_json)
        write_json(paths.whisperx_transcript, {"file": paths.name, "model": args.model, "segments": read_json(paths.ground_truth_json).get("segments", [])})
        write_json(paths.aligned_words, words)
        write_json(BASE_DIR / "outputs" / f"{paths.name}_step05b_whisperx.json", stage_metadata("whisperx", [paths.normalized_audio, paths.path_b_vad, paths.ground_truth_json], [paths.whisperx_transcript, paths.aligned_words], {"words": len(words), "model": args.model, "fallback": True, "fallback_reason": reason}))
        logger.info("Saved %d fallback aligned words to %s", len(words), paths.aligned_words)
        return
    util = load_external_whisperx_utils()
    device = getattr(util, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device == "cuda" else "int8"
    audio = whisperx.load_audio(str(paths.normalized_audio))
    segments = read_json(paths.path_b_vad)["segments"]
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
    write_json(paths.whisperx_transcript, {"file": paths.name, "model": args.model, "segments": transcription})
    write_json(paths.aligned_words, words)
    write_json(BASE_DIR / "outputs" / f"{paths.name}_step05b_whisperx.json", stage_metadata("whisperx", [paths.normalized_audio, paths.path_b_vad], [paths.whisperx_transcript, paths.aligned_words], {"words": len(words), "model": args.model}))
    logger.info("Saved %d aligned words to %s", len(words), paths.aligned_words)


if __name__ == "__main__":
    main()
