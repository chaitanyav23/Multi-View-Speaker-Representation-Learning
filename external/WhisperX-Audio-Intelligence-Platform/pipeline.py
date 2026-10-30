from utils import (
    load_models,
    format_transcript,
    DEVICE
)

from transcribe import transcribe_audio
from align import align_transcript
from diarize import (
    perform_diarization,
    assign_speakers
)
from utils import validate_audio_length
from utils import load_diarization_model

# Load globally ONCE
asr_model = load_models()
diarize_model = None

def run_pipeline(audio_path, enable_diarization=False):

    global diarize_model
    validate_audio_length(audio_path)

    print("Starting transcription...")

    # 1. Transcription
    audio, result = transcribe_audio(
        audio_path,
        asr_model
    )

    print("Aligning transcript...")

    # 2. Alignment
    aligned_result = align_transcript(
        audio,
        result,
        DEVICE
    )
    if enable_diarization:

        if diarize_model is None:

            print("Loading diarization model...")

            diarize_model = load_diarization_model()

        print("Running diarization...")

        # 3. Diarization
        diarize_segments = perform_diarization(
            audio,
            diarize_model
        )

        # 4. Speaker assignment
        final_result = assign_speakers(
            diarize_segments,
            aligned_result
        )

        # 5. Formatting
        transcript = format_transcript(
            final_result["segments"]
        )

    else:

        print("Skipping diarization...")

        transcript = ""

        for seg in aligned_result["segments"]:

            start = round(seg["start"], 2)
            end = round(seg["end"], 2)

            transcript += (
                f"[{start}s - {end}s] "
                f"{seg['text']}\n"
            )

    return transcript


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Run WhisperX Audio Intelligence Pipeline."
    )
    parser.add_argument(
        "audio",
        help="Path to the input audio file."
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        help="Enable speaker diarization."
    )
    args = parser.parse_args()

    output = run_pipeline(
        audio_path=args.audio,
        enable_diarization=args.diarization
    )

    print(output)
