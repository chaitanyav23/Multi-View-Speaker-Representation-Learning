import whisperx


def align_transcript(audio, result, device):
    """
    Align transcript to obtain word-level timestamps.
    """

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )

    aligned_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device
    )

    return aligned_result
