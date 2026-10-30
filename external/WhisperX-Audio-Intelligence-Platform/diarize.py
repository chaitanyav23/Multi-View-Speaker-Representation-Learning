import whisperx


def perform_diarization(audio, diarize_model):
    """
    Run speaker diarization.
    """

    diarize_segments = diarize_model(audio)

    return diarize_segments


def assign_speakers(diarize_segments, aligned_result):
    """
    Assign speaker labels to transcript segments.
    """

    final_result = whisperx.assign_word_speakers(
        diarize_segments,
        aligned_result
    )

    return final_result
