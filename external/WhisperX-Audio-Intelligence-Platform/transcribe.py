import whisperx


def transcribe_audio(audio_path, model):
    """
    Perform ASR transcription.
    """

    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio)

    return audio, result
