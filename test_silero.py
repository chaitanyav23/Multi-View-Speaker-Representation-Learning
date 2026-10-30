import torch
from pathlib import Path

# Load Silero VAD
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

audio_path = "experiments/karen_11A_audit/audio/karen_normalized.wav"
wav = read_audio(audio_path, sampling_rate=16000)
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, threshold=0.3)

print(f"Found {len(speech_timestamps)} segments with Silero.")
if speech_timestamps:
    print(f"First segment: {speech_timestamps[0]}")
