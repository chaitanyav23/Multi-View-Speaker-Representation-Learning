from pydub import AudioSegment
import sys

try:
    audio = AudioSegment.from_file("raw_audio/karen.mp3")
    print(f"Duration: {len(audio) / 1000.0}s")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export("experiments/phase13_karen_generalization/audio/karen_normalized.wav", format="wav")
    print("Exported successfully.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
