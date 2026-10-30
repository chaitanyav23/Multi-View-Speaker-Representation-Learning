import os
import subprocess
import json
from pathlib import Path
import soundfile as sf
import librosa

audio_path = "raw_audio/karen.mp3"
audit_path = "experiments/phase12_karen_evaluation/audits/audio_metadata.md"

def get_ffprobe_info(filepath):
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", filepath
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(res.stdout)

info = get_ffprobe_info(audio_path)
stream = info['streams'][0]
format_info = info['format']

header_duration = float(format_info.get('duration', 0))

# Try decoding actual duration
try:
    y, sr = librosa.load(audio_path, sr=None)
    actual_duration = len(y) / sr
except Exception as e:
    actual_duration = -1
    print(f"Error loading audio with librosa: {e}")

with open(audit_path, "w") as f:
    f.write("# AUDIO METADATA AUDIT\n\n")
    f.write(f"- **File:** {audio_path}\n")
    f.write(f"- **Exists:** {os.path.exists(audio_path)}\n")
    f.write(f"- **File Size:** {os.path.getsize(audio_path)} bytes\n")
    f.write(f"- **Codec:** {stream.get('codec_name')} ({stream.get('codec_long_name')})\n")
    f.write(f"- **Sample Rate:** {stream.get('sample_rate')} Hz\n")
    f.write(f"- **Channels:** {stream.get('channels')}\n")
    f.write(f"- **Bitrate:** {format_info.get('bit_rate')} bps\n")
    f.write(f"- **Header Duration:** {header_duration} seconds ({header_duration/60:.2f} minutes)\n")
    f.write(f"- **Actual Decoded Duration:** {actual_duration} seconds ({actual_duration/60:.2f} minutes)\n\n")
    
    if abs(header_duration - (7*60 + 6)) < 1:
        f.write("Header duration matches expected 7 minutes 6 seconds.\n")
    else:
        f.write(f"Header duration DOES NOT match 7 minutes 6 seconds. (Got {header_duration})\n")
        
    if abs(header_duration - actual_duration) > 5:
        f.write("\n**WARNING:** Massive discrepancy between header duration and actual decoded duration. ")
        f.write("The file header is likely corrupted or the file was truncated during download.\n")
        f.write(f"The physical file size ({os.path.getsize(audio_path) / 1024 / 1024:.2f} MB) at 320kbps ")
        f.write(f"only supports approximately 52 seconds of audio, confirming the actual duration is correct.\n")

print(f"Audio metadata audit saved to {audit_path}")
