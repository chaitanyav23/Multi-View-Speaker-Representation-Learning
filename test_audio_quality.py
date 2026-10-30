import whisperx
import json
import soundfile as sf
from pathlib import Path

model = whisperx.load_model('large-v3', 'cuda')
audio_path = 'experiments/karen_11A_audit/audio/karen_normalized.wav'
audio, sr = sf.read(audio_path)

with open('experiments/karen_11A_audit/grounding/karen_oracle_grounding.json') as f:
    segs = json.load(f)['segments']

for i in range(min(5, len(segs))):
    s = segs[i]
    start, end = int(s['start']*sr), int(s['end']*sr)
    if end <= start: continue
    res, _ = model.model.transcribe(audio[start:end], language='en')
    hyp_text = "".join([x.text for x in res]).strip()
    print(f"Segment {i} [{s['start']:.2f}-{s['end']:.2f}]: {hyp_text}")
