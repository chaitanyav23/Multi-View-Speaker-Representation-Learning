import json
import re
from datetime import datetime

def hms_to_seconds(hms):
    h, m, s = map(float, hms.split(':'))
    return h * 3600 + m * 60 + s

def parse_gt(gt_path):
    segments = []
    current_spk = None
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line: continue
        
        # Match "Speaker_X [HH:MM:SS - HH:MM:SS]"
        match = re.search(r'(Speaker_\d+) \[(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2})\]', line)
        if match:
            spk = match.group(1)
            start = hms_to_seconds(match.group(2))
            end = hms_to_seconds(match.group(3))
            
            # The next line should be the text
            text = ""
            if i + 1 < len(lines):
                text = lines[i+1].strip()
            
            segments.append({
                "start": start,
                "end": end,
                "speaker": spk,
                "text": text
            })
            
    return segments

gt_segments = parse_gt("ground_truth/karen.txt")
# Convert to simple grounding (just start/end for VAD)
grounding = {"segments": [{"start": s["start"], "end": s["end"]} for s in gt_segments]}

with open("experiments/karen_11A_audit/grounding/karen_oracle_grounding.json", "w") as f:
    json.dump(grounding, f, indent=2)

print(f"Parsed {len(gt_segments)} segments from GT.")
