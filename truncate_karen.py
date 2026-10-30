import json
from pathlib import Path

path = Path("experiments/karen_11A_audit/outputs/karen_final.json")
with open(path) as f:
    data = json.load(f)

# Truncate segments to 426 seconds
filtered_segments = []
for s in data["segments"]:
    if s["start"] < 426:
        if s["end"] > 426:
            s["end"] = 426
        filtered_segments.append(s)

data["segments"] = filtered_segments

with open(path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Truncated segments to 426s. Remaining segments: {len(filtered_segments)}")
