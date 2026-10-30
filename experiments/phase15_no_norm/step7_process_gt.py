import re
import json
from pathlib import Path

def parse_timestamp(ts):
    parts = ts.split(':')
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60 + s
    return float(parts[0])

def main():
    BASE_DIR = Path("experiments/phase15_no_norm")
    GT_PATH = Path("ground_truth/karen.txt")
    GT_JSON_DIR = Path("ground_truth_json")
    GT_RTTM_DIR = Path("ground_truth_rttm")
    
    GT_JSON_DIR.mkdir(exist_ok=True)
    GT_RTTM_DIR.mkdir(exist_ok=True)

    with open(GT_PATH, "r") as f:
        lines = f.readlines()

    segments = []
    current_speaker = None
    current_start = None
    current_end = None
    
    # Regex for "Speaker_N [HH:MM:SS - HH:MM:SS]"
    header_re = re.compile(r"(Speaker_\d+) \[([\d:]+) - ([\d:]+)\]")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        match = header_re.match(line)
        if match:
            speaker = match.group(1)
            start = parse_timestamp(match.group(2))
            end = parse_timestamp(match.group(3))
            
            # The next line is the text
            i += 1
            text = ""
            while i < len(lines) and not header_re.match(lines[i].strip()) and lines[i].strip():
                text += lines[i].strip() + " "
                i += 1
            
            segments.append({
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text.strip()
            })
        else:
            i += 1

    # Save JSON
    with open(GT_JSON_DIR / "karen.json", "w") as f:
        json.dump({"file": "karen", "segments": segments}, f, indent=2)

    # Save RTTM
    with open(GT_RTTM_DIR / "karen.rttm", "w") as f:
        for s in segments:
            duration = s["end"] - s["start"]
            f.write(f"SPEAKER karen 1 {s['start']:.3f} {duration:.3f} <NA> <NA> {s['speaker']} <NA> <NA>\n")

    print(f"Processed {len(segments)} segments from GT.")
    print(f"Saved to {GT_JSON_DIR}/karen.json and {GT_RTTM_DIR}/karen.rttm")

if __name__ == "__main__":
    main()
