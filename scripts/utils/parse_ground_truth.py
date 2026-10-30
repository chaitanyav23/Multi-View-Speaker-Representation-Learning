import re
import json
import argparse
from pathlib import Path

def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

def parse_ground_truth(file_path):
    # Pattern: Speaker X [HH:MM:SS - HH:MM:SS]: transcript
    pattern = re.compile(r"Speaker (\d+) \[([\d:]+) - ([\d:]+)\]: (.*)")
    
    segments = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                spk_id = match.group(1)
                start_str = match.group(2)
                end_str = match.group(3)
                text = match.group(4)
                
                start = time_to_seconds(start_str)
                end = time_to_seconds(end_str)
                
                segments.append({
                    "start": start,
                    "end": end,
                    "speaker": f"SPEAKER_{spk_id.zfill(2)}",
                    "text": text
                })
    return segments

def main():
    parser = argparse.ArgumentParser(description="Parse BWC Ground Truth TXT to JSON/RTTM")
    parser.add_argument("--input-dir", default="ground_truth")
    parser.add_argument("--output-dir", default="ground_truth/parsed")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for txt_path in input_dir.glob("*.txt"):
        print(f"Parsing: {txt_path.name}")
        segments = parse_ground_truth(txt_path)
        
        # Save as JSON
        json_path = output_dir / f"{txt_path.stem}.json"
        with open(json_path, 'w') as f:
            json.dump(segments, f, indent=2)
            
        # Save as RTTM (for standard tools)
        rttm_path = output_dir / f"{txt_path.stem}.rttm"
        with open(rttm_path, 'w') as f:
            for seg in segments:
                duration = seg['end'] - seg['start']
                # RTTM format: SPEAKER <file> <chnl> <start> <dur> <ortho> <stype> <name> <conf>
                f.write(f"SPEAKER {txt_path.stem} 1 {seg['start']:.3f} {duration:.3f} <NA> <NA> {seg['speaker']} <NA>\n")

if __name__ == "__main__":
    main()
