import re

def hms_to_seconds(hms):
    h, m, s = map(float, hms.split(':'))
    return h * 3600 + m * 60 + s

def format_gt(in_path, out_path):
    with open(in_path, 'r') as f:
        lines = f.readlines()
    
    formatted = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line: continue
        
        # Match "Speaker_X [HH:MM:SS - HH:MM:SS]"
        match = re.search(r'(Speaker)_(\d+) \[(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2})\]', line)
        if match:
            prefix = match.group(1)
            num = match.group(2)
            ts1 = match.group(3)
            ts2 = match.group(4)
            
            text = ""
            if i + 1 < len(lines):
                text = lines[i+1].strip()
                # Skip if the next line is also a speaker block (empty text)
                if re.search(r'Speaker_\d+ \[', text):
                    text = ""
            
            formatted.append(f"{prefix} {num} [{ts1} - {ts2}]: {text}")
            
    with open(out_path, 'w') as f:
        f.write("\n\n".join(formatted))

format_gt("ground_truth/karen.txt", "ground_truth/karen_fixed.txt")
print("Saved fixed GT to ground_truth/karen_fixed.txt")
