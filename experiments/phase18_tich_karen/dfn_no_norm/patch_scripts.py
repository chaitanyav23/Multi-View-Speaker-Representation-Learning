import os
from pathlib import Path

def main():
    target_dir = Path("experiments/phase18_tich_karen/dfn_no_norm")
    for script_path in target_dir.glob("*.py"):
        with open(script_path, "r") as f:
            content = f.read()
            
        # Replace the directory path
        content = content.replace("phase18_tich_karen/dfn_no_norm", "phase18_tich_karen/dfn_no_norm")
        
        # We don't need to change the file names, because the DFN output 
        # is still named `rich_karen_dfn_no_norm.wav` in the `dfn_no_norm/audio` folder.
        # But wait, to prevent confusion let's rename the file and the paths to "dfn_no_norm"
        content = content.replace("rich_karen_dfn_no_norm.wav", "rich_karen_dfn_no_norm.wav")
        content = content.replace("rich_karen_dfn_no_norm.rttm", "rich_karen_dfn_no_norm.rttm")
        content = content.replace("rich_karen_dfn_no_norm", "rich_karen_dfn_no_norm")
        
        with open(script_path, "w") as f:
            f.write(content)
            
    print("Paths updated in Python scripts.")

if __name__ == "__main__":
    main()
