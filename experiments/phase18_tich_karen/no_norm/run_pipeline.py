import subprocess
import sys
from pathlib import Path

def run_step(script_name):
    print(f"\n" + "="*50)
    print(f"RUNNING: {script_name}")
    print("="*50 + "\n")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with exit code {result.returncode}")
        # sys.exit(result.returncode) # Stop on failure

def main():
    BASE_DIR = Path("experiments/phase18_tich_karen/no_norm")
    scripts = [
        "step1_preprocess.py",
        "step2_process_gt.py",
        "step3_vad_grounding.py",
        "step4_oracle_rttm.py",
        "step5_whisperx.py",
        "step6_msdd.py",
        "step7_fusion.py",
        "step8_vad_eval.py",
        "step9_asr_eval.py",
        "step10_diar_eval.py"
    ]
    
    for s in scripts:
        run_step(str(BASE_DIR / s))

    # Diagnostics: Run embedding analysis
    print("\nRunning Embedding Analysis for Diagnostics...")
    run_step(str(BASE_DIR / "analyze_embeddings.py")) 

if __name__ == "__main__":
    main()
