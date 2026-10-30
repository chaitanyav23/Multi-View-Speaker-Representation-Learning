#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
import sys

# Expected documentation files per phase
EXPECTED_FILES = [
    "README.md",
    "DIRECTORY_TREE.md",
    "ARCHITECTURE.md",
    "SCRIPT_ANALYSIS.md",
    "PIPELINE_FLOW.md",
    "EVALUATION_ANALYSIS.md",
    "RESULTS_ANALYSIS.md",
    "ERROR_ANALYSIS.md",
    "METRICS_TABLES.md",
    "INFERENCES.md",
    "CONCLUSIONS.md",
    "FUTURE_DIRECTION.md",
]

def extract_phase_number(dir_name):
    """Extracts the phase number from the directory name for numerical sorting."""
    match = re.search(r'phase(\d+)', dir_name.lower())
    if match:
        return int(match.group(1))
    return 999999  # Put non-matching at the end

def analyze_phases(phases_dir):
    """Analyzes the phase directories for expected files."""
    phases_dir = Path(phases_dir)
    if not phases_dir.exists() or not phases_dir.is_dir():
        print(f"Directory {phases_dir} not found.")
        return []

    phase_dirs = [d for d in phases_dir.iterdir() if d.is_dir()]
    # Sort phases numerically
    phase_dirs.sort(key=lambda d: extract_phase_number(d.name))
    
    results = []
    for p_dir in phase_dirs:
        missing_files = []
        for file in EXPECTED_FILES:
            if not (p_dir / file).exists():
                missing_files.append(file)
        results.append({
            'path': p_dir,
            'name': p_dir.name,
            'phase_num': extract_phase_number(p_dir.name),
            'missing': missing_files,
            'is_complete': len(missing_files) == 0
        })
    return results

def get_phase_display_name(res):
    """Returns the phase display string."""
    if res['phase_num'] != 999999:
        return f"Phase {res['phase_num']}"
    return res['name']

def print_normal_report(results):
    """Prints the comprehensive documentation completion report."""
    print("==================================================")
    print("Documentation Completion Report")
    print("===============================\n")

    completed_count = 0
    incomplete_count = 0
    total_missing = 0
    highest_complete = None
    first_incomplete = None

    for res in results:
        phase_str = get_phase_display_name(res)
        if res['is_complete']:
            print(f"{phase_str:<10} \u2713 COMPLETE\n")
            completed_count += 1
            highest_complete = phase_str
        else:
            print(f"{phase_str:<10} \u2717 INCOMPLETE")
            print("Missing:")
            for m in res['missing']:
                print(m)
            print()
            incomplete_count += 1
            total_missing += len(res['missing'])
            if first_incomplete is None:
                first_incomplete = phase_str

    print("## Summary\n")
    print(f"Completed phases : {completed_count}\n")
    print(f"Incomplete phases : {incomplete_count}\n")
    print(f"Highest fully documented phase\n")
    print(f"{highest_complete if highest_complete else 'None'}\n")
    print(f"First incomplete phase\n")
    print(f"{first_incomplete if first_incomplete else 'None'}\n")
    print(f"Missing files total\n")
    print(f"{total_missing}\n")

def print_brief_report(results):
    """Prints a brief status report of all phases."""
    for res in results:
        phase_str = get_phase_display_name(res)
        status = "COMPLETE" if res['is_complete'] else "INCOMPLETE"
        print(f"{phase_str:<8} {status}\n")

def print_next_report(results):
    """Prints only the next incomplete phase to resume work on."""
    for res in results:
        if not res['is_complete']:
            print("Next phase to resume:\n")
            print(f"{res['path']}/\n")
            print("Missing files:\n")
            for m in res['missing']:
                print(f"{m}\n")
            return
    print("All phases are complete!\n")

def generate_csv(results, output_path):
    """Generates a CSV file containing the documentation status."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['phase', 'status', 'missing_count', 'missing_files'])
        for res in results:
            phase_str = get_phase_display_name(res)
            status = "COMPLETE" if res['is_complete'] else "INCOMPLETE"
            missing_count = len(res['missing'])
            missing_files_str = "|".join(res['missing'])
            writer.writerow([phase_str, status, missing_count, missing_files_str])

def main():
    parser = argparse.ArgumentParser(description="Check documentation completion status.")
    parser.add_argument('--brief', action='store_true', help="Print brief report")
    parser.add_argument('--next', action='store_true', help="Output only the next phase to resume")
    parser.add_argument('--csv', action='store_true', help="Generate documentation_status.csv")
    args = parser.parse_args()

    # Relative to the root directory where the script is run
    phases_dir = Path("documentation/phases")
    results = analyze_phases(phases_dir)
    
    if args.csv:
        generate_csv(results, Path("documentation/documentation_status.csv"))
    elif args.brief:
        print_brief_report(results)
    elif args.next:
        print_next_report(results)
    else:
        print_normal_report(results)
        
    print("==================================================")
    print("Documentation audit complete")
    print("============================")

if __name__ == "__main__":
    main()