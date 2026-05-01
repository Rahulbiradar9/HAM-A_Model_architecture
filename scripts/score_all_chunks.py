import os
import glob
import subprocess

def score_all():
    # Find all the chunk files
    chunk_files = glob.glob("combo_response/combined_responses_part*.json")
    
    # Sort them by part number
    def get_part_num(filename):
        # Extract the number from "combined_responses_partX.json"
        basename = os.path.basename(filename)
        num_str = basename.replace("combined_responses_part", "").replace(".json", "")
        return int(num_str) if num_str.isdigit() else 0

    chunk_files.sort(key=get_part_num)

    print(f"Found {len(chunk_files)} chunk files to process.")

    for filepath in chunk_files:
        basename = os.path.basename(filepath)
        # Create an output filename like "scored_partX.json"
        out_name = basename.replace("combined_responses_", "scored_")
        out_path = os.path.join("combo_response", out_name)
        
        # Skip if it already exists (resume capability)
        if os.path.exists(out_path):
            print(f"Skipping {basename} - already scored.")
            continue
            
        print(f"\n==========================================")
        print(f"Scoring {basename}...")
        print(f"==========================================")
        
        cmd = [
            "python", "scripts/score_combined_responses.py",
            "--input", filepath,
            "--output", out_path
        ]
        
        # Run the command
        subprocess.run(cmd)

if __name__ == "__main__":
    score_all()
