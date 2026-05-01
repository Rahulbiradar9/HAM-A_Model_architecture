import os
import json

def create_pairs():
    input_dir = r"d:\Rahul_Intern\convo_model\combined_transcripts"
    output_file = r"d:\Rahul_Intern\convo_model\all_conversation_pairs.json"

    all_pairs = []

    for filename in sorted(os.listdir(input_dir)):
        if filename[0].isdigit() and filename.endswith(".json"):
            file_id = filename.split("_")[0]
            filepath = os.path.join(input_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading {filename}")
                    continue
                
            i = 0
            while i < len(data) - 1:
                current = data[i]
                next_turn = data[i+1]
                
                if current.get("speaker") == "Ellie" and next_turn.get("speaker") == "Participant":
                    all_pairs.append({
                        "ellie": current.get("value", ""),
                        "participant": next_turn.get("value", "")
                    })
                    i += 2  # Skip the participant we just paired
                else:
                    i += 1  # If not Ellie followed by Participant, just move forward one step

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, indent=4)

    print(f"Successfully generated {len(all_pairs)} conversation pairs from {input_dir}.")
    print(f"Saved output to {output_file}")

if __name__ == "__main__":
    create_pairs()
