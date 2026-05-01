import json
import os

def combine_responses():
    ellie_file = os.path.join('combo_response', 'ellie_responses.json')
    participant_file = os.path.join('combo_response', 'participant_responses.json')
    output_file = os.path.join('combo_response', 'combined_responses.json')

    print(f"Reading {ellie_file}...")
    with open(ellie_file, 'r', encoding='utf-8') as f:
        ellie_data = json.load(f)
        
    print(f"Reading {participant_file}...")
    with open(participant_file, 'r', encoding='utf-8') as f:
        participant_data = json.load(f)
        
    combined = {}
    global_idx = 1
    
    # Iterate through all transcripts
    for transcript_id in ellie_data:
        if transcript_id in participant_data:
            ellie_pairs = ellie_data[transcript_id]
            participant_pairs = participant_data[transcript_id]
            
            # Sort the keys numerically just to ensure they are in dialogue order
            sorted_keys = sorted(ellie_pairs.keys(), key=lambda x: int(x))
            
            for key in sorted_keys:
                ellie_text = ellie_pairs[key]
                participant_text = participant_pairs.get(key, "")
                
                combined[str(global_idx)] = {
                    "ellie": ellie_text,
                    "participant": participant_text
                }
                global_idx += 1
                
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=4)
        
    print(f"Successfully combined {global_idx - 1} conversation pairs.")

if __name__ == '__main__':
    combine_responses()
