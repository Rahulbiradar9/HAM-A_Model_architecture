import os
import json
import glob

def process_transcripts(input_dir='json_transcripts'):
    all_ellie = {}
    all_participant = {}
    
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return
        
    filepaths = glob.glob(os.path.join(input_dir, '*.json'))
    print(f"Found {len(filepaths)} transcripts.")
    
    for file_path in filepaths:
        filename = os.path.basename(file_path).replace('.json', '')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {file_path}")
                continue
                
        ellie_dict = {}
        participant_dict = {}
        
        pair_idx = 1
        current_ellie_text = []
        current_participant_text = []
        last_speaker = None
        
        for item in data:
            # Safely handle missing keys depending on JSON structure
            speaker = item.get('speaker')
            value = item.get('value', '').strip()
            
            # Skip entries that don't have a speaker
            if not speaker:
                continue
                
            if speaker == 'Ellie':
                # If Ellie speaks right after Participant, it means a new exchange pair has started
                if last_speaker == 'Participant':
                    if current_ellie_text or current_participant_text:
                        ellie_dict[str(pair_idx)] = " ".join(current_ellie_text)
                        participant_dict[str(pair_idx)] = " ".join(current_participant_text)
                        pair_idx += 1
                    current_ellie_text = [value]
                    current_participant_text = []
                else:
                    # Ellie continues speaking
                    current_ellie_text.append(value)
                    
            elif speaker == 'Participant':
                # Participant is speaking (or continues speaking)
                current_participant_text.append(value)
                
            last_speaker = speaker
            
        # Add the final trailing exchange pair for this file
        if current_ellie_text or current_participant_text:
            ellie_dict[str(pair_idx)] = " ".join(current_ellie_text)
            participant_dict[str(pair_idx)] = " ".join(current_participant_text)
            
        all_ellie[filename] = ellie_dict
        all_participant[filename] = participant_dict
        
    # Write to new JSON files
    ellie_output = 'ellie_responses.json'
    participant_output = 'participant_responses.json'
    
    with open(ellie_output, 'w', encoding='utf-8') as f:
        json.dump(all_ellie, f, indent=4)
        
    with open(participant_output, 'w', encoding='utf-8') as f:
        json.dump(all_participant, f, indent=4)
        
    print(f"Processing complete.")
    print(f"Saved Ellie's lines to: {ellie_output}")
    print(f"Saved Participant's lines to: {participant_output}")

if __name__ == '__main__':
    process_transcripts()
