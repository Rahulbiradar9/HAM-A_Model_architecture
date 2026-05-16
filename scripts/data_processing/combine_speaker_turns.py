import os
import json

def combine_transcripts():
    input_dir = r"d:\Rahul_Intern\convo_model\json_transcripts"
    output_dir = r"d:\Rahul_Intern\convo_model\combined_transcripts"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_count = 0

    for filename in os.listdir(input_dir):
        # Check if file starts with a number and is a JSON file
        if filename[0].isdigit() and filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading {filename}. Skipping.")
                    continue
            
            if not data:
                continue
                
            combined_data = []
            
            # Initialize with the first item
            current_item = data[0]
            current_speaker = current_item.get("speaker")
            current_value = current_item.get("value", "").strip()
            current_start_time = current_item.get("start_time")
            current_stop_time = current_item.get("stop_time")
            
            # Iterate through the rest of the items
            for item in data[1:]:
                speaker = item.get("speaker")
                value = item.get("value", "").strip()
                
                if speaker == current_speaker:
                    # Same speaker, combine sentences
                    if value: # Only append if value is not empty
                        current_value = f"{current_value} {value}".strip()
                    # Update stop time to the latest stop time
                    if item.get("stop_time"):
                        current_stop_time = item.get("stop_time")
                else:
                    # Different speaker, save the combined item
                    combined_data.append({
                        "start_time": current_start_time,
                        "stop_time": current_stop_time,
                        "speaker": current_speaker,
                        "value": current_value
                    })
                    
                    # Reset current item to the new speaker
                    current_speaker = speaker
                    current_value = value
                    current_start_time = item.get("start_time")
                    current_stop_time = item.get("stop_time")
                    
            # Append the very last item after the loop finishes
            combined_data.append({
                "start_time": current_start_time,
                "stop_time": current_stop_time,
                "speaker": current_speaker,
                "value": current_value
            })
            
            out_filepath = os.path.join(output_dir, filename)
            with open(out_filepath, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=4)
                
            processed_count += 1
            
    print(f"Successfully processed {processed_count} files and saved to {output_dir}")

if __name__ == "__main__":
    combine_transcripts()
