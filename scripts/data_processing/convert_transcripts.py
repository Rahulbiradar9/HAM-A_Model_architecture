import os
import csv
import json

src_dir = r"D:\Rahul_Intern\Daic-woz_dataset\daicwoz\daicwoz\transcript"
dst_dir = r"D:\Rahul_Intern\convo_model\json_transcripts"

os.makedirs(dst_dir, exist_ok=True)

files_converted = 0

for filename in os.listdir(src_dir):
    if filename.endswith(".csv"):
        src_path = os.path.join(src_dir, filename)
        dst_filename = filename.replace(".csv", ".json")
        dst_path = os.path.join(dst_dir, dst_filename)
        
        data = []
        with open(src_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                data.append(row)
                
        with open(dst_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        files_converted += 1

print(f"Successfully converted {files_converted} transcript files to JSON format inside {dst_dir}.")
