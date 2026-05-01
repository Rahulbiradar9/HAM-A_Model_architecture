import json
import os
import random
import argparse

def split_json_random(filepath, min_size, max_size):
    print(f"Reading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    items = list(data.items())
    total = len(items)
    
    base_dir = os.path.dirname(filepath)
    base_name = os.path.basename(filepath).replace('.json', '')
    
    print(f"Splitting {total} items into random chunks (size between {min_size} and {max_size})...")
    
    current_idx = 0
    chunk_num = 1
    
    while current_idx < total:
        chunk_size = random.randint(min_size, max_size)
        end_idx = min(current_idx + chunk_size, total)
        
        chunk_data = dict(items[current_idx:end_idx])
        
        chunk_file = os.path.join(base_dir, f"{base_name}_part{chunk_num}.json")
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=4)
            
        print(f"Saved {chunk_file} ({len(chunk_data)} items)")
        
        current_idx = end_idx
        chunk_num += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a large JSON dict into randomly sized chunks.")
    parser.add_argument("--input", default="combo_response/combined_responses.json")
    parser.add_argument("--min", type=int, default=100, help="Minimum items per chunk")
    parser.add_argument("--max", type=int, default=1000, help="Maximum items per chunk")
    args = parser.parse_args()
    
    split_json_random(args.input, args.min, args.max)
