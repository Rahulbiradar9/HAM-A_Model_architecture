import csv
import json
import argparse
import sys
import os

def csv_to_json(csv_filepath, json_filepath):
    """Converts a CSV file to a JSON file."""
    if not os.path.exists(csv_filepath):
        print(f"Error: CSV file not found at {csv_filepath}")
        sys.exit(1)

    data = []
    try:
        # Read the CSV file using DictReader to map column names to dictionary keys
        with open(csv_filepath, mode='r', encoding='utf-8-sig') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    try:
        # Write the data to a JSON file
        with open(json_filepath, mode='w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Successfully converted '{csv_filepath}' to '{json_filepath}'")
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Set up argument parsing for command line usage
    parser = argparse.ArgumentParser(description='Convert a CSV file to JSON format.')
    parser.add_argument('csv_file', help='Path to the input CSV file')
    parser.add_argument('json_file', nargs='?', help='Path to the output JSON file (optional)')
    
    args = parser.parse_args()
    
    csv_path = args.csv_file
    json_path = args.json_file
    
    # If no output path is provided, use the CSV filename with a .json extension
    if not json_path:
        base_name, _ = os.path.splitext(csv_path)
        json_path = f"{base_name}.json"
        
    csv_to_json(csv_path, json_path)
