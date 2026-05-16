import json
import os

# Define the file paths
qwen_file = '../data/m_chat16k_combined_scored.json'
llama_file = '../data/m_chat16k_combined_scored_llama.json'
mistral_file = '../data/m_chat16k_combined_scored_mistral.json'
output_file = '../data/m_chat16k_combined_scored_weighted_rank1.json'

print("Loading JSON files...")
with open(qwen_file, 'r', encoding='utf-8') as f:
    qwen_data = json.load(f)
    
with open(llama_file, 'r', encoding='utf-8') as f:
    llama_data = json.load(f)

with open(mistral_file, 'r', encoding='utf-8') as f:
    mistral_data = json.load(f)

# Convert to dictionaries by id for easy lookup
llama_dict = {item['id']: item for item in llama_data}
mistral_dict = {item['id']: item for item in mistral_data}

# Weights for Rank 1
w_qwen = 0.3
w_llama = 0.2
w_mistral = 0.5

combined_data = []

print("Processing and calculating weighted scores...")
for qwen_item in qwen_data:
    item_id = qwen_item['id']
    
    if item_id in llama_dict and item_id in mistral_dict:
        llama_item = llama_dict[item_id]
        mistral_item = mistral_dict[item_id]
        
        q_score = qwen_item.get('score', {})
        l_score = llama_item.get('score', {})
        m_score = mistral_item.get('score', {})
        
        # New combined score dictionary
        combined_score = {}
        
        # We assume all models output the same score keys. Let's use q_score keys
        for key in q_score.keys():
            val_q = q_score.get(key, 0)
            val_l = l_score.get(key, 0)
            val_m = m_score.get(key, 0)
            
            # Weighted average
            weighted_val = (val_q * w_qwen) + (val_l * w_llama) + (val_m * w_mistral)
            
            # Round to nearest integer
            combined_score[key] = int(round(weighted_val))
            
        # Create the final combined item
        # Copy original fields from one of them (e.g. Qwen)
        final_item = {
            "id": item_id,
            "source": qwen_item.get("source", ""),
            "input_text": qwen_item.get("input_text", ""),
            "output_text": qwen_item.get("output_text", ""),
            "score": combined_score
        }
        
        combined_data.append(final_item)
    else:
        # If an ID is missing in Llama or Mistral, just keep Qwen's but warn
        print(f"Warning: ID {item_id} missing in Llama or Mistral")
        combined_data.append(qwen_item)

print(f"Writing {len(combined_data)} records to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

print("Finished!")
