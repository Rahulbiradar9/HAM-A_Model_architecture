import json

with open(r"d:\Rahul_Intern\convo_model\data\Interview_Data_6K_llama3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Scored so far: {len(data)}")
last = data[-1]
print(f"Last conversation_index: {last['conversation_index']}")
print(f"Last patient_input: {last.get('patient_input', 'N/A')[:150]}")
