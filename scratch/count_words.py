import json

def count_words(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        total_words = 0
        for transcript_id, pairs in data.items():
            for index, text in pairs.items():
                # Only count if text is a string
                if isinstance(text, str):
                    total_words += len(text.split())
        return total_words
    except Exception as e:
        return f"Error: {e}"

print(f"Ellie word count: {count_words('ellie_responses.json')}")
print(f"Participant word count: {count_words('participant_responses.json')}")
