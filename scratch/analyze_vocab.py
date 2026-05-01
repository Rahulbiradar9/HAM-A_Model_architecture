import json
import re
from collections import Counter

def analyze_words(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_words = []
        for transcript_id, pairs in data.items():
            for index, text in pairs.items():
                if isinstance(text, str):
                    # Clean and tokenize words (lowercase, remove punctuation)
                    words = re.findall(r'\b\w+\b', text.lower())
                    all_words.extend(words)
        
        return all_words
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []

ellie_words_list = analyze_words('ellie_responses.json')
participant_words_list = analyze_words('participant_responses.json')

ellie_counter = Counter(ellie_words_list)
participant_counter = Counter(participant_words_list)

ellie_unique = set(ellie_words_list)
participant_unique = set(participant_words_list)

common_vocab = ellie_unique.intersection(participant_unique)
only_ellie = ellie_unique - participant_unique
only_participant = participant_unique - ellie_unique

print("--- Frequency Analysis ---")
print(f"Ellie Top 10: {ellie_counter.most_common(10)}")
print(f"Participant Top 10: {participant_counter.most_common(10)}")

print("\n--- Vocabulary Stats ---")
print(f"Unique words in Ellie: {len(ellie_unique)}")
print(f"Unique words in Participant: {len(participant_unique)}")
print(f"Words common to both scripts: {len(common_vocab)}")
print(f"Words unique to Ellie: {len(only_ellie)}")
print(f"Words unique to Participant: {len(only_participant)}")

# Example common words that are frequent in both
shared_freq = {word: (ellie_counter[word] + participant_counter[word]) for word in common_vocab}
sorted_shared = sorted(shared_freq.items(), key=lambda x: x[1], reverse=True)
print(f"\nMost frequent words shared by both: {sorted_shared[:10]}")
