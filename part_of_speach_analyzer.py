import glob
import os
import json
from collections import defaultdict

def pos_for_file(file_data:dict) -> dict:
    for file_entry, summaries in file_data.items():
        file_path = file_entry
        for model,summary in summaries.items():
            summary_text = summary['file_summary']
            pos_for_model = pos_for_text(summary_text)
            summary['pos'] = pos_for_model
    return file_data
def pos_for_text(text:str) -> dict:
    import nltk
    from nltk import pos_tag, word_tokenize
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
    # Tokenize and tag parts of speech
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Count frequency of each POS tag
    pos_counts = defaultdict(lambda: set())
    for word, pos in pos_tags:
        pos_counts[pos].add(word)

    return pos_counts

if __name__ == "__main__":
    cwd = os.getcwd()
    json_files = glob.glob(f'{cwd}/results/*.json')
    print(f'{len(json_files)} json files found')
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        new_data = pos_for_file(data)
        with open(json_file, 'w') as f:
            json.dump(new_data, f, indent=4)

