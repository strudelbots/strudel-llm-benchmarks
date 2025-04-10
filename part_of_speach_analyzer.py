# This file seems like a failed attempt to use nltk to define similarities 
# between models. 
import glob
import os
import json
from collections import defaultdict
import itertools
from pprint import pprint
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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
        pos_counts[pos].add(word.lower())
    result = {}
    for pos, words in pos_counts.items():
        result[pos] = list(words)

    return result

def add_pos_diff_to_data(data:dict, pos) -> dict:
    for file_entry, summaries in data.items():
        skip_nn = {'module', 'function', 'result', 'functionality', 'support', 'part', 'system', 
                   'example', 'output', 'usage', 'input', 'script', 'file', 'use', 'python', 'Python', 'need'}
        models = list(summaries.keys())
        model_pairs = list(itertools.combinations(models, 2))
        for model_pair in model_pairs:
            try:
                model1, model2 = model_pair
                pos1 = summaries[model1]['pos'].get(pos, [])
                pos2 = summaries[model2]['pos'].get(pos, [])
                pos_1_minus_2_diff = (set(pos1) - set(pos2)) - skip_nn
                pos_2_minus_1_diff = (set(pos2) - set(pos1)) - skip_nn

                summaries[model1][f'pos_diff_{pos}_{model1}-minus-{model2}'] = list(pos_1_minus_2_diff)
                summaries[model2][f'pos_diff_{pos}_{model2}-minus-{model1}'] = list(pos_2_minus_1_diff)
            except KeyError:
                raise KeyError(f'KeyError: {model_pair} {summaries[model1]['pos']['NNP']}')
    return data

def add_pos_intersection_to_data(data:dict, pos) -> dict:
    for file_entry, summaries in data.items():
        models = list(summaries.keys())
        model_pairs = list(itertools.combinations(models, 2))
        for model_pair in model_pairs:
            model1, model2 = model_pair
            pos1 = summaries[model1]['pos'].get(pos, [])
            pos2 = summaries[model2]['pos'].get(pos, [])
            intersection = list(set(pos1) & set(pos2))
            summaries[model1][f'pos_intersection_{pos}_{model1}-intersect-{model2}'] = intersection
    return data

if __name__ == "__main__":
    cwd = os.getcwd()
    json_files = glob.glob(f'{cwd}/results/*summary.json')
    print(f'{len(json_files)} json files found')
    total_file_summaries = 0
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        new_data = pos_for_file(data)
        new_data = add_pos_diff_to_data(new_data, 'NN')
        new_data = add_pos_diff_to_data(new_data, 'NNP')
        new_data = add_pos_intersection_to_data(new_data, 'NN')
        new_data = add_pos_intersection_to_data(new_data, 'NNP')
        for file_entry, summaries in new_data.items():
            for model,summary in summaries.items():
                del summary['pos']
            file_name = file_entry.split('/')[-1].replace('.py', '.json')
            diff_file = f'/tmp/{file_name.replace('.json', '_pos_diff.json')}'
            with open(diff_file, 'w') as f:
                print(f'{total_file_summaries}. Generating: {diff_file}')
                json.dump({file_entry:summaries}, f, indent=4)
                total_file_summaries += 1

