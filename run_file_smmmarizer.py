import glob
import json
import os
import random
from random import randint
from time import sleep
from math import floor
from benchmark_code import REPO_DIRECTORY, OUT_FILES_DIRECTORY, file_date_prefix
from benchmark_code.create_summary_file import CollectFileSummaryData
from benchmark_code.file_smmmarizer import FileSummarizer
from benchmark_code.s3_accessor import store_results_for_model
from benchmark_code.utils import clean_outputs_dir
from benchmark_code.llm_model import AVAILABLE_MODELS, LlmModel

def _generate_file_summaries_for_model(model: LlmModel,python_files):
    index =0
    summarizer = FileSummarizer(model)
    for _, file in enumerate(python_files):
        if any(elem in file for elem in file_keywords_to_skip):
            continue
        index += 1
        print(f'{index}. Generate summary for {file=}, {model.known_name=}, ')
        try:
            summarizer.summarize_file(file)
        except Exception as e:
            print(f'Failed to summarize file: {file}, {model.known_name=}')
            raise e
    store_results_for_model(model.known_name, 'ai-llm-experiments')


def _combine_summaries_into_single_json(file_name):
    file_name = file_name.replace(':', '_')
    os.chdir(OUT_FILES_DIRECTORY)
    glob_pattern = f'{OUT_FILES_DIRECTORY}/{file_date_prefix}summary__*.json'
    summary_files = glob.glob(glob_pattern)
    data_collector = CollectFileSummaryData(summary_files)
    comparable_summaries = data_collector.merged_summaries()
    with open(OUT_FILES_DIRECTORY +'/' + file_name, 'w') as f:
        json.dump(comparable_summaries, f, indent=4)

def get_models():
    run_on = ['nova-lite-v1',
              'Claude3.5',
              'Llama3.3',
              'titan_premier',
              'nova-pro-v1'
              ]
    assert all(elem in [model.known_name for model in AVAILABLE_MODELS] for elem in run_on)
    models = AVAILABLE_MODELS
    models = [model for model in models if model.known_name in run_on]    
    return models

if __name__ == "__main__":
    clean_outputs_dir()
    # Skip files that their full-path contains one of the following.
    file_keywords_to_skip = [ '__init__.py', 'test']
    models = get_models()
    sample_factor = 0.5 # Controls of the percentage of files that would be summarized.
    random.seed(25) # This ensures we will get the same files to analyze for each model.
    python_files = glob.glob(f'{REPO_DIRECTORY}/**/*.py', recursive=True)
    python_files = random.sample(python_files, floor(len(python_files)*sample_factor/100.0))
    for model in models:
        _generate_file_summaries_for_model(model,python_files)
    summary_file_name = file_date_prefix+ '_'.join([x.known_name for x in models])+'__summary.json'
    _combine_summaries_into_single_json(summary_file_name)

