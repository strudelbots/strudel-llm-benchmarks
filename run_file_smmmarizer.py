
import glob
import json
import os
import random
from random import randint

from benchmark_code import REPO_DIRECTORY, OUT_FILES_DIRECTORY, file_date_prefix
from benchmark_code.create_summary_file import CollectFileSummaryData
from benchmark_code.file_smmmarizer import FileSummarizer
from benchmark_code.s3_accessor import store_results_for_model
from benchmark_code.utils import clean_outputs_dir


def _generate_file_summaries_for_model():
    index =0
    summarizer = FileSummarizer('AZURE', model)
    for _, file in enumerate(python_files):
        for keyword in file_keywords_to_skip:
            if keyword in file:
                continue
        i = randint(1, 100)
        if i > sample_factor or any(elem in file for elem in file_keywords_to_skip):
            continue
        index += 1
        print(f'{index}. Generate summary for {file=}, {model=}, ')
        try:
            summarizer.summarize_file(file)
        except Exception as e:
            print(f'Failed to summarize file: {file}')
            print(e)
    store_results_for_model(model, 'ai-llm-experiments')


if __name__ == "__main__":
    clean_outputs_dir()
    # Skip files that their full-path contains one of the following.
    file_keywords_to_skip = [ '__init__.py']
    models = ['gpt-35-turbo', 'gpt-4', 'gpt-4o']
    # Controls of the percentage of files that would be summarized.
    sample_factor = 5
    python_files = glob.glob(f'{REPO_DIRECTORY}/**/*.py', recursive=True)
    for model in models:
        random.seed(42)
        _generate_file_summaries_for_model()
    os.chdir(OUT_FILES_DIRECTORY)
    glob_pattern = f'{OUT_FILES_DIRECTORY}/{file_date_prefix}summary__*.json'
    summary_files = glob.glob(glob_pattern)
    data_collector = CollectFileSummaryData(summary_files)
    comparable_summaries = data_collector.merged_summaries()


    with open(OUT_FILES_DIRECTORY + '/gpt35_gpt4_gpt4o_pytorch.py', 'w') as f:
        json.dump(comparable_summaries, f, indent=4)

