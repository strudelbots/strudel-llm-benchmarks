
import glob
import random
from random import randint

from benchmark_code import REPO_DIRECTORY
from benchmark_code.file_smmmarizer import FileSummarizer
from benchmark_code.s3_accessor import upload_results_to_s3
from benchmark_code.utils import clean_outputs_dir

if __name__ == "__main__":
    clean_outputs_dir(exclude_str = '__summary__')
    # Skip files that their full-path contains one of the following.
    file_keywords_to_skip = [ '__init__.py']
    # Controls of the percentage of files that would be summarized.
    sample_factor = 1
#    model_name = 'gpt-35-turbo'
#    model_name = 'gpt-4o'
    model_name = 'gpt-4'
    summarizer = FileSummarizer('AZURE', model_name)
    python_files = glob.glob(f'{REPO_DIRECTORY}/**/*.py', recursive=True)
    random.seed(42)
    index = 0
    for _, file in enumerate(python_files):
        for keyword in file_keywords_to_skip:
            if keyword in file:
                continue
        i = randint(1, 100)
        if i  > sample_factor or any(elem in file for elem in file_keywords_to_skip):
            continue
        index += 1
        print(f'file index: {index}')
        try:
            summarizer.summarize_file(file)
        except Exception as e:
            print(f'Failed to summarize file: {file}')
            print(e)
    upload_results_to_s3(model_name, 'ai-llm-experiments')
    