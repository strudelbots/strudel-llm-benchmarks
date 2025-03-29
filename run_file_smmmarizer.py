
import glob
import os
from random import randint
from code.file_smmmarizer import FileSummarizer
from code.s3_accessor import upload_results_to_s3
from code.utils import clean_tmp_dir



REPO_DIR = os.getenv('REPO_DIR')
if __name__ == "__main__":
    clean_tmp_dir()
    # Skip files that their full-path contains one of the following.
    file_keywords_to_skip = ['pytorch', '__init__.py']
    # Controls of the percentage of files that would be summarized.
    sample_factor = 10
    model_name = 'gpt-4'
    summarizer = FileSummarizer('AZURE', model_name)
    python_files = glob.glob(f'{REPO_DIR}/**/*.py', recursive=True)
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
        summarizer.summarize_file(file)
    upload_results_to_s3(model_name, 'ai-llm-experiments')
    