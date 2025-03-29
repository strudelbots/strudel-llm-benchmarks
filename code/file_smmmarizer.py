import datetime
import glob
import json
import os
from random import randint

from code import JSON_FILES_DIRECTORY
from code.llm_factory import get_llm_accessor
from code.llm_response import FileSummary
from code.s3_accessor import upload_results_to_s3
from code.utils import clean_tmp_dir


class FileSummarizer:
    system_context = ("You are a senior developer and an expert in Python. "
                      "Your task is to analyze the given python file below and summarize in up to "
                      "three sentences what is the main functionality of the file. "
                      "The user input is given in a form of the following json with "
                      "two fields: "
                      "{'file_name': 'example name', 'file_content': 'example content'}")

    def __init__(self, provider_name, model_name):
        self.llm_accessor = get_llm_accessor(provider_name, self.system_context, model_name)

    def summarize_file(self, full_path):
        file_name = os.path.basename(full_path)
        file_text = self._get_file_text(filename=full_path)
        llm_input = '{' + f'"file_name": "{file_name}", "file_content": "{file_text}"' + '}'
        llm_response = self.llm_accessor._get_llm_response(llm_input)
        file_summary = FileSummary(llm_response, full_path)
        self._save_response_in_file(file_summary, full_path)


    def _get_file_text(self, filename):
        print(f'Getting file name: {filename}')
        text_file = open(filename, 'r')
        python_string = text_file.read()
        text_file.close()
        return python_string


    def _save_response_in_file(self, llm_response, full_path):
        model_name = self.llm_accessor.model_name
        full_path = os.path.abspath(full_path).replace('/', '-')[1:]
        now = datetime.datetime.now()  # current date and time
        output_file = f'{JSON_FILES_DIRECTORY}/'+ now.strftime("%m-%d-%Y__") + model_name+'__'+full_path.replace('.py', '.json')
        dict_data = llm_response.to_dict()
        with open(output_file, 'w') as outfile:
            json.dump(dict_data, outfile, indent=4)


REPO_DIR = os.getenv('REPO_DIR')
if __name__ == "__main__":
    clean_tmp_dir()
    # Skip files that their full-path contains one of the following.
    file_keywords_to_skip = ['pytorch', '__init__.py']
    # Controls of the percentage of files that would be summarized.
    sample_factor = 20
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
    