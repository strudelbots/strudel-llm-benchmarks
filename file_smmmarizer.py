import datetime
import glob
import json
import os
from random import randint

import boto3

from llm_factory import get_llm_accessor
from llm_response import FileSummary


class FileSummarizer:
    system_context = ("You are a senior developer and an expert in Python. "
                      "Your task is to analyze the given python file below and summarize in up to "
                      "three sentences what is the main functionality of the file. "
                      "The user input is given in a form of the following json with two fields: "
                      "{'file_name': 'example name', 'file_content': 'example content'}")

    def __init__(self):
        self.llm_accessor = get_llm_accessor('AZURE', self.system_context, 'gpt-4')

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
        output_file = '/tmp/'+ now.strftime("%m-%d-%Y__") + model_name+'__'+full_path.replace('.py', '.json')
        dict_data = llm_response.to_dict()
        with open(output_file, 'w') as outfile:
            json.dump(dict_data, outfile, indent=4)

    def upload_results_to_s3(self):
        model_name = self.llm_accessor.model_name
        all_data = []
        json_files = glob.glob('/tmp/*.json')
        now = datetime.datetime.now()  # current date and time
        for file in json_files:
            if model_name in file and now.strftime("%m-%d-%Y__") in file:
                with open(file, 'r') as infile:
                    data = json.load(infile)
                    all_data.append(data)
        all_data = sorted(all_data, key=lambda x: x["file_name"])
        boto3.setup_default_session(profile_name='s3_accessor')
        s3 = boto3.client('s3')
        object_name = now.strftime("%m-%d-%Y__") + model_name + '__'+ ROOT_DIR.replace('/', '-') + ".json"
        s3.put_object(Body=json.dumps(all_data, indent=4), Bucket='ai-llm-experiments', Key=object_name)

ROOT_DIR = '/home/shai/make-developers-brighter'
if __name__ == "__main__":
    file_keywords_to_skip = ['pytorch', '__init__.py']
    sample_factor = 1
    summarizer = FileSummarizer()
    python_files = glob.glob(f'{ROOT_DIR}/**/*.py', recursive=True)
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
    summarizer.upload_results_to_s3()
    