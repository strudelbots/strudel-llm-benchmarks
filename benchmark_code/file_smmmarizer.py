import datetime
import json
import os

from benchmark_code import OUT_FILES_DIRECTORY, file_date_prefix
from benchmark_code.llm_factory import get_llm_accessor
from benchmark_code.llm_response import FileSummary
from benchmark_code.llm_model import LlmModel
from benchmark_code import REPO_DIRECTORY

class FileSummarizer:
    system_context = ("You are a senior developer and an expert in Python. "
                      "Your task is to analyze the given python file below and summarize in up to "
                      "three sentences what is the main functionality of the file. "
                      "The user input is given in a form of the following json with "
                      "two fields: "
                      "{'file_name': 'example name', 'file_content': 'example content'}")
    def __init__(self, model: LlmModel,provider_name= 'AWS'):
        self.llm_accessor = get_llm_accessor(self.system_context, model,provider_name)
        self.file_prefix = file_date_prefix+"single__"

    def summarize_file(self, full_path):
        file_name = os.path.basename(full_path)
        file_text = self._get_file_text(filename=full_path)
        llm_input = '{' + f'"file_name": "{file_name}", "file_content": "{file_text}"' + '}'
        llm_response = self.llm_accessor._get_llm_response(llm_input)
        file_summary = FileSummary(llm_response, full_path)
        self._save_response_in_file(file_summary, full_path)


    def _get_file_text(self, filename):
        text_file = open(filename, 'r')
        python_string = text_file.read()
        text_file.close()
        return python_string


    def _save_response_in_file(self, llm_response, full_path):
        model_name = self.llm_accessor.model.known_name
        out_name = full_path.removeprefix(REPO_DIRECTORY)
        out_name = out_name.replace('/', '**')
        output_file = f'{OUT_FILES_DIRECTORY}/'+ self.file_prefix + model_name+'__'+out_name.replace('.py', '.json')
        dict_data = llm_response.to_dict()
        with open(output_file, 'w') as outfile:
            json.dump(dict_data, outfile, indent=4)

