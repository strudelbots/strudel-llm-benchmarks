import json
import os
import shutil
import hashlib
from benchmark_code import OUT_FILES_DIRECTORY, file_date_prefix, project_name
from benchmark_code.llm_factory import get_llm_accessor
from benchmark_code.llm_response import FileSummary
from benchmark_code.llm_model import LlmModel
from benchmark_code import REPO_DIRECTORY, OUT_FILES_DIRECTORY_CACHE

class FileSummarizer:
    system_context = ("You are a senior developer and an expert in Python. "
                      "Your task is to analyze the given python file below and summarize in up to "
                      "three sentences what is the main functionality of the file. "
                      "The user input is given in a form of the following json with "
                      "two fields: "
                      "{'file_name': 'example name', 'file_content': 'example content'}")
    with open('./results/pytorch_DB.json', 'r') as file:
        file_data = json.load(file)
    
    def __init__(self, model: LlmModel, use_file_cache=True):
        self.model = model
        self.llm_accessor = get_llm_accessor(self.system_context, model)
        self.file_prefix = file_date_prefix+"single__"
        self.use_file_cache = use_file_cache

    def summarize_file(self, full_path):
        output_file_name, hased_name = self._get_out_file_name(full_path, self.llm_accessor.model.known_name)
        if self._sumarization_exists(output_file_name, hased_name, full_path):
            #print(f'{full_path} already exists in the database')
            return
        #print(f'{full_path} does not exist in the database')
        print(f'{full_path} is being summarized by model: {self.llm_accessor.model.known_name}')
        file_name = os.path.basename(full_path)
        file_text = self._get_file_text(filename=full_path)
        llm_input = '{' + f'"file_name": "{file_name}", "file_content": "{file_text}"' + '}'
        llm_response = self.llm_accessor._get_llm_response(llm_input)
        file_summary = FileSummary(llm_response, full_path, len(file_text.split('\n')))
        self._save_response_in_file(file_summary, output_file_name, hased_name)

    def summarize_file_multi_repeat(self, full_path, repeat_count=1):
        file_name = os.path.basename(full_path)
        file_data = self._get_single_file_db(full_path)        
        file_text = self._get_file_text(filename=full_path)
        for index in range(repeat_count):
            print(f'{full_path} is being summarized by model: {self.llm_accessor.model.known_name} for the {index} time')
            llm_input = '{' + f'"file_name": "{file_name}", "file_content": "{file_text}"' + '}'
            llm_response = self.llm_accessor._get_llm_response(llm_input)
            file_summary = FileSummary(llm_response, full_path, len(file_text.split('\n')))
            file_data.append(file_summary.to_dict())
        self._save_single_file_db(full_path, file_data)

    def _save_single_file_db(self, full_path, file_data):
        db_file = self._get_single_file_db_name(full_path)
        with open(db_file, 'w') as file:
            json.dump(file_data, file, indent=4)

    
    def _get_single_file_db(self, full_path):
        db_file = self._get_single_file_db_name(full_path)
        if os.path.exists(db_file):
            with open(db_file, 'r') as file:
                file_data = json.load(file)
            return file_data
        return []

    def _get_single_file_db_name(self, full_path):
        file_key = full_path.removeprefix(REPO_DIRECTORY).replace('/', '_').replace('.py', '')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_file = os.path.join(current_dir, '../results', f'{project_name}_{file_key}.json')
        return db_file


    def _get_file_text(self, filename):
        text_file = open(filename, 'r')
        python_string = text_file.read()
        text_file.close()
        return python_string


    def _save_response_in_file(self, llm_response, output_file_name, hased_name):
        dict_data = llm_response.to_dict()
        with open(output_file_name, 'w') as outfile:
            json.dump(dict_data, outfile, indent=4)
        shutil.copy(output_file_name, f'{OUT_FILES_DIRECTORY_CACHE}/{hased_name}')

    def _get_out_file_name(self, full_path, model_name):
        out_name = full_path.removeprefix(REPO_DIRECTORY)
        out_name = out_name.replace('/', '**')
        output_file = f'{OUT_FILES_DIRECTORY}/'+ self.file_prefix + model_name+'__'+out_name.replace('.py', '.json')
        hash_file_name = hashlib.sha256(output_file.encode()).hexdigest()+ '.json'
        return output_file, hash_file_name

    def _sumarization_exists(self, output_file_name, hased_name, full_path):
        relative_path = full_path.removeprefix(REPO_DIRECTORY)
        summary_data = self.file_data.get(relative_path, None)
        if summary_data:
            if self.model.known_name in summary_data:
                return True
        if os.path.exists(f'{OUT_FILES_DIRECTORY_CACHE}/{hased_name}'):
            shutil.copy(f'{OUT_FILES_DIRECTORY_CACHE}/{hased_name}', output_file_name)
            return True
        return False
