import json
import os
import shutil
from pathlib import Path
from benchmark_code import OUT_FILES_DIRECTORY_CACHE, REPO_DIRECTORY
from collections import defaultdict
from db_entry import SingleModelDBEntry
from llm_response import LlmResponse
from llm_model import LlmModel
from db_entry import SingleFileDBEntry
class CacheManager:
    def __init__(self, db_file_path, clear_cache=False):
        self.db_file_path = db_file_path
        self.cache_dir = OUT_FILES_DIRECTORY_CACHE
        self._ensure_db_exists()
        self.force_clear_cache = clear_cache


    def _ensure_db_exists(self):
        """Ensure the database file exists, create it if it doesn't"""
        if not os.path.exists(self.db_file_path):
            raise FileNotFoundError(f"Database file not found at {self.db_file_path}")

    def _load_db(self):
        """Load the current database content"""
        with open(self.db_file_path, 'r') as f:
            return json.load(f)

    def _save_db(self, data):
        """Save data to the database file"""
        with open(self.db_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _collect_summaries_per_target_file(self):
        """Collect all summaries into a single file"""
        summaries_per_file = defaultdict(list)
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        for cache_file in cache_files:
            with open(os.path.join(self.cache_dir, cache_file), 'r') as f:
                cache_data = json.load(f)
                file = cache_data['file_name']
                summaries_per_file[file].append(cache_data['llm_result'])
        return summaries_per_file
 
    def force_clear_cache(self):
        if not self.force_clear_cache:
            return
        """Clear all files from the cache directory"""
        for file in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file}: {str(e)}") 

    def _collect_single_file_entries(self, summaries_per_file):
        """Collect all summaries into a single file"""
        single_file_entries = SingleFileDBEntry()
        for file, summaries in summaries_per_file.items():
            key = file.removeprefix('/home/shai/pytorch')

            for summary in summaries:
                llm_model = LlmModel(summary['model']['known_name'], summary['model']['provider_name'], 
                                     summary['model']['aws_model_id'], summary['model']['aws_region'], 
                                     summary['model']['azure_deployment_name'], summary['model']['azure_region'], 
                                     summary['model']['delay_time'], summary['model']['langchain_ready'], 
                                     summary['model']['price_per_1000_input_tokens'], 
                                     summary['model']['price_per_1000_output_tokens'])
                llm_result = LlmResponse(summary['message'], summary['total_tokens'], llm_model, summary['latency'])
                number_of_lines = -1
                project_name = 'pytorch'
                model_key = llm_model.known_name
                if model_key in single_file_entries.data.get(key, {}):
                    raise ValueError(f"Model {model_key} already exists in {key}")
                else:
                    file_summary = SingleModelDBEntry(llm_result,key,number_of_lines,project_name)
                    current_file_data = single_file_entries.data.get(key, {})

                    if not current_file_data:
                        single_file_entries.data[key] = {model_key: file_summary}
                    else:
                        single_file_entries.data[key] = {**current_file_data, model_key: file_summary}
        return single_file_entries

    def update_db(self, db_dict_format):
        db = self._load_db()
        for file, db_entry in db_dict_format.items():
            if file in db.keys():
                raise NotImplementedError(f"File {file} already exists in the database")
            db[file] = db_entry
        self._save_db(db)

if __name__ == "__main__":
    cache_manager = CacheManager(db_file_path='./results/pytorch_DB.json')
    summaries_per_file = cache_manager._collect_summaries_per_target_file()
    single_file_entries = cache_manager._collect_single_file_entries(summaries_per_file)
    #data_dict = single_file_entries.to_dict()
    db_dict_format = single_file_entries.to_db_dict()
    str_json = json.dumps(db_dict_format, indent=4)
    with open('/tmp/single_file_entries.json', 'w') as f:
        f.write(str_json)
    for key, value in single_file_entries.data.items():
        print(key)
    cache_manager.update_db(db_dict_format)
    #cache_manager.process_cache_files()
    #cache_manager.force_lear_cache()