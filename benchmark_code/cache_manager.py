import json
import os
import re
from pathlib import Path
from benchmark_code import OUT_FILES_DIRECTORY_CACHE, REPO_DIRECTORY, project_name
from collections import defaultdict
from benchmark_code.db_entry import SingleModelDBEntry, SingleFileDBEntry
from benchmark_code.llm_response import LlmResponse
from benchmark_code.llm_model import LlmModel

class CacheManager:
    hash_pattern = re.compile(r'^[a-f0-9]{64}\.json$')
    
    def __init__(self, db_file_path, clear_cache=False):
        self.db_file_path = db_file_path
        self.cache_dir = OUT_FILES_DIRECTORY_CACHE
        self._ensure_db_exists()
        self.force_clear_cache = clear_cache
        if REPO_DIRECTORY is None:
            raise ValueError("REPO_DIRECTORY is not set")
        self.analyzed_repo_dir = REPO_DIRECTORY

    def _ensure_db_exists(self):
        """Ensure the database file exists, create it if it doesn't"""
        if not os.path.exists(self.db_file_path):
            raise FileNotFoundError(f"Database file not found at {self.db_file_path}")

    def _load_db(self):
        """Load the current database content"""
        with open(self.db_file_path, 'r') as f:
            return json.load(f)

    def _save_db(self, data, target_file):
        """Save data to the database file"""
        with open(target_file, 'w') as f:
            json.dump(data, f, indent=4)

    def _collect_summaries_per_target_file(self):
        """Collect all summaries into a single file"""
        summaries_per_file = defaultdict(list)
        cache_files = self._get_cached_summary_files()
        for cache_file in cache_files:
            with open(os.path.join(self.cache_dir, cache_file), 'r') as f:
                cache_data = json.load(f)
                file = cache_data['file_name']
                summaries_per_file[file].append(cache_data['llm_result'])
        return summaries_per_file

    def _get_cached_summary_files(self):
        hash_files = []
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                if self.hash_pattern.match(filename):
                    hash_files.append(filename)
        return hash_files
 
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
            key = file.removeprefix(self.analyzed_repo_dir)

            for summary in summaries:
                llm_model = LlmModel(summary['model']['known_name'], summary['model']['provider_name'], 
                                     summary['model']['aws_model_id'], summary['model']['aws_region'], 
                                     summary['model']['azure_deployment_name'], summary['model']['azure_region'], 
                                     summary['model']['delay_time'], summary['model']['langchain_ready'], 
                                     summary['model']['price_per_1000_input_tokens'], 
                                     summary['model']['price_per_1000_output_tokens'])
                llm_result = LlmResponse(summary['file_summary'], summary['total_tokens'], llm_model, summary['latency'])
                with open(file, 'r') as f:
                    file_content = f.read()
                number_of_lines = len(file_content.split('\n'))
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

    def _update_db(self, new_db_entries, dry_run=True):
        db = self._load_db()
        for file, new_db_entry in new_db_entries.items():
            if file in db.keys():
                self._merge_file_entries(db, file, new_db_entry)
            else:
                db[file] = new_db_entry
        target_file = self.db_file_path if not dry_run else os.path.join('/tmp', 'pytorch_DB.json')
        self._save_db(db, target_file)

    def _merge_file_entries(self, db, file, new_db_entry):
        file_data_in_db = db[file]
        db_entry_keys = set(file_data_in_db.keys())
        new_db_entry_keys = set(new_db_entry.keys())
        if db_entry_keys != new_db_entry_keys:
            keys_to_update = new_db_entry_keys - db_entry_keys
            if len(keys_to_update) >= 1:
                for key in keys_to_update:
                    file_data_in_db[key] = new_db_entry[key]
            else:
                print(f"Keys from new db entry {new_db_entry_keys} are already in the database for file {file}")
        return

    def from_cache_to_db(self, dry_run=True):
        summaries_per_file = self._collect_summaries_per_target_file()
        single_file_entries = self._collect_single_file_entries(summaries_per_file)
        db_dict_format = single_file_entries.to_db_dict()
        self._update_db(db_dict_format, dry_run)
if __name__ == "__main__":
    cache_manager = CacheManager(db_file_path='./results/pytorch_DB.json')
    cache_manager.from_cache_to_db(dry_run=False)