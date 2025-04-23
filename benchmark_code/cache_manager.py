import json
import os
import shutil
from pathlib import Path
from benchmark_code import OUT_FILES_DIRECTORY_CACHE

class CacheManager:
    def __init__(self, db_file_path):
        self.db_file_path = db_file_path
        self.cache_dir = OUT_FILES_DIRECTORY_CACHE
        self._ensure_db_exists()

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

    def process_cache_files(self):
        """Process all JSON files in the cache directory and update the database"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        if not cache_files:
            return

        db_data = self._load_db()
        
        for cache_file in cache_files:
            cache_path = os.path.join(self.cache_dir, cache_file)
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                
                # Update database with cache data
                # Assuming cache data is a dictionary that should be merged into the database
                if isinstance(cache_data, dict):
                    db_data.update(cache_data)
                
                # Delete the processed cache file
                os.remove(cache_path)
                
            except Exception as e:
                print(f"Error processing cache file {cache_file}: {str(e)}")
                continue

        # Save the updated database
        self._save_db(db_data)

    def clear_cache(self):
        """Clear all files from the cache directory"""
        for file in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file}: {str(e)}") 

if __name__ == "__main__":
    cache_manager = CacheManager(db_file_path='./results/pytorch_DB.json')
    cache_manager.process_cache_files()
    cache_manager.clear_cache()