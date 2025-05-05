from benchmark_code.cache_manager import CacheManager
if __name__ == "__main__":
    cache_manager = CacheManager(db_file_path='./results/pytorch_DB.json')
    cache_manager.from_cache_to_db(dry_run=False)