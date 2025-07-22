import os
import json
from benchmark_code.embedding_generator import EmbeddingGenerator
from benchmark_code.utils import clean_outputs_dir

def generate_embeddings_for_DB_json():
    clean_outputs_dir()
    embedding_models = ['all-mpnet-base-v2']
    for model in embedding_models:
        embedding_generator = EmbeddingGenerator(model)
        print(f'Generating embeddings for {model}')
        print(f'Cache directory: {embedding_generator.cache_directory}')
        print(f'Output file: {embedding_generator.out_file}')
        embedding_generator._generate_embeddings()
    
def generate_embedding_for_single_file_results(results_json_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_json_file = os.path.join(current_dir, 'results', results_json_file)
    if not os.path.exists(results_json_file):
        raise FileNotFoundError(f'File {results_json_file} not found')
    with open(results_json_file, 'r') as f:
        single_file_results = json.load(f)
    embedding_generator = EmbeddingGenerator('all-mpnet-base-v2')
    all_embeddings = {}
    for result in single_file_results:
        file_summary = result['llm_result']['file_summary']
        file_summary_embedding = embedding_generator.embedding_accessor.get_embeddings(file_summary)
        assert len(file_summary_embedding) == 1
        file_summary_embedding = file_summary_embedding[0]
        assert len(file_summary_embedding) == 768
        uuid = result['llm_result']['uuid']
        print(f'Generating embeddings for {uuid}')
        all_embeddings[uuid] = file_summary_embedding
    embedding_db_path = os.path.join(current_dir, 'results', 'all_embeddings_db.json')  
    with open(embedding_db_path, 'r') as f:
        embedding_db = json.load(f)
    embedding_db.update(all_embeddings)
    with open(embedding_db_path, 'w') as f:
        json.dump(embedding_db, f)


if __name__ == "__main__":
    #generate_embeddings_for_DB_json()
    generate_embedding_for_single_file_results('pytorch__torch_cuda_graphs.json')