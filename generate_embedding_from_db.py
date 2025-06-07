from benchmark_code.embedding_generator import EmbeddingGenerator
from benchmark_code.utils import clean_outputs_dir
if __name__ == "__main__":
    clean_outputs_dir()
    embedding_models = ['all-mpnet-base-v2']
    for model in embedding_models:
        embedding_generator = EmbeddingGenerator(model)
        print(f'Generating embeddings for {model}')
        print(f'Cache directory: {embedding_generator.cache_directory}')
        print(f'Output file: {embedding_generator.out_file}')
        embedding_generator.generate_embeddings()
    