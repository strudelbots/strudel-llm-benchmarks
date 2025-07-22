import json
import os
from benchmark_code.similarity_matrix_generator import SimilarityMatrixGenerator
from benchmark_code.charting.chart_generator import ChartGenerator
from benchmark_code.embedding_generator import EmbeddingGenerator
from benchmark_code.similatiry_analyzer import SimilarityAnalyzer
from benchmark_code.llm_model import AVAILABLE_MODELS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_embedding_for_file_and_model(file_db, embedding_db_for_single_model, model_known_name):
    embedding_for_model = []
    for file_result in file_db:
        current_model = file_result['llm_result']['model']['known_name']
        if current_model != model_known_name:
            continue
        file_summary = file_result['llm_result']['file_summary']
        uuid = file_result['llm_result']['uuid']
        embedding = embedding_db_for_single_model[uuid]
        assert len(embedding) == 768
        embedding_for_model.append(embedding)
    return embedding_for_model

def generate_similarity_matrix_for_file_and_model(file_name, model_known_name):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_path = os.path.join(current_dir, 'results', 'all_embeddings_db.json')
    with open(embedding_path, 'r') as f:
        embedding_db_for_single_model = json.load(f)
    file_db_path = os.path.join(current_dir, 'results', file_name)
    with open(file_db_path, 'r') as f:
        file_db = json.load(f)
    embedding_for_model = get_embedding_for_file_and_model(file_db, embedding_db_for_single_model, model_known_name)
    print(f'{model_known_name} has {len(embedding_for_model)} embeddings')
    if len(embedding_for_model) == 0:
        print(f'{model_known_name} has no embeddings for {file_name}')
        return
    embedding_matrix = np.array(embedding_for_model)
    first_embedding = embedding_matrix[0].reshape(1, -1)
    similarities = cosine_similarity(first_embedding, embedding_matrix)
    similarities_array = similarities.flatten()
    for i, score in enumerate(similarities_array):
        print(f"Similarity with embedding[{i}]: {score:.4f}")
    
    chart_generator = ChartGenerator()
    chart_generator.create_heat_map(similarities, f'/tmp/{model_known_name}_{file_name}_similarity.png', 
                                    f'file: {file_name} model: {model_known_name}', mask_lower_triangle=False, 
                                    x_label='run #', y_label='')

if __name__ == '__main__':
    models = [ 'nova-pro-v1', 'nova-lite-v1', 'gpt-4o', 'gpt-4.1', 'Claude3.7', 'Claude3.5', 'cohere-v1']
    files = ['pytorch__torch_cuda_graphs.json', 'pytorch__torch_onnx__internal_exporter__tensors.json']
    for model_known_name in models:
        for file_name in files:
            generate_similarity_matrix_for_file_and_model(file_name, model_known_name)





#                                                             exclude_models=exclude_models) 
#     print("--------------------------------")                                                        
#     print(f'Cache directory for similarity matrix: {similarity_matrix_generator.cache_directory}')
#     print(f'Output file for similarity matrix: {similarity_matrix_generator.out_file}')
#     print("--------------------------------")                                                        
#     similarity_matrix_generator.build_db(write_to_file=False)
#     chart_generator = ChartGenerator()
#     upper_triangle_average = 1.0
#     data_frames = similarity_matrix_generator.get_dataframes()
#     similarity_analyzer = SimilarityAnalyzer(data_frames.values())
#     while upper_triangle_average > 0.27:
#         random_similarity_matrix = similarity_matrix_generator.get_random_similarity_matrix()
#         upper_triangle_average = similarity_analyzer.get_upper_triangle_average(random_similarity_matrix)
#         print(f'Upper triangle average: {upper_triangle_average:.2f}')
#     title = f'Random files similarity matrix (average: {upper_triangle_average:.2f})'
#     chart_generator.create_heat_map(random_similarity_matrix, 
#                                     f'/tmp/random_similarity_matrix.png',
#                                     title, mask_lower_triangle=False)
#     average_similarity_matrix = similarity_analyzer.get_avarage_similary_matrix()
#     upper_triangle_average = similarity_analyzer.get_upper_triangle_average(average_similarity_matrix)
#     title = f'Average similarity matrix (average: {upper_triangle_average:.2f})'
#     chart_generator.create_heat_map(average_similarity_matrix, 
#                                     f'/tmp/average_similarity_matrix.png',
#                                     title, mask_lower_triangle=False)
#     assert len(data_frames) >= 65
#     for file_name, similarity_df in data_frames.items():
#         base_name = os.path.basename(file_name)
#         base_name = base_name.replace('.py', '')
#         average = similarity_analyzer.get_upper_triangle_average(similarity_df)
#         print(f'{base_name}.py similarity matrix (average: {average:.2f})')
#         title = f'{base_name}.py similarity matrix (average: {average:.2f})'
#         chart_generator.create_heat_map(similarity_df, f'/tmp/{base_name}_{average:.2f}.png', title)