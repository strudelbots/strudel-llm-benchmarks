import os
from benchmark_code.similarity_matrix_generator import SimilarityMatrixGenerator
from benchmark_code.charting.chart_generator import ChartGenerator
from benchmark_code.embedding_generator import EmbeddingGenerator
from benchmark_code.similatiry_analyzer import SimilarityAnalyzer
from benchmark_code.llm_model import AVAILABLE_MODELS
if __name__ == '__main__':
    embedding_db = EmbeddingGenerator('all-mpnet-base-v2')
    embedding_db.build_db()
    exclude_models = ['titan_premier', 'mistral-small']
    #exclude_models = [x.known_name for x in AVAILABLE_MODELS if 'gpt' not in x.known_name and \
    #                  x.known_name != 'nova-pro-v1']
    similarity_matrix_generator = SimilarityMatrixGenerator(embedding_db,
                                                            exclude_models=exclude_models) 
    print("--------------------------------")                                                        
    print(f'Cache directory for similarity matrix: {similarity_matrix_generator.cache_directory}')
    print(f'Output file for similarity matrix: {similarity_matrix_generator.out_file}')
    print("--------------------------------")                                                        
    similarity_matrix_generator.build_db(write_to_file=False)
    chart_generator = ChartGenerator()
    upper_triangle_average = 1.0
    data_frames = similarity_matrix_generator.get_dataframes()
    similarity_analyzer = SimilarityAnalyzer(data_frames.values())
    while upper_triangle_average > 0.27:
        random_similarity_matrix = similarity_matrix_generator.get_random_similarity_matrix()
        upper_triangle_average = similarity_analyzer.get_upper_triangle_average(random_similarity_matrix)
        print(f'Upper triangle average: {upper_triangle_average:.2f}')
    title = f'Random files similarity matrix (average: {upper_triangle_average:.2f})'
    chart_generator.create_heat_map(random_similarity_matrix, 
                                    f'/tmp/random_similarity_matrix.png',
                                    title, mask_lower_triangle=False)
    average_similarity_matrix = similarity_analyzer.get_avarage_similary_matrix()
    upper_triangle_average = similarity_analyzer.get_upper_triangle_average(average_similarity_matrix)
    title = f'Average similarity matrix (average: {upper_triangle_average:.2f})'
    chart_generator.create_heat_map(average_similarity_matrix, 
                                    f'/tmp/average_similarity_matrix.png',
                                    title, mask_lower_triangle=False)
    assert len(data_frames) >= 65
    for file_name, similarity_df in data_frames.items():
        base_name = os.path.basename(file_name)
        base_name = base_name.replace('.py', '')
        average = similarity_analyzer.get_upper_triangle_average(similarity_df)
        print(f'{base_name}.py similarity matrix (average: {average:.2f})')
        title = f'{base_name}.py similarity matrix (average: {average:.2f})'
        chart_generator.create_heat_map(similarity_df, f'/tmp/{base_name}_{average:.2f}.png', title)