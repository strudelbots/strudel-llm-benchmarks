import glob
from benchmark_code.utils import clean_outputs_dir, OUT_FILES_DIRECTORY
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def generate_matrix_for_single_file(file_data):
    enbeddings = []
    labels = []
    metadata = file_data['metadata']
    metadata['file_type'] = 'similarity_matrix'
    try:
        for model, summary in file_data.items():
            if model == 'metadata':
                continue
            enbeddings.append(summary['embedding_of_'+model][0])
            labels.append(model)
    except Exception as e:
        print(f'Error generating matrix for {metadata["input_file_name"]}: {e}')
        return None

    similarity_matrix = cosine_similarity(enbeddings)
    assert similarity_matrix.shape == (len(enbeddings), len(enbeddings)), "Similarity matrix should be a square matrix"
    all_values = similarity_matrix.flatten()
    for value in all_values:
        assert 0 <= round(value, 6) <= 1, f"All values in the similarity matrix should be between 0 and 1, but got {value}"
    similarity_matrix[np.tril_indices(similarity_matrix.shape[0], k=-1)] = 0
    labeled_matrix = {
        'labels': labels, 
        'similarity_matrix': similarity_matrix.tolist(),
        'metadata': metadata
    }
    return labeled_matrix

def create_heat_map(s_m):    # Create a heatmap
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(s_m['similarity_matrix'], annot=True, xticklabels= s_m['labels'], yticklabels= s_m['labels'],
                cmap='jet', center=0)
    plt.title('Similarity Matrix Heatmap')
    plt.tight_layout()
    fig_file = s_m['metadata']['input_file_name'].replace('/', '_')
    fig_file = f'{OUT_FILES_DIRECTORY}/{fig_file.replace('.py', '_similarity_matrix.png')}'
    plt.savefig(fig_file)
    plt.close()
if __name__ == "__main__":
    json_files = glob.glob(f'{OUT_FILES_DIRECTORY}/*.json')
    for file in json_files:
        with open(file, 'r') as f:
            file_data = json.load(f)
            if file_data.get('metadata', {}).get('file_type', 'NA') == 'embedding':
                s_m = generate_matrix_for_single_file(file_data)
                outfile_name = file.replace('.json', '_similarity_matrix.json')
                with open(outfile_name, 'w') as f:
                    json.dump(s_m, f)
                create_heat_map(s_m)
            else:
                continue
