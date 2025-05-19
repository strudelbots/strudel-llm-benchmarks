import glob
from math import floor
from benchmark_code import REPO_DIRECTORY, OUT_FILES_DIRECTORY, file_date_prefix
from benchmark_code.utils import clean_outputs_dir
from benchmark_code.accessors.embedding_accessor_langchain_HF import EmbeddingAccessorLangchainHF
import json
from collections import defaultdict
from datetime import datetime
def embedding_for_files(json_files, embedding_accessor, embedding_model_name):
    for file in json_files:
        with open(file, 'r') as f:
            file_data = json.load(f)
        embedding_for_single_file(embedding_accessor, file_data, embedding_model_name, generate_files=True)


def embedding_for_single_file(embedding_accessor, file_data, embedding_model_name, generate_files):
    for input_file, summaries in file_data.items():
        file_name = input_file.split('/')[-1]
        results_for_file_name = defaultdict(dict, {})
        results_for_file_name['metadata']['input_file_name'] = input_file
        results_for_file_name['metadata']['embedding_model_name'] = embedding_model_name
        results_for_file_name['metadata']['generation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_for_file_name['metadata']['file_type'] = 'embedding'
        for model, summary in summaries.items():
            summary_text = summary['file_summary']
            embeddings = embedding_accessor.get_embeddings(summary_text)
            results_for_file_name[model]['embedding'+'_of_'+model] = embeddings
            results_for_file_name[model]['summary_text'+'_of_'+model] = summary_text
        _generate_embedding_file(embedding_model_name, generate_files, file_name, results_for_file_name)

def _generate_embedding_file(embedding_model_name, generate_files, file_name, results_for_file_name):
    if generate_files:
        out_file = f'{OUT_FILES_DIRECTORY}/{file_name.replace(".py", "")}__{embedding_model_name}.json'
        print(f'Generating embeddings file: {out_file}')
        with open(out_file, 'w') as f:
            json.dump(results_for_file_name, f, indent=4)

if __name__ == "__main__":
    clean_outputs_dir()
    embedding_models = [('all-mpnet-base-v2', 'Hugging Face')]
    json_files = glob.glob(f'results/*.json')
    for model in embedding_models:
        embedding_accessor = EmbeddingAccessorLangchainHF(model[0])
        embedding_for_files(json_files, embedding_accessor, model[0])