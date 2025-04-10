import glob
from math import floor
from benchmark_code import REPO_DIRECTORY, OUT_FILES_DIRECTORY, file_date_prefix
from benchmark_code.utils import clean_outputs_dir
from benchmark_code.embedding_accessor_langchain_HF import EmbeddingAccessorLangchainHF
import json
from collections import defaultdict

def embedding_for_files(json_files, embedding_accessor, embedding_model_name):
    for file in json_files:
        with open(file, 'r') as f:
            file_data = json.load(f)
        embedding_for_single_file(embedding_accessor, file_data, embedding_model_name, generate_files=False)


def embedding_for_single_file(embedding_accessor, file_data, embedding_model_name, generate_files):
    for input_file, summaries in file_data.items():
        file_name = input_file.split('/')[-1]
        results_for_file_name = defaultdict(dict, {})
        results_for_file_name['input_file_name'] = input_file
        for model, summary in summaries.items():
            summary_text = summary['file_summary']
            embeddings = embedding_accessor.get_embeddings(summary_text)
            results_for_file_name[model]['embedding'+'_of_'+model] = embeddings
            results_for_file_name[model]['summary_text'+'_of_'+model] = summary_text
        if generate_files:
            out_file = f'{OUT_FILES_DIRECTORY}/{file_name.replace(".py", "")}__{embedding_model_name}.json'
            with open(out_file, 'w') as f:
                json.dump(results_for_file_name, f, indent=4)

if __name__ == "__main__":
    clean_outputs_dir()
    embedding_models = [('all-mpnet-base-v2', 'Hugging Face')]
    json_files = glob.glob(f'results/*.json')
    for model in embedding_models:
        embedding_accessor = EmbeddingAccessorLangchainHF(model[0])
        embedding_for_files(json_files, embedding_accessor, model[0])