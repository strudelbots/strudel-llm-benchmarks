import glob
from benchmark_code.utils import clean_outputs_dir, OUT_FILES_DIRECTORY

import json

if __name__ == "__main__":
    json_files = glob.glob(f'{OUT_FILES_DIRECTORY}/*.json')
    for file in json_files:
        with open(file, 'r') as f:
            file_data = json.load(f)
            if file_data.get('file_type', 'NA') == 'embedding':
                print(f'file: {file} is embedding file')
            else:
                continue
