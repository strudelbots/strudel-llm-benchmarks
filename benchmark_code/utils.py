import os
import json
from benchmark_code import OUT_FILES_DIRECTORY, project_name


def clean_outputs_dir(exclude_str=None):
    for file in os.listdir(OUT_FILES_DIRECTORY):
        if exclude_str and exclude_str in  file:
            continue
        if file.endswith(".json") or file.endswith(".png"):
            os.remove(os.path.join(OUT_FILES_DIRECTORY, file))

def get_db():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, f'../results/{project_name}_DB.json'), 'r') as f:
        db = json.load(f)
    return db

