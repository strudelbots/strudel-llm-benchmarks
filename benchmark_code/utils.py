import os
import json
from benchmark_code import OUT_FILES_DIRECTORY, project_name, REPO_DIRECTORY


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

def update_results_db():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, f'../results/{project_name}_DB.json'), 'r') as f:
        db = json.load(f)
    for index, (key, value) in enumerate(db.items()):
        print(f'{index}: {key}')
        for index_2, (key_2, value_2) in enumerate(value.items()):
            print(f'    {index_2}: {key_2}')
            assert 'number_of_lines' in value_2
            full_path = os.path.join(REPO_DIRECTORY, key.removeprefix('/'))
            with open(full_path, 'r') as text_file:
                python_string = text_file.read()
                number_of_lines = len(python_string.split('\n'))
                current_lines= db[key][key_2]['number_of_lines']
                if current_lines != number_of_lines:
                    assert current_lines == -1
                    db[key][key_2]['number_of_lines'] = number_of_lines
                else:
                    assert current_lines == number_of_lines
    with open(os.path.join(current_dir, f'../results/{project_name}_DB.json'), 'w') as f:
        json.dump(db, f, indent=4)

        
    #with open(os.path.join(current_dir, f'../results/{project_name}_DB.json'), 'w') as f:
    #    json.dump(db, f)
if __name__ == "__main__":
    update_results_db()