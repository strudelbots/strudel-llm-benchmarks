import os

from benchmark_code import OUT_FILES_DIRECTORY


def clean_outputs_dir(exclude_str=None):
    for file in os.listdir(OUT_FILES_DIRECTORY):
        if exclude_str and exclude_str in  file:
            continue
        if file.endswith(".json"):
            os.remove(os.path.join(OUT_FILES_DIRECTORY, file))
