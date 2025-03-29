import os

from code import JSON_FILES_DIRECTORY


def clean_tmp_dir():
    for file in os.listdir(JSON_FILES_DIRECTORY):
        if file.endswith(".json"):
            os.remove(os.path.join(JSON_FILES_DIRECTORY, file))
