import datetime
import os
from pathlib import Path

REPO_DIRECTORY=os.getenv('REPO_DIRECTORY') # points to the root directory of repository you want to analyze
if REPO_DIRECTORY is None:
    raise ValueError('you need to set the REPO_DIRECTORY environment ' \
    'variable to point to the root directory of the repository you want to analyze')
OUT_FILES_DIRECTORY=f'{Path.home()}/tmp' # where all files will be writen, feel free to change
OUT_FILES_DIRECTORY_CACHE=f'{Path.home()}/cache'

file_date_prefix=datetime.datetime.now().strftime("%m-%Y__") # do not change unless you are a developer
project_name = 'pytorch'
file_date_prefix=f'{project_name}__{file_date_prefix}'

DEFAULT_TEMPERATURE=0.3
DEFAULT_TOP_P=0.2
DEFAULT_TOP_K=10
DEFAULT_MAX_TOKENS=3072 # should go into a specific model configuration
DEFAULT_TIMEOUT=30