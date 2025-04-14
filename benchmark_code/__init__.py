import datetime
import os
from pathlib import Path

REPO_DIRECTORY=os.getenv('REPO_DIR') # points to the root directory of repository you want to analyze
OUT_FILES_DIRECTORY=f'{Path.home()}/tmp' # where all files will be writen, feel free to change
OUT_FILES_DIRECTORY_CACHE=f'{Path.home()}/cache'

file_date_prefix=datetime.datetime.now().strftime("%m-%Y__") # do not change unless you are a developer
DEFAULT_TEMPERATURE=0.3
DEFAULT_TOP_P=0.2
DEFAULT_TOP_K=10
DEFAULT_MAX_TOKENS=500
DEFAULT_TIMEOUT=30