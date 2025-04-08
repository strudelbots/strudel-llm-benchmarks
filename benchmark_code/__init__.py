import datetime
import os
REPO_DIRECTORY=os.getenv('REPO_DIR') # points to the root directory of repository you want to analyze
if not REPO_DIRECTORY:
    raise ValueError('REPO_DIR environment variable is not set')

OUT_FILES_DIRECTORY='/home/shai/tmp' # where all files will be writen, feel free to change
file_date_prefix=datetime.datetime.now().strftime("%m-%d-%Y__") # do not change unless you are a developer
DEFAULT_TEMPERATURE=0.3
DEFAULT_TOP_P=0.2
DEFAULT_TOP_K=10
DEFAULT_MAX_TOKENS=500
DEFAULT_TIMEOUT=30