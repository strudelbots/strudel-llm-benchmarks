import datetime
import os
REPO_DIRECTORY=os.getenv('REPO_DIR')
if not REPO_DIRECTORY:
    raise ValueError('REPO_DIR environment variable is not set')
OUT_FILES_DIRECTORY='/home/shai/tmp'
file_date_prefix=datetime.datetime.now().strftime("%m-%d-%Y__")