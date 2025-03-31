import datetime
import glob
import json
import os

import boto3

from benchmark_code import OUT_FILES_DIRECTORY

REPO_DIR = os.getenv('REPO_DIR')
if REPO_DIR is None:
    raise ValueError('REPO_DIR environment variable is not set')

def upload_results_to_s3(model_name, bucket_name):
    all_data = []
    json_files = glob.glob(f'{OUT_FILES_DIRECTORY}/*.json')
    now = datetime.datetime.now()  # current date and time
    for file in json_files:
        if model_name in file and now.strftime("%m-%d-%Y__single__") in file:
            with open(file, 'r') as infile:
                data = json.load(infile)
                all_data.append(data)
    all_data = sorted(all_data, key=lambda x: x["file_name"])
    if not all_data:
        return
    boto3.setup_default_session(profile_name='s3_accessor')
    s3 = boto3.client('s3')
    object_name = now.strftime("%m-%d-%Y__summary__") + model_name + '__' + REPO_DIR.replace('/', '-') + ".json"
    with open (f'{OUT_FILES_DIRECTORY}/{object_name}', 'w') as outfile:
        outfile.write(json.dumps(all_data, indent=4))
    print(f'Writing {object_name} to {bucket_name}')
    s3.put_object(Body=json.dumps(all_data, indent=4), Bucket=bucket_name, Key=object_name)
