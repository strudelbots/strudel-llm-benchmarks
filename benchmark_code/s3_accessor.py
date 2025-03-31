import datetime
import glob
import json

import boto3

from benchmark_code import OUT_FILES_DIRECTORY, REPO_DIRECTORY, file_date_prefix




def store_results_for_model(model_name, bucket_name):
    all_data = []
    json_files = glob.glob(f'{OUT_FILES_DIRECTORY}/*.json')
    now = datetime.datetime.now()  # current date and time
    for file in json_files:
        if model_name in file and file_date_prefix+"single__" in file:
            with open(file, 'r') as infile:
                data = json.load(infile)
                all_data.append(data)
    all_data = sorted(all_data, key=lambda x: x["file_name"])
    if not all_data:
        return
    boto3.setup_default_session(profile_name='s3_accessor')
    s3 = boto3.client('s3')
    object_name = file_date_prefix +"summary__"+ model_name + '__' + REPO_DIRECTORY.replace('/', '-') + ".json"
    with open (f'{OUT_FILES_DIRECTORY}/{object_name}', 'w') as outfile:
        outfile.write(json.dumps(all_data, indent=4))
    print(f'Writing {object_name} to bucket={bucket_name}, full_path={OUT_FILES_DIRECTORY}/{object_name}')
    s3.put_object(Body=json.dumps(all_data, indent=4), Bucket=bucket_name, Key=object_name)
