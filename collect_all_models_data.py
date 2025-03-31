import glob
import json
import os

from benchmark_code import OUT_FILES_DIRECTORY
from benchmark_code.create_summary_file import CollectFileSummaryData

if __name__ == "__main__":
    os.chdir(OUT_FILES_DIRECTORY)
    summary_files = glob.glob("*__summary__*.json")
    all_data = CollectFileSummaryData(summary_files)

    all_summaries_all_models  =  all_data._collect_all_summaries_into_a_single_file()
    comparable_summaries = all_data._merge_summaries_per_file(all_summaries_all_models)
    with open(OUT_FILES_DIRECTORY+'/comparable_summaries.json', 'w') as f:
        json.dump(comparable_summaries, f, indent=4)
    