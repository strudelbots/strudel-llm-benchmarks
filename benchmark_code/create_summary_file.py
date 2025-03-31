import json
import os

from benchmark_code import JSON_FILES_DIRECTORY
from benchmark_code.llm_response import ModelSummaries, FileSummary

REPO_DIR = os.getenv("REPO_DIR")
if not REPO_DIR:
    raise ValueError("REPO_DIR environment variable is not set")

class CollectFileSummaryData:

    def __init__(self, summary_files:list):
        self.summary_files = summary_files


    def _collect_all_summaries_into_a_single_file(self):
        all_summaries = []
        for file in self.summary_files:
            with open(file) as f:
                file_data =json.load(f)
                summaries = FileSummary.schema().load(file_data, many=True)
                all_summaries.extend(summaries)
        return all_summaries

    def _merge_summaries_per_file(self, all_summaries_all_models):
        unique_files = {x.file_name for x in all_summaries_all_models}
        merged_results = {}
        for unique_file in unique_files:
            file_entry = {}
            for summary in all_summaries_all_models:
                if summary.file_name == unique_file:
                    file_entry[summary.model_name] = {"file_summary": summary.summary,
                                                      "total_tokens": summary.total_tokens,
                                                      "latency": summary.latency,}
            file_key = unique_file.replace(REPO_DIR, "")
            merged_results[file_key] = file_entry
        return merged_results


if __name__ == "__main__":
    root_dir = '/home/shai/Downloads'
    os.chdir(root_dir)
    all_data = CollectFileSummaryData(['03-31-2025__gpt-4o__-home-shai-make-developers-brighter.json',
                                       '03-31-2025__gpt-4__-home-shai-make-developers-brighter.json',
                                       '03-31-2025__gpt-35-turbo__-home-shai-make-developers-brighter.json'])

    all_summaries_all_models  =  all_data._collect_all_summaries_into_a_single_file()
    comparable_summaries = all_data._merge_summaries_per_file(all_summaries_all_models)
    with open(JSON_FILES_DIRECTORY+'/comparable_summaries.json', 'w') as f:
        json.dump(comparable_summaries, f, indent=4)
    