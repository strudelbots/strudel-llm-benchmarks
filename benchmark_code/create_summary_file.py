import datetime
import json

from benchmark_code import REPO_DIRECTORY, file_date_prefix
from benchmark_code.llm_response import FileSummary

class CollectFileSummaryData:

    def __init__(self, summary_files:list):
        self.summary_files = summary_files
        self.file_prefix = file_date_prefix+"summary__"


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
            file_key = unique_file.replace(REPO_DIRECTORY, "")
            merged_results[file_key] = file_entry
        return merged_results

    def merged_summaries(self):
        all_summaries_all_models = self._collect_all_summaries_into_a_single_file()
        comparable_summaries = self._merge_summaries_per_file(all_summaries_all_models)
        return comparable_summaries
    