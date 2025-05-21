import os
from collections import defaultdict
from pprint import pprint
from math import floor
from dataclasses import dataclass
from benchmark_code.llm_model import AVAILABLE_MODELS
from benchmark_code.utils import get_db
from chart_generator import ChartGenerator
from benchmark_code import project_name

debug = True
supported_models = [x.known_name for x in AVAILABLE_MODELS]
        

def default_model_stats():
    return {
        "avg_words_per_summary": float('0.0'),
        "avg_words_per_file_line": float('0.0'),
        "max_words_per_file_line": float('0.0'),
    }

def get_model_stats(model_data):
    
    model_stats = defaultdict(lambda: {"words_per_summary": [], "avg_words_per_file_line": []})
    for file_name, file_sumarization_data in model_data.items():
        for model_name, model_sumarization_data in file_sumarization_data.items():
            if not model_name in supported_models:
                raise ValueError(f"{model_name=}  is not supported")
            input_file_summary = model_sumarization_data.get("file_summary", None)
            if input_file_summary is None:
                raise ValueError(f"{file_name=} has no input file summary")
            number_of_words = len(input_file_summary.split())
            assert number_of_words > 0, f"{file_name=} has no words in input file summary"
            words_per_line = number_of_words / model_sumarization_data["number_of_lines"]
            model_stats[model_name]["words_per_summary"].append(number_of_words)
            model_stats[model_name]["avg_words_per_file_line"].append(words_per_line)
    return model_stats


def _calculate_metric_per_model(data_for_models):
    model_stats = defaultdict(default_model_stats)
    for model_name, stats in data_for_models.items():
        assert model_name in supported_models
        avg_words_per_summary = sum(stats["words_per_summary"]) / len(stats["words_per_summary"])
        model_stats[model_name]["avg_words_per_summary"] = avg_words_per_summary
        avg_words_per_file_line = sum(stats["avg_words_per_file_line"]) / len(stats["avg_words_per_file_line"])
        model_stats[model_name]["avg_words_per_file_line"] = avg_words_per_file_line
        max_words_per_file_line = max(stats["avg_words_per_file_line"])
        model_stats[model_name]["max_words_per_file_line"] = max_words_per_file_line
        if debug:
            print(f"*** {model_name} avg words per summary: {model_stats[model_name]['avg_words_per_summary']}")
            print(f"*** {model_name} avg words per file line: {model_stats[model_name]['avg_words_per_file_line']}")
            print(f"*** {model_name} max words per file line: {model_stats[model_name]['max_words_per_file_line']}")
    return model_stats

def _avg_words_per_summary(model_stats):
    chart_data = [(x[0], x[1]["avg_words_per_summary"]) for x in model_stats.items()]
    chart_data.sort(key=lambda x: x[0])
    chart_generator = ChartGenerator()
    chart_generator.generate_bar_chart(chart_data,f"/tmp/{project_name}_avg_words_per_summary.png", 
                            "Average Words per Summary", "Model Name", "Words (avg)")
    
def _avg_words_per_file_line(model_stats):
    chart_data = [(x[0], x[1]["avg_words_per_file_line"]) for x in model_stats.items()]
    chart_data.sort(key=lambda x: x[0])
    chart_generator = ChartGenerator()
    chart_generator.generate_bar_chart(chart_data,f"/tmp/{project_name}_avg_words_per_file_line.png", 
                        "Average Words per File Line", "Model Name", "Words (avg)")

def _max_words_per_file_line(model_stats):
    chart_data = [(x[0], x[1]["max_words_per_file_line"]) for x in model_stats.items()]
    chart_data.sort(key=lambda x: x[0])
    chart_generator = ChartGenerator()
    chart_generator.generate_bar_chart(chart_data,f"/tmp/{project_name}_max_words_per_file_line.png", 
                        "Max Words per File Line", "Model Name", "Words (max)")

def _find_top_verbosity_for_model(db):
    results = []
    top_verbosity_files = []  
    for file_name, all_model_results in db.items():
        current_file_total_verbosity = 0
        for model_name, model_results in all_model_results.items():
            words_in_summary = len(model_results["file_summary"].split())
            number_of_lines = model_results["number_of_lines"]
            verbosity = words_in_summary / number_of_lines
            #print(f"*** {file_name} {model_name} {words_in_summary} {number_of_lines}, {verbosity}")
            results.append((file_name, model_name, verbosity))
            current_file_total_verbosity += verbosity
        models_in_entry = len(all_model_results.keys())
        file_avg_verbosity = current_file_total_verbosity / models_in_entry
        top_verbosity_files.append((file_name, file_avg_verbosity))
    
    top_verbosity_files.sort(key=lambda x: x[1], reverse=True)
    with open(f"/tmp/{project_name}_files_sorted_by_verbosity.csv", "w") as f:
        for file_name, file_avg_verbosity in top_verbosity_files:
            f.write(f"{file_name}, {file_avg_verbosity}\n")
    results.sort(key=lambda x: x[2], reverse=True)
    with open(f"/tmp/{project_name}_top_verbosity.csv", "w") as f:
        for file_name, model_name, verbosity in results:
            f.write(f"{file_name}, {model_name}, {verbosity}, {words_in_summary}, {number_of_lines}\n")



if __name__ == "__main__":
    model_stats = defaultdict(default_model_stats)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db = get_db()
    data_for_models = get_model_stats(db)
    model_stats = _calculate_metric_per_model(data_for_models)
    _avg_words_per_summary(model_stats)
    _avg_words_per_file_line(model_stats)
    _max_words_per_file_line(model_stats)
    _find_top_verbosity_for_model(db)