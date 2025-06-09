import os
from collections import defaultdict
from math import floor

from benchmark_code.llm_model import AVAILABLE_MODELS
from benchmark_code.utils import get_db
from chart_generator import ChartGenerator
from benchmark_code import project_name

debug = True
supported_models = [x.known_name for x in AVAILABLE_MODELS]
def default_model_stats():
    return {
        "latency_ms": [],
        "average_latency_ms": 0.0,
        "words_per_summary": [],
         "avg_words_per_summary": float('0.0'),
        "chattiness_score": 0,
        "latency_score": 0,
        "quality_score": (-1,-1)
    }

def get_model_stats(model_data):
    
    model_stats = defaultdict(default_model_stats)
    for file_name, file_data in model_data.items():
        for model_name, m_data in file_data.items():
            if not model_name in supported_models:
                raise ValueError(f"{model_name=}  is not supported")
            model_stats[model_name]["latency_ms"].append(m_data["latency"])
            number_of_words = len(m_data["file_summary"].split())
            model_stats[model_name]["words_per_summary"].append(number_of_words)
    return model_stats


def _calculate_metric_per_model(model_stats):
    for model_name, stats in model_stats.items():
        assert model_name in supported_models
        stats["average_latency_ms"] = sum(stats["latency_ms"]) / len(stats["latency_ms"])
        stats["avg_words_per_summary"] = sum(stats["words_per_summary"]) / len(stats["words_per_summary"])
        if debug:
            print(f"--------------------------------")
            print(f"{model_name=}")
            print(f"{stats['average_latency_ms']=}")


def _calculate_quality_score(model_stats):
    model_latencies = {x[0]: x[1]["average_latency_ms"] for x in model_stats.items()}
    min_latency = min(model_latencies.values())
    model_latencies = {x: y/min_latency  for x,y in model_latencies.items()}
    model_chattiness = {x[0]: x[1]["avg_words_per_summary"] for x in model_stats.items()}
    min_chattiness = min(model_chattiness.values())
    model_chattiness = {x: y/min_chattiness  for x,y in model_chattiness.items()}
    for model_name in model_stats.keys():
        model_stats[model_name]["quality_score"] = (model_chattiness[model_name], 
                                                    model_latencies[model_name])
def _chart_quality_score(model_stats, exclude_models=[]):
    chart_data = {x:y  for x,y in model_stats.items() if x not in exclude_models}
    labels = [x for x in chart_data]
    latencies = [chart_data[x]['quality_score'][1] for x in labels]
    chattiness = [chart_data[x]['quality_score'][0] for x in labels]
    chart_generator = ChartGenerator(x_ticks_rotation=60, add_annotations=True)
    labels = [x.replace('-turbo', '') for x in chart_data]
    chart_generator.create_scatter_plot(latencies, chattiness, labels, 
                                        f"/tmp/{project_name}_quality_score_chart.png",
                                        "Relative Latency", "Relative Conciseness", 
                                        "Latency vs. Conciseness Comparison")
if __name__ == "__main__":
    model_stats = defaultdict(dict)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db = get_db()
    model_stats = get_model_stats(db)
    _calculate_metric_per_model(model_stats)
    _calculate_quality_score(model_stats)
    _chart_quality_score(model_stats, exclude_models=["titan_premier", 'mistral-small'])