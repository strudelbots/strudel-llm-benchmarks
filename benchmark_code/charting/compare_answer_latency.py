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
    }

def get_model_stats(model_data):
    
    model_stats = defaultdict(default_model_stats)
    for file_name, file_data in model_data.items():
        for model_name, m_data in file_data.items():
            if not model_name in supported_models:
                raise ValueError(f"{model_name=}  is not supported")
            model_stats[model_name]["latency_ms"].append(m_data["latency"])
    return model_stats


def _calculate_metric_per_model(model_stats):
    for model_name, stats in model_stats.items():
        assert model_name in supported_models
        stats["average_latency_ms"] = sum(stats["latency_ms"]) / len(stats["latency_ms"])
        if debug:
            print(f"--------------------------------")
            print(f"{model_name=}")
            print(f"{stats['average_latency_ms']=}")

def _avg_latency_per_file(model_stats):
    chart_data = [(x[0][:12], x[1]["average_latency_ms"]) for x in model_stats.items()]
    chart_data.sort(key=lambda x: x[0])
    chart_generator = ChartGenerator(x_ticks_rotation=60)
    chart_generator.generate_bar_chart(chart_data,f"/tmp/{project_name}_latency_chart.png", 
                            "Average Summarization Latency per File", "Model Name", "Latency (sec)")


if __name__ == "__main__":
    model_stats = defaultdict(dict)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db = get_db()
    model_stats = get_model_stats(db)
    _calculate_metric_per_model(model_stats)
    _avg_latency_per_file(model_stats)