import os
from collections import defaultdict
from math import floor
from pprint import pprint
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
    return model_stats


def _calculate_metric_per_model(model_stats):
    for model_name, stats in model_stats.items():
        assert model_name in supported_models
        stats["average_latency_ms"] = sum(stats["latency_ms"]) / len(stats["latency_ms"])
        if debug:
            print(f"--------------------------------")
            print(f"{model_name=}")
            print(f"{stats['average_latency_ms']=}")


def _calculate_quality_score(model_stats, exclude_models):
    model_latencies = {x[0]: x[1]["average_latency_ms"] for x in model_stats.items() if x[0] not in exclude_models}
    #min_latency = min(model_latencies.values())
    #model_latencies = {x: y/min_latency  for x,y in model_latencies.items()}
    model_costs = {x[0]: x[1]["avg_file_cost"] for x in model_stats.items() if x[0] not in exclude_models}
    #min_cost = min(model_costs.values())
    #model_costs = {x: y/min_cost  for x,y in model_costs.items()}
    for model_name in [x for x in model_stats.keys() if x not in exclude_models]:
        model_stats[model_name]["quality_score"] = (model_costs[model_name], 
                                                    model_latencies[model_name])
def _chart_quality_score(model_stats, file_ext, exclude_models, x_position, y_position):
    chart_data = {x:y  for x,y in model_stats.items() if x not in exclude_models}
    labels = [x for x in chart_data]
    latencies = [chart_data[x]['quality_score'][1] for x in labels]
    costs = [chart_data[x]['quality_score'][0] for x in labels]
    chart_generator = ChartGenerator(x_ticks_rotation=60, add_annotations=True)
    labels = [x.replace('-turbo', '') for x in chart_data]
    chart_generator.create_scatter_plot(latencies, costs, labels, 
                                        f"/tmp/{project_name}_latency_cost_score_chart_{file_ext}.png",
                                        "Avg. Latency (Sec)", "Avg. Cost per File ($)", 
                                        "Latency vs. Cost Comparison", x_position=x_position, y_position=y_position)
def merge_nested_dicts(d1, d2):
    merged = dict(d1)  # make a shallow copy of d1
    for key, value in d2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_nested_dicts(merged[key], value)
        else:
            # Overwrite or add
            merged[key] = value
    return merged
if __name__ == "__main__":
    #model_stats = defaultdict(dict)
    exclude_models = ["titan_premier", "mistral-small", "gpt-4.5"]
    #exclude_models = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db = get_db()
    latency_model_stats = get_model_stats(db)
    _calculate_metric_per_model(latency_model_stats)
    from compare_costs_chart import get_model_stats_costs, _calculate_metric_per_model_costs
    costs_model_stats = get_model_stats_costs(db)
    _calculate_metric_per_model_costs(costs_model_stats)    
    for model_name, stats in costs_model_stats.items():
        print(f"{model_name=}, {stats['avg_file_cost']=}")
    merged = merge_nested_dicts(latency_model_stats, costs_model_stats)
    _calculate_quality_score(merged, exclude_models=[])
    _chart_quality_score(merged, file_ext="1", exclude_models=[], x_position=0.3, y_position=0.01)
    _calculate_quality_score(merged, exclude_models=exclude_models)
    _chart_quality_score(merged, file_ext="2", exclude_models=exclude_models, x_position=0.3, 
                         y_position=0.002)
    exclude_models += [ "Claude3.5", "gpt-4", "Llama3.1", "Claude3.7", "gemini-2.5"]
    _calculate_quality_score(merged, exclude_models=exclude_models)
    _chart_quality_score(merged, file_ext="3", exclude_models=exclude_models, 
                         x_position=0.1, y_position=0.0002)