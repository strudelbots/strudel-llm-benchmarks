from chart_generator import ChartGenerator
import json
import os
from benchmark_code import project_name
from collections import defaultdict
from benchmark_code.llm_model import AVAILABLE_MODELS
from math import floor
supported_models = [x.known_name for x in AVAILABLE_MODELS]
def default_model_stats():
    return {
        "tokens_used": [],
        "input_token_cost": 0.0,
        "output_token_cost": 0.0,
        "total_input_cost": 0.0,
        "total_output_cost": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "avg_file_cost": 0.0,
    }
def get_model_stats(model_data):
    
    model_stats = defaultdict(default_model_stats)
    for file_name, file_data in model_data.items():
        for model_name, m_data in file_data.items():
            if not model_name in supported_models:
                raise ValueError(f"{model_name=}  is not supported")
            model_stats[model_name]["tokens_used"].append(m_data["total_tokens"])
            if model_stats[model_name]["input_token_cost"] == 0.0:
                model_stats[model_name]["input_token_cost"] = m_data['model']["price_per_1000_input_tokens"] / 1000
            if model_stats[model_name]["output_token_cost"] == 0.0:
                model_stats[model_name]["output_token_cost"] = m_data['model']["price_per_1000_output_tokens"] / 1000
            assert model_name == m_data['model']['known_name']
    return model_stats


def _calculate_metric_per_model(model_stats):
    for model_name, stats in model_stats.items():
        assert model_name in supported_models
        stats["total_input_tokens"] = floor(sum(stats["tokens_used"]) * 0.9)
        stats["total_output_tokens"] = floor(sum(stats["tokens_used"]) * 0.1)
        stats["total_input_cost"] = stats["input_token_cost"] * stats["total_input_tokens"]
        stats["total_output_cost"] = stats["output_token_cost"] * stats["total_output_tokens"]
        stats["total_cost"] = stats["total_input_cost"] + stats["total_output_cost"]
        stats["avg_file_cost"] = stats["total_cost"] / len(stats["tokens_used"])
        print(f"--------------------------------")
        print(f"{model_name=}")
        print(f"{stats['total_input_tokens']=}")
        print(f"{stats['total_output_tokens']=}")
        print(f"{stats['total_input_cost']=}")
        print(f"{stats['total_output_cost']=}")
        print(f"{stats['total_cost']=}")
        print(f"{stats['avg_file_cost']=}")
if __name__ == "__main__":
    model_stats = defaultdict(dict)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, f'../../results/{project_name}_DB.json'), 'r') as f:
        db = json.load(f)
    model_stats = get_model_stats(db)
    _calculate_metric_per_model(model_stats)
    chart_data = [(x[0], x[1]["avg_file_cost"]) for x in model_stats.items()]
    chart_data.sort(key=lambda x: x[0])
    chart_generator = ChartGenerator()
    chart_generator.generate_bar_chart(chart_data,f"/tmp/{project_name}_costs_chart.png", 
                                       "Avg Analysis Cost for File", "Model Name", "Cost $")
