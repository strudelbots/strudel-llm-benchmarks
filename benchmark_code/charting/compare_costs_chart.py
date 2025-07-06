import os
from collections import defaultdict
from math import floor

from benchmark_code.llm_model import AVAILABLE_MODELS
from benchmark_code.utils import get_db
from chart_generator import ChartGenerator
from benchmark_code import project_name
exclude_models = []
debug = True
supported_models = [x.known_name for x in AVAILABLE_MODELS if x.known_name not in exclude_models]
def default_model_stats_costs():
    return {
        "tokens_used": [],
        "single_input_token_cost": None,
        "single_output_token_cost": None,
        "total_input_cost": 0.0,
        "total_output_cost": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "avg_file_cost": 0.0,
        
    }

def get_model_stats_costs(model_data):
    
    model_stats = defaultdict(default_model_stats_costs)
    for file_name, file_data in model_data.items():
        for model_name, m_data in file_data.items():
            if not model_name in supported_models and model_name not in exclude_models:
                raise ValueError(f"{model_name=}  is not supported")
            if model_name in exclude_models:
                continue
            model_stats[model_name]["tokens_used"].append(m_data["total_tokens"])
            if model_stats[model_name]["single_input_token_cost"] == None:
                model_stats[model_name]["single_input_token_cost"] = m_data['model']["price_per_1000_input_tokens"] / 1000
            if model_stats[model_name]["single_output_token_cost"] == None:
                model_stats[model_name]["single_output_token_cost"] = m_data['model']["price_per_1000_output_tokens"] / 1000
            assert model_name == m_data['model']['known_name']
    for model_name, stats in model_stats.items():
        if stats["single_input_token_cost"] == None or stats["single_output_token_cost"] == None:
            raise ValueError(f"{model_name=} has no valid cost data")
    return model_stats


def _calculate_metric_per_model_costs(model_stats):
    for model_name, stats in model_stats.items():
        if model_name in exclude_models:
            assert False, f"{model_name=} is excluded"
        if stats["single_input_token_cost"] == None or stats["single_output_token_cost"] == None:
            raise ValueError(f"{model_name=} has no valid cost data")
        stats["total_input_tokens"] = floor(sum(stats["tokens_used"]) * 0.9)
        stats["total_output_tokens"] = floor(sum(stats["tokens_used"]) * 0.1)
        stats["total_input_cost"] = stats["single_input_token_cost"] * stats["total_input_tokens"] 
        stats["total_output_cost"] = stats["single_output_token_cost"] * stats["total_output_tokens"]
        stats["total_cost"] = stats["total_input_cost"] + stats["total_output_cost"]
        stats["avg_file_cost"] = stats["total_cost"] / len(stats["tokens_used"])
        if debug:
            print(f"--------------------------------")
            print(f"{model_name=}")
            print(f"{stats['total_input_tokens']=}")
            print(f"{stats['total_output_tokens']=}")
            print(f"{stats['total_input_cost']=}")
            print(f"{stats['total_output_cost']=}")
            print(f"{stats['total_cost']=}")
            print(f"{stats['avg_file_cost']=}")
            print(f"{stats['single_input_token_cost']=:.10f}")
            print(f"{stats['single_output_token_cost']=:.10f}")
    for model_name in exclude_models:
        model_stats[model_name]["total_input_tokens"] = 0
        model_stats[model_name]["total_output_tokens"] = 0
        model_stats[model_name]["total_input_cost"] = 0
        model_stats[model_name]["total_output_cost"] = 0
        model_stats[model_name]["total_cost"] = 0
        model_stats[model_name]["avg_file_cost"] = 0
def _avg_tokens_per_file(model_stats):
    chart_data = []
    for model_name, stats in model_stats.items():
        if model_name in exclude_models:
            continue
        n_files = len(stats['tokens_used'])
        avg_tokens = sum(stats['tokens_used']) / n_files
        #print(f"{model_name=} {n_files=} {avg_tokens=}")
        chart_data.append((model_name, avg_tokens))
    chart_data.sort(key=lambda x: x[0])
    chart_generator = ChartGenerator()
    chart_generator.generate_bar_chart(chart_data,f"/tmp/{project_name}_tokens_per_file_chart.png", 
                                       "Avg. Tokens Per File", "Model Name", "Tokens", annotation=0)

def _avg_cost_per_file(model_stats):
    chart_data = [(x[0][:12], x[1]["avg_file_cost"]) for x in model_stats.items()]
    
    chart_data.sort(key=lambda x: x[0])
    chart_generator = ChartGenerator(x_ticks_rotation=60)
    chart_generator.generate_bar_chart(chart_data,f"/tmp/{project_name}_costs_chart.png", 
                            "Average Summarization Cost per File", "Model Name", "Cost $")



if __name__ == "__main__":
    model_stats = defaultdict(dict)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db = get_db()
    model_stats = get_model_stats_costs(db)
    _calculate_metric_per_model_costs(model_stats)
    _avg_cost_per_file(model_stats)
    _avg_tokens_per_file(model_stats)