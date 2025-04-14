from dataclasses import dataclass, field
from typing import List
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class LlmModel():
    known_name: str
    aws_model_id: str
    aws_region: str 
    azure_model_id: str = ''
    azure_region: str = ''
    delay_time: int = 3 
    langchain_ready: bool = False
    price_per_1000_input_tokens: float = 0.0
    price_per_1000_output_tokens: float = 0.0

AVAILABLE_MODELS = [
    LlmModel(known_name="Claude3.5", aws_model_id="eu.anthropic.claude-3-5-sonnet-20240620-v1:0", 
             aws_region="eu-central-1", langchain_ready=True,
             price_per_1000_input_tokens=0.003, price_per_1000_output_tokens=0.015),
    LlmModel(known_name="nova-lite-v1", aws_model_id="eu.amazon.nova-lite-v1:0", 
             aws_region="eu-central-1", langchain_ready=True,
             price_per_1000_input_tokens=0.000078, price_per_1000_output_tokens=0.000312),
    LlmModel(known_name="nova-pro-v1", aws_model_id="eu.amazon.nova-pro-v1:0", 
             aws_region="eu-central-1", langchain_ready=True,
             price_per_1000_input_tokens=0.00105, price_per_1000_output_tokens=0.0042),
    LlmModel(known_name="titan_premier", aws_model_id="amazon.titan-text-premier-v1:0", 
             aws_region="us-east-1", langchain_ready=False, 
             price_per_1000_input_tokens=0.0005, price_per_1000_output_tokens=0.0015, delay_time=10),
    LlmModel(known_name="Llama3.3", aws_model_id="us.meta.llama3-3-70b-instruct-v1:0", 
             aws_region="us-east-1", langchain_ready=True, delay_time=10,
             price_per_1000_input_tokens=0.00072, price_per_1000_output_tokens=0.00072),
    LlmModel(known_name="Llama3.1", aws_model_id="us.meta.llama3-1-70b-instruct-v1:0", 
             aws_region="us-east-1", langchain_ready=True, delay_time=20,
             price_per_1000_input_tokens=0.0009, price_per_1000_output_tokens=0.0009),
    
]
