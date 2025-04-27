from dataclasses import dataclass, field
from typing import List
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class LlmModel():
    known_name: str
    provider_name: str = ''
    aws_model_id: str = ''
    aws_region: str = ''
    azure_deployment_name: str = ''
    azure_region: str = ''
    delay_time: int = 3 
    langchain_ready: bool = False
    price_per_1000_input_tokens: float = 0.0
    price_per_1000_output_tokens: float = 0.0

AVAILABLE_MODELS = [
    LlmModel(known_name="Claude3.5", aws_model_id="eu.anthropic.claude-3-5-sonnet-20240620-v1:0", 
             aws_region="eu-central-1", langchain_ready=True, provider_name="AWS",
             price_per_1000_input_tokens=0.003, price_per_1000_output_tokens=0.015),
    LlmModel(known_name="nova-lite-v1", aws_model_id="eu.amazon.nova-lite-v1:0", 
             aws_region="eu-central-1", langchain_ready=True, provider_name="AWS",
             price_per_1000_input_tokens=0.000078, price_per_1000_output_tokens=0.000312),
    LlmModel(known_name="nova-pro-v1", aws_model_id="eu.amazon.nova-pro-v1:0", 
             aws_region="eu-central-1", langchain_ready=True, provider_name="AWS",
             price_per_1000_input_tokens=0.00105, price_per_1000_output_tokens=0.0042),
    LlmModel(known_name="titan_premier", aws_model_id="amazon.titan-text-premier-v1:0", 
             aws_region="us-east-1", langchain_ready=False, provider_name="AWS",
             price_per_1000_input_tokens=0.0005, price_per_1000_output_tokens=0.0015, delay_time=20),
    LlmModel(known_name="Llama3.3", aws_model_id="us.meta.llama3-3-70b-instruct-v1:0", 
             aws_region="us-east-1", langchain_ready=True, provider_name="AWS", delay_time=10,
             price_per_1000_input_tokens=0.00072, price_per_1000_output_tokens=0.00072),
    LlmModel(known_name="Llama3.1", aws_model_id="us.meta.llama3-1-70b-instruct-v1:0", 
             aws_region="us-east-1", langchain_ready=True, provider_name="AWS", delay_time=20,
             price_per_1000_input_tokens=0.0009, price_per_1000_output_tokens=0.0009),
    LlmModel(known_name="gpt-3.5-turbo", azure_deployment_name="gpt-35-turbo",
             azure_region="eastus", langchain_ready=True, provider_name="AZURE",
             price_per_1000_input_tokens=0.5/1000.0, price_per_1000_output_tokens=1.5/1000.0),
    LlmModel(known_name="gpt-4o", azure_deployment_name="gpt-4o",
             azure_region="eastus", langchain_ready=True, provider_name="AZURE",
             price_per_1000_input_tokens=2.5/1000.0, price_per_1000_output_tokens=10/1000.0),
    LlmModel(known_name="gpt-4", azure_deployment_name="gpt-4",
             azure_region="eastus", langchain_ready=True, provider_name="AZURE",
             price_per_1000_input_tokens=10/1000.0, price_per_1000_output_tokens=30/1000.0),
    LlmModel(known_name="gpt-4.5", azure_deployment_name="gpt-4.5-preview",
             azure_region="eastus2", langchain_ready=True, provider_name="AZURE",
             price_per_1000_input_tokens=75/1000.0, price_per_1000_output_tokens=150/1000.0),
    LlmModel(known_name="gpt-4.1", azure_deployment_name="gpt-4.1",
             azure_region="eastus2", langchain_ready=True, provider_name="AZURE",
             price_per_1000_input_tokens=2/1000.0, price_per_1000_output_tokens=8/1000.0),
    LlmModel(known_name="gemini-2.5", azure_deployment_name="",
             azure_region="", langchain_ready=True, provider_name="GOOGLE",
             price_per_1000_input_tokens=1.25/1000.0, price_per_1000_output_tokens=10/1000.0,
             delay_time=20),
    LlmModel(known_name="gemini-2.5-flash", azure_deployment_name="",
             azure_region="", langchain_ready=True, provider_name="GOOGLE",
             price_per_1000_input_tokens=0.15/1000.0, price_per_1000_output_tokens=0.6/1000.0,
             delay_time=15),

]


