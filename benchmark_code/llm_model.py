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
    delay_time = 3 
    bedrock_ready: bool = False

AVAILABLE_MODELS = [
    LlmModel(known_name="sonnet-v1", aws_model_id="eu.anthropic.claude-3-5-sonnet-20240620-v1:0", 
             aws_region="eu-central-1", bedrock_ready=True),
#    LlmModel(known_name="nova-lite-v1", aws_model_id="eu.amazon.nova-lite-v1:0", aws_region="eu-central-1"),
]
