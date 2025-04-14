from dataclasses import dataclass, field
from typing import List
from benchmark_code.llm_model import LlmModel
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(slots=True)
class LlmResponse():
    message: str
    total_tokens: int
    model: LlmModel
    latency: float = -1.0

@dataclass_json
@dataclass
class FileSummary:
    llm_result: LlmResponse
    file_name: str
    number_of_lines: int = -1 
    project_name: str = 'pytorch'
    
    @property
    def model_name(self):
        return self.llm_result.model.known_name
    @property
    def summary(self):
        return self.llm_result.message
    @property
    def total_tokens(self):
        return self.llm_result.total_tokens
    @property
    def latency(self):
        return self.llm_result.latency
@dataclass_json
@dataclass
class ModelSummaries:
    diffs: List[FileSummary]=field(default_factory=list)


