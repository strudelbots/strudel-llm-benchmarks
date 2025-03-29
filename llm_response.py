from dataclasses import dataclass, field
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(slots=True)
class LlmResponse():
    message: str
    total_tokens: int
    model_name: str
    latency: float = -1.0

@dataclass_json
@dataclass
class FileSummary:
    llm_result: LlmResponse
    file_name: str

@dataclass_json
@dataclass
class FileSummeryDiff:
    model1: LlmResponse
    model2: LlmResponse
    diff: str

@dataclass_json
@dataclass
class RepoSummeryDiff:
    diffs: List[FileSummeryDiff]=field(default_factory=list)


