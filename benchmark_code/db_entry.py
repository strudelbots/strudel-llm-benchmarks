from dataclasses import dataclass
from dataclasses_json import dataclass_json
from benchmark_code.llm_response import LlmResponse
from typing import Dict
from dataclasses import field
@dataclass_json
@dataclass

class SingleModelDBEntry:
    llm_result: LlmResponse
    file_name: str
    number_of_lines: int 
    project_name: str 
    
@dataclass_json
@dataclass
class SingleFileDBEntry:
    data: Dict[str, Dict[str, SingleModelDBEntry]] = field(default_factory=dict)
    def to_db_dict(self):
        orig_dict = self.to_dict()
        tmp_dict = orig_dict['data'].copy()
        for file_entry, model_value in orig_dict['data'].items():
            for model_known_name, params in model_value.items():
                for param_key, param_value in params['llm_result'].items():
                    tmp_dict[file_entry][model_known_name][param_key] = param_value
                y = tmp_dict[file_entry][model_known_name]['llm_result']
                assert y is not None
                del tmp_dict[file_entry][model_known_name]['llm_result']
                x = tmp_dict[file_entry][model_known_name].get('llm_result', None)
                assert x is None
        return tmp_dict
