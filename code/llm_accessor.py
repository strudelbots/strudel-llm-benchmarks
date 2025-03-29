import datetime
import os

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

AZURE_LLM_KEY = os.getenv("AZURE_LLM_FOUNDRY_KEY")
if not AZURE_LLM_KEY:
    raise ValueError("AZURE_LLM_KEY environment variable not set")

class LlmAccessor:
    def __init__(self, system_content, model_name="gpt-4"):
        self.model_name = model_name
        self.system_content = system_content


    def _get_llm_response(self, user_input:str):
            start_time = datetime.datetime.now()
            response = self._invoke_llm(user_input)
            end_time = datetime.datetime.now()
            latency = end_time - start_time
            print(f'request Latency: {latency}, total tokens: {response.total_tokens}, '
                  f'response length: {len(response.message)}')
            response.latency = latency.total_seconds()
            return response

    def _invoke_llm(self, user_input):
        raise NotImplementedError('Abstract method - please implement')