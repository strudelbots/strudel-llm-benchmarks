import os

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from benchmark_code.llm_accessor import LlmAccessor
from benchmark_code.llm_response import LlmResponse
from benchmark_code import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_MAX_TOKENS

AZURE_LLM_KEY = os.getenv("AZURE_LLM_FOUNDRY_KEY")
if not AZURE_LLM_KEY:
    raise ValueError("AZURE_LLM_KEY environment variable not set")

class AzureLlmAccessor(LlmAccessor):
    def __init__(self, system_content, model_name, sleep_time):
        super().__init__(system_content, model_name, sleep_time)
        self.endpoint = f"https://strudel-azure-opnai.openai.azure.com/openai/deployments/{model_name}"
        self.azure_llm_client = ChatCompletionsClient(
                                    endpoint=self.endpoint,
                                    credential=AzureKeyCredential(AZURE_LLM_KEY),
                                    max_tokens=DEFAULT_MAX_TOKENS,
                                    temperature=DEFAULT_TEMPERATURE,
                                    top_p=DEFAULT_TOP_P,
                                    top_k=DEFAULT_TOP_K
                                    )



    def _invoke_llm(self, user_input):
        try:
            response = self.azure_llm_client.complete(
                messages=[
                    SystemMessage(content=self.system_context),
                    UserMessage(content=user_input),
                ],
                temperature=DEFAULT_TEMPERATURE,
                model=self.model_name,
                timeout=30,
            )
        except Exception as e:
            raise e
        llm_response = LlmResponse( message= response.choices[0].message.content,
                                   total_tokens=response.usage.total_tokens,
                                   model_name=self.model_name)

        return llm_response