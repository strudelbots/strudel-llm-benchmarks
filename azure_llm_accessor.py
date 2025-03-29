import datetime
import os

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from llm_accessor import LlmAccessor
from llm_response import LlmResponse

AZURE_LLM_KEY = os.getenv("AZURE_LLM_FOUNDRY_KEY")
if not AZURE_LLM_KEY:
    raise ValueError("AZURE_LLM_KEY environment variable not set")

class AzureLlmAccessor(LlmAccessor):
    def __init__(self, system_content, model_name="gpt-4"):
        super().__init__(system_content, model_name)
        self.endpoint = f"https://strudel-azure-opnai.openai.azure.com/openai/deployments/{model_name}"
        self.azure_llm_client = ChatCompletionsClient(
                                    endpoint=self.endpoint,
                                    credential=AzureKeyCredential(AZURE_LLM_KEY),
                                    )



    def _invoke_llm(self, user_input):
        try:
            response = self.azure_llm_client.complete(
                messages=[
                    SystemMessage(content=self.system_content),
                    UserMessage(content=user_input),
                ],
                temperature=1.0,
                top_p=1.0,
                model=self.model_name,
                timeout=3,
            )
        except Exception as e:
            raise e
        llm_response = LlmResponse( message= response.choices[0].message.content,
                                   total_tokens=response.usage.total_tokens,
                                   model_name=self.model_name)

        return llm_response