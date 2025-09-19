import os
from benchmark_code.llm_response import LlmResponse
from benchmark_code.accessors.llm_accessor import LlmAccessor
from benchmark_code import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TOP_P, DEFAULT_TOP_K
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

# secrets are defined in the launch.json file
AZURE_LLM_KEY = os.getenv("AZURE_LLM_FOUNDRY_KEY")
if not AZURE_LLM_KEY:
    raise ValueError("AZURE_LLM_KEY environment variable not set")

class AzureLangchainLlmAccessor(LlmAccessor):
    supported_models = ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4', 'gpt-4.5', 'gpt-4.1']
    def __init__(self, system_context, model):
        assert model.known_name in self.supported_models, f"Model {model.known_name} is not supported for Azure Langchain."
        super().__init__(system_context, model)
        self.endpoint = f"https://shai-m9i812r1-eastus2.cognitiveservices.azure.com/openai/deployments/{model.azure_deployment_name}"
        kwargs = {"top_p": DEFAULT_TOP_P, "top_k": DEFAULT_TOP_K}
        self.chat = AzureAIChatCompletionsModel(
            endpoint=self.endpoint,
            credential=AZURE_LLM_KEY,
            api_version="2024-05-01-preview",
            model_name=model.known_name,
        )
            

    def _invoke_llm(self, user_input):
        messages = [
            ("system",self.system_context),
            ("human", user_input),
        ]
        try:
            response = self.chat.invoke(messages)
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            raise e
        response = self.chat.invoke(messages)
        llm_response = LlmResponse(file_summary=response.content, 
                                   total_tokens=response.usage_metadata["total_tokens"], 
                                   model=self.model)
        return llm_response
    