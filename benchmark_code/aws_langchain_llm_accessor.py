from langchain_aws import ChatBedrock
from benchmark_code.llm_response import LlmResponse
from benchmark_code.llm_accessor import LlmAccessor


class AWSLangchainLlmAccessor(LlmAccessor):
    """
    This class is used to access the AWS Bedrock LLM.
    """
    def __init__(self, system_context, model_name):
        super().__init__(system_context, model_name)
        self.chat = ChatBedrock(
            credentials_profile_name="bedrock_admin", region_name="eu-central-1", model_id=model_name)
            

    def _invoke_llm(self, user_input):
        messages = [
            ("system",self.system_content),
            ("human", user_input),
        ]
        response = self.chat.invoke(messages)
        print(response.content)
        llm_response = LlmResponse(message=response.content, 
                                   total_tokens=response.usage_metadata["total_tokens"], 
                                   model_name=self.model_name)
        return llm_response
    