from langchain_aws import ChatBedrock
from benchmark_code.llm_response import LlmResponse
from benchmark_code.llm_accessor import LlmAccessor
from benchmark_code import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TOP_P, DEFAULT_TOP_K

class AWSLangchainLlmAccessor(LlmAccessor):
    """
    This class is used to access the AWS Bedrock LLM.
    """
    def __init__(self, system_context, model):
        super().__init__(system_context, model)
        if model.aws_model_id == "eu.anthropic.claude-3-5-sonnet-20240620-v1:0":
            model.delay_time = 20
        self.chat = ChatBedrock(
            credentials_profile_name="bedrock_admin", region_name=model.aws_region, model_id=model.aws_model_id,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            model_kwargs={"top_p": DEFAULT_TOP_P, "top_k": DEFAULT_TOP_K}
            )
            

    def _invoke_llm(self, user_input):
        messages = [
            ("system",self.system_context),
            ("human", user_input),
        ]
        response = self.chat.invoke(messages)
        llm_response = LlmResponse(message=response.content, 
                                   total_tokens=response.usage_metadata["total_tokens"], 
                                   model=self.model)
        return llm_response
    