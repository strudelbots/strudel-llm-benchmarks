from langchain_aws import BedrockLLM

from benchmark_code.llm_accessor import LlmAccessor


class AWSLangchainLlmAccessor(LlmAccessor):
    """
    This class is used to access the AWS Bedrock LLM.
    """
    llm = BedrockLLM(
        credentials_profile_name="bedrock-admin", model_id="amazon.titan-text-express-v1"
    )

    def _invoke_llm(self, user_input):
        raise NotImplementedError("Not implemented")

    