import boto3
from benchmark_code.llm_response import LlmResponse
from benchmark_code.accessors.llm_accessor import LlmAccessor
from benchmark_code import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_MAX_TOKENS
class AWSBoto3Accessor(LlmAccessor):
    """
    This class is used to access the AWS Bedrock LLM.
    """
    def __init__(self, system_context, model):
        super().__init__(system_context, model)
        boto3.setup_default_session(profile_name="bedrock_admin", region_name=model.aws_region)
        self.llm = boto3.client("bedrock-runtime" )
            

    def _invoke_llm(self, user_input):
        system = [{ "text": self.system_context }]

        messages = [
            {"role": "user", "content": [{"text": user_input}]},
        ]

        inf_params = {"maxTokens": DEFAULT_MAX_TOKENS, "topP":DEFAULT_TOP_P, "temperature": DEFAULT_TEMPERATURE}

        additionalModelRequestFields = {
            "inferenceConfig": {
            "topK": DEFAULT_TOP_K
            }
        }

        model_response = self.llm.converse(
            modelId=self.model.aws_model_id,
            messages=messages, 
            system=system, 
            inferenceConfig=inf_params,
            additionalModelRequestFields=additionalModelRequestFields
        )
        llm_response = LlmResponse(message=model_response["output"]["message"]["content"][0]["text"], 
                                   total_tokens=model_response['usage']["totalTokens"], 
                                   model_name=self.model_name)
        return llm_response

    