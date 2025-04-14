import boto3
from benchmark_code.llm_response import LlmResponse
from benchmark_code.aws_boto3_llm_accessor import AWSBoto3Accessor
from benchmark_code import DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_MAX_TOKENS
import json
class AWSBoto3TitanLlmAccessor(AWSBoto3Accessor):
    """
    This class is used to access the AWS Bedrock LLM.
    """
    def __init__(self, system_context, model):
        super().__init__(system_context, model)
            

    def _invoke_llm(self, user_input):
        body = json.dumps({
            "inputText": self.system_context + "\n" + user_input,
            "textGenerationConfig": {
                "maxTokenCount": DEFAULT_MAX_TOKENS,
                "stopSequences": [],
                "temperature": DEFAULT_TEMPERATURE,
                "topP": DEFAULT_TOP_P,
            }
        })

        model_response = self.llm.invoke_model(
            modelId=self.model.aws_model_id,
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(model_response.get("body").read())

        finish_reason = response_body.get("error")

        if finish_reason is not None:
            raise Exception(f"Text generation error. Error is {finish_reason}")
        total_tokens = response_body['results'][0]['tokenCount']
        llm_response = LlmResponse(message=response_body["results"][0]['outputText'], 
                                   total_tokens=total_tokens, 
                                   model=self.model)
        return llm_response

    