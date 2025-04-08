from benchmark_code.azure_llm_accessor import AzureLlmAccessor
from benchmark_code.aws_langchain_llm_accessor import AWSLangchainLlmAccessor
from benchmark_code.aws_boto3_llm_accessor import AWSBoto3Accessor

def get_llm_accessor(type: str, system_context, model_name):
    if type == "AZURE":
        return AzureLlmAccessor(system_context, model_name, sleep_time=1)
    elif type == "AWS-BOTO3":
        return AWSBoto3Accessor(system_context, model_name, sleep_time=3)
    elif type == "AWS-LANGCHAIN":
        return AWSLangchainLlmAccessor(system_context, model_name, sleep_time=10)
    else:
        raise Exception("Unknown accessor type {}".format(type))