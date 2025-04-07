from benchmark_code.azure_llm_accessor import AzureLlmAccessor
from benchmark_code.aws_langchain_llm_accessor import AWSLangchainLlmAccessor


def get_llm_accessor(type: str, system_context, model_name):
    if type == "AZURE":
        return AzureLlmAccessor(system_context, model_name)
    elif type == "AWS":
        return AWSLangchainLlmAccessor(system_context, model_name)
    else:
        raise Exception("Unknown accessor type {}".format(type))