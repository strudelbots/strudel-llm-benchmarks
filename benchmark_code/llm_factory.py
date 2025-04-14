from benchmark_code.azure_llm_accessor import AzureLlmAccessor
from benchmark_code.aws_langchain_llm_accessor import AWSLangchainLlmAccessor
from benchmark_code.aws_boto3_llm_accessor import AWSBoto3Accessor
from benchmark_code.aws_boto3_titan_llm_accessor import AWSBoto3TitanLlmAccessor
from benchmark_code.azure_langchain_llm_accessor import AzureLangchainLlmAccessor
def get_llm_accessor(system_context, model):
#    if type == "AZURE":
#        return AzureLlmAccessor(system_context, model, sleep_time=1)
    
    if model.provider_name == "AWS":
        if model.langchain_ready:
            return AWSLangchainLlmAccessor(system_context, model)
        elif model.known_name == "titan_premier":
            return AWSBoto3TitanLlmAccessor(system_context, model)
        else:
            return AWSBoto3Accessor(system_context, model)
    elif model.provider_name == "AZURE":
        return AzureLangchainLlmAccessor(system_context, model)
    else:
        raise Exception("Unknown accessor type {}".format(model.provider_name))