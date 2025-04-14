from benchmark_code.azure_llm_accessor import AzureLlmAccessor
from benchmark_code.aws_langchain_llm_accessor import AWSLangchainLlmAccessor
from benchmark_code.aws_boto3_llm_accessor import AWSBoto3Accessor
from benchmark_code.aws_boto3_titan_llm_accessor import AWSBoto3TitanLlmAccessor

def get_llm_accessor(system_context, model, provider_name):
#    if type == "AZURE":
#        return AzureLlmAccessor(system_context, model, sleep_time=1)
    
    if provider_name == "AWS":
        if model.langchain_ready:
            return AWSLangchainLlmAccessor(system_context, model)
        elif model.known_name == "titan_premier":
            return AWSBoto3TitanLlmAccessor(system_context, model)
        else:
            return AWSBoto3Accessor(system_context, model)

    elif provider_name == "AWS-LANGCHAIN":
        raise NotImplementedError("AWS Langchain is not supported yet.")
    else:
        raise Exception("Unknown accessor type {}".format(type))