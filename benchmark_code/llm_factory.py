from benchmark_code.azure_llm_accessor import AzureLlmAccessor
from benchmark_code.aws_langchain_llm_accessor import AWSLangchainLlmAccessor
from benchmark_code.aws_boto3_llm_accessor import AWSBoto3Accessor


def get_llm_accessor(system_context, model, provider_name):
#    if type == "AZURE":
#        return AzureLlmAccessor(system_context, model, sleep_time=1)
    
    if provider_name == "AWS":
        if model.bedrock_ready:
        #return AWSBoto3Accessor(system_context, model)
            return AWSLangchainLlmAccessor(system_context, model)
        else:
            raise Exception("AWS Boto3 is not supported for this model {}".format(model.known_name))
    elif provider_name == "AWS-LANGCHAIN":
        raise NotImplementedError("AWS Langchain is not supported yet.")
    else:
        raise Exception("Unknown accessor type {}".format(type))