from azure_llm_accessor import AzureLlmAccessor


def get_llm_accessor(type: str, system_context, model_name):
    if type == "AZURE":
        return AzureLlmAccessor(system_context, model_name)
    else:
        raise Exception("Unknown accessor type {}".format(type))