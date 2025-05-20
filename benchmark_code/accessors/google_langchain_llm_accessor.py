from langchain_aws import ChatBedrock
from benchmark_code.llm_response import LlmResponse
from benchmark_code.accessors.llm_accessor import LlmAccessor
from benchmark_code import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TOP_P, DEFAULT_TOP_K
from langchain_google_genai import ChatGoogleGenerativeAI
class GoogleLangchainLlmAccessor(LlmAccessor):
    # titan express is not supported for langchain
    def __init__(self, system_context, model):
        if model.known_name == "gemini-2.5-flash":
            model_name = "gemini-2.5-flash-preview-04-17"
        elif model.known_name == "gemini-2.5":
            model_name = "gemini-2.5-pro-exp-03-25"
        else:
            raise ValueError(f"Unknown model: {model.known_name}")
        super().__init__(system_context, model)

        self.chat = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS * 4,
            top_p = DEFAULT_TOP_P, 
            top_k =  DEFAULT_TOP_K
            )
            

    def _invoke_llm(self, user_input):
        messages = [
            ("system",self.system_context),
            ("human", user_input),
        ]
        response = self.chat.invoke(messages)
        if response.response_metadata.get("finish_reason") != "STOP":
            raise ValueError("Length exceeded")
        llm_response = LlmResponse(file_summary=response.content, 
                                   total_tokens=response.usage_metadata["total_tokens"], 
                                   model=self.model)
        return llm_response
    