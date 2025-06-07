from langchain_huggingface import HuggingFaceEmbeddings
from benchmark_code.accessors.embedding_accessor import EmbeddingAccessor

class EmbeddingAccessorLangchainHF(EmbeddingAccessor):
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model_name = model_name
        self.embeddings_model = HuggingFaceEmbeddings(model_name=self.model_name)
        if model_name != 'all-mpnet-base-v2':
            raise ValueError(f'{model_name} is not supported')
    
    def get_embeddings(self, text:str) -> list[float]:
        embeddings = self.embeddings_model.embed_documents([text])
        assert len(embeddings) == 1, "Langchain HF embeddings model should return a single embedding for single paragraph"
        assert len(embeddings[0]) == 768, "Langchain HF embeddings model should return an embedding of size 768"
        return embeddings
