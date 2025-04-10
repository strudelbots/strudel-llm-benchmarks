from langchain_huggingface import HuggingFaceEmbeddings
from benchmark_code.embedding_accessor import EmbeddingAccessor

class EmbeddingAccessorLangchainHF(EmbeddingAccessor):
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model_name = model_name
        self.embeddings_model = HuggingFaceEmbeddings(model_name=self.model_name)
    def get_embeddings(self, text:str) -> list[float]:
        embeddings = self.embeddings_model.embed_documents([text])
        return embeddings

# if __name__ == "__main__":
#     texts_to_embed = [
#         "The quick brown rabbit jumps over the lazy frogs.",
#         "A fast tan hare leaps above slow green toads.",
#         "This is a completely different sentence."
#     ]

#     print("--- Using Hugging Face Sentence-Transformer ---")
#     huggingface_embeddings = get_huggingface_embeddings(texts_to_embed)
#     for i, embedding in enumerate(huggingface_embeddings):
#         print(f"Embedding for '{texts_to_embed[i]}':")
#         print(f"  Shape: {len(embedding)}")
#         print(f"  First 5 elements: {embedding[:5]}... Total embeeding length: {len(embedding)}")
#         print("-" * 20)