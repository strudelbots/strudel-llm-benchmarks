class EmbeddingAccessor:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model_name = model_name

    def get_embeddings(self, texts):
        raise NotImplementedError("Subclasses must implement this method")