import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from benchmark_code import OUT_FILES_DIRECTORY, OUT_FILES_DIRECTORY_CACHE
from benchmark_code.utils import get_db
from benchmark_code.embedding_generator import EmbeddingGenerator


class SimilarityMatrixGenerator():
    def __init__(self, embedding_db):
        today = datetime.now().strftime("%Y-%m-%d")
        self.today = today
        self.db = get_db()
        self.embedding_db = embedding_db
        self._cache_directory = f'{OUT_FILES_DIRECTORY_CACHE}/similarity_matrix_{today}'
        self._out_file = f'{OUT_FILES_DIRECTORY}/similarity_matrix_{self.embedding_db.embedding_model_name}_{today}.json'
        if not os.path.exists(self._cache_directory):
            os.makedirs(self._cache_directory)
        self._similarity_matrix_db = {}
    def generate_similarity_matrix(self):
        for file_name, file_data in self.db.items():
            self._similarity_matrix_for_file(file_name)
        return
    def _similarity_matrix_for_file(self, file_name):
        uuids = self._get_uuids_for_file(file_name)
        embeddings = self._get_embeddings_for_uuids(file_name, uuids)
        similarity_matrix = self._get_similarity_matrix(embeddings)
        self._similarity_matrix_db[file_name] = similarity_matrix.tolist()

    def _get_similarity_matrix(self, embeddings):
        similarity_matrix = cosine_similarity(embeddings)
        assert similarity_matrix.shape == (len(embeddings), len(embeddings))
        return similarity_matrix

    def _get_embeddings_for_uuids(self, file_name, uuids):
        embeddings = []
        for uuid in uuids:
            embedding = self.embedding_db.get_embedding(uuid)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                raise ValueError(f'Embedding for file {file_name} and uuid {uuid} not found')
        return embeddings

    def _get_uuids_for_file(self, file_name):
        uuids = []
        summaries = self.db[file_name]
        assert isinstance(summaries, dict)
        for summary in summaries.values():
            uuids.append(summary['uuid'])
        return uuids


    @property
    def out_file(self):
        return self._out_file
    
    @property
    def cache_directory(self):
        return self._cache_directory
    
    @out_file.setter
    def out_file(self, value):
        self._out_file = value

    @cache_directory.setter
    def cache_directory(self, value):
        self._cache_directory = value

if __name__ == '__main__':
    embedding_db = EmbeddingGenerator('all-mpnet-base-v2')
    embedding_db.build_db()
    similarity_matrix_generator = SimilarityMatrixGenerator(embedding_db) 
    print("--------------------------------")                                                        
    print(f'Cache directory for similarity matrix: {similarity_matrix_generator.cache_directory}')
    print(f'Output file for similarity matrix: {similarity_matrix_generator.out_file}')
    print("--------------------------------")                                                        
    similarity_matrix_generator.generate_similarity_matrix()