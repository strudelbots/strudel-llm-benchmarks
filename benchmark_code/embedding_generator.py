import os
from benchmark_code import OUT_FILES_DIRECTORY, OUT_FILES_DIRECTORY_CACHE
from benchmark_code.utils import get_db
from benchmark_code.accessors.embedding_accessor_langchain_HF import EmbeddingAccessorLangchainHF
import json
from datetime import datetime

class EmbeddingGenerator():
    def __init__(self, embedding_model_name):
        today = datetime.now().strftime("%Y-%m-%d")
        self.today = today
        self.db = get_db()
        self.embedding_accessor = EmbeddingAccessorLangchainHF(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self._cache_directory = f'{OUT_FILES_DIRECTORY_CACHE}/{self.embedding_model_name}_{today}'
        self._out_file = f'{OUT_FILES_DIRECTORY}/embeddings_{self.embedding_model_name}_{today}.json'
        if not os.path.exists(self._cache_directory):
            os.makedirs(self._cache_directory)
        self._embeddings_db = None

    def build_db(self):
        embeddings_db  = self._generate_embeddings(write_to_file=False)
        self._embeddings_db = embeddings_db

    def _get_embedding(self, summary):
        uuid = summary['uuid']
        summary_file_name = f'{self.cache_directory}/{uuid}_{self.embedding_model_name}_{self.today}.json'
        text = summary['file_summary']
        if not os.path.exists(summary_file_name):
            embeddings = self.embedding_accessor.get_embeddings(text)
            with open(summary_file_name, 'w') as f:
                json.dump(embeddings, f)
            return embeddings
        else:
            with open(summary_file_name, 'r') as f:
                return json.load(f)
    def _generate_embeddings(self, write_to_file=True):
        today = datetime.now().strftime("%Y-%m-%d")
        embeddings_db = {}
        for file_name, file_data in self.db.items():
            print(f'Generating embeddings for {file_name}')
            for model, summary in file_data.items():
                uuid = summary['uuid']
                embeddings = self._get_embedding(summary)
                embeddings_db[uuid] = embeddings
        if write_to_file:
            with open(self.out_file, 'w') as f:
                json.dump(embeddings_db, f, indent=4)
        return embeddings_db

    def get_embedding(self, uuid):
        embedding = self._embeddings_db.get(uuid)
        assert len(embedding) == 1
        assert len(embedding[0]) == 768
        return embedding[0]

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
