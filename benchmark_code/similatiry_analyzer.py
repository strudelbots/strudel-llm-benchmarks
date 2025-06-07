import numpy as np
from collections.abc import Iterable
import pandas as pd
class SimilarityAnalyzer():

    def __init__(self, similarity_matrixes: Iterable[pd.DataFrame]):
        assert len(similarity_matrixes) > 0
        self.similarity_matrixes = list(similarity_matrixes)

    def _get_matixes_with_same_size(self):
        sizes = [df.shape for df in self.similarity_matrixes]
        max_size = max(sizes)
        return [df for df in self.similarity_matrixes if df.shape == max_size]

    def get_upper_triangle_average(self, similarity_df):
        mat = similarity_df.values
        # Mask to get upper triangle excluding diagonal
        upper_vals = mat[np.triu_indices_from(mat, k=1)]
        return np.mean(upper_vals)
    
    def get_avarage_similary_matrix(self):
        matrixes = self._get_matixes_with_same_size()
        stacked = np.stack([df.values for df in matrixes])
        average = np.mean(stacked, axis=0)
        result = pd.DataFrame(average, index=matrixes[0].index, 
                              columns=matrixes[0].columns)
        return result

        return np.mean([similarity_df for similarity_df in self.similarity_matrixes] )
    
    def get_average_upper_triangle_average(self):
        return np.mean([self.get_upper_triangle_average(similarity_df) for similarity_df in self.similarity_matrixes])
