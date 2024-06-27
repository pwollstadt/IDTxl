import numpy as np

from idtxl.knn.knn_finder import KnnFinder


class TreeKnnFinder(KnnFinder):

    def __init__(self, data: np.ndarray, num_threads: str='USE_ALL', metric: str='chebyshev', leaf_size: int=40):
        """Initialise the KnnFinder with settings.

        Args:
            data (np.ndarray): The points to find neighbors for. Shape is (n_points, n_dimensions).
            num_threads (int): The number of threads to use. If -1 or "USE_ALL", use all available threads.
            metric (str): The metric to use for finding neighbors.
            leaf_size (int): The leaf size to use for the tree.
        """
        super().__init__(data, num_threads, metric)

        self._leaf_size = leaf_size
