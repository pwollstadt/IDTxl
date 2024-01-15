import numpy as np
from sklearn.neighbors import BallTree, KDTree

from idtxl.knn.tree_knn_finder import TreeKnnFinder


class _SklearnTreeKnnFinder(TreeKnnFinder):
    def __init__(self, data: np.ndarray, **kwargs):
        """Initialise the KnnFinder with settings.

        Args:
            data : np.ndarray
                The points to find neighbors for.
            kwargs : dict
                Settings for the KnnFinder.
        """

        super().__init__(**kwargs)

        self._tree = self._get_tree(data)

    def find_neighbors(self, x: np.ndarray, k: int) -> np.ndarray:
        return self._tree.query(x, k=k)

    def find_neighbors_within(self, x: np.array, r: float) -> np.ndarray:
        return self._tree.query_radius(x, np.nextafter(r, 0))

    def count_neighbors(self, x: np.ndarray, r: float) -> np.ndarray:
        return self._tree.query_radius(x, np.nextafter(r, 0), count_only=True)

    def _get_tree(self, data: np.ndarray):
        raise NotImplementedError


class SklearnKDTreeKnnFinder(_SklearnTreeKnnFinder):
    def _get_tree(self, data: np.ndarray):
        return KDTree(data, leaf_size=self._leaf_size, metric=self._metric)


class SklearnBallTreeKnnFinder(_SklearnTreeKnnFinder):
    def _get_tree(self, data: np.ndarray):
        return BallTree(data, leaf_size=self._leaf_size, metric=self._metric)
