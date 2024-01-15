import numpy as np
from scipy.spatial import KDTree

from idtxl.knn.tree_knn_finder import TreeKnnFinder


class ScipyKDTreeKnnFinder(TreeKnnFinder):
    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(**kwargs)

        if self._metric == "chebyshev":
            self._p = np.inf
        elif self._metric == "euclidean":
            self._p = 2
        else:
            raise ValueError(f"Unsupported metric {self._metric}")

        self._tree = KDTree(data, leafsize=self._leaf_size)

    def find_neighbors(self, x: np.ndarray, k: int) -> np.ndarray:
        return self._tree.query(x, k=k, p=np.inf, workers=self._num_threads)

    def find_neighbors_within(self, x: np.array, r: float) -> np.ndarray:
        return self._tree.query_ball_point(
            x=x, r=np.nextafter(r, 0), p=self._p, workers=self._num_threads
        )

    def count_neighbors(self, x: np.ndarray, r: float) -> np.ndarray:
        return self._tree.query_ball_point(
            x=x,
            r=np.nextafter(r, 0),
            p=self._p,
            return_length=True,
            workers=self._num_threads,
        )
