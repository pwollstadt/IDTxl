import numpy as np


class KnnFinder:
    def __init__(self, num_threads="USE_ALL", metric="chebyshev"):
        """Initialise the KnnFinder with settings.

        Args:
            num_threads : int
                The number of threads to use. If -1 or "USE_ALL", use all
                available threads.
            metric : str
                The metric to use for finding neighbors.
        """
        if num_threads == "USE_ALL":
            num_threads = -1

        self._num_threads = num_threads
        self._metric = metric

    def find_neighbors(self, x: np.ndarray, k: int) -> np.ndarray:
        """Find the k nearest neighbors to each point in x.
        May include x itself if it is in the data.

        Args:
            x : np.ndarray
                The points to find neighbors for.
            k : int
                The number of neighbors to find.

        Returns:
            np.ndarray
                Array of lists of distances to the k nearest neighbors for
                each point in x
            np.ndarray
                Array of lists of indices of the k nearest neighbors for each
                point in x.
        """
        raise NotImplementedError

    def find_neighbors_within(self, x: np.array, r: float) -> np.ndarray:
        """Find the neighbors strictly within (<) a given radius for each point in x.
        May include x itself if it is in the data.

        Args:
            x : np.ndarray
                The points to find neighbors for.
            r : float
                The radius to find neighbors within.

        Returns:
            np.ndarray
                Array of lists of indices of the neighbors within the given
                radius for each point in x.
        """
        raise NotImplementedError

    def find_dist_to_kth_neighbor(self, x: np.ndarray, k: int) -> np.ndarray:
        """Find the distance to the kth nearest neighbor for each point in x.

        May include x itself if it is in the data.

        Default implementation uses find_neighbors and returns the distance to
        the kth neighbor.

        Args:
            x : np.ndarray
                The points to find the kth nearest neighbor for.
            k : int
                The kth nearest neighbor to find.
        """
        return self.find_neighbors(x, k)[0][:, k - 1]

    def count_neighbors_within(self, x: np.ndarray, r: float) -> np.ndarray:
        """Count the number of neighbors strictly within (<) a given radius for each point in x.

        May include x itself if it is in the data.

        The default implementation uses find_neighbors and counts the number of
        neighbors in each list.

        Args:
            x : np.ndarray
                The points to count neighbors for.
            r : float
                The radius to count neighbors within.

        Returns:
            np.ndarray
                The number of neighbors within the given radius for each point
                in x.
        """
        return np.array([len(neighbors) for neighbors in self.find_neighbors(x, r)])
