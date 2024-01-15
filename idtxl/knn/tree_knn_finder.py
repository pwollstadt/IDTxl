from idtxl.knn.knn_finder import KnnFinder


class TreeKnnFinder(KnnFinder):
    def __init__(self, num_threads="USE_ALL", metric="chebyshev", leaf_size=40):
        """Initialise the KnnFinder with settings.

        Args:
            num_threads : int
                The number of threads to use. If -1 or "USE_ALL", use all
                available threads.
            metric : str
                The metric to use for finding neighbors.
            leaf_size : int
                The leaf size to use for the tree.
        """
        super().__init__(num_threads, metric)

        self._leaf_size = leaf_size
