from idtxl.knn.knn_finder import KnnFinder


def get_knn_finder(name: str) -> type[KnnFinder]:
    """Find a KnnFinder Subclass by name and import only the required modules.

    Args:
        name : str
            The name of the KnnFinder to return.
    """

    if name == "scipy_kdtree":
        from .knn_finder_scipy import ScipyKDTreeKnnFinder

        return ScipyKDTreeKnnFinder
    elif name == "sklearn_kdtree":
        from .knn_finder_sklearn import SklearnKDTreeKnnFinder

        return SklearnKDTreeKnnFinder
    elif name == "sklearn_balltree":
        from .knn_finder_sklearn import SklearnBallTreeKnnFinder

        return SklearnBallTreeKnnFinder
    else:
        raise KeyError(f"Unknown KnnFinder {name}")
