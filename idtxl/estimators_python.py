import numpy as np

from scipy.special import digamma

from idtxl.estimator import Estimator
from idtxl.knn.knn_finder_factory import get_knn_finder


class PythonKraskovCMI(Estimator):
    """Estimate conditional mutual information using Kraskov's first estimator.

    Args:
        settings : dict [optional]
            set estimator parameters:

            - kraskov_k : int [optional] - no. nearest neighbours for KNN
              search (default=4)
            - base : float - base of returned values (default=np=e)
            - normalise : bool [optional] - z-standardise data (default=False)
            - noise_level : float [optional] - random noise added to the data
              (default=1e-8)
            - rng_seed : int | None [optional] - random seed if noise level > 0
            - num_threads : int | str [optional] - number of threads used for
              estimation (default='USE_ALL', note that this uses *all*
              available threads on the current machine)
            - knn_finder : str [optional] - knn algorithm to use, can be
              'scipy_kdtree' (default), 'sklearn_kdtree', or 'sklearn_balltree'
    """

    def __init__(self, settings):
        """Initialise estimator with settings."""

        # Check for currently unsupported settings
        if (
            "local_values" in settings
            or "theiler_t" in settings
            or "algorithm_num" in settings
        ):
            raise ValueError(
                "This estimator currently does not support local_values, theiler_t or algorithm_num arguments."
            )

        self._knn_finder_settings = settings.get("knn_finder_settings", {})

        self._kraskov_k = settings.get("kraskov_k", 4)
        self._base = settings.get("base", np.e)
        self._normalise = settings.get("normalise", False)

        # Set number of threads
        num_threads = settings.get("num_threads", -1)
        if num_threads == "USE_ALL":
            num_threads = -1
        self._knn_finder_settings["num_threads"] = num_threads

        # Init rng for added gaussian noise
        self._noise_level = settings.get("noise_level", 1e-8)
        if self._noise_level > 0:
            rng_seed = settings.get("rng_seed", None)
            self._rng = np.random.default_rng(rng_seed)

        # Get KNN finder class
        self._knn_finder_name = settings.get("knn_finder", "scipy_kdtree")
        self._knn_finder_class = get_knn_finder(self._knn_finder_name)

    def estimate(self, var1: np.ndarray, var2: np.ndarray, conditional=None):
        """Estimate conditional mutual information between var1 and var2, given
        conditional.
        """

        if conditional is None:
            conditional = np.empty((len(var1), 0))

        # Check the input data
        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)

        assert (
            var1.shape[0] == var2.shape[0] == conditional.shape[0]
        ), f"Unequal number of observations (var1: {var1.shape[0]}, var2: {var2.shape[0]}, conditional: {conditional.shape[0]})"

        # Check if number of points is sufficient for estimation.
        if var1.shape[0] - 1 < self._kraskov_k:
            raise ValueError(
                f"Not enough observations for Kraskov estimator (need at least {self._kraskov_k + 1}, got {var1.shape[0]})."
            )

        # Normalise data
        if self._normalise:
            var1 = self._normalise_data(var1)
            var2 = self._normalise_data(var2)
            conditional = self._normalise_data(conditional)

        # Add noise to avoid duplicate points
        # Do not add noise inplace, because it would change the input data
        if self._noise_level > 0:
            var1 = var1 + self._rng.normal(0, self._noise_level, var1.shape)
            var2 = var2 + self._rng.normal(0, self._noise_level, var2.shape)
            conditional = conditional + self._rng.normal(
                0, self._noise_level, conditional.shape
            )

        # Compute distances to kth nearest neighbors in the joint space
        epsilon = self._compute_epsilon(
            np.concatenate((var1, var2, conditional), axis=1), self._kraskov_k
        )

        # Count neighbors in the conditional space
        if conditional.shape[1] > 0:
            n_c = self._compute_n(conditional, epsilon)
            mean_digamma_nc = np.mean(digamma(n_c))
            del n_c

        n_c_var1 = self._compute_n(np.concatenate((var1, conditional), axis=1), epsilon)
        mean_digamma_nc_var1 = np.mean(digamma(n_c_var1))
        del n_c_var1

        n_c_var2 = self._compute_n(np.concatenate((var2, conditional), axis=1), epsilon)
        mean_digamma_nc_var2 = np.mean(digamma(n_c_var2))
        del n_c_var2

        if conditional.shape[1] > 0:
            # Compute CMI
            return (
                digamma(self._kraskov_k)
                + mean_digamma_nc
                - mean_digamma_nc_var1
                - mean_digamma_nc_var2
            ) / np.log(self._base)
        else:
            # Compute MI
            return (
                digamma(self._kraskov_k)
                + digamma(len(var1))
                - mean_digamma_nc_var1
                - mean_digamma_nc_var2
            ) / np.log(self._base)

    def _normalise_data(self, data: np.ndarray):
        """Standardise data to zero mean and unit variance."""
        return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    def _compute_epsilon(self, data: np.ndarray, k: int):
        """Compute the distance to the kth nearest neighbor for each point in x."""
        knn_finder = self._knn_finder_class(data, **self._knn_finder_settings)
        return knn_finder.find_dist_to_kth_neighbor(
            data, k + 1
        )  # +1 because the point itself is included in the data

    def _compute_n(self, data: np.ndarray, r: np.ndarray):
        """Count the number of neighbors strictly within a given radius r for each point in x.
        Returns the number of neighbors plus one, because the point itself is included in the data.
        """
        knn_finder = self._knn_finder_class(data, **self._knn_finder_settings)
        return knn_finder.count_neighbors(data, r)

    def is_analytic_null_estimator(self):
        return False

    def is_parallel(self):
        return False
