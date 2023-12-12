import numpy as np

from scipy.special import digamma

from idtxl.estimator import Estimator
from idtxl.knn.knn_finder_factory import get_knn_finder
from idtxl import idtxl_utils as utils

class PythonKraskovCMI(Estimator):
    """Estimate conditional mutual information using Kraskov's first estimator.
    """

    def __init__(self, settings):
        """Initialise estimator with settings.
        """

        # Check for currently unsupported settings
        if 'local_values' in settings or 'theiler_t' in settings or 'algorithm_num' in settings:
            raise ValueError('This estimator currently does not support local_values, theiler_t or algorithm_num arguments.')

        self._knn_finder_settings = settings.get('knn_finder_settings', {})

        self._kraskov_k = settings.get('kraskov_k', 4)
        self._base = settings.get('base', np.e)
        self._normalise = settings.get('normalise', False)

        # Set number of threads
        num_threads = settings.get('num_threads', -1)
        if num_threads == 'USE_ALL':
            num_threads = -1
        self._knn_finder_settings['num_threads'] = num_threads

        # Init rng for added gaussian noise
        self._noise_level = settings.get('noise_level', 1e-8)
        if self._noise_level > 0:
            rng_seed = settings.get('rng_seed', None)
            self._rng = np.random.default_rng(rng_seed)

        # Get KNN finder class
        self._knn_finder_name = settings.get('knn_finder', 'scipy_kdtree')
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

        assert var1.shape[0] == var2.shape[0] == conditional.shape[0], \
            f'Unequal number of observations (var1: {var1.shape[0]}, var2: {var2.shape[0]}, conditional: {conditional.shape[0]})'
        

        # Check if number of points is sufficient for estimation.
        if var1.shape[0] - 1 < self._kraskov_k:
            raise ValueError(f'Not enough observations for Kraskov estimator (need at least {self._kraskov_k + 1}, got {var1.shape[0]}).')

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
            conditional = conditional + self._rng.normal(0, self._noise_level, conditional.shape)

        # Compute distances to kth nearest neighbors in the joint space
        epsilon = self._compute_epsilon(np.concatenate((var1, var2, conditional), axis=1), self._kraskov_k)

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
            return (digamma(self._kraskov_k)
                 + mean_digamma_nc
                 - mean_digamma_nc_var1
                 - mean_digamma_nc_var2
                ) / np.log(self._base)
        else:
            # Compute MI
            return (digamma(self._kraskov_k)
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
        return knn_finder.find_dist_to_kth_neighbor(data, k + 1) # +1 because the point itself is included in the data
    
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
    
class PythonDiscreteCMI(Estimator):

    def __init__(self, settings=None):

        if settings is None:
            settings = {}

        self._discretise_method = settings.get('discretise_method', 'none')
        self._alph1 = settings.get('alph1', 2)
        self._alph2 = settings.get('alph2', 2)
        self._alphc = settings.get('alphc', 2)

        self._base = settings.get('base', 2)

    def _discretise_vars(self, var1, var2, conditional=None):
        # Discretise variables if requested. Otherwise assert data are discrete
        # and provided alphabet sizes are correct.
        if self._discretise_method == 'equal':
            var1 = utils.discretise(var1, self._alph1)
            var2 = utils.discretise(var2, self._alph2)
            if conditional is not None:
                conditional = utils.discretise(conditional, self._alphc)

        elif self._discretise_method == 'max_ent':
            var1 = utils.discretise_max_ent(var1, self._alph1)
            var2 = utils.discretise_max_ent(var2, self._alph2)
            if not (conditional is None):
                conditional = utils.discretise_max_ent(conditional, self._alphc)

        elif self._discretise_method == 'none':
            assert issubclass(var1.dtype.type, np.integer), (
                'Var1 is not an integer numpy array. '
                'Discretise data to use this estimator.')
            assert issubclass(var2.dtype.type, np.integer), (
                'Var2 is not an integer numpy array. '
                'Discretise data to use this estimator.')
            assert np.min(var1) >= 0, 'Minimum of var1 is smaller than 0.'
            assert np.min(var2) >= 0, 'Minimum of var2 is smaller than 0.'
            assert np.max(var1) < self._alph1, (
                        'Maximum of var1 is larger than the alphabet size.')
            assert np.max(var2) < self._alph2, (
                        'Maximum of var2 is larger than the alphabet size.')
            if conditional.shape[1] > 0:
                assert np.min(conditional) >= 0, (
                        'Minimum of conditional is smaller than 0.')
                assert issubclass(conditional.dtype.type, np.integer), (
                    'Conditional is not an integer numpy array. '
                    'Discretise data to use this estimator.')
                assert np.max(conditional) < self._alphc, (
                    'Maximum of conditional is larger than the alphabet size.')
        else:
            raise ValueError('Unkown discretisation method.')

        if conditional is not None:
            return var1, var2, conditional
        else:
            return var1, var2
        
    def estimate(self, var1: np.ndarray, var2: np.ndarray, conditional=None):

        if conditional is None:
            conditional = np.empty((len(var1), 0))

        var1 = self._ensure_two_dim_input(var1)
        var2 = self._ensure_two_dim_input(var2)
        conditional = self._ensure_two_dim_input(conditional)

        # Discretise if requested.
        var1, var2, conditional = self._discretise_vars(var1, var2,
                                                        conditional)
        
        
        
        # Compute entropies
        H12c, H1c, H2c, Hc = self._estimate_entropies(var1, var2, conditional)

        # Compute CMI
        return (H1c + H2c - H12c - Hc)  / np.log(self._base)
    
    def is_parallel(self):
        return False
    
    def is_analytic_null_estimator(self):
        return False

    def _estimate_entropy(self, data: np.ndarray):
        raise NotImplementedError

class PythonDiscretePluginCMI(PythonDiscreteCMI):
    """Estimate conditional mutual information using the plugin estimator.

    """

    def __init__(self, settings=None):

        if settings is None:
            settings = {}

        self._sparsity = settings.get('sparsity', 'auto')
        assert self._sparsity in ['auto', 'dense', 'sparse'], (
            'Unknown sparsity option. Choose from "auto", "dense" or "sparse".')
        self._sparsity_memory_limit = settings.get('memory_limit', 1e9) # 1 GB

        super().__init__(settings)

    def _estimate_entropies(self, V1: np.ndarray, V2: np.ndarray, C: np.ndarray):
    
        if self._sparsity == 'auto':
            # Compute number of bins for dense estimation
            n_bins = (self._alph1 ** V1.shape[1] * self._alph2 ** V2.shape[1] * self._alphc ** C.shape[1]) * 8

            if n_bins * 8 > self._sparsity_memory_limit:
                use_sparse = True
            elif n_bins > len(V1):
                use_sparse = True
            else:
                use_sparse = False
        else:
            use_sparse = self._sparsity == 'sparse'

        if use_sparse:
            return self._estimate_entropies_sparse(V1, V2, C)
        else:
            return self._estimate_entropies_dense(V1, V2, C)


    def _estimate_entropies_sparse(self, V1: np.ndarray, V2: np.ndarray, C: np.ndarray):
        """ Estimate entropy using a sparse pmf."""

        N = len(V1)
        l1 = V1.shape[1]
        lc = C.shape[1]

        data = np.concatenate((V1, C, V2), axis=1)

        _, counts12c = np.unique(data, axis=0, return_counts=True)
        _, counts1c = np.unique(data[:, :l1 + lc], axis=0, return_counts=True)
        _, counts2c = np.unique(data[:, l1:], axis=0, return_counts=True)
        _, countsc = np.unique(data[:, l1:l1 + lc], axis=0, return_counts=True)

        pmf12c = counts12c / N
        pmf1c = counts1c / N
        pmf2c = counts2c / N
        pmfc = countsc / N

        return  - pmf12c @ np.log(pmf12c), \
                - pmf1c @ np.log(pmf1c), \
                - pmf2c @ np.log(pmf2c), \
                - pmfc @ np.log(pmfc)
        
    def _estimate_entropies_dense(self, V1: np.ndarray, V2: np.ndarray, C: np.ndarray):
        """ Estimate entropy using a dense pmf."""

        N = len(V1)
        l1 = V1.shape[1]
        l2 = V2.shape[1]
        lc = C.shape[1]
        min_dtype = np.min_scalar_type(N)

        counts12c = np.zeros((self._alph1,) * l1 + (self._alph2,) * l2 + (self._alphc,) * lc, dtype=min_dtype)

        np.add.at(counts12c, tuple(v1 for v1 in V1.T) + tuple(c for c in C.T) + tuple(v2 for v2 in V2.T), 1)

        counts1c = np.sum(counts12c, axis=tuple(a for a in range(l1+lc, l1+lc+l2)), dtype=min_dtype)
        counts2c = np.sum(counts12c, axis=tuple(a for a in range(0, l1)), dtype=min_dtype)
        countsc = np.sum(counts1c, axis=tuple(a for a in range(0, l1)), dtype=min_dtype)

        pmf12c = counts12c[counts12c != 0] / N
        pmf1c = counts1c[counts1c != 0] / N
        pmf2c = counts2c[counts2c != 0] / N
        pmfc = countsc[countsc != 0] / N

        return  - pmf12c @ np.log(pmf12c), \
                - pmf1c @ np.log(pmf1c), \
                - pmf2c @ np.log(pmf2c), \
                - pmfc @ np.log(pmfc)
