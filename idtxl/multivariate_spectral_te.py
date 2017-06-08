"""Estimate multivarate spectral TE.

Note:
    Written for Python 3.4+

@author: edoardo
"""
import numpy as np
from . import stats
from .network_analysis import NetworkAnalysis
from .estimator import find_estimator

VERBOSE = True


class MultivariateSpectralTE(NetworkAnalysis):
    """Set up a network analysis using multivariate spectral transfer entropy.

    Set parameters necessary for inference of spectral components of
    multivariate transfer entropy (TE). To perform network inference call
    analyse_network() on an instance of the data class.

    Args:
        options : dict
            parameters for estimator use and statistics:

            - 'cmi_calc_name' - estimator to be used for CMI calculation
              (For estimator options see the respective documentation.)
            - 'n_perm_spec' - number of permutations (default=200)
            - 'alpha_spec' - critical alpha level for statistical significance
              (default=0.05)
            - 'cmi_calc_name' - estimator to be used for CMI calculation
              (For estimator options see the respective documentation.)
            - 'permute_in_time' - force surrogate creation by shuffling
              realisations in time instead of shuffling replications; see
              documentation of Data.permute_samples() for further options
              (default=False)

    """

    # TODO right now 'options' holds all optional params (stats AND estimator).
    # We could split this up by adding the stats options to the analyse_*
    # methods?
    def __init__(self, options):
        # Set estimator in the child class for network inference because the
        # estimated quantity may be different from CMI in other inference
        # algorithms. (Everything else can be done in the parent class.)
        try:
            EstimatorClass = find_estimator(options['cmi_estimator'])
        except KeyError:
            raise KeyError('Estimator was not specified!')
        self._cmi_estimator = EstimatorClass(options)
        self.n_permutations = options.get('n_perm_spec', 200)
        self.alpha = options.get('alpha_spec', 0.05)
        self.tail = options.get('tail', 'two')
        self.cmi_opts = options

    def analyse_network(self, res_network, data, targets='all', sources='all'):
        """Find multivariate spectral transfer entropy between all nodes.

        Estimate multivariate transfer entropy (TE) between all nodes in the
        network or between selected sources and targets.

        Example:

            >>> dat = Data()
            >>> dat.generate_mute_data(100, 5)
            >>> max_lag = 5
            >>> min_lag = 4
            >>> analysis_opts = {
            >>>     'cmi_estimator':  'JidtKraskovCMI',
            >>>     'n_perm_max_stat': 200,
            >>>     'n_perm_min_stat': 200,
            >>>     'n_perm_omnibus': 500,
            >>>     'n_perm_max_seq': 500,
            >>>     }
            >>> network_analysis = MultivariateTE(max_lag, min_lag,
            >>>                                   analysis_opts)
            >>> res = network_analysis.analyse_network(dat)
            >>>
            >>> spectral_opts = {
            >>>     'cmi_estimator':  'JidtKraskovCMI',
            >>>     'n_perm_spec': 200,
            >>>     'alpha_spec': 0.05
            >>>     }
            >>> spectral_analysis = Multivariate_spectral_te(spectral_opts)
            >>> res_spec = spectral_analysis.analyse_network(res)

        Note:
            For more details on the estimation of multivariate transfer entropy
            see documentation of class method 'analyse_single_target'.

        Args:
            res_network: dict
                results from multivariate network inference, e.g., using TE
            data : Data instance
                raw data from which the network was inferred
            targets : list of int | 'all' [optinal]
                index of target processes (default='all')
            sources : list of int | list of list | 'all' [optional]
                indices of source processes for each target (default='all');
                if 'all', all identified sources in the network are tested for
                spectral TE;
                if list of int, sources specified in the list are tested for
                each target;
                if list of list, sources specified in each inner list are
                tested for the corresponding target

        Returns:
            dict
                results consisting of

                - TODO to be specified ...

                for each target
        """
        # TODO see MultivariateTE.analyse_network()
        return 1

    def analyse_single_target(self, res_target, data, sources='all'):
        """Find multivariate spectral transfer entropy into a target.

        Test multivariate spectral transfer entropy (TE) between all source
        identified using multivariate TE and a target:

        (1) take one relevant variable s
        (2) perform a maximal overlap discrete wavelet transform (MODWT)
        (3) destroy information carried by a single frequency band by
            scrambling the coefficients in the respective scale
        (4) perform the inverse of the MODWT, iMODWT, to get back the time-
            domain representation of the variable
        (5) calculate multivariate TE between s and the target, conditional on
            all other relevant sources
        (6) repeat (3) to (5) n_perm number of times to build a test
            distribution
        (7) test original multivariate TE against the test distribution

        Example:

            >>> dat = Data()
            >>> dat.generate_mute_data(100, 5)
            >>> max_lag = 5
            >>> min_lag = 4
            >>> analysis_opts = {
            >>>     'cmi_estimator':  'JidtKraskovCMI',
            >>>     'n_perm_max_stat': 200,
            >>>     'n_perm_min_stat': 200,
            >>>     'n_perm_omnibus': 500,
            >>>     'n_perm_max_seq': 500,
            >>>     }
            >>> target = 0
            >>> sources = [1, 2, 3]
            >>> network_analysis = MultivariateTE(max_lag, min_lag,
            >>>                                   analysis_opts)
            >>> res = network_analysis.analyse_single_target(dat, target,
            >>>                                              sources)
            >>>
            >>> spectral_opts = {
            >>>     'cmi_estimator':  'JidtKraskovCMI',
            >>>     'n_perm_spec': 200,
            >>>     'alpha_spec': 0.05
            >>>     }
            >>> spectral_analysis = Multivariate_spectral_te(spectral_opts)
            >>> res_spec = spectral_analysis.analyse_single_target(res, dat)

            Note:
            For more details on the estimation of multivariate transfer entropy
            see documentation of class method 'analyse_single_target'.

            Args:
            res_network: dict
                results from multivariate network inference, e.g., using TE
            data : Data instance
                raw data from which the network was inferred
            sources : list of int | int | 'all' [optional]
                single index or list of indices of source processes
                (default='all'), if 'all', all possible sources for the given
                target are tested

        Returns:
            dict
                results consisting of sets of selected variables as (full, from
                sources only, from target only), pvalues and TE for each
                significant source variable, the current value for this
                analysis, results for omnibus test (joint influence of all
                selected source variables on the target, omnibus TE, p-value,
                and significance); NOTE that all variables are listed as tuples
                (process, lag wrt. current value)
        """
        # Convert lags in the results structure to absolute indices
        idx_list_sources = self._lag_to_idx(
                    lag_list=res_target[target]['selected_vars_sources'],
                    current_value_sample=res_target[target]['current_value'])

        # Main algorithm.
        for s in idx_list_sources:
            # TODO do stuff
            # new methods in class Data():
            # dat._get_data_slice
            # dat.slice_permute_samples
            # dat.slice_permute_replications
            # new method in module stats:
            # stats._generate_spectral_surrogates

        return 1
