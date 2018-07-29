"""Perform network inference using multivarate mutual information.

Estimate multivariate mutual information (MI) for network inference using a
greedy approach with maximum statistics to generate a non-uniform embedding
(Faes, 2011; Lizier, 2012).

Note:
    Written for Python 3.4+
"""
from .stats import network_fdr
from .network_inference import NetworkInferenceMI, NetworkInferenceMultivariate
from .results import ResultsNetworkInference


class MultivariateMI(NetworkInferenceMI, NetworkInferenceMultivariate):
    """Perform network inference using multivariate mutual information.

    Perform network inference using multivariate mutual information (MI). To
    perform network inference call analyse_network() on the whole network or a
    set of nodes or call analyse_single_target() to estimate MI for a single
    target. See docstrings of the two functions for more information.

    References:

    - Lizier, J. T., & Rubinov, M. (2012). Multivariate construction of
      effective computational networks from observational data. Max Planck
      Institute: Preprint. Retrieved from
      http://www.mis.mpg.de/preprints/2012/preprint2012_25.pdf
    - Faes, L., Nollo, G., & Porta, A. (2011). Information-based detection
      of nonlinear Granger causality in multivariate processes via a
      nonuniform embedding technique. Phys Rev E, 83, 1–15.
      http://doi.org/10.1103/PhysRevE.83.051112

    Attributes:
        source_set : list
            indices of source processes tested for their influence on the
            target
        target : list
            index of target process
        settings : dict
            analysis settings
        current_value : tuple
            index of the current value in MI estimation, (idx process,
            idx sample)
        selected_vars_full : list of tuples
            samples in the full conditional set, (idx process, idx sample)
        selected_vars_sources : list of tuples
            source samples in the conditional set, (idx process, idx sample)
        pvalue_omnibus : float
            p-value of the omnibus test
        pvalues_sign_sources : numpy array
            array of p-values for MI from individual sources to the target
        mi_omnibus : float
            joint MI from all sources to the target
        mi_sign_sources : numpy array
            raw MI values from individual sources to the target
        sign_ominbus : bool
            statistical significance of the over-all MI
    """

    def __init__(self):
        super().__init__()

    def analyse_network(self, settings, data, targets='all', sources='all'):
        """Find multivariate mutual information between nodes in the network.

        Estimate multivariate mutual information (MI) between all nodes in the
        network or between selected sources and targets.

        Note:
            For a detailed description of the algorithm and settings see
            documentation of the analyse_single_target() method and references
            in the class docstring.

        Example:

            >>> data = Data()
            >>> data.generate_mute_data(100, 5)
            >>> # The algorithm uses a conditional mutual information to
            >>> # construct a non-uniform embedding, hence a CMI- not MI-
            >>> # estimator has to be specified:
            >>> settings = {
            >>>     'cmi_estimator':  'JidtKraskovCMI',
            >>>     'n_perm_max_stat': 200,
            >>>     'n_perm_min_stat': 200,
            >>>     'n_perm_omnibus': 500,
            >>>     'n_perm_max_seq': 500,
            >>>     'max_lag_sources': 5,
            >>>     'min_lag_sources': 2
            >>>     }
            >>> network_analysis = MultivariateMI()
            >>> results = network_analysis.analyse_network(settings, data)

        Args:
            settings : dict
                parameters for estimation and statistical testing, see
                documentation of analyse_single_target() for details, settings
                can further contain

                - verbose : bool [optional] - toggle console output
                  (default=True)
                - fdr_correction : bool [optional] - correct results on the
                  network level, see documentation of stats.network_fdr() for
                  details (default=True)

            data : Data instance
                raw data for analysis
            targets : list of int | 'all' [optional]
                index of target processes (default='all')
            sources : list of int | list of list | 'all' [optional]
                indices of source processes for each target (default='all');
                if 'all', all network nodes excluding the target node are
                considered as potential sources and tested;
                if list of int, the source specified by each int is tested as
                a potential source for the target with the same index or a
                single target;
                if list of list, sources specified in each inner list are
                tested for the target with the same index

        Returns:
            dict
                results for each target, see documentation of
                analyse_single_target(); results FDR-corrected, see
                documentation of stats.network_fdr()
        """
        # Set defaults for network inference.
        settings.setdefault('verbose', True)
        settings.setdefault('fdr_correction', True)

        # Check which targets and sources are requested for analysis.
        if targets == 'all':
            targets = [t for t in range(data.n_processes)]
        if sources == 'all':
            sources = ['all' for t in targets]
        if (type(sources) is list) and (type(sources[0]) is int):
            sources = [sources for t in targets]
        if (type(sources) is list) and (type(sources[0]) is list):
            pass
        else:
            ValueError('Sources was not specified correctly: {0}.'.format(
                                                                    sources))
        assert(len(sources) == len(targets)), ('List of targets and list of '
                                               'sources have to have the same '
                                               'same length')

        # Perform MI estimation for each target individually
        results = ResultsNetworkInference(n_nodes=data.n_processes,
                                          n_realisations=data.n_realisations(),
                                          normalised=data.normalise)
        for t in range(len(targets)):
            if settings['verbose']:
                print('\n####### analysing target with index {0} from list {1}'
                      .format(t, targets))
            res_single = self.analyse_single_target(
                    settings, data, targets[t], sources[t])
            results.combine_results(res_single)

        # Get no. realisations actually used for estimation from single target
        # analysis.
        results.data_properties.n_realisations = (
            res_single.data_properties.n_realisations)

        # Perform FDR-correction on the network level. Add FDR-corrected
        # results as an extra field. Network_fdr/combine_results internally
        # creates a deep copy of the results.
        if settings['fdr_correction']:
            results = network_fdr(settings, results)
        return results

    def analyse_single_target(self, settings, data, target, sources='all'):
        """Find multivariate mutual information between sources and a target.

        Find multivariate mutual information (MI) between all source processes
        and the target process. Uses multivariate, non-uniform embedding found
        through information maximisation .

        Multivariate MI is calculated in four steps (see Lizier and Faes for
        details):

        Note:
            For a further description of the algorithm see references in the
            class docstring.

        (1) Find all relevant samples in the source processes' past, by
            iteratively adding candidate samples that have significant
            conditional mutual information (CMI) with the current value
            (conditional on all samples that were added previously)
        (2) Prune the final conditional set by testing the CMI between each
            sample in the final set and the current value, conditional on all
            other samples in the final set
        (3) Statistics on the final set of sources (test for over-all transfer
            between the final conditional set and the current value, and for
            significant transfer of all individual samples in the set)

        Example:

            >>> data = Data()
            >>> data.generate_mute_data(100, 5)
            >>> # The algorithm uses a conditional mutual information to
            >>> # construct a non-uniform embedding, hence a CMI- not MI-
            >>> # estimator has to be specified:
            >>> settings = {
            >>>     'cmi_estimator':  'JidtKraskovCMI',
            >>>     'n_perm_max_stat': 200,
            >>>     'n_perm_min_stat': 200,
            >>>     'n_perm_omnibus': 500,
            >>>     'n_perm_max_seq': 500,
            >>>     'max_lag_sources': 5,
            >>>     'min_lag_sources': 2
            >>>     }
            >>> target = 0
            >>> sources = [1, 2, 3]
            >>> network_analysis = MultivariateMI()
            >>> results = network_analysis.analyse_single_target(settings,
            >>>                                                  data, target,
            >>>                                                  sources)

        Args:
            settings : dict
                parameters for estimation and statistical testing:

                - cmi_estimator : str - estimator to be used for CMI
                  calculation (for estimator settings see the documentation in
                  the estimators_* modules)
                - max_lag_sources : int - maximum temporal search depth for
                  candidates in the sources' past in samples
                - min_lag_sources : int - minimum temporal search depth for
                  candidates in the sources' past in samples
                - tau_sources : int [optional] - spacing between candidates in
                  the sources' past in samples (default=1)
                - n_perm_* : int [optional] - number of permutations, where *
                  can be 'max_stat', 'min_stat', 'omnibus', and 'max_seq'
                  (default=500)
                - alpha_* : float [optional] - critical alpha level for
                  statistical significance, where * can be 'max_stats',
                  'min_stats', 'omnibus', and 'max_seq' (default=0.05)
                - add_conditionals : list of tuples | str [optional] - force
                  the estimator to add these conditionals when estimating MI;
                  can either be a list of variables, where each variable is
                  described as (idx process, lag wrt to current value) or can
                  be a string: 'faes' for Faes-Method (see references)
                - permute_in_time : bool [optional] - force surrogate
                  creation by shuffling realisations in time instead of
                  shuffling replications; see documentation of
                  Data.permute_samples() for further settings (default=False)
                - verbose : bool [optional] - toggle console output
                  (default=True)

            data : Data instance
                raw data for analysis
            target : int
                index of target process
            sources : list of int | int | 'all' [optional]
                single index or list of indices of source processes
                (default='all'), if 'all', all network nodes excluding the
                target node are considered as potential sources

        Returns:
            dict
                results consisting of sets of selected variables as (full set,
                variables from the sources' past), pvalues and MI for each
                selected variable, the current value for this analysis, results
                for omnibus test (joint MI between all selected source
                variables and the target, omnibus MI, p-value, and
                significance); NOTE that all variables are listed as tuples
                (process, lag wrt. current value)
        """
        # Check input and clean up object if it was used before.
        self._initialise(settings, data, sources, target)

        # Main algorithm.
        print('\n---------------------------- (1) include source candidates')
        self._include_source_candidates(data)
        print('\n---------------------------- (2) prune source candidate')
        self._prune_candidates(data)
        print('\n---------------------------- (3) final statistics')
        self._test_final_conditional(data)

        # Clean up and return results.
        if self.settings['verbose']:
            print('final source samples: {0}'.format(
                    self._idx_to_lag(self.selected_vars_sources)))
        results = ResultsNetworkInference(
            n_nodes=data.n_processes,
            n_realisations=data.n_realisations(self.current_value),
            normalised=data.normalise)
        results._add_single_result(
            target=self.target,
            settings=self.settings,
            results={
                'sources_tested': self.source_set,
                'current_value': self.current_value,
                'selected_vars_target': self._idx_to_lag(
                    self.selected_vars_target),
                'selected_vars_sources': self._idx_to_lag(
                    self.selected_vars_sources),
                'selected_sources_pval': self.pvalues_sign_sources,
                'selected_sources_mi': self.statistic_sign_sources,
                'omnibus_mi': self.statistic_omnibus,
                'omnibus_pval': self.pvalue_omnibus,
                'omnibus_sign': self.sign_omnibus,
                'mi': self.statistic_single_link
            })
        self._reset()  # remove attributes
        return results
