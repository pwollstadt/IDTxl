"""Estimate multivarate TE.

Created on Thu Mar 10 14:24:31 2016

Iterative greedy algorithm for multivariate network inference using transfer
entropy. For details see Lizier 2012 and Faes 2011.

Note:
    Written for Python 3.4+

@author: patricia
"""
import numpy as np
from . import stats
from .network_inference import NetworkInference
from .stats import network_fdr


class MultivariateTE(NetworkInference):
    """Perform network inference using multivariate transfer entropy.

    Perform network inference using multivariate transfer entropy (TE). To
    perform network inference call analyse_network() on the whole network or a
    set of nodes or call analyse_single_target() to estimate TE for a single
    target. See docstrings of the two functions for more information.

    References:

        - Schreiber, T. (2000). Measuring Information Transfer. Phys Rev Lett,
          85(2), 461–464. http://doi.org/10.1103/PhysRevLett.85.461
        - Vicente, R., Wibral, M., Lindner, M., & Pipa, G. (2011). Transfer
          entropy-a model-free measure of effective connectivity for the
          neurosciences. J Comp Neurosci, 30(1), 45–67.
          http://doi.org/10.1007/s10827-010-0262-3
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
            index of the current value in TE estimation, (idx process,
            idx sample)
        selected_vars_full : list of tuples
            samples in the full conditional set, (idx process, idx sample)
        selected_vars_sources : list of tuples
            source samples in the conditional set, (idx process, idx sample)
        selected_vars_target : list of tuples
            target samples in the conditional set, (idx process, idx sample)
        pvalue_omnibus : float
            p-value of the omnibus test
        pvalues_sign_sources : numpy array
            array of p-values for TE from individual sources to the target
        te_omnibus : float
            joint TE from all sources to the target
        te_sign_sources : numpy array
            raw TE values from individual sources to the target
        sign_ominbus : bool
            statistical significance of the over-all TE
    """

    def __init__(self):
        super().__init__()

    def analyse_network(self, settings, data, targets='all', sources='all'):
        """Find multivariate transfer entropy between all nodes in the network.

        Estimate multivariate transfer entropy (TE) between all nodes in the
        network or between selected sources and targets.

        Note:
            For a detailed description see the documentation of the
            analyse_single_target() method of this class and the references.

        Example:

            >>> dat = Data()
            >>> dat.generate_mute_data(100, 5)
            >>> settings = {
            >>>     'cmi_estimator':  'JidtKraskovCMI',
            >>>     'n_perm_max_stat': 200,
            >>>     'n_perm_min_stat': 200,
            >>>     'n_perm_omnibus': 500,
            >>>     'n_perm_max_seq': 500,
            >>>     'max_lag_sources': 5,
            >>>     'min_lag_sources': 2
            >>>     }
            >>> network_analysis = MultivariateTE()
            >>> res = network_analysis.analyse_network(settings, dat)

        Args:
            settings : dict
                parameters for estimation and statistical testing, see
                documentation of analyse_single_target() for details, settings
                can further contain

                - 'verbose' : bool [optional] - toggle console output
                  (default=True)
                - 'fdr_correction' : bool [optional] - correct results on the
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

        # Perform TE estimation for each target individually
        results = {}
        for t in range(len(targets)):
            if settings['verbose']:
                print('####### analysing target with index {0} from list {1}'
                      .format(t, targets))
            results[targets[t]] = self.analyse_single_target(settings,
                                                             data,
                                                             targets[t],
                                                             sources[t])

        # Perform FDR-correction on the network level. Add FDR-corrected
        # results as an extra field. Network_fdr/combine_results internally
        # creates a deep copy of the results.
        if settings['fdr_correction']:
            results['fdr_corrected'] = network_fdr(settings, results)

        return results

    def analyse_single_target(self, settings, data, target, sources='all'):
        """Find multivariate transfer entropy between sources and a target.

        Find multivariate transfer entropy (TE) between all source processes
        and the target process. Uses multivariate, non-uniform embedding found
        through information maximisation (see Faes et al., 2011, Phys Rev E 83,
        051112 and Lizier & Rubinov, 2012, Max Planck Institute: Preprint.
        Retrieved from
        http://www.mis.mpg.de/preprints/2012/preprint2012_25.pdf). Multivariate
        TE is calculated in four steps (see Lizier and Faes for details):

        (1) find all relevant samples in the target processes' own past, by
            iteratively adding candidate samples that have significant
            conditional mutual information (CMI) with the current value
            (conditional on all samples that were added previously)
        (2) find all relevant samples in the source processes' pasts (again
            by finding all candidates with significant CMI)
        (3) prune the final conditional set by testing the CMI between each
            sample in the final set and the current value, conditional on all
            other samples in the final set
        (4) statistics on the final set of sources (test for over-all transfer
            between the final conditional set and the current value, and for
            significant transfer of all individual samples in the set)

        Example:

            >>> dat = Data()
            >>> dat.generate_mute_data(100, 5)
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
            >>> network_analysis = MultivariateTE()
            >>> res = network_analysis.analyse_single_target(settings,
            >>>                                              dat, target,
            >>>                                              sources)

        Args:
            settings : dict
                parameters for estimation and statistical testing:

                - 'cmi_estimator' : str - estimator to be used for CMI
                  calculation (for estimator settings see the documentation in
                  the estimators_* modules)
                - 'max_lag_sources' : int - maximum temporal search depth for
                  candidates in the sources' past in samples
                - 'min_lag_sources' : int - minimum temporal search depth for
                  candidates in the sources' past in samples
                - 'max_lag_target' : int [optional] - maximum temporal search
                  depth for candidates in the target's past in samples
                  (default=same as max_lag_sources)
                - 'tau_sources' : int [optional] - spacing between candidates in
                  the sources' past in samples (default=1)
                - 'tau_target' : int [optional] - spacing between candidates in
                  the target's past in samples (default=1)
                - 'n_perm_*' : int [optional] - number of permutations, where *
                  can be 'max_stat', 'min_stat', 'omnibus', and 'max_seq'
                  (default=500)
                - 'alpha_*' : float [optional] - critical alpha level for
                  statistical significance, where * can be 'max_stats',
                  'min_stats', 'omnibus', and 'max_seq' (default=0.05)
                - 'add_conditionals' : list of tuples | str [optional] - force
                  the estimator to add these conditionals when estimating TE;
                  can either be a list of variables, where each variable is
                  described as (idx process, lag wrt to current value) or can
                  be a string: 'faes' for Faes-Method (see references)
                - 'permute_in_time' : bool [optional] - force surrogate
                  creation by shuffling realisations in time instead of
                  shuffling replications; see documentation of
                  Data.permute_samples() for further settings (default=False)
                - 'verbose' : bool [optional] - toggle console output
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
                variables from the sources' past, variables from the target's
                past), pvalues and TE for each selected variable, the current
                value for this analysis, results for omnibus test (joint
                influence of all selected source variables on the target,
                omnibus TE, p-value, and significance); NOTE that all variables
                are listed as tuples (process, lag wrt. current value)
        """
        # Check input and clean up object if it was used before.
        self._initialise(settings, data, sources, target)

        # Main algorithm.
        print('\n---------------------------- (1) include target candidates')
        self._include_target_candidates(data)
        print('\n---------------------------- (2) include source candidates')
        self._include_source_candidates(data)
        print('\n---------------------------- (3) prune source candidate')
        self._prune_candidates(data)
        print('\n---------------------------- (4) final statistics')
        self._test_final_conditional(data)

        # Clean up and return results.
        if self.settings['verbose']:
            print('final source samples: {0}'.format(
                    self._idx_to_lag(self.selected_vars_sources)))
            print('final target samples: {0}'.format(
                    self._idx_to_lag(self.selected_vars_target)))
        results = {
            'target': self.target,
            'sources_tested': self.source_set,
            'settings': self.settings,
            'current_value': self.current_value,
            'selected_vars_full': self._idx_to_lag(self.selected_vars_full),
            'selected_vars_target': self._idx_to_lag(
                                                self.selected_vars_target),
            'selected_vars_sources': self._idx_to_lag(
                                                self.selected_vars_sources),
            'selected_sources_pval': self.pvalues_sign_sources,
            'selected_sources_te': self.te_sign_sources,
            'omnibus_te': self.te_omnibus,
            'omnibus_pval': self.pvalue_omnibus,
            'omnibus_sign': self.sign_omnibus
            }
        self._reset()  # remove attributes
        return results

    def _prune_candidates(self, data):
        """Remove uninformative candidates from the final conditional set.

        For each sample in the final conditioning set, check if it is
        informative about the current value given all other samples in the
        final set. If a sample is not informative, it is removed from the
        final set.

        Args:
            data : Data instance
                raw data
        """
        # FOR LATER we don't need to test the last included in the first round
        print(self.selected_vars_sources)
        while self.selected_vars_sources:
            # Find the candidate with the minimum TE into the target.
            temp_te = np.empty(len(self.selected_vars_sources))
            cond_dim = len(self.selected_vars_full) - 1
            candidate_realisations = np.empty(
                                (data.n_realisations(self.current_value) *
                                 len(self.selected_vars_sources), 1))
            conditional_realisations = np.empty(
                (data.n_realisations(self.current_value) *
                 len(self.selected_vars_sources), cond_dim))

            # calculate TE simultaneously for all candidates
            i_1 = 0
            i_2 = data.n_realisations(self.current_value)
            for candidate in self.selected_vars_sources:
                # Separate the candidate realisations and all other
                # realisations to test the candidate's individual contribution.
                [temp_cond, temp_cand] = self._separate_realisations(
                                                    self.selected_vars_full,
                                                    candidate)
                if temp_cond is None:
                    conditional_realisations = None
                else:
                    conditional_realisations[i_1:i_2, ] = temp_cond
                candidate_realisations[i_1:i_2, ] = temp_cand
                i_1 = i_2
                i_2 += data.n_realisations(self.current_value)

            print(('var1, candidate_realisations: {0}, var2, current_value: '
                   '{1}, cond: {2}').format(
                            candidate_realisations.shape,
                            self._current_value_realisations.shape,
                            conditional_realisations.shape))
            temp_te = self._cmi_estimator.estimate_mult(
                                n_chunks=len(self.selected_vars_sources),
                                re_use=['var2'],
                                var1=candidate_realisations,
                                var2=self._current_value_realisations,
                                conditional=conditional_realisations)

            # Test min TE for significance with minimum statistics.
            te_min_candidate = min(temp_te)
            min_candidate = self.selected_vars_sources[np.argmin(temp_te)]
            if self.settings['verbose']:
                print('testing {0} from candidate set {1}'.format(
                                self._idx_to_lag([min_candidate])[0],
                                self._idx_to_lag(self.selected_vars_sources)),
                      end='')
            [significant, p, surr_table] = stats.min_statistic(
                                              self, data,
                                              self.selected_vars_sources,
                                              te_min_candidate)

            # Remove the minimum it is not significant and test the next min.
            # candidate. If the minimum is significant, break, all other
            # sources will be significant as well (b/c they have higher TE).
            if not significant:
                if self.settings['verbose']:
                    print(' -- not significant')
                self._remove_selected_var(min_candidate)
            else:
                if self.settings['verbose']:
                    print(' -- significant')
                self._min_stats_surr_table = surr_table
                break

    def _test_final_conditional(self, data):  # TODO test this!
        """Perform statistical test on the final conditional set."""
        self.te_omnibus = None
        self.sign_omnibus = False
        self.pvalue_omnibus = None
        self.pvalues_sign_sources = None
        self.te_sign_sources = None

        if not self.selected_vars_sources:
            print('---------------------------- no sources found')
        else:
            print(self._idx_to_lag(self.selected_vars_full))
            [s, p, te] = stats.omnibus_test(self, data)
            self.te_omnibus = te
            self.sign_omnibus = s
            self.pvalue_omnibus = p
            # Test individual links if the omnibus test is significant.
            if self.sign_omnibus:
                [s, p, te] = stats.max_statistic_sequential(self, data)
                # Remove non-significant sources from the candidate set. Loop
                # backwards over the candidates to remove them iteratively.
                for i in range(s.shape[0] - 1, -1, -1):
                    if not s[i]:
                        self._remove_selected_var(
                                                self.selected_vars_sources[i])
                        p = np.delete(p, i)
                        te = np.delete(te, i)
                self.pvalues_sign_sources = p
                self.te_sign_sources = te
            else:
                self.selected_vars_sources = []
                self.selected_vars_full = self.selected_vars_target
