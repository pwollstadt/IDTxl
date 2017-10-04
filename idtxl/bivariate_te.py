"""Estimate bivariate transfer entropy.

Calculate bivariate transfer entropy (TE) using the maximum statistic.

Created on Wed Apr 06 17:58:31 2016

Note:
    Written for Python 3.4+

@author: patricia
"""
import numpy as np
import itertools as it
from . import stats
from .network_inference import NetworkInference


class BivariateTE(NetworkInference):
    """Perform network inference using bivariate transfer entropy.

    Perform network inference using bivariate transfer entropy (TE). To
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
        """Find bivariate transfer entropy between all nodes in the network.

        Estimate bivariate transfer entropy (TE) between all nodes in the
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
            >>>     'max_lag': 5,
            >>>     'min_lag': 4
            >>>     }
            >>> network_analysis = Bivariate_te()
            >>> res = network_analysis.analyse_network(settings, dat)

        Args:
            settings : dict
                parameters for estimation and statistical testing, see
                documentation of analyse_single_target() for details, settings
                can further contain

                - verbose : bool [optional] - toggle console output
                  (default=True)

            data : Data instance
                raw data for analysis
            targets : list of int | 'all' [optional]
                index of target processes (default='all')
            sources : list of int | list of list | 'all'  [optional]
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
                analyse_single_target()
        """
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
        settings.setdefault('verbose', True)
        results = {}
        for t in range(len(targets)):
            if settings['verbose']:
                print('####### analysing target {0} of {1}'.format(t, targets))
            r = self.analyse_single_target(settings, data,
                                           targets[t], sources[t])
            r['target'] = targets[t]
            r['sources'] = sources[t]
            results[targets[t]] = r
        return results

    def analyse_single_target(self, settings, data, target, sources='all'):
        """Find bivariate transfer entropy between sources and a target.

        Find bivariate transfer entropy (TE) between all potential source
        processes and the target process. Uses bivariate, non-uniform embedding
        found through information maximisation (see Faes et al., 2011, Phys Rev
        E 83, 051112 and Lizier & Rubinov, 2012, Max Planck Institute:
        Preprint. Retrieved from
        http://www.mis.mpg.de/preprints/2012/preprint2012_25.pdf). Bivariate
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
            >>>     'max_lag': 5,
            >>>     'min_lag': 4
            >>>     }
            >>> target = 0
            >>> sources = [1, 2, 3]
            >>> network_analysis = Bivariate_te()
            >>> res = network_analysis.analyse_single_target(settings,
            >>>                                              dat, target,
            >>>                                              sources)

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
                - max_lag_target : int [optional] - maximum temporal search
                  depth for candidates in the target's past in samples
                  (default=same as max_lag_sources)
                - tau_sources : int [optional] - spacing between candidates in
                  the sources' past in samples (default=1)
                - tau_target : int [optional] - spacing between candidates in
                  the target's past in samples (default=1)
                - n_perm_* : int - number of permutations, where * can be
                  'max_stat', 'min_stat', 'omnibus', and 'max_seq'
                  (default=500)
                - alpha_* : float - critical alpha level for statistical
                  significance, where * can be 'max_stats',  'min_stats', and
                  'omnibus' (default=0.05)
                - add_conditionals : list of tuples | str [optional] - force
                  the estimator to add these conditionals when estimating TE;
                  can either be a list of variables, where each variable is
                  described as (idx process, lag wrt to current value) or can
                  be a string: 'faes' for Faes-Method (see references)
                - permute_in_time : bool [optional] - force surrogate creation
                  by shuffling realisations in time instead of shuffling
                  replications; see documentation of Data.permute_samples() for
                  further settings (default=False)
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
        print('\n---------------------------- (3) omnibus test')
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
            'selected_vars_sources': self._idx_to_lag(
                                                self.selected_vars_sources),
            'selected_vars_target': self._idx_to_lag(
                                                self.selected_vars_target),
            'selected_sources_pval': self.pvalues_sign_sources,
            'selected_sources_te': self.te_sign_sources,
            'omnibus_te': self.te_omnibus,
            'omnibus_pval': self.pvalue_omnibus,
            'omnibus_sign': self.sign_omnibus}
        self._reset()  # remove attributes
        return results

    def _include_source_candidates(self, data):
        """Inlcude informative candidates into the conditioning set.

        Loop over each candidate in the candidate set and test if it has
        significant mutual information with the current value, conditional
        on all samples that were informative in previous rounds and are already
        in the conditioning set. If this conditional mutual information is
        significant using maximum statistics, add the current candidate to the
        conditional set.

        Args:
            data : Data instance
                raw data
        """
        # Define candidate set and get realisations.
        procs = self.source_set
        samples = np.arange(
                self.current_value[1] - self.settings['min_lag_sources'],
                self.current_value[1] - self.settings['max_lag_sources'],
                -self.settings['tau_sources'])
        candidate_set = self._define_candidates(procs, samples)
        self._append_selected_vars(
                candidate_set,
                data.get_realisations(self.current_value, candidate_set)[0])

        # Perform one round of sequential max statistics.
        [s, p, te] = stats.max_statistic_sequential(self, data)

        # Remove non-significant sources from the candidate set. Loop
        # backwards over the candidates to remove them iteratively.
        for i in range(s.shape[0] - 1, -1, -1):
            if not s[i]:
                self._remove_selected_var(self.selected_vars_sources[i])
                p = np.delete(p, i)
                te = np.delete(te, i)
        self.pvalues_sign_sources = p
        self.te_sign_sources = te

    def _define_candidates(self, processes, samples):
        """Build a list of candidate indices.

        Args:
            processes : list of int
                process indices
            samples: list of int
                sample indices

        Returns:
            a list of tuples, where each tuple holds the index of one
            candidate and has the form (process index, sample index)
        """
        candidate_set = []
        for idx in it.product(processes, samples):
            candidate_set.append(idx)
        return candidate_set

    def _test_final_conditional(self, data):
        """Perform statistical test on the final conditional set."""
        if not self.selected_vars_sources:
            print('---------------------------- no sources found')
            self.te_omnibus = None
            self.sign_omnibus = False
            self.pvalue_omnibus = None
        else:
            print(self._idx_to_lag(self.selected_vars_full))
            [s, p, te] = stats.omnibus_test(self, data)
            self.te_omnibus = te
            self.sign_omnibus = s
            self.pvalue_omnibus = p
