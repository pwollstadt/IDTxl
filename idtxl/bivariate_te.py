# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 17:58:31 2016

Greedy algorithm for multivariate network inference using transfer entropy.
For details see Lizier ??? and Faes ???.

If executed as standalone, the script applies the algorithm to example data
presented in Montalto, PLOS ONE, 2014, (eq. 14).

Eample:
    python multivariate_te.py

Note:
    Written for Python 3.4+

@author: patricia
"""
import copy as cp
import numpy as np
import itertools as it
import stats
from network_analysis import Network_analysis
from set_estimator import Estimator_cmi

VERBOSE = True


class Bivariate_te(Network_analysis):
    """Set up a network analysis using bivariate transfer entropy.

    Set parameters necessary for network inference using transfer entropy (TE).
    To perform network inference call analyse_network() on an instance of the
    data class.

    Args:
        max_lag : int
            maximum temporal search depth
        min_lag : int
            minimum temporal search depth
        options : dict [optional]
            parameters for estimator use and statistics:

            - 'n_perm_*' - number of permutations, where * can be 'max_stat',
            - 'min_stat', 'omnibus', and 'max_seq' (default=500)
            - 'alpha_*' - critical alpha level for statistical significance,
              where * can be 'max_stats',  'min_stats', and 'omnibus'
              (default=0.05)
            - 'cmi_calc_name' - estimator to be used for CMI calculation
              (For estimator options see the respective documentation.)
            - 'add_conditionals' - force the estimator to add these
              conditionals when estimating TE; can either be a list of
              variables, where each variable is described as (idx process, lag
              wrt to current value) or can be a string: 'faes' for Faes-Method

    Attributes:
        selected_vars_full : list of tuples
            samples in the full conditional set, (idx process, idx sample)
        selected_vars_sources : list of tuples
            source samples in the conditional set, (idx process, idx sample)
        selected_vars_target : list of tuples
            target samples in the conditional set, (idx process, idx sample)
        current_value : tuple
            index of the current value in TE estimation, (idx process,
            idx sample)
        calculator_name : string
            calculator used for TE estimation
        max_lag_target : int
            maximum temporal search depth for candidates in the target's past
            (default=same as max_lag_sources)
        max_lag_sources : int
            maximum temporal search depth for candidates in the sources' past
        min_lag_sources : int
            minimum temporal search depth for candidates in the sources' past
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
        source_set : list
            list with indices of source processes
        target : list
            index of target process

    """
    # TODO right now 'options' holds all optional params (stats AND estimator).
    # We could split this up by adding the stats options to the analyse_*
    # methods?
    def __init__(self, max_lag_sources, min_lag_sources, max_lag_target,
                 tau_sources=1, tau_target=1, options=None):

        if max_lag_target is None:
            self.max_lag_target = max_lag_sources
        else:
            self.max_lag_target = max_lag_target
        self.max_lag_sources = max_lag_sources
        self.min_lag_sources = min_lag_sources
        self.tau_sources = tau_sources
        self.tau_target = tau_target
        self.te_omnibus = None
        self.te_sign_sources = None
        self.sign_omnibus = False
        self.sign_sign_sources = None
        self.pvalue_omnibus = None
        self.pvalues_sign_sources = None
        self.min_stats_surr_table = None
        self.options = options
        try:
            self.calculator_name = options['cmi_calc_name']
        except KeyError:
            raise KeyError('Calculator name was not specified!')
        self._cmi_calculator = Estimator_cmi(self.calculator_name)  # TODO should be 'calculator'
        super().__init__()

    def analyse_network(self, data, targets='all', sources='all'):
        """Find multivariate transfer entropy between all nodes in the network.

        Estimate multivariate transfer entropy between provided sources and
        each target. Custom source sets can be provided for each target, as
        lists of lists of nodes.

        Example:

            >>> dat = Data()
            >>> dat.generate_mute_data(100, 5)
            >>> max_lag = 5
            >>> min_lag = 4
            >>> analysis_opts = {
            >>>     'cmi_calc_name': 'jidt_kraskov',
            >>>     'n_perm_max_stat': 200,
            >>>     'n_perm_min_stat': 200,
            >>>     'n_perm_omnibus': 500,
            >>>     'n_perm_max_seq': 500,
            >>>     }
            >>> target = 0
            >>> sources = [1, 2, 3]
            >>> network_analysis = Multivariate_te(max_lag, analysis_opts,
            >>>                                    min_lag)
            >>> res = network_analysis.analyse_single_target(dat, target,
            >>>                                              sources)

        Note:
            For more details on the estimation of multivariate transfer entropy
            see documentation of class method 'analyse_single_target'.

        Args:
            data : Data instance
                raw data for analysis
            targets : list of int
                index of target processes
            sources : list of int | list of list | 'all'
                indices of source processes for each target;
                if 'all', all sources are tested for each target;
                if list of int, sources specified in the list are tested for
                each target;
                if list of list, sources specified in each inner list are
                tested for the corresponding target
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

        # Perform TE estimation for each target individually. FDR-correct
        # overall results.
        results = {}
        for t in range(len(targets)):
            if VERBOSE:
                print('####### analysing target {0} of {1}'.format(t, targets))
            r = self.analyse_single_target(data, targets[t], sources[t])
            r['target'] = targets[t]
            r['sources'] = sources[t]
            results[targets[t]] = r
        results['fdr'] = stats.network_fdr(results)
        return results

    def analyse_single_target(self, data, target, sources='all'):
        """Find multivariate transfer entropy between sources and a target.

        Find multivariate transfer entropy between all source processes and the
        target process. Uses multivariate, non-uniform embedding found through
        information maximisation (see Faes, ???, and Lizier, 2012). This is
        done in four steps (see Lizier and Faes for details):

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

        Args:
            data : Data instance
                raw data for analysis
            target : int
                index of target process
            sources : list of int, int, or 'all'
                single index or list of indices of source processes, if 'all',
                all possible sources for the given target are tested

        Returns:
            dict
                results consisting of
                conditional sets (full, from sources, from target),
                results for omnibus test (joint influence of source cands.),
                pvalues for each significant source candidate
        """
        # Check input and clean up object if it was used before.
        self._initialise(data, sources, target)

        # Main algorithm.
        print('\n---------------------------- (1) include target candidates')
        self._include_target_candidates(data)
        print('\n---------------------------- (2) include source candidates')
        self._include_source_candidates(data)
        print('\n---------------------------- (3) omnibus test')
        [s, p, te] = stats.omnibus_test(self, data, self.options)
        self.te_omnibus = te
        self.sign_omnibus = s
        self.pvalue_omnibus = p

        # Clean up and return results.
        if VERBOSE:
            print('final source samples: {0}'.format(
                    self._idx_to_lag(self.selected_vars_sources)))
            print('final target samples: {0}'.format(
                    self._idx_to_lag(self.selected_vars_target)))
        self._clean_up()
        results = {
            'current_value': self.current_value,
            'selected_vars_full': self._idx_to_lag(self.selected_vars_full),
            'selected_vars_sources': self._idx_to_lag(
                                                self.selected_vars_sources),
            'selected_vars_target': self._idx_to_lag(
                                                self.selected_vars_target),
            'omnibus_te': self.te_omnibus,
            'omnibus_pval': self.pvalue_omnibus,
            'omnibus_sign': self.sign_omnibus,
            'cond_sources_pval': self.pvalues_sign_sources,
            'cond_sources_te': self.te_sign_sources}
        return results

    def _initialise(self, data, sources, target):
        """Check input and set everything to initial values."""

        # Check the provided target and sources.
        self.target = target
        self._check_source_set(sources, data.n_processes)

        # Check provided search depths for source and target
        assert(self.min_lag_sources <= self.max_lag_sources), (
            'min_lag_sources ({0}) must be smaller or equal to max_lag_sources'
            ' ({1}).'.format(self.min_lag_sources, self.max_lag_sources))
        max_lag = max(self.max_lag_sources, self.max_lag_target)
        assert(data.n_samples >= max_lag + 1), (
            'Not enough samples in data ({0}) to allow for the chosen maximum '
            'lag ({1})'.format(data.n_samples, max_lag))
        self._current_value = (target, max_lag)
        [cv_realisation, repl_idx] = data.get_realisations(
                                             current_value=self.current_value,
                                             idx=[self.current_value])
        self._current_value_realisations = cv_realisation

        # Remember which realisations come from which replication. This may be
        # needed for surrogate creation at a later point.
        self._replication_index = repl_idx

        # Check the permutation type and no. permutations requested by the
        # user. This tests if there is sufficient data to do all tests.
        # surrogates.check_permutations(self, data)

        # Reset all attributes to inital values if the instance has been used
        # before.
        if self.selected_vars_full:
            self.selected_vars_full = []
            self._selected_vars_realisations = None
            self.selected_vars_sources = []
            self.selected_vars_target = []
            self.te_omnibus = None
            self.sign_sign_sources = None
            self.pvalue_omnibus = None
            self.pvalues_sign_sources = None
            self.te_sign_sources = None
            self.min_stats_surr_table = None

        # Check if the user provided a list of candidates that must go into
        # the conditioning set. These will be added and used for TE estimation,
        # but never tested for significance.
        try:
            cond = self.options['add_conditionals']
            self._force_conditionals(cond, data)
        except KeyError:
            pass

    def _check_source_set(self, sources, n_processes):
        """Set default if no source set was provided by the user."""
        if sources == 'all':
            sources = [x for x in range(n_processes)]
            sources.pop(self.target)
        elif type(sources) is int:
            sources = [sources]

        if self.target in sources:
            raise RuntimeError('The target {0} should not be in the list '
                               'of sources {1}.'.format(self.target,
                                                        sources))
        else:
            self.source_set = sources
            if VERBOSE:
                print('Testing sources {0}'.format(self.source_set))

    def _include_target_candidates(self, data):
        """Test candidates from the target's past."""
        procs = [self.target]
        samples = np.arange(self.current_value[1] - 1,
                            self.current_value[1] - self.max_lag_target - 1,
                            -self.tau_target)
        candidates = self._define_candidates(procs, samples)
        sources_found = self._include_candidates(candidates, data)

        # If no candidates were found in the target's past, add at least one
        # sample so we are still calculating a proper TE.
        if not sources_found:  # TODO put a flag in to make this optional
            print(('No informative sources in the target''s past - ' +
                   'adding point at t-1 in the target'))
            idx = (self.current_value[0], self.current_value[1] - 1)
            realisations = data.get_realisations(self.current_value, [idx])[0]
            self._append_selected_vars_idx([idx])
            self._append_selected_vars_realisations(realisations)

    def _include_source_candidates(self, data):
        """Test candidates in the source's past."""
        procs = self.source_set
        samples = np.arange(self.current_value[1] - self.min_lag_sources,
                            self.current_value[1] - self.max_lag_sources,
                            -self.tau_sources)
        candidates = self._define_candidates(procs, samples)
        # TODO include non-selected target candidates as further candidates, they may get selected due to synergies
        self._include_candidates(candidates, data)

    def _include_candidates(self, candidate_set, data):
        """Inlcude informative candidates into the conditioning set.

        Loop over each candidate in the candidate set and test if it has
        significant mutual information with the current value, conditional
        on all samples that were informative in previous rounds and are already
        in the conditioning set. If this conditional mutual information is
        significant using maximum statistics, add the current candidate to the
        conditional set.

        Args:
            candidate_set : list of tuples
                candidate set to be tested, where each entry is a tuple
                (process index, sample index)
            data : Data instance
                raw data
            options : dict [optional]
                parameters for estimation and statistical testing

        Returns:
            list of tuples
                indices of the conditional set created from the candidate set
            selected_vars_realisations : numpy array
                realisations of the conditional set
        """
        self._append_selected_vars_idx(candidate_set)
        self._append_selected_vars_realisations(
                data.get_realisations(self.current_value, candidate_set)[0])
        [s, p, te] = stats.max_statistic_sequential(self, data, self.options)
        # Remove non-significant sources from the candidate set. Loop
        # backwards over the candidates to remove them iteratively.
        for i in range(s.shape[0] - 1, -1, -1):
            if not s[i]:
                self._remove_candidate(self.selected_vars_sources[i])
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

