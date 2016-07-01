# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:24:31 2016

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
from . import stats
from .network_analysis import Network_analysis
from .set_estimator import Estimator_cmi
from .set_estimator import Estimator_ais

VERBOSE = True


class Single_process_storage(Network_analysis):
    """Set up analysis of storage in each process of the network.

    Set parameters necessary for active information storage (AIS) in every
    process of a network. To perform network inference call analyse_network() on
    an instance of the data class.

    Args:
        max_lag : int
            maximum temporal search depth
        tau : int
            spacing between samples analyzed for information contribution
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
        current_value : tuple
            index of the current value in TE estimation, (idx process,
            idx sample)
        calculator_name : string
            calculator used for TE estimation
        max_lag : int
            maximum temporal search depth for candidates in the target's past
            (default=same as max_lag_sources)
        tau : int
            spacing between samples analyzed for information contribution
        ais_sign_processes : numpy array
            raw AIS values from individual processes
        pvalues_sign_processes : numpy array
            p-values of significant AIS
        process_set : list
            list with indices of analyzed processes
    """
    # TODO right now 'options' holds all optional params (stats AND estimator).
    # We could split this up by adding the stats options to the analyse_*
    # methods?
    def __init__(self, max_lag, options, tau=1):
        self.max_lag = max_lag
        self.tau = tau
        self.pvalue = None
        self.sign = False
        self.ais = None
        self.min_stats_surr_table = None
        self.options = options
        try:
            self.calculator_name = options['cmi_calc_name']
        except KeyError:
            raise KeyError('Calculator name was not specified!')
        self._cmi_calculator = Estimator_cmi(self.calculator_name)  # TODO should be 'calculator'
        super().__init__()

    def analyse_network(self, data, processes='all'):
        """Estimate active information storage for all processes in the network.

        Estimate active information storage for all or selected processes in the
        network.

        Example:

            >>> dat = Data()
            >>> dat.generate_mute_data(100, 5)
            >>> max_lag = 5
            >>> analysis_opts = {
            >>>     'cmi_calc_name': 'jidt_kraskov',
            >>>     'n_perm_max_stat': 200,
            >>>     'n_perm_min_stat': 200,
            >>>     'n_perm_omnibus': 500,
            >>>     'n_perm_max_seq': 500,
            >>>     }
            >>> processes = [1, 2, 3]
            >>> network_analysis = Single_process_storage(max_lag,
                                                          analysis_opts,
                                                          tau=1)
            >>> res = network_analysis.analyse_network(dat, processes)

        Note:
            For more details on the estimation of active information storage
            see documentation of class method 'analyse_single_process'.

        Args:
            data : Data instance
                raw data for analysis
            process : list of int | 'all'
                index of processes
                if 'all', AIS is estimated for all processes;
                if list of int, AIS is estimated for processes specified in the
                list.
        """
        if processes == 'all':
            processes = [t for t in range(data.n_processes)]
        if (type(processes) is list) and (type(processes[0]) is int):
            pass
        else:
            ValueError('Processes were not specified correctly: {0}.'.format(
                                                                    processes))

        # Perform AIS estimation for each target individually.
        results = {}
        for t in range(len(processes)):
            if VERBOSE:
                print('####### analysing process {0} of {1}'.format(t,
                                                                    processes))
            r = self.analyse_single_process(data, processes[t])
            r['process'] = processes[t]
            results[processes[t]] = r
            # TODO FDR correct this
        return results

    def analyse_single_process(self, data, process):
        """Estimate active information storage for a single process.

        Estimate active information storage for one processes in the network.
        Uses non-uniform embedding found through information maximisation (see
        Faes, 2011, and Lizier, ???). This is
        done in three steps (see Lizier and Faes for details):

        (1) find all relevant samples in the processes' own past, by iteratively
            adding candidate samples that have significant conditional mutual
            information (CMI) with the current value (conditional on all samples
            that were added previously)
        (3) prune the final conditional set by testing the CMI between each
            sample in the final set and the current value, conditional on all
            other samples in the final set
        (4) statistics on the final set of sources (test for over-all transfer
            between the final conditional set and the current value, and for
            significant transfer of all individual samples in the set)

        Args:
            data : Data instance
                raw data for analysis
            process : int
                index of process

        Returns:
            dict
                results consisting of conditional sets (full, from sources,
                from target), results for omnibus test (joint influence of
                source cands.), pvalues for each significant source candidate
        """
        # Check input and clean up object if it was used before.
        self._initialise(data, process)

        # Main algorithm.
        print('\n---------------------------- (1) include candidates')
        procs = [self.process]
        samples = np.arange(self.current_value[1] - 1,
                            self.current_value[1] - self.max_lag - 1,
                            -self.tau)
        candidates = self._define_candidates(procs, samples)
        samples_found = self._include_candidates(candidates, data)


        print('\n---------------------------- (2) prune source candidate')
        # If no candidates were found in the process' past, return 0.
        if samples_found:
            self._prune_candidates(data)
            print('\n---------------------------- (3) final statistics')
            if self._selected_vars_full:
                self._test_final_conditional(data)

        # Clean up and return results.
        if VERBOSE:
            print('final conditional samples: {0}'.format(
                    self._idx_to_lag(self.selected_vars_full)))
        self._clean_up()
        results = {
            'current_value': self.current_value,
            'selected_vars': self._idx_to_lag(self.selected_vars_full),
            'ais': self.ais,
            'ais_pval': self.pvalue,
            'ais_sign': self.sign}
        return results

    def _initialise(self, data, process):
        """Check input and set everything to initial values."""

        # Check the provided process.
        self.process = process

        # Check provided search depths for source and target
        assert(data.n_samples >= self.max_lag + 1), (
            'Not enough samples in data ({0}) to allow for the chosen maximum '
            'lag ({1})'.format(data.n_samples, self.max_lag))
        self._current_value = (process, self.max_lag)
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
            self.pvalue = None
            self.sign = False
            self.ais = None
            self.min_stats_surr_table = None

        # Check if the user provided a list of candidates that must go into
        # the conditioning set. These will be added and used for TE estimation,
        # but never tested for significance.
        try:
            cond = self.options['add_conditionals']
            self._force_conditionals(cond, data)
        except KeyError:
            pass

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
        success = False
        while candidate_set:
            # Find the candidate with maximum TE.
            temp_te = np.empty(len(candidate_set))
            i = 0
            for candidate in candidate_set:
                candidate_realisations = data.get_realisations(
                                                            self.current_value,
                                                            [candidate])[0]
                temp_te[i] = self._cmi_calculator.estimate(
                                        candidate_realisations,
                                        self._current_value_realisations,
                                        self._selected_vars_realisations,
                                        self.options)
                i += 1

            # Test max TE for significance with maximum statistics.
            te_max_candidate = max(temp_te)
            max_candidate = candidate_set[np.argmax(temp_te)]
            if VERBOSE:
                print('testing {0} from candidate set {1}'.format(
                                    self._idx_to_lag([max_candidate])[0],
                                    self._idx_to_lag(candidate_set)), end='')
            significant = stats.max_statistic(self, data, candidate_set,
                                              te_max_candidate,
                                              self.options)[0]

            # If the max is significant keep it and test the next candidate. If
            # it is not significant break. There will be no further significant
            # sources b/c they all have lesser TE.
            if significant:
                if VERBOSE:
                    print(' -- significant')
                success = True
                candidate_set.pop(np.argmax(temp_te))
                self._append_selected_vars_idx([max_candidate])
                self._append_selected_vars_realisations(
                            data.get_realisations(self.current_value,
                                                  [max_candidate])[0])
            else:
                if VERBOSE:
                    print(' -- not significant')
                break

        return success

    def _prune_candidates(self, data):
        """Remove uninformative candidates from the final conditional set.

        For each sample in the final conditioning set, check if it is
        informative about the current value given all other samples in the
        final set. If a sample is not informative, it is removed from the
        final set.

        Args:
            data : Data instance
                raw data
            options : dict [optional]
                parameters for estimation and statistical testing

        """
        # FOR LATER we don't need to test the last included in the first round
        while self.selected_vars_sources:
            # Find the candidate with the minimum TE into the target.
            temp_te = np.empty(len(self.selected_vars_sources))
            i = 0
            for candidate in self.selected_vars_sources:
                # Separate the candidate realisations and all other
                # realisations to test the candidate's individual contribution.
                [temp_cond, temp_cand] = self._separate_realisations(
                                                    self.selected_vars_full,
                                                    candidate)
                temp_te[i] = self._cmi_calculator.estimate(
                                            temp_cand,
                                            self._current_value_realisations,
                                            temp_cond,
                                            self.options)
                i += 1

            # Test min TE for significance with minimum statistics.
            te_min_candidate = min(temp_te)
            min_candidate = self.selected_vars_sources[np.argmin(temp_te)]
            if VERBOSE:
                print('testing {0} from candidate set {1}'.format(
                                self._idx_to_lag([min_candidate])[0],
                                self._idx_to_lag(self.selected_vars_sources)),
                      end='')
            [significant, p, surr_table] = stats.min_statistic(
                                              self, data,
                                              self.selected_vars_sources,
                                              te_min_candidate,
                                              self.options)

            # Remove the minimum it is not significant and test the next min.
            # candidate. If the minimum is significant, break, all other
            # sources will be significant as well (b/c they have higher TE).
            if not significant:
                if VERBOSE:
                    print(' -- not significant')
                self._remove_candidate(min_candidate)
            else:
                if VERBOSE:
                    print(' -- significant')
                self.min_stats_surr_table = surr_table
                break
            i += 1

    def _test_final_conditional(self, data):  # TODO test this!
        """Perform statistical test on AIS using the final conditional set."""

        print(self._idx_to_lag(self.selected_vars_full))
        [ais, s, p] = stats.mi_against_surrogates(self, data)

        self.ais = ais
        self.sign = s
        self.pvalue = p

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

    def _separate_realisations(self, idx_full, idx_single):
        """Separate a single indexes' realisations from a set of realisations.

        Return the realisations of a single index and the realisations of the
        remaining set of indexes. The function takes realisations from the
        array in self._selected_vars_realisations. This allows to reuse the
        collected realisations when pruning the conditional set after
        candidates have been included.

        Args:
            idx_full : list of tuples
                indices indicating the full set
            idx_single : tuple
                index to be removed

        Returns:
            numpy array
                realisations of the set without the single index
            numpy array
                realisations of the variable at the single index
        """
        # Get realisations for all indices from the class attribute
        # ._selected_vars_realisations. Find the respective columns.
        idx_remaining = cp.copy(idx_full)
        idx_remaining.pop(idx_remaining.index(idx_single))
        array_col_single = self.selected_vars_full.index(idx_single)
        array_col_full = np.zeros(len(idx_full)).astype(int)
        i = 0
        for idx in idx_remaining:  # find the columns with realisations
            array_col_full[i] = self.selected_vars_full.index(idx)
            i += 1

        real_single = np.expand_dims(
                        self._selected_vars_realisations[:, array_col_single],
                        axis=1)
        if len(idx_full) == 1:
            real_remaining = None  # so the JIDT estimator doesn't break
        else:
            real_remaining = self._selected_vars_realisations[:,
                                                              array_col_full]
        return real_remaining, real_single

    def _clean_up(self):
        """Remove temporary data at the end of the analysis."""
        self._current_value_realisations = None
        self._selected_vars_sources_realisations = None
        self._selected_vars_target_realisations = None
        self._current_value_realisations = None
        self.min_stats_surr_table = None

    def _idx_to_lag(self, idx_list):
        """Change sample indices to lags for each index in the list."""
        lag_list = cp.copy(idx_list)
        for c in idx_list:
            lag_list[idx_list.index(c)] = (c[0], self.current_value[1] - c[1])
        return lag_list

    def _force_conditionals(self, cond, data):
        """Enforce a given conditioning set."""
        if type(cond) is tuple:  # easily add single variable
            cond = [cond]
        elif type(cond) is str:
            if cond == 'faes':
                cond = self._define_candidates(self.source_set,
                                               [self.current_value[1]])

        print('Adding the following variables to the conditioning set: {0}.'.
              format(self._idx_to_lag(cond)))
        self._append_selected_vars_idx(cond)
        self._append_selected_vars_realisations(
                        data.get_realisations(self.current_value, cond)[0])

