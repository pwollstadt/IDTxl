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
        print('\n\nSetting calculator to: {0}'.format(self.calculator_name))
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
                print('\n####### analysing process {0} of {1}'.format(processes[t],
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
        candidates = self._define_candidates()
        samples_found = self._include_candidates(candidates, data)

        print('\n---------------------------- (2) prune source candidates')
        # If no candidates were found in the process' past, return 0.
        if samples_found:
            self._prune_candidates(data)
        print('\n---------------------------- (3) final statistics')
        if self._selected_vars_full:
            self._test_final_conditional(data)
        else:
            self.ais = np.nan
            self.sign = False
            self.pvalue = 1

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
            self.selected_vars_sources = []
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
            candidate_realisations = np.empty((data.n_realisations(self.current_value) *
                                               len(candidate_set), 1))
            i_1 = 0
            i_2 = data.n_realisations(self.current_value)
            for candidate in candidate_set:
                candidate_realisations[i_1:i_2, 0] = data.get_realisations(
                                                            self.current_value,
                                                            [candidate])[0].reshape(data.n_realisations(self.current_value),)
                i_1 = i_2
                i_2 += data.n_realisations(self.current_value)
            temp_te = self._cmi_calculator.estimate_mult(
                                        n_chunks=len(candidate_set),
                                        options=self.options,
                                        re_use=['var2', 'conditional'],
                                        var1=candidate_realisations,
                                        var2=self._current_value_realisations,
                                        conditional=self._selected_vars_realisations)

            # Test max TE for significance with maximum statistics.
            te_max_candidate = max(temp_te)
            max_candidate = candidate_set[np.argmax(temp_te)]
            if VERBOSE:
                print('testing candidate {0} from candidate set {1}'.format(
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
            cond_dim = len(self.selected_vars_sources) - 1
            candidate_realisations = np.empty((data.n_realisations(self.current_value) *
                                               len(self.selected_vars_sources), 1))
            conditional_realisations = np.empty((data.n_realisations(self.current_value) *
                                                 len(self.selected_vars_sources), cond_dim ))
            i_1 = 0
            i_2 = data.n_realisations(self.current_value)            
            for candidate in self.selected_vars_sources:
                # Separate the candidate realisations and all other
                # realisations to test the candidate's individual contribution.
                [temp_cond, temp_cand] = self._separate_realisations(
                                                    self.selected_vars_sources,
                                                    candidate)
                if temp_cond is None:
                    conditional_realisations = None
                else:
                    conditional_realisations[i_1:i_2, ] = temp_cond
                candidate_realisations[i_1:i_2, 0] = temp_cand
                i_1 = i_2
                i_2 += data.n_realisations(self.current_value)
            temp_te = self._cmi_calculator.estimate_mult(
                                            n_chunks=len(self.selected_vars_sources),
                                            options=self.options,
                                            re_use=['var2'],
                                            var1=candidate_realisations,
                                            var2=self._current_value_realisations,
                                            conditional=conditional_realisations)

            # Test min TE for significance with minimum statistics.
            te_min_candidate = min(temp_te)
            min_candidate = self.selected_vars_sources[np.argmin(temp_te)]
            if VERBOSE:
                print('testing candidate {0} from candidate set {1}'.format(
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

    def _test_final_conditional(self, data):  # TODO test this!
        """Perform statistical test on AIS using the final conditional set."""

        print(self._idx_to_lag(self.selected_vars_full))
        [ais, s, p] = stats.mi_against_surrogates(self, data)

        # If a parallel estimator was used, an array of AIS estimates is returned.
        # Make the output uniform for both estimator types.
        if type(ais) is np.ndarray:
            assert ais.shape[0] == 1, 'AIS estimation returned more than one value.'
            ais = ais[0]

        self.ais = ais
        self.sign = s
        self.pvalue = p

    def _define_candidates(self):
        """Build a list of candidate indices.
        
        Note that for AIS estimation, the candidate set is defined as the past
        of the process up to the max_lag defined by the user (samples spaced
        by tau if requested). This function thus does not need to take 
        arguments as the same function in multivariate TE estimation. 

        Returns:
            a list of tuples, where each tuple holds the index of one
            candidate and has the form (process index, sample index)
        """
        process = [self.process]
        samples = np.arange(self.current_value[1] - 1,
                            self.current_value[1] - self.max_lag - 1,
                            -self.tau)
        candidate_set = []
        for idx in it.product(process, samples):
            candidate_set.append(idx)
        return candidate_set
