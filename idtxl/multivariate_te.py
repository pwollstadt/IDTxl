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
import numpy as np
import itertools as it
import copy as cp
import stats as stats
import utils as utils
from data import Data
from network_analysis import Network_analysis
from set_estimator import Estimator_cmi

VERBOSE = True


class Multivariate_te(Network_analysis):
    """Set up a network analysis using multivariate transfer entropy.

    Set parameters necessary for network inference using transfer entropy (TE).
    To perform network inference call analyse_network() on an instance of the
    data class.

    Args:
        max_lag : int
            maximum temporal search depth
        min_lag : int
            minimum temporal search depth
        options : dict [optional]
            parameters for estimator use and statistics
            'n_perm_*' - number of permutations, where * can be 'max_stats',
            'min_stats', and 'omnibus' (default=500)
            'alpha_*' - critical alpha level for statistical significance,
            where * can be 'max_stats',  'min_stats', and 'omnibus'
            (default=0.05)
            'cmi_calc_name' - estimator to be used for CMI calculation
            (For estimator options see the respective documentation.)

    Attributes:
        conditional_full : list of tuples
            samples in the full conditional set, (idx process, idx sample)
        conditional_sources : list of tuples
            source samples in the conditional set, (idx process, idx sample)
        conditional_target : list of tuples
            target samples in the conditional set, (idx process, idx sample)
        current_value : tuple
            index of the current value in TE estimation, (idx process,
            idx sample)
        calculator_name : string
            calculator used for TE estimation
        max_lag : int
            maximum temporal search depth
        min_lag : int
            minimum temporal search depth
        pvalue_omnibus : float
            p-value of the omnibus test
        pvalue_individual_sources : numpy array
            array of p-values for TE from
            individual sources to the target
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
    def __init__(self, max_lag, min_lag, options):
        self.max_lag = max_lag
        self.min_lag = min_lag
        self.sign_omnibus = None
        self.sign_individual_sources = None
        self.pvalue_omnibus = None
        self.pvalues_sign_sources = None
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
            dat = Data()
            dat.generate_mute_data()
            max_lag = 5
            min_lag = 4
            cmi_calculator = 'jidt_kraskov'
            targets = [0, 1, 2]
            sources = 'all'
            network_analysis = Multivariate_te(max_lag, min_lag, cmi_calculator)
            network_analysis.analyse_network(dat, targets, sources)
            sources = [[1, 2, 3], ['all'], [1]]  # set sources for each target
            network_analysis.analyse_network(dat, targets, sources)

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
        if type(targets) is not list:
            raise TypeError('target should be a list of integers.')
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
            sources : list of int or 'all'
                indices of source processes, if 'all', all sources are tested

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

        print('\n---------------------------- (3) pruning step for {0} '
              'candidates'.format(len(self.conditional_sources)))
        self._prune_candidates(data)

        print('\n---------------------------- (4) final statistics')
        self._test_final_conditional(data)

        # Clean up and return results.
        if VERBOSE:
            print('final source samples: {0}'.format(self.conditional_sources))
            print('final target samples: {0}'.format(self.conditional_target))
        self._clean_up()
        [cond_full, cond_sources, cond_target] = self._indices_to_lags()

        results = {
            'conditional_full': cond_full,
            'conditional_sources': cond_sources,
            'conditional_target': cond_target,
            'omnibus_sign': self.sign_omnibus,
            'omnibus_pval': self.pvalue_omnibus,
            'cond_sources_pval': self.pvalues_sign_sources}
        return results

    def _initialise(self, data, sources, target):
        """Check input and set everything to initial values."""
        self.target = target
        self._check_source_set(sources, data.n_processes)
        self._current_value = (target, self.max_lag)
        [cv_realisation, repl_idx] = data.get_realisations(
                                             current_value=self.current_value,
                                             idx=[self.current_value])
        self._current_value_realisations = cv_realisation
        self._replication_index = repl_idx  # remember which realisations come
                                            # from which replication

        if self.conditional_full is not None:
            self.conditional_full = []
            self._conditional_realisations = None
        if self.conditional_sources is not None:
            self.conditional_sources = []
        if self.conditional_target is not None:
            self.conditional_target = []

    def _check_source_set(self, sources, n_processes):
        """Set default if no source set was provided by the user."""
        if sources == 'all':
            self.source_set = [x for x in range(n_processes)]
            self.source_set.pop(self.target)
            if VERBOSE:
                print('Testing sources {0}'.format(self.source_set))
        elif type(sources) is not list:
            raise TypeError('Source set has to be a list.')
        else:
            if self.target in sources:
                raise RuntimeError('The target {0} should not be in the list '
                                   'of sources {1}.'.format(self.target,
                                                            sources))
            else:
                self.source_set = sources

    def _include_target_candidates(self, data):
        """Test candidates from the target's past."""
        candidates = self._define_candidates(processes=[self.target],
                                             samples=np.arange(self.max_lag))
        sources_found = self._find_conditional(candidates, data, self.options)
        if not sources_found:
            print(('No informative sources in the target''s past - ' +
                   'adding point at t-1 in the target'))
            idx = (self.current_value[0], self.current_value[1] - 1)
            realisations = data.get_realisations(self.current_value, [idx])
            self._append_conditional_idx(idx)
            self._append_conditional_realisations(realisations)

    def _include_source_candidates(self, data):  # TODO something's slow here
        """Test candidates in the source's past."""
        candidates = self._define_candidates(
                                         processes=self.source_set,
                                         samples=np.arange(self.max_lag -
                                                           self.min_lag + 1))
        _ = self._find_conditional(candidates, data, self.options)

    def _test_final_conditional(self, data):  # TODO test this!
        """Perform statistical test on the final conditional set."""
        if not self.conditional_sources:
            print('---------------------------- no sources found')
        else:
            print(self.conditional_full)
            [s, p] = stats.omnibus_test(self, data, self.options)
            self.sign_omnibus = s
            self.pvalue_omnibus = p
            if self.sign_omnibus:
                [s, p] = stats.max_statistic_sequential(self, data,
                                                        self.options)
                # Remove non-significant sources from the candidate set.
                for i in range(s.shape[0] - 1, -1, -1):
                    cand = self.conditional_sources[i]
                    self._remove_candidate(cand)
                    p = np.delete(p, i)
                self.pvalues_sign_sources = p
            else:
                self.conditional_sources = []
                self.conditional_full = self.conditional_target

    def _define_candidates(self, processes, samples):
        """Build a list of candidates' indices.

            Args:
                processes: a list of integers representing process indices
                samples: a list of integers representing sample indices of the
                    process

            Returns:
                a list of tuples, where each tuple holds the index of one
                candidate and has the form (process index, sample index)
        """
        candidate_set = []
        for idx in it.product(processes, samples):
            candidate_set.append(idx)
        return candidate_set

    def _find_conditional(self, candidate_set, data, options):
        """Find informative conditioning set from a set of candidate samples.

        Loop over each candidate in the candidate set and test if it has
        significant mutual information with the current value, conditional
        on all samples that were informative in previous rounds. If this
        conditional mutual information is significant, add it to the
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
            conditional_realisations: numpy array
                realisations of the conditional set
        """
        success = False
        while candidate_set:
            if VERBOSE:
                print('find conditionals in set {0}'.format(candidate_set))
            temp_te = np.empty(len(candidate_set))
            i = 0
            for candidate in candidate_set:
                candidate_realisations = data.get_realisations(
                                                            self.current_value,
                                                            [candidate])[0]
                temp_te[i] = self._cmi_calculator.estimate(
                                        candidate_realisations,
                                        self._current_value_realisations,
                                        self._conditional_realisations,
                                        options)
            te_max_candidate = max(temp_te)
            max_candidate = candidate_set.pop(np.argmax(temp_te))
            significant = stats.max_statistic(self, data, candidate_set,
                                              te_max_candidate, options)
            if significant:
                success = True
                self._append_conditional_idx([max_candidate])
                self._append_conditional_realisations(
                            data.get_realisations(self.current_value,
                                                  [max_candidate])[0])
            else:
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
        significant = True
        while self.conditional_sources:
            temp_te = np.empty(len(self.conditional_sources))
            i = 0
            for candidate in self.conditional_sources: # TODO I only loop over source candidates, ok?
                [temp_cond, temp_cand] = self._remove_realisation(
                                                    self.conditional_sources,
                                                    candidate)
                temp_te[i] = self._cmi_calculator.estimate(
                                            temp_cand,
                                            self._current_value_realisations,
                                            temp_cond,
                                            self.options)

            te_min_candidate = min(temp_te)
            test_set = cp.copy(self.conditional_sources)  # TODO check this
            test_set.pop(test_set.index(candidate))
            significant = stats.min_statistic(self, data, test_set,
                                              te_min_candidate,
                                              self.options)
            if not significant:
                self._remove_candidate(candidate)
            else:
                break
            i += 1

    def _remove_realisation(self, idx_full, idx_single):
        """Remove single indexes' realisations from a set of realisations.

        Remove the realisations of a single index from a set of realisations.
        Return both the single realisation and realisations for the remaining
        set. This allows us to reuse the collected realisations when pruning
        the conditional set after candidates have been included.

        Args:
            idx_full: list of indices indicating the full set
            idx_single: index to be removed

        Returns:
            realisation_single: numpy array with realisations for single index
            realisations_remaining: numpy array with remaining realisations
        """
        assert(len(idx_full) > 1), ('No remaining realisations after removal '
                                    'of single realisation.')
        index_single = idx_full.index(idx_single)
        indices_full = np.zeros(len(idx_full)).astype(int)
        i = 0
        for idx in idx_full:
            indices_full[i] = self.conditional_full.index(idx)
            i += 1
        realisations_full = self._conditional_realisations[:, indices_full]
        realisations_single = np.expand_dims(
                                            realisations_full[:, index_single],
                                            axis=1)
        realisations_remaining = utils.remove_column(realisations_full,
                                                     index_single)
        return realisations_remaining, realisations_single

    def _clean_up(self):
        """Remove temporary data at the end of the analysis."""
        self._current_value_realisations = None
        self._conditional_sources_realisations = None
        self._conditional_target_realisations = None
        self._current_value_realisations = None

    def _indices_to_lags(self):
        """Change sample indices to lags for each candidate set."""
        conditional_full = []
        conditional_sources = []
        conditional_target = []
        for cond in self.conditional_full:
            conditional_full.append((cond[0], self.max_lag - cond[1]))
        for cond in self.conditional_sources:
            conditional_sources.append((cond[0], self.max_lag - cond[1]))
        for cond in self.conditional_target:
            conditional_target.append((cond[0], self.max_lag - cond[1]))
        return conditional_full, conditional_sources, conditional_target

if __name__ == '__main__':
    dat = Data()
    dat.generate_mute_data(100, 5)
    max_lag = 5
    min_lag = 4
    analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov'
        }
    target = 0
    sources = [1, 2, 3]

    network_analysis = Multivariate_te(max_lag, min_lag, analysis_opts)
    # res = network_analysis.analyse_single_target(dat, target, sources)

    d = np.arange(2000).reshape((2, 1000))
    dat2 = Data(d, dim_order='ps')
    # res2 = network_analysis.analyse_single_target(dat2, target)

    targets = [0, 2, 3]
    # res3 = network_analysis.analyse_network(dat, targets, sources='all')
    sources = [[1, 2, 3], 'all', [1]]  # set sources for each target
    res4 = network_analysis.analyse_network(dat, targets, sources=sources)
