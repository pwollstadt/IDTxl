# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:24:31 2016

@author: patricia
"""
import numpy as np
import itertools as it
import copy as cp
import statistics as stats
import utils as utils
from data import Data
from network_analysis import Network_analyses
from set_estimator import Estimator_cmi

VERBOSE = True

# TODO create surrogates from embeddings/realisations and surrogates from data/
# replications


class Multivariate_te(Network_analyses):
    """Set up a network analysis using multivariate transfer entropy.

    Set parameters necessary for network inference using transfer entropy (TE).
    To perform network inference call analyse_network() on an instance of the
    data class.

    Args:
        max_lag: maximum number of steps into the past to look for informative
            samples (maximum temporal search depth)
        min_lag: minimum number of steps into the past to look for informative
            samples (minimum temporal search depth)
        cmi_calculator_name: string with the name of the calculator to be used
            for TE estimation
        target: index of the target process
        source_set: list of process indices used as potential sources (default:
            all possible processes, i.e., all processes other than the target
            process)

    Attributes:
        analyse_network: perform network inference on data, has to be run to
            first to write results to other attributes
        conditional_full: samples in the full conditional set
        conditional_sources: samples in the conditional set coming from souces
        conditional_target: samples in the conditional set coming from target
        current_value: index of the current value in TE estimation
        estimator_name: estimator used for TE estimation
        max_lag: maximum temporal search depth
        min_lag: minimum temporal search depth
        pvalue_omnibus: p-value of the omnibus test
        pvalue_individual_sources: array of p-values for TE from individual
            sources to the target
        sign_ominbus: statistical significance of the over-all TE
        sign_individual: array of booleans, indicates statistical significance
            of TE from individual sources to the target
        source_set: list with indices of source processes
        target: index of target process
    """
    def __init__(self, max_lag, min_lag, cmi_calculator_name, target,
                 source_set=None):
        self.max_lag = max_lag
        self.min_lag = min_lag
        self.estimator_name = cmi_calculator_name
        self._cmi_estimator = Estimator_cmi(cmi_calculator_name)  # TODO should be 'calculator'
        # TODO add kwargs for the estimator
        self.source_set = source_set
        self.target = target
        self.sign_omnibus = None
        self.sign_individual_sources = None
        self.pvalue_omnibus = None
        self.pvalues_individual_sources = None
        super(Multivariate_te, self).__init__(max_lag, target)

    def analyse_network(self, data):  # this should allow for multiple targets
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
        """
        self._current_value_realisations = data.get_realisations(
                                                    analysis_setup=self,
                                                    idx=[self.current_value])
        self._check_source_set(data.n_processes)

        print('---------------------------- (1) include target candidates')
        self._include_target_candidates(data)

        print('---------------------------- (2) include source candidates')
        self._include_source_candidates(data)

        print('---------------------------- (3) pruning step for {0} '
              'candidates'.format(len(self.conditional_sources)))
        self._prune_candidates(data)

        print('---------------------------- (4) final statistics')
        self._test_final_conditional(data)

        if VERBOSE:
            print('finalsource samples: {0}'.format(self.conditional_sources))
            print('final target samples: {0}'.format(self.conditional_target))
        self._clean_up()
        return

    def _check_source_set(self, n_processes):
        """Set default if no source set was provided by the user."""
        if self.source_set is None:
            self.source_set = [x for x in range(n_processes)]
            self.source_set.pop(self.target)
            if VERBOSE:
                print('Testing sources {0}'.format(self.source_set))
        else:
            if type(self.source_set) is not list:
                raise TypeError('Source set has to be a list.')

    def _include_target_candidates(self, data):
        """Test candidates from the target's past."""
        candidates = self._define_candidates(processes=[self.target],
                                             samples=np.arange(self.max_lag))  # TODO switch back to actual *lags'*
                                             # samples=np.arange(self.max_lag, 0, -1))
        sources_found = self._find_conditional(candidates, data)
        if not sources_found:
            print(('No informative sources in the target''s past - ' +
                   'adding point at t-1 in the target'))
            idx = (self.current_value[0], self.current_value[1] - 1)
            realisations = data.get_realisations(self, [idx])
            self._append_conditional_idx(idx)
            self._append_conditional_realisations(realisations)

    def _include_source_candidates(self, data):  # TODO something's slow here
        """Test candidates in the source's past."""
        candidates = self._define_candidates(
                                         processes=self.source_set,
                                         samples=np.arange(self.max_lag -
                                                           self.min_lag + 1))
        _ = self._find_conditional(candidates, data)

    def _test_final_conditional(self, data):
        """Perform statistical test on the final conditional set."""
        if not self.conditional_sources:
            print('---------------------------- no sources found')
        else:
            print(self.conditional_full)
            [s, p] = stats.omnibus_test(self, data)
            self.sign_omnibus = s
            self.pvalue_omnibus = p
            if self.sign_omnibus:
                [s, p] = stats.max_statistic_sequential(self, data)
                self.sign_individual_sources = s
                self.pvalues_individual_sources = p

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

    def _find_conditional(self, candidate_set, data):
        """Find informative conditioning set from a set of candidate samples.

        Loop over each candidate in the candidate set and test if it has
        significant mutual information with the current value, conditional
        on all samples that were informative in previous rounds. If this
        conditional mutual information is significant, add it to the
        conditional set.

        Args:
            data: instance of Data class
            cmi_estimator: instance of Estimator_cmi class
            candidate_set: list of tuples, where each tuple is an index with
                entries (process index, sample index)
            conditional_set: if available, list with indices of samples
                already in the conditional set (default is an empty numpy
                array), new sources are added to the array
            conditional_realisations: if available, realisations of samples
                already in the conditional set (default is an empty numpy
                array), realisations of new sources are added to the array

        Returns:
            conditional_set: list of indices with informative sources from
                the candidate set
            conditional_realisations: numpy array with realisations of the
                conditional set
        """
        success = False
        while candidate_set:
            if VERBOSE:
                print('find conditionals in set {0}'.format(candidate_set))
            temp_te = np.empty(len(candidate_set))
            i = 0
            for candidate in candidate_set:
                candidate_realisations = data.get_realisations(self,
                                                               [candidate])
                temp_te[i] = self._cmi_estimator.estimate(
                                        candidate_realisations,
                                        self._current_value_realisations,
                                        self._conditional_realisations)
            te_max_candidate = max(temp_te)
            max_candidate = candidate_set.pop(np.argmax(temp_te)) # TO_DO correct?
            significant = stats.max_statistic(self, data, candidate_set,
                                              te_max_candidate)
            if significant:
                success = True
                self._append_conditional_idx([max_candidate])
                self._append_conditional_realisations(
                                data.get_realisations(self, [max_candidate]))
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
            data: instance of Data class
            conditional_set: set of informative sources about the target

        Returns:
            conditional_set_pruned: list of indices of samples in the
            conditional set after removal of spurious samples
        """
        test_set_1 = cp.copy(self.conditional_sources)
        i = 0
        for candidate in reversed(test_set_1):  # TODO loop over all, take minimum (see steps 1 and 2)
            if len(test_set_1) == 1:  # TODO is this alright?
                break

            if VERBOSE:
                print('pruning candidate {0}'.format(i + 1))
            # TODO 2 options: (1) use the conditional realisations from
            # earlier, cut current candidate from the array and use the
            # rest as temp conditional; (2) stick with the current
            # implementation and always read stuff from data
            # we currently do option (1) -> profile this
            [temp_cond, temp_cand] = self._remove_realisation(
                                                self.conditional_sources,
                                                candidate)
            temp_te = self._cmi_estimator.estimate(
                                        temp_cand,
                                        self._current_value_realisations,
                                        temp_cond)
            test_set_2 = cp.copy(self.conditional_sources)  # TODO check this
            test_set_2.pop(test_set.index(candidate))
            significant = stats.min_statistic(self, data, test_set_2, temp_te)
            if not significant:
                self._remove_candidate(candidate)
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


if __name__ == '__main__':
    dat = Data()  # initialise an empty data object
    dat.generate_mute_data()
    max_lag = 5
    min_lag = 4
    cmi_estimator = 'jidt_kraskov'
    target = 0
    sources = [1, 2, 3]
    # for t in range(dat.n_processes):
    #     multivariate_te(dat, max_lag, min_lag, cmi_estimator, t)
    network_analysis = Multivariate_te(max_lag, min_lag, cmi_estimator, target,
                                       sources)
    network_analysis.analyse_network(dat)


    # test cases:
    #   - bivariately coupled Lorenz
    #   - independent random samples (for false positives)
    #   - lagged copy of white noise

    min_lag2 = 1
    network_analysis2 = Multivariate_te(max_lag, min_lag2, cmi_estimator,
                                        target)
    network_analysis2.analyse_network(dat)
