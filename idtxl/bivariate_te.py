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
from .set_estimator import Estimator_cmi

VERBOSE = True


class BivariateTE(NetworkInference):
    """Set up a network analysis using bivariate transfer entropy.

    Set parameters necessary for network inference using transfer entropy (TE).
    To perform network inference call analyse_network() on an instance of the
    data class.

    Args:
        max_lag_sources : int
            maximum temporal search depth for candidates in the sources' past
        min_lag_sources : int
            minimum temporal search depth for candidates in the sources' past
        options : dict
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
            - 'permute_in_time' - force surrogate creation by shuffling
              realisations in time instead of shuffling replications; see
              documentation of Data.permute_samples() for further options
              (default=False)

        max_lag_target : int [optional]
            maximum temporal search depth for candidates in the target's past
            (default=same as max_lag_sources)
        tau_sources : int [optinal]
            spacing between candidates in the sources' past (default=1)
        tau_target : int [optinal]
            spacing between candidates in the target's past (default=1)

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
        tau_sources : int
            spacing between candidates in the sources' past
        tau_target : int
            spacing between candidates in the target's past
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
        options : dict
            dictionary with the analysis options
    """

    # TODO right now 'options' holds all optional params (stats AND estimator).
    # We could split this up by adding the stats options to the analyse_*
    # methods?
    def __init__(self, max_lag_sources, min_lag_sources, options,
                 max_lag_target=None, tau_sources=1, tau_target=1):
        # Set estimator in the child class for network inference because the
        # estimated quantity may be different from CMI in other inference
        # algorithms. (Everything else can be done in the parent class.)
        try:
            self.calculator_name = options['cmi_calc_name']
        except KeyError:
            raise KeyError('Calculator name was not specified!')
        self._cmi_calculator = Estimator_cmi(self.calculator_name)
        super().__init__(max_lag_sources, min_lag_sources, options,
                         max_lag_target, tau_sources, tau_target)

    def analyse_network(self, data, targets='all', sources='all'):
        """Find bivariate transfer entropy between all nodes in the network.

        Estimate bivariate transfer entropy (TE) between all nodes in the
        network or between selected sources and targets.

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
            >>> network_analysis = Bivariate_te(max_lag, min_lag,
            >>>                                 analysis_opts)
            >>> res = network_analysis.analyse_network(dat)

        Note:
            For more details on the estimation of bivariate transfer entropy
            see documentation of class method 'analyse_single_target'.

        Args:
            data : Data instance
                raw data for analysis
            targets : list of int | 'all' [optinal]
                index of target processes (default='all')
            sources : list of int | list of list | 'all'  [optional]
                indices of source processes for each target (default='all');
                if 'all', all sources are tested for each target;
                if list of int, sources specified in the list are tested for
                each target;
                if list of list, sources specified in each inner list are
                tested for the corresponding target

        Returns:
            dict
                results consisting of

                - conditional sets (full, from sources, from target),
                - results for omnibus test (joint influence of source cands.),
                - pvalues for each significant source candidate

                for each target
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
            >>> network_analysis = Bivariate_te(max_lag, min_lag,
            >>>                                 analysis_opts)
            >>> res = network_analysis.analyse_single_target(dat, target,
            >>>                                              sources)

        Args:
            data : Data instance
                raw data for analysis
            target : int
                index of target process
            sources : list of int | int | 'all' [optional]
                single index or list of indices of source processes
                (default='all'), if 'all', all possible sources for the given
                target are tested

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
        self._append_selected_vars(
                candidate_set,
                data.get_realisations(self.current_value, candidate_set)[0])
        [s, p, te] = stats.max_statistic_sequential(self, data, self.options)
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
