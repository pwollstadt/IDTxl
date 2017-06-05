"""Analysis of AIS in a network of processes.

Analysis of active information storage (AIS) in individual processes of a
network. The algorithm uses non-uniform embedding as described in Faes ???.

Note:
    Written for Python 3.4+

@author: patricia
"""
import numpy as np
from . import stats
from .single_process_analysis import SingleProcessAnalysis
from .set_estimator import Estimator_cmi

VERBOSE = True

# TODO use target instead of process to define the process that is analyzed.
# This would reuse an attribute set in the parent class.


class ActiveInformationStorage(SingleProcessAnalysis):
    """Set up analysis of storage in each process of the network.

    Set parameters necessary for active information storage (AIS) in every
    process of a network. To perform AIS estimation call analyse_network() on
    the whole network or a set of nodes or call analyse_single_process() to
    estimate AIS for a single process. See docstrings of the two functions
    for more information.

    Args:
        max_lag : int
            maximum temporal search depth
        tau : int [optional]
            spacing between samples analyzed for information contribution
            (default=1)
        options : dict
            parameters for estimator use and statistics:

            - 'n_perm_*' - number of permutations, where * can be 'max_stat',
              'min_stat', 'mi' (default=500)
            - 'alpha_*' - critical alpha level for statistical significance,
              where * can be 'max_stat', 'min_stat', 'mi' (default=0.05)
            - 'cmi_calc_name' - estimator to be used for CMI calculation. Note
              that this estimator is also used to estimate MI later on.
              (For estimator options see the respective documentation.)
            - 'add_conditionals' - force the estimator to add these
              conditionals when estimating AIS; can be a list of
              variables, where each variable is described as (idx process, lag
              wrt to current value)

    Attributes:
        selected_vars_full : list of tuples
            samples in the past state, (idx process, idx sample)
        current_value : tuple
            index of the current value in AIS estimation, (idx process,
            idx sample)
        calculator_name : string
            calculator used for CMI/MI estimation
        max_lag : int
            maximum temporal search depth for candidates in the processes' past
            (default=same as max_lag_sources)
        tau : int
            spacing between samples analyzed for information contribution
        ais : float
            raw AIS value
        sign : bool
            true if AIS is significant
        pvalue: float
            p-value of AIS
        process_set : list
            list with indices of analyzed processes
    """

    def __init__(self, max_lag, options, tau=1):
        # Check user input
        if type(max_lag) is not int or max_lag < 0:
            raise RuntimeError('max_lag has to be an integer >= 0.')
        if type(tau) is not int or tau <= 0:
            raise RuntimeError('tau has to be an integer > 0.')
        if tau >= max_lag:
            raise RuntimeError('tau ({0}) has to be smaller than max_lag '
                               '({1}).'.format(tau, max_lag))

        # Set user-specified estimation parameters
        self.max_lag = max_lag
        self.tau = tau
        self.pvalue = None
        self.sign = False
        self.ais = None
        self.options = options
        self._min_stats_surr_table = None
        try:
            self.calculator_name = options['cmi_calc_name']
        except KeyError:
            raise KeyError('Calculator name was not specified!')
        print('\n\nSetting calculator to: {0}'.format(self.calculator_name))
        self._cmi_calculator = Estimator_cmi(self.calculator_name)
        super().__init__()

    def analyse_network(self, data, processes='all'):
        """Estimate active information storage for multiple network processes.

        Estimate active information storage for all or a subset of processes in
        the network.

        Example:

            >>> dat = Data()
            >>> dat.generate_mute_data(100, 5)
            >>> max_lag = 5
            >>> analysis_opts = {
            >>>     'cmi_calc_name': 'jidt_kraskov',
            >>>     'n_perm_max_stat': 200,
            >>>     'n_perm_min_stat': 200,
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
                index of processes (default='all');
                if 'all', AIS is estimated for all processes;
                if list of int, AIS is estimated for processes specified in the
                list.
        """
        # Check provided processes for analysis.
        if processes == 'all':
            processes = [t for t in range(data.n_processes)]
        if (type(processes) is list) and (type(processes[0]) is int):
            pass
        else:
            raise ValueError('Processes were not specified correctly: '
                             '{0}.'.format(processes))

        # Perform AIS estimation for each target individually.
        results = {}
        for t in range(len(processes)):
            if VERBOSE:
                print('\n####### analysing process {0} of {1}'.format(
                                                processes[t], processes))
            r = self.analyse_single_process(data, processes[t])
            r['process'] = processes[t]
            results[processes[t]] = r
            # TODO FDR correct this
        return results

    def analyse_single_process(self, data, process):
        """Estimate active information storage for a single process.

        Estimate active information storage for one process in the network.
        Uses non-uniform embedding found through information maximisation (see
        Faes, 2011, and Lizier, ???). This is
        done in three steps (see Lizier and Faes for details):

        (1) find all relevant samples in the processes' own past, by
            iteratively adding candidate samples that have significant
            conditional mutual information (CMI) with the current value
            (conditional on all samples that were added previously)
        (3) prune the final conditional set by testing the CMI between each
            sample in the final set and the current value, conditional on all
            other samples in the final set
        (4) calculate AIS using the final set of candidates as the past state
            (calculate MI between samples in the past and the current value);
            test for statistical significance using a permutation test

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
        self._include_process_candidates(data)
        print('\n---------------------------- (2) prune source candidates')
        self._prune_candidates(data)
        print('\n---------------------------- (3) final statistics')
        self._test_final_conditional(data)

        # Clean up and return results.
        if VERBOSE:
            print('final conditional samples: {0}'.format(
                    self._idx_to_lag(self.selected_vars_full)))
        self._clean_up()  # remove realisations and min_stats surrogate table
        results = {
            'current_value': self.current_value,
            'selected_vars': self._idx_to_lag(self.selected_vars_full),
            'ais': self.ais,
            'ais_pval': self.pvalue,
            'ais_sign': self.sign}
        return results

    def _initialise(self, data, process):
        """Check input and set everything to initial values."""
        # Check user input
        if type(process) is not int or process < 0:
            raise RuntimeError('The index of the process ({0}) has to be an '
                               'int >= 0.'.format(process))
        if process > data.n_processes:
            raise RuntimeError('Trying to analyse process with index {0}, '
                               'which greater than the number of processes in '
                               'the data ({1}).'.format(process,
                                                        data.n_processes))
        self.process = process

        # Check provided search depths for source and target
        assert(data.n_samples >= self.max_lag + 1), (
            'Not enough samples in data ({0}) to allow for the chosen maximum '
            'lag ({1})'.format(data.n_samples, self.max_lag))
        self.current_value = (process, self.max_lag)
        [cv_realisation, repl_idx] = data.get_realisations(
                                             current_value=self.current_value,
                                             idx_list=[self.current_value])
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
            self._min_stats_surr_table = None

        # Check if the user provided a list of candidates that must go into
        # the conditioning set. These will be added and used for TE estimation,
        # but never tested for significance.
        try:
            cond = self.options['add_conditionals']
            self._force_conditionals(cond, data)
        except KeyError:
            pass

    def _include_process_candidates(self, data):
        """Test candidates in the process's past."""
        process = [self.process]
        samples = np.arange(self.current_value[1] - 1,
                            self.current_value[1] - self.max_lag - 1,
                            -self.tau)
        candidates = self._define_candidates(process, samples)
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
        success = False
        while candidate_set:
            # Get realisations for all candidates.
            cand_real = data.get_realisations(self.current_value,
                                              candidate_set)[0]
            cand_real = cand_real.T.reshape(cand_real.size, 1)

            # Calculate the (C)MI for each candidate and the target.
            temp_te = self._cmi_calculator.estimate_mult(
                                n_chunks=len(candidate_set),
                                options=self.options,
                                re_use=['var2', 'conditional'],
                                var1=cand_real,
                                var2=self._current_value_realisations,
                                conditional=self._selected_vars_realisations)

            # Test max CMI for significance with maximum statistics.
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
                # Remove candidate from candidate set and add it to the
                # selected variables (used as the conditioning set).
                candidate_set.pop(np.argmax(temp_te))
                self._append_selected_vars(
                        [max_candidate],
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
            candidate_realisations = np.empty(
                                (data.n_realisations(self.current_value) *
                                 len(self.selected_vars_sources), 1))
            conditional_realisations = np.empty(
                                (data.n_realisations(self.current_value) *
                                 len(self.selected_vars_sources), cond_dim))
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
                    re_use = ['var2', 'conditional']
                else:
                    conditional_realisations[i_1:i_2, ] = temp_cond
                    re_use = ['var2']
                candidate_realisations[i_1:i_2, ] = temp_cand
                i_1 = i_2
                i_2 += data.n_realisations(self.current_value)

            temp_te = self._cmi_calculator.estimate_mult(
                                    n_chunks=len(self.selected_vars_sources),
                                    options=self.options,
                                    re_use=re_use,
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
                self._remove_selected_var(min_candidate)
            else:
                if VERBOSE:
                    print(' -- significant')
                self._min_stats_surr_table = surr_table
                break

    def _test_final_conditional(self, data):  # TODO test this!
        """Perform statistical test on AIS using the final conditional set."""
        if self._selected_vars_full:
            print(self._idx_to_lag(self.selected_vars_full))
            [ais, s, p] = stats.mi_against_surrogates(self, data)

            # If a parallel estimator was used, an array of AIS estimates is
            # returned. Make the output uniform for both estimator types.
            if type(ais) is np.ndarray:
                assert ais.shape[0] == 1, 'AIS result is not a scalar.'
                ais = ais[0]

            self.ais = ais
            self.sign = s
            self.pvalue = p
        else:
            self.ais = np.nan
            self.sign = False
            self.pvalue = 1.0
