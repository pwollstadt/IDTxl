"""Parent class for all network inference."""
import numpy as np
from .network_analysis import NetworkAnalysis
from . import stats
from . import idtxl_exceptions as ex


class NetworkInference(NetworkAnalysis):
    """Parent class for network inference algorithms.

    Hold variables that are relevant for network inference using for example
    bivariate and multivariate transfer entropy.

    Attributes:
        settings : dict
            settings for estimation of information theoretic measures and
            statistical testing, see child classes for documentation
        target : int
            target process of analysis
        current_value : tuple
            index of the current value
        selected_vars_full : list of tuples
            indices of the full set of random variables to be conditioned on
        selected_vars_target : list of tuples
            indices of the set of conditionals coming from the target process
        selected_vars_sources : list of tuples
            indices of the set of conditionals coming from source processes
    """

    def __init__(self):
        # Create class attributes for estimation
        self.statistic_omnibus = None
        self.sign_omnibus = False
        self.pvalue_omnibus = None
        self.statistic_sign_sources = None
        self.pvalues_sign_sources = None
        super().__init__()

    def _check_target(self, target, n_processes):
        """Set and check the target provided by the user."""
        if type(target) is not int or target < 0:
            raise RuntimeError('The index of the target process ({0}) has to '
                               'be an int >= 0.'.format(target))
        if target > n_processes:
            raise RuntimeError('Trying to analyse target with index {0}, '
                               'which greater than the number of processes in '
                               'the data ({1}).'.format(target, n_processes))
        self.target = target

    def _check_source_set(self, sources, n_processes):
        """Set default if no source set was provided by the user."""
        if sources == 'all':
            sources = [x for x in range(n_processes)]
            sources.pop(self.target)
        elif type(sources) is int:
            sources = [sources]
        elif type(sources) is list:
            assert type(sources[0]) is int, 'Source list has to contain ints.'
        else:
            raise TypeError('Sources have to be passes as a single int, list '
                            'of ints or "all".')

        if self.target in sources:
            raise RuntimeError('The target ({0}) should not be in the list '
                               'of sources ({1}).'.format(self.target,
                                                          sources))
        if max(sources) > n_processes:
            raise RuntimeError('The list of sources {0} contains indices '
                               'greater than the number of processes {1} in '
                               'the data.'.format(sources, n_processes))
        if min(sources) < 0:
            raise RuntimeError('The source list ({0}) can not contain negative'
                               ' indices.'.format(sources))

        self.source_set = sources
        if self.settings['verbose']:
            print('\nTarget: {0} - testing sources {1}'.format(
                self.target, self.source_set))

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

        Returns:
            bool
                True if a candidate with significant MI was found
        """
        success = False
        if self.settings['verbose']:
                print('candidate set: {0}'.format(
                    self._idx_to_lag(candidate_set)))
        while candidate_set:
            # Get realisations for all candidates.
            cand_real = data.get_realisations(self.current_value,
                                              candidate_set)[0]
            # Reshape candidates to a 1D-array, where realisations for a single
            # candidate are treated as one chunk.
            cand_real = cand_real.T.reshape(cand_real.size, 1)

            # Calculate the (C)MI for each candidate and the target.
            try:
                temp_te = self._cmi_estimator.estimate_parallel(
                                n_chunks=len(candidate_set),
                                re_use=['var2', 'conditional'],
                                var1=cand_real,
                                var2=self._current_value_realisations,
                                conditional=self._selected_vars_realisations)
            except ex.AlgorithmExhaustedError as aee:
                # The algorithm cannot continue here, so
                #  we'll terminate the search for more candidates,
                #  though those identified already remain valid
                print('AlgorithmExhaustedError encountered in '
                    'estimations: ' + aee.message)
                print('Halting current estimation set.')
                # For now we don't need a stack trace:
                # traceback.print_tb(aee.__traceback__)
                break

            # Test max CMI for significance with maximum statistics.
            te_max_candidate = max(temp_te)
            max_candidate = candidate_set[np.argmax(temp_te)]
            if self.settings['verbose']:
                print('testing candidate: {0} '.format(
                    self._idx_to_lag([max_candidate])[0]), end='')
            try:
                significant = stats.max_statistic(self, data, candidate_set,
                                              te_max_candidate)[0]
            except ex.AlgorithmExhaustedError as aee:
                # The algorithm cannot continue here, so
                #  we'll terminate the check of significance for this candidate,
                #  though those identified already remain valid
                print('AlgorithmExhaustedError encountered in '
                    'estimations: ' + aee.message)
                print('Halting candidate max stats test')
                # For now we don't need a stack trace:
                # traceback.print_tb(aee.__traceback__)
                break

            # If the max is significant keep it and test the next candidate. If
            # it is not significant break. There will be no further significant
            # sources b/c they all have lesser TE.
            if significant:
                # if self.settings['verbose']:
                #     print(' -- significant')
                success = True
                candidate_set.pop(np.argmax(temp_te))
                self._append_selected_vars(
                        [max_candidate],
                        data.get_realisations(self.current_value,
                                              [max_candidate])[0])
            else:
                if self.settings['verbose']:
                    print(' -- not significant')
                break
        return success

    def _force_conditionals(self, cond, data):
        """Enforce a given conditioning set."""
        if type(cond) is str:
            # Get realisations and indices of source variables with lag 0. Note
            # that _define_candidates returns tuples with absolute indices and
            # not lags.
            if cond == 'faes':
                cond = self._build_variable_list(self.source_set,
                                                 [self.current_value[1]])
                self._append_selected_vars(
                        cond,
                        data.get_realisations(self.current_value, cond)[0])
        else:
            # If specific variables for conditioning were provided, convert
            # lags to absolute sample indices and add variables.
            if type(cond) is tuple:  # easily add single variable
                cond = [cond]
            print('Adding the following variables to the conditioning set: '
                  '{0}.'.format(cond))
            cond_idx = self._lag_to_idx(cond)
            self._append_selected_vars(
                        cond_idx,
                        data.get_realisations(self.current_value, cond_idx)[0])

    def _remove_non_significant(self, s, p, stat):
        # Remove non-significant sources from the candidate set. Loop
        # backwards over the candidates to remove them iteratively.
        for i in range(s.shape[0] - 1, -1, -1):
            if not s[i]:
                self._remove_selected_var(self.selected_vars_sources[i])
                p = np.delete(p, i)
                stat = np.delete(stat, i)
        return p, stat


class NetworkInferenceMI(NetworkInference):
    """Parent class for mutual information network inference algorithms."""

    def __init__(self):
        self.measure = 'mi'
        super().__init__()

    def _initialise(self, settings, data, sources, target):
        """Check input, set initial or default values for analysis settings."""
        # Check analysis settings and set defaults.
        self.settings = settings.copy()
        self.settings.setdefault('verbose', True)
        self.settings.setdefault('add_conditionals', None)
        self.settings.setdefault('tau_sources', 1)
        self.settings.setdefault('local_values', False)

        # Check lags and taus for multivariate embedding.
        if 'max_lag_sources' not in self.settings:
            raise RuntimeError('The maximum lag for source embedding '
                               '(''max_lag_sources'') needs to be specified.')
        if 'min_lag_sources' not in self.settings:
            raise RuntimeError('The minimum lag for source embedding '
                               '(''min_lag_sources'') needs to be specified.')

        if (type(self.settings['min_lag_sources']) is not int or
                self.settings['min_lag_sources'] < 0):
            raise RuntimeError('min_lag_sources has to be an integer >= 0.')
        if (type(self.settings['max_lag_sources']) is not int or
                self.settings['max_lag_sources'] < 0):
            raise RuntimeError('max_lag_sources has to be an integer >= 0.')
        if (type(self.settings['tau_sources']) is not int or
                self.settings['tau_sources'] < 0):
            raise RuntimeError('tau_sources must be an integer >= 0.')
        if self.settings['min_lag_sources'] > self.settings['max_lag_sources']:
            raise RuntimeError('min_lag_sources ({0}) must be smaller or equal'
                               ' to max_lag_sources ({1}).'.format(
                                   self.settings['min_lag_sources'],
                                   self.settings['max_lag_sources']))
        # max_lag_sources can be 0 for MI estimation, in this case we don't
        # require the tau to be larger than the max lag. Still, tau has to be
        # one to later generate the candidate set via enumerating all samples.
        if (self.settings['max_lag_sources'] > 0 and
                self.settings['tau_sources'] > self.settings['max_lag_sources']):
            raise RuntimeError('tau_sources ({0}) has to be smaller than '
                               'max_lag_sources ({1}).'.format(
                                   self.settings['tau_sources'],
                                   self.settings['max_lag_sources']))

        # Set CMI estimator.
        self._set_cmi_estimator()

        # Check the provided target and sources.
        self._check_target(target, data.n_processes)
        self._check_source_set(sources, data.n_processes)

        # Check provided search depths (lags) for sources, set the
        # current_value.
        assert(data.n_samples >= self.settings['max_lag_sources'] + 1), (
            'Not enough samples in data ({0}) to allow for the chosen maximum '
            'lag ({1})'.format(
                data.n_samples, self.settings['max_lag_sources']))

        self.current_value = (self.target, self.settings['max_lag_sources'])
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

        # Reset all attributes to inital values if the instance of
        # MultivariateTE has been used before.
        if self.selected_vars_full:
            self.selected_vars_full = []
            self._selected_vars_realisations = None
            self.selected_vars_sources = []
            self.mi_omnibus = None
            self.pvalue_omnibus = None
            self.pvalues_sign_sources = None
            self.mi_sign_sources = None
            self._min_stats_surr_table = None

        # Check if the user provided a list of candidates that must go into
        # the conditioning set. These will be added and used for TE estimation,
        # but never tested for significance.
        if self.settings['add_conditionals'] is not None:
            self._force_conditionals(self.settings['add_conditionals'], data)

    def _reset(self):
        """Reset instance after analysis."""
        self.__init__()
        del self.settings
        del self.source_set
        del self.pvalues_sign_sources
        del self.statistic_sign_sources
        del self.statistic_omnibus
        del self.pvalue_omnibus
        del self.sign_omnibus
        del self._cmi_estimator


class NetworkInferenceTE(NetworkInference):
    """Parent class for transfer entropy network inference algorithms."""

    def __init__(self):
        self.measure = 'te'
        super().__init__()

    def _initialise(self, settings, data, sources, target):
        """Check input, set initial or default values for analysis settings."""
        # Check analysis settings and set defaults.
        self.settings = settings.copy()
        self.settings.setdefault('verbose', True)
        self.settings.setdefault('add_conditionals', None)
        self.settings.setdefault('tau_target', 1)
        self.settings.setdefault('tau_sources', 1)
        self.settings.setdefault('local_values', False)

        # Check lags and taus for multivariate embedding.
        if 'max_lag_sources' not in self.settings:
            raise RuntimeError('The maximum lag for source embedding '
                               '(''max_lag_sources'') needs to be specified.')
        if 'min_lag_sources' not in self.settings:
            raise RuntimeError('The minimum lag for source embedding '
                               '(''min_lag_sources'') needs to be specified.')
        self.settings.setdefault('max_lag_target', settings['max_lag_sources'])

        if (type(self.settings['min_lag_sources']) is not int or
                self.settings['min_lag_sources'] < 0):
            raise RuntimeError('min_lag_sources has to be an integer >= 0.')
        if (type(self.settings['max_lag_sources']) is not int or
                self.settings['max_lag_sources'] < 0):
            raise RuntimeError('max_lag_sources has to be an integer >= 0.')
        if (type(self.settings['max_lag_target']) is not int or
                self.settings['max_lag_target'] <= 0):
            raise RuntimeError('max_lag_target must be an integer > 0.')
        if (type(self.settings['tau_sources']) is not int or
                self.settings['tau_sources'] < 0):
            raise RuntimeError('tau_sources must be an integer >= 0.')
        if (type(self.settings['tau_target']) is not int or
                self.settings['tau_target'] < 1):
            raise RuntimeError('tau_sources must be an integer > 0.')
        if self.settings['min_lag_sources'] > self.settings['max_lag_sources']:
            raise RuntimeError('min_lag_sources ({0}) must be smaller or equal'
                               ' to max_lag_sources ({1}).'.format(
                                   self.settings['min_lag_sources'],
                                   self.settings['max_lag_sources']))
        if self.settings['tau_sources'] > self.settings['max_lag_sources']:
            raise RuntimeError('tau_sources ({0}) has to be smaller than '
                               'max_lag_sources ({1}).'.format(
                                   self.settings['tau_sources'],
                                   self.settings['max_lag_sources']))
        if self.settings['tau_target'] > self.settings['max_lag_target']:
            raise RuntimeError('tau_target ({0}) has to be smaller than '
                               'max_lag_target ({1}).'.format(
                                   self.settings['tau_target'],
                                   self.settings['max_lag_target']))

        # Set CMI estimator.
        self._set_cmi_estimator()

        # Check the provided target and sources.
        self._check_target(target, data.n_processes)
        self._check_source_set(sources, data.n_processes)

        # Check provided search depths (lags) for source and target, set the
        # current_value.
        max_lag = max(self.settings['max_lag_sources'],
                      self.settings['max_lag_target'])

        assert(data.n_samples >= max_lag + 1), (
            'Not enough samples in data ({0}) to allow for the chosen maximum '
            'lag ({1})'.format(data.n_samples, max_lag))

        self.current_value = (self.target, max_lag)
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

        # Reset all attributes to inital values if the instance of
        # MultivariateTE has been used before.
        if self.selected_vars_full:
            self.selected_vars_full = []
            self._selected_vars_realisations = None
            self.selected_vars_sources = []
            self.selected_vars_target = []
            self.statistic_omnibus = None
            self.pvalue_omnibus = None
            self.pvalues_sign_sources = None
            self.te_sign_sources = None
            self._min_stats_surr_table = None

        # Check if the user provided a list of candidates that must go into
        # the conditioning set. These will be added and used for TE estimation,
        # but never tested for significance.
        if self.settings['add_conditionals'] is not None:
            self._force_conditionals(self.settings['add_conditionals'], data)

    def _include_target_candidates(self, data):
        """Test candidates from the target's past."""
        procs = [self.target]
        # Make samples
        samples = np.arange(
                self.current_value[1] - 1,
                self.current_value[1] - self.settings['max_lag_target'] - 1,
                -self.settings['tau_target']).tolist()
        candidates = self._define_candidates(procs, samples)
        sources_found = self._include_candidates(candidates, data)

        # If no candidates were found in the target's past, add at least one
        # sample so we are still calculating a proper TE.
        if not sources_found:
            print('\nNo informative sources in the target\'s past - '
                  'adding target sample with lag 1.')
            idx = (self.current_value[0], self.current_value[1] - 1)
            realisations = data.get_realisations(self.current_value, [idx])[0]
            self._append_selected_vars([idx], realisations)

    def _reset(self):
        """Reset instance after analysis."""
        self.__init__()
        del self.settings
        del self.source_set
        del self.pvalues_sign_sources
        del self.statistic_sign_sources
        del self.statistic_omnibus
        del self.pvalue_omnibus
        del self.sign_omnibus
        del self._cmi_estimator


class NetworkInferenceBivariate(NetworkInference):
    """Parent class for bivariate network inference algorithms."""

    def __init__(self):
        super().__init__()

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
        # Define samples for candidate sets.
        if self.settings['max_lag_sources'] == 0:
            samples = np.zeros(1).astype(int)
        else:
            samples = np.arange(
                self.current_value[1] - self.settings['min_lag_sources'],
                self.current_value[1] - self.settings['max_lag_sources'] - 1,
                -self.settings['tau_sources'])

        # Check if target variables were selected to distinguish between TE
        # and MI analysis.
        if len(self._selected_vars_target) == 0:
            conditional_realisations_target = None
        else:
            conditional_realisations_target = (
                self._selected_vars_target_realisations)

        # Iterate over all potential sources in the analysis. This way, the
        # conditioning uses past variables from the current source only
        # (opposed to past variables from all sources as in multivariate
        # network inference).
        success = False
        for source in self.source_set:
            candidate_set = self._define_candidates([source], samples)
            if self.settings['verbose']:
                    print('candidate set current source: {0}\n'.format(
                            self._idx_to_lag(candidate_set)), end='')

            # Initialise conditional realisations. This gets updated if sources
            # are selected in the iterative conditioning.
            conditional_realisations = conditional_realisations_target

            while candidate_set:
                # Get realisations for all candidates.
                cand_real = data.get_realisations(self.current_value,
                                                  candidate_set)[0]
                # Reshape candidates to a 1D-array, where realisations for a
                # single candidate are treated as one chunk.
                cand_real = cand_real.T.reshape(cand_real.size, 1)

                # Calculate the (C)MI for each candidate and the target.
                try:
                    temp_te = self._cmi_estimator.estimate_parallel(
                                n_chunks=len(candidate_set),
                                re_use=['var2', 'conditional'],
                                var1=cand_real,
                                var2=self._current_value_realisations,
                                conditional=conditional_realisations)
                except ex.AlgorithmExhaustedError as aee:
                    # The algorithm cannot continue here, so
                    #  we'll terminate the search for more candidates,
                    #  though those identified already remain valid
                    print('AlgorithmExhaustedError encountered in '
                        'estimations: ' + aee.message)
                    print('Halting current estimation set.')
                    # For now we don't need a stack trace:
                    # traceback.print_tb(aee.__traceback__)
                    break

                # Test max CMI for significance with maximum statistics.
                te_max_candidate = max(temp_te)
                max_candidate = candidate_set[np.argmax(temp_te)]
                if self.settings['verbose']:
                    print('testing candidate: {0} '.format(
                        self._idx_to_lag([max_candidate])[0]), end='')
                try:
                    significant = stats.max_statistic(
                        self, data, candidate_set,
                        te_max_candidate, conditional_realisations)[0]
                except ex.AlgorithmExhaustedError as aee:
                    # The algorithm cannot continue here, so
                    #  we'll terminate the significance check for this candidate,
                    #  though those identified already remain valid
                    print('AlgorithmExhaustedError encountered in '
                        'estimations: ' + aee.message)
                    print('Halting candidate max stats test')
                    # For now we don't need a stack trace:
                    # traceback.print_tb(aee.__traceback__)
                    break

                # If the max is significant move it from the candidate set to
                # the set of selected sources and test the next candidate. If
                # it is not significant break. There will be no further
                # significant sources b/c they all have lesser TE.
                if significant:
                    success = True
                    candidate_set.pop(np.argmax(temp_te))
                    candidate_realisations = data.get_realisations(
                        self.current_value, [max_candidate])[0]
                    self._append_selected_vars(
                            [max_candidate], candidate_realisations)
                    # Update conditioning set for max. statistics in the next
                    # round.
                    if conditional_realisations is None:
                        conditional_realisations = candidate_realisations
                    else:
                        conditional_realisations = np.hstack((
                            conditional_realisations, candidate_realisations))
                else:
                    if self.settings['verbose']:
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
        """
        # FOR LATER we don't need to test the last included in the first round
        if self.settings['verbose']:
            if not self.selected_vars_sources:
                print('no sources selected, nothing to prune ...')

        # Check if target variables were selected to distinguish between TE
        # and MI analysis.
        if len(self._selected_vars_target) == 0:
            conditional_realisations_target = None
            cond_target_dim = 0
        else:
            conditional_realisations_target = (
                self._selected_vars_target_realisations)
            cond_target_dim = conditional_realisations_target.shape[1]
        # Prune all selected sources separately. This way, the conditioning
        # uses past variables from the current source only (opposed to past
        # variables from all sources as in multivariate network inference).
        significant_sources = np.unique(
            [s[0] for s in self.selected_vars_sources])
        for source in significant_sources:
            # Find selected past variables for current source
            print('selected vars sources {0}'.format(self.selected_vars_sources))
            source_vars = [s for s in self.selected_vars_sources if
                           s[0] == source]
            print('selected candidates current source: {0}'.format(
                        self._idx_to_lag(source_vars)))
            # If only a single variable was selected for the current source, no
            # pruning is necessary. The minimum statistic would be equal to the
            # maximum statistic for this variable.
            if len(source_vars) == 1:
                if self.settings['verbose']:
                        print(' -- significant')
                continue

            # Find the candidate with the minimum TE/MI into the target.
            while source_vars:
                # Allocate memory, collect realisations, and calculate TE/MI
                # in parallel for all selected variables in the current
                # process.
                temp_te = np.empty(len(source_vars))
                cond_dim = cond_target_dim + len(source_vars) - 1
                candidate_realisations = np.empty(
                    (data.n_realisations(self.current_value) *
                     len(source_vars), 1)).astype(data.data_type)
                conditional_realisations = np.empty(
                    (data.n_realisations(self.current_value) *
                     len(source_vars),
                     cond_dim)).astype(data.data_type)

                i_1 = 0
                i_2 = data.n_realisations(self.current_value)
                for candidate in source_vars:
                    temp_cond = data.get_realisations(
                        self.current_value,
                        set(source_vars).difference(set([candidate])))[0]
                    temp_cand = data.get_realisations(
                        self.current_value, [candidate])[0]

                    if temp_cond is None:
                        conditional_realisations = conditional_realisations_target
                        re_use = ['var2', 'conditional']
                    else:
                        re_use = ['var2']
                        if conditional_realisations_target is None:
                            conditional_realisations[i_1:i_2, ] = temp_cond
                        else:
                            conditional_realisations[i_1:i_2, ] = np.hstack((
                                temp_cond, conditional_realisations_target))
                    candidate_realisations[i_1:i_2, ] = temp_cand
                    i_1 = i_2
                    i_2 += data.n_realisations(self.current_value)

                try:
                    temp_te = self._cmi_estimator.estimate_parallel(
                                    n_chunks=len(source_vars),
                                    re_use=re_use,
                                    var1=candidate_realisations,
                                    var2=self._current_value_realisations,
                                    conditional=conditional_realisations)
                except ex.AlgorithmExhaustedError as aee:
                    # The algorithm cannot continue here, so
                    #  we'll terminate the pruning check,
                    #  assuming that we need not prune any more
                    print('AlgorithmExhaustedError encountered in '
                        'estimations: ' + aee.message)
                    print('Halting current pruning and allowing others to'
                        ' remain.')
                    # For now we don't need a stack trace:
                    # traceback.print_tb(aee.__traceback__)
                    break

                # Find variable with minimum MI/TE. Test min TE/MI for
                # significance with minimum statistics. Build conditioning set
                # for minimum statistics by removing the minimum candidate.
                te_min_candidate = min(temp_te)
                min_candidate = source_vars[np.argmin(temp_te)]
                if self.settings['verbose']:
                    print('testing candidate: {0} '.format(
                        self._idx_to_lag([min_candidate])[0]), end='')

                remaining_candidates = set(source_vars).difference(
                    set([min_candidate]))
                conditional_realisations_sources = data.get_realisations(
                        self.current_value, remaining_candidates)[0]
                if conditional_realisations_target is None:
                    conditional_realisations = conditional_realisations_sources
                elif conditional_realisations_sources is None:
                    conditional_realisations = conditional_realisations_target
                else:
                    conditional_realisations = np.hstack((
                        conditional_realisations_target,
                        conditional_realisations_sources))
                try:
                    [significant, p, surr_table] = stats.min_statistic(
                                                self, data,
                                                source_vars,
                                                te_min_candidate,
                                                conditional_realisations)
                except ex.AlgorithmExhaustedError as aee:
                    # The algorithm cannot continue here, so
                    #  we'll terminate the pruning check,
                    #  assuming that we need not prune any more
                    print('AlgorithmExhaustedError encountered in '
                        'estimations: ' + aee.message)
                    print('Halting current pruning and allowing others to'
                        ' remain.')
                    # For now we don't need a stack trace:
                    # traceback.print_tb(aee.__traceback__)
                    break

                # Remove the minimum it is not significant and test the next
                # min. candidate. If the minimum is significant, break. All
                # other sources will be significant as well (b/c they have
                # higher TE/MI).
                if not significant:
                    self._remove_selected_var(min_candidate)
                    source_vars.pop(np.argmin(temp_te))
                    if len(source_vars) == 0:
                        print('No remaining candidates after pruning.')
                else:
                    if self.settings['verbose']:
                        print(' -- significant')
                    break

    def _test_final_conditional(self, data):
        """Perform statistical test on the final conditional set."""
        if not self.selected_vars_sources:
            if self.settings['verbose']:
                print('no sources selected ...')
            self.statistic_omnibus = None
            self.sign_omnibus = False
            self.pvalue_omnibus = None
            self.pvalues_sign_sources = None
            self.statistic_sign_sources = None
            self.statistic_single_link = None
        else:
            if self.settings['verbose']:
                print('selected variables: {0}'.format(
                    self._idx_to_lag(self.selected_vars_full)))
            try:
                [s, p, stat] = stats.omnibus_test(self, data)
            except ex.AlgorithmExhaustedError as aee:
                # The algorithm cannot continue here, so
                #  we'll set the results to zero
                print('AlgorithmExhaustedError encountered in '
                    'estimations: ' + aee.message)
                print('Halting omnibus test and setting to not significant.')
                # For now we don't need a stack trace:
                # traceback.print_tb(aee.__traceback__)
                stat = 0
                s = False
                p = 1
            self.statistic_omnibus = stat
            self.sign_omnibus = s
            self.pvalue_omnibus = p
            # Test individual links if the omnibus test is significant using
            # the sequential max stats. Remove non-significant links.
            if self.sign_omnibus:
                # If there is an ex.AlgorithmExhaustedError exception inside
                #  max_stats_sequential, it will catch it and return
                #  everything as not significant:
                [s, p, stat] = stats.max_statistic_sequential_bivariate(
                    self, data)
                p, stat = self._remove_non_significant(s, p, stat)
                self.pvalues_sign_sources = p
                self.statistic_sign_sources = stat
                if self.measure == 'te':
                    conditioning = 'target'
                elif self.measure == 'mi':
                    conditioning = 'none'
                try:
                    self.statistic_single_link = self._calculate_single_link(
                        data=data,
                        current_value=self.current_value,
                        source_vars=self.selected_vars_sources,
                        target_vars=self.selected_vars_target,
                        sources='all',
                        conditioning=conditioning)
                except ex.AlgorithmExhaustedError as aee:
                    # The algorithm cannot continue here, so
                    #  we'll terminate the computation of single link stats.
                    #  Since max stats sequential etc all passed up to here,
                    #  it seems ok to let everything through still but
                    #  just write a 0 for final values
                    print('AlgorithmExhaustedError encountered in '
                        'final_conditional estimations: ' + aee.message)
                    print('Halting final_conditional estimations')
                    # For now we don't need a stack trace:
                    # traceback.print_tb(aee.__traceback__)
                    self.statistic_single_link = \
                        np.zeros(len(self.selected_vars_sources))
            else:
                self.selected_vars_sources = []
                self.selected_vars_full = self.selected_vars_target
                self.pvalues_sign_sources = None
                self.statistic_sign_sources = None
                self.statistic_single_link = None


class NetworkInferenceMultivariate(NetworkInference):
    """Parent class for multivariate network inference algorithms."""

    def __init__(self):
        super().__init__()

    def _include_source_candidates(self, data):
        """Test candidates in the source's past."""
        procs = self.source_set
        if self.settings['max_lag_sources'] == 0:
            samples = np.zeros(1).astype(int)
        else:
            samples = np.arange(
                self.current_value[1] - self.settings['min_lag_sources'],
                self.current_value[1] - self.settings['max_lag_sources'] - 1,
                -self.settings['tau_sources'])
        candidates = self._define_candidates(procs, samples)
        # Possible extension in the future: include non-selected target
        # candidates as further candidates, # they may get selected due to
        # synergies.
        self._include_candidates(candidates, data)

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
        if self.settings['verbose']:
            if self.selected_vars_sources:
                print('selected candidates: {0}'.format(
                        self._idx_to_lag(self.selected_vars_sources)))
            else:
                print('no sources selected, nothing to prune ...')
        # If only a single variable was selected, no pruning is necessary. The
        # minimum statistic would be equal to the maximum statistic for this
        # variable.
        if len(self.selected_vars_sources) == 1:
            if self.settings['verbose']:
                print(' -- significant')
            return
        while self.selected_vars_sources:
            # Find the candidate with the minimum TE into the target.
            temp_te = np.empty(len(self.selected_vars_sources))
            cond_dim = len(self.selected_vars_full) - 1
            candidate_realisations = np.empty(
                (data.n_realisations(self.current_value) *
                 len(self.selected_vars_sources), 1)).astype(data.data_type)
            conditional_realisations = np.empty(
                (data.n_realisations(self.current_value) *
                 len(self.selected_vars_sources),
                 cond_dim)).astype(data.data_type)

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
                    re_use = ['var2', 'conditional']
                else:
                    conditional_realisations[i_1:i_2, ] = temp_cond
                    re_use = ['var2']
                candidate_realisations[i_1:i_2, ] = temp_cand
                i_1 = i_2
                i_2 += data.n_realisations(self.current_value)

            try:
                temp_te = self._cmi_estimator.estimate_parallel(
                                n_chunks=len(self.selected_vars_sources),
                                re_use=re_use,
                                var1=candidate_realisations,
                                var2=self._current_value_realisations,
                                conditional=conditional_realisations)
            except ex.AlgorithmExhaustedError as aee:
                # The algorithm cannot continue here, so
                #  we'll terminate the pruning check,
                #  assuming that we need not prune any more
                print('AlgorithmExhaustedError encountered in '
                    'estimations: ' + aee.message)
                print('Halting current pruning and allowing others to'
                    ' remain.')
                # For now we don't need a stack trace:
                # traceback.print_tb(aee.__traceback__)
                break

            # Find variable with minimum MI/TE. Test min TE/MI for significance
            # with minimum statistics. Build conditioning set for minimum
            # statistics by removing the minimum candidate.
            te_min_candidate = min(temp_te)
            min_candidate = self.selected_vars_sources[np.argmin(temp_te)]
            if self.settings['verbose']:
                print('testing candidate: {0} '.format(
                    self._idx_to_lag([min_candidate])[0]), end='')

            remaining_candidates = set(self.selected_vars_full).difference(
                    set([min_candidate]))
            conditional_realisations = data.get_realisations(
                        self.current_value, remaining_candidates)[0]
            try:
                [significant, p, surr_table] = stats.min_statistic(
                                              self, data,
                                              self.selected_vars_sources,
                                              te_min_candidate,
                                              conditional_realisations)
            except ex.AlgorithmExhaustedError as aee:
                # The algorithm cannot continue here, so
                #  we'll terminate the pruning check,
                #  assuming that we need not prune any more
                print('AlgorithmExhaustedError encountered in '
                    'estimations: ' + aee.message)
                print('Halting current pruning and allowing others to'
                    ' remain.')
                # For now we don't need a stack trace:
                # traceback.print_tb(aee.__traceback__)
                break

            # Remove the minimum it is not significant and test the next min.
            # candidate. If the minimum is significant, break, all other
            # sources will be significant as well (b/c they have higher TE).
            if not significant:
                # if self.settings['verbose']:
                #     print(' -- not significant\n')
                self._remove_selected_var(min_candidate)
                if len(self.selected_vars_sources) == 0:
                        print('No remaining candidates after pruning.')
            else:
                if self.settings['verbose']:
                    print(' -- significant')
                self._min_stats_surr_table = surr_table
                break

    def _test_final_conditional(self, data):
        """Perform statistical test on the final conditional set."""
        if not self.selected_vars_sources:
            if self.settings['verbose']:
                print('no sources selected ...')
            self.statistic_omnibus = None
            self.sign_omnibus = False
            self.pvalue_omnibus = None
            self.pvalues_sign_sources = None
            self.statistic_sign_sources = None
            self.statistic_single_link = None
        else:
            if self.settings['verbose']:
                print('selected variables: {0}'.format(
                    self._idx_to_lag(self.selected_vars_full)))
            try:
                [s, p, stat] = stats.omnibus_test(self, data)
            except ex.AlgorithmExhaustedError as aee:
                # The algorithm cannot continue here, so
                #  we'll set the results to zero
                print('AlgorithmExhaustedError encountered in '
                    'estimations: ' + aee.message)
                print('Halting omnibus test and setting to not significant.')
                # For now we don't need a stack trace:
                # traceback.print_tb(aee.__traceback__)
                stat = 0
                s = False
                p = 1
            self.statistic_omnibus = stat
            self.sign_omnibus = s
            self.pvalue_omnibus = p
            # Test individual links if the omnibus test is significant using
            # the sequential max stats. Remove non-significant links.
            if self.sign_omnibus:
                # If there is an ex.AlgorithmExhaustedError exception inside
                #  max_stats_sequential, it will catch it and return
                #  everything as not significant:
                [s, p, stat] = stats.max_statistic_sequential(self, data)
                p, stat = self._remove_non_significant(s, p, stat)
                self.pvalues_sign_sources = p
                self.statistic_sign_sources = stat
                # Calculate TE for all links in the network. Calculate local TE
                # if requested by the user.
                if self.measure == 'te':
                    conditioning = 'target'
                elif self.measure == 'mi':
                    conditioning = 'none'
                try:
                    self.statistic_single_link = self._calculate_single_link(
                        data=data,
                        current_value=self.current_value,
                        source_vars=self.selected_vars_sources,
                        target_vars=self.selected_vars_target,
                        sources='all',
                        conditioning=conditioning)
                except ex.AlgorithmExhaustedError as aee:
                    # The algorithm cannot continue here, so
                    #  we'll terminate the computation of single link stats.
                    #  Since max stats sequential etc all passed up to here,
                    #  it seems ok to let everything through still but
                    #  just write a 0 for final values
                    print('AlgorithmExhaustedError encountered in '
                        'final_conditional estimations: ' + aee.message)
                    print('Halting final_conditional estimations')
                    # For now we don't need a stack trace:
                    # traceback.print_tb(aee.__traceback__)
                    self.statistic_single_link = \
                        np.zeros(len(self.selected_vars_sources))
            else:
                self.selected_vars_sources = []
                self.selected_vars_full = self.selected_vars_target
                self.pvalues_sign_sources = None
                self.statistic_sign_sources = None
                self.statistic_single_link = None
