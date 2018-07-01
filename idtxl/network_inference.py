"""Parent class for all network inference.

Created on Mon Mar  7 18:13:27 2016

@author: patricia
"""
import numpy as np
from .network_analysis import NetworkAnalysis
from .estimator import find_estimator
from . import stats


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
            print('Testing sources {0}'.format(self.source_set))

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
        while candidate_set:
            # Get realisations for all candidates.
            cand_real = data.get_realisations(self.current_value,
                                              candidate_set)[0]
            cand_real = cand_real.T.reshape(cand_real.size, 1)

            # Calculate the (C)MI for each candidate and the target.
            temp_te = self._cmi_estimator.estimate_parallel(
                                n_chunks=len(candidate_set),
                                re_use=['var2', 'conditional'],
                                var1=cand_real,
                                var2=self._current_value_realisations,
                                conditional=self._selected_vars_realisations)

            # Test max CMI for significance with maximum statistics.
            te_max_candidate = max(temp_te)
            max_candidate = candidate_set[np.argmax(temp_te)]
            if self.settings['verbose']:
                print('testing {0} from candidate set {1}'.format(
                                    self._idx_to_lag([max_candidate])[0],
                                    self._idx_to_lag(candidate_set)), end='')
            significant = stats.max_statistic(self, data, candidate_set,
                                              te_max_candidate)[0]

            # If the max is significant keep it and test the next candidate. If
            # it is not significant break. There will be no further significant
            # sources b/c they all have lesser TE.
            if significant:
                if self.settings['verbose']:
                    print(' -- significant')
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
                cond = self._define_candidates(self.source_set,
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
                               '(''max_lag_sources'') needs to be specified.')

        if (type(self.settings['min_lag_sources']) is not int or
                self.settings['min_lag_sources'] < 0):
            raise RuntimeError('min_lag_sources has to be an integer >= 0.')
        if (type(self.settings['max_lag_sources']) is not int or
                self.settings['max_lag_sources'] < 0):
            raise RuntimeError('max_lag_sources has to be an integer >= 0.')
        if (type(self.settings['tau_sources']) is not int or
                self.settings['tau_sources'] <= 0):
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

        # Check if the user requested the estimation of local values.
        # Internally, the estimator uses the user settings for building the
        # non-uniform embedding, etc. Remember the user setting and set
        # local_values to False temporarily.
        if self.settings['local_values']:
            self._local_values = True
            self.settings['local_values'] = False
        else:
            self._local_values = False

        # Set CMI estimator.
        try:
            EstimatorClass = find_estimator(self.settings['cmi_estimator'])
        except KeyError:
            raise RuntimeError('Please provide an estimator class or name!')
        self._cmi_estimator = EstimatorClass(self.settings)

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
            self._local_values = None

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
                               '(''max_lag_sources'') needs to be specified.')
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
                self.settings['tau_sources'] <= 0):
            raise RuntimeError('tau_sources must be an integer > 0.')
        if (type(self.settings['tau_target']) is not int or
                self.settings['tau_target'] <= 0):
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

        # Check if the user requested the estimation of local values.
        # Internally, the estimator uses the user settings for building the
        # non-uniform embedding, etc. Remember the user setting and set
        # local_values to False temporarily.
        if self.settings['local_values']:
            self._local_values = True
            self.settings['local_values'] = False
        else:
            self._local_values = False

        # Set CMI estimator.
        try:
            EstimatorClass = find_estimator(self.settings['cmi_estimator'])
        except KeyError:
            raise RuntimeError('Please provide an estimator class or name!')
        self._cmi_estimator = EstimatorClass(self.settings)

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
            self._local_values = False

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
            print(('No informative sources in the target''s past - ' +
                   'adding point at t-1 in the target'))
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
        if self.measure == 'te':
            conditioning = 'target'
        elif self.measure == 'mi':
            conditioning = 'none'
        [s, p, stat] = stats.max_statistic_sequential(self, data, conditioning)

        # Remove non-significant links from the source set
        p, stat = self._remove_non_significant(s, p, stat)
        self.pvalues_sign_sources = p
        self.statistic_sign_sources = stat

    def _test_final_conditional(self, data):
        """Perform statistical test on the final conditional set."""
        if not self.selected_vars_sources:
            print('---------------------------- no sources found')
            self.statistic_omnibus = None
            self.sign_omnibus = False
            self.pvalue_omnibus = None
            if self._local_values:
                    self.settings['local_values'] = True
            self.statistic_single_link = None
        else:
            print(self._idx_to_lag(self.selected_vars_full))
            [s, p, stat] = stats.omnibus_test(self, data)
            self.statistic_omnibus = stat
            self.sign_omnibus = s
            self.pvalue_omnibus = p
            if self._local_values:
                    self.settings['local_values'] = True
            if self.measure == 'te':
                conditioning = 'target'
            elif self.measure == 'mi':
                conditioning = 'none'
            self.statistic_single_link = self._calculate_single_link(
                    data=data,
                    current_value=self.current_value,
                    source_vars=self.selected_vars_sources,
                    target_vars=self.selected_vars_target,
                    sources='all',
                    conditioning=conditioning)


class NetworkInferenceMultivariate(NetworkInference):

    def __init__(self):
        super().__init__()

    def _include_source_candidates(self, data):
        """Test candidates in the source's past."""
        procs = self.source_set
        samples = np.arange(
                    self.current_value[1] - self.settings['min_lag_sources'],
                    self.current_value[1] - self.settings['max_lag_sources'],
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
        print(self.selected_vars_sources)
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
                else:
                    conditional_realisations[i_1:i_2, ] = temp_cond
                candidate_realisations[i_1:i_2, ] = temp_cand
                i_1 = i_2
                i_2 += data.n_realisations(self.current_value)

            temp_te = self._cmi_estimator.estimate_parallel(
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
                    print(' -- not significant\n')
                self._remove_selected_var(min_candidate)
            else:
                if self.settings['verbose']:
                    print(' -- significant\n')
                self._min_stats_surr_table = surr_table
                break

    def _test_final_conditional(self, data):
        """Perform statistical test on the final conditional set."""
        if not self.selected_vars_sources:
            print('---------------------------- no sources found')
            self.statistic_omnibus = None
            self.sign_omnibus = False
            self.pvalue_omnibus = None
            self.pvalues_sign_sources = None
            self.statistic_sign_sources = None
            if self._local_values:
                    self.settings['local_values'] = True
            self.statistic_single_link = None
        else:
            print(self._idx_to_lag(self.selected_vars_full))
            [s, p, stat] = stats.omnibus_test(self, data)
            self.statistic_omnibus = stat
            self.sign_omnibus = s
            self.pvalue_omnibus = p
            # Test individual links if the omnibus test is significant using
            # the sequential max stats. Remove non-significant links.
            if self.sign_omnibus:
                [s, p, stat] = stats.max_statistic_sequential(self, data)
                p, stat = self._remove_non_significant(s, p, stat)
                self.pvalues_sign_sources = p
                self.statistic_sign_sources = stat
                # Calculate TE for all links in the network. Calculate local TE
                # if requested by the user.
                if self._local_values:
                    self.settings['local_values'] = True
                self.statistic_single_link = self._calculate_single_link(
                    data=data,
                    current_value=self.current_value,
                    source_vars=self.selected_vars_sources,
                    target_vars=self.selected_vars_target,
                    sources='all')
            else:
                self.selected_vars_sources = []
                self.selected_vars_full = self.selected_vars_target
                self.pvalues_sign_sources = None
                self.statistic_sign_sources = None
                self.statistic_single_link = None
                if self._local_values:
                    self.settings['local_values'] = True
