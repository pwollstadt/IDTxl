"""Parent class for all network inference.

Created on Mon Mar  7 18:13:27 2016

@author: patricia
"""
import numpy as np
from .network_analysis import NetworkAnalysis
from . import stats

VERBOSE = True


class NetworkInference(NetworkAnalysis):
    """Provide an analysis setup for multivariate network inference.

    Hold variables that are relevant for multivariate network inference.
    The class holds

    (1) analysis parameters
    (2) 'analysis pattern', i.e., indices of random variables used for
    network inference (e.g. current value and conditional in transfer entropy
    estimation)
    (3) temporary data for analysis, i.e., realisations of the variables

    The class provides routines to check user input and set defaults. The
    'analysis pattern' is represented by tuples or list of tuples (process
    index, sample index), where a tuple indicates where to find realisations in
    the data.

    Attributes:
        options : dict
            options for estimation of information theoretic measures and
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

    def __init__(self, max_lag_sources, min_lag_sources, options,
                 max_lag_target=None, tau_sources=1, tau_target=1):
        # Check user input. Allow for unembedded sources (max_lag_sources=0)
        if not max_lag_target:
            max_lag_target = max_lag_sources
        if type(min_lag_sources) is not int or min_lag_sources < 0:
            raise RuntimeError('min_lag_sources has to be an integer >= 0.')
        if type(max_lag_sources) is not int or max_lag_sources < 0:
            raise RuntimeError('max_lag_sources has to be an integer >= 0.')
        if type(max_lag_target) is not int or max_lag_target <= 0:
            raise RuntimeError('max_lag_target must be an integer > 0.')
        if type(tau_sources) is not int or tau_sources <= 0:
            raise RuntimeError('tau_sources must be an integer > 0.')
        if type(tau_target) is not int or tau_target <= 0:
            raise RuntimeError('tau_sources must be an integer > 0.')
        if min_lag_sources > max_lag_sources:
            raise RuntimeError('min_lag_sources ({0}) must be smaller or equal'
                               ' to max_lag_sources ({1}).'.format(
                                   min_lag_sources, max_lag_sources))
        if tau_sources > max_lag_sources:
            raise RuntimeError('tau_sources ({0}) has to be smaller than '
                               'max_lag_sources ({1}).'.format(
                                   tau_sources, max_lag_sources))
        if tau_target > max_lag_target:
            raise RuntimeError('tau_target ({0}) has to be smaller than '
                               'max_lag_target ({1}).'.format(
                                   tau_target, max_lag_target))

        # Set default options
        self.options = options
        self.options.setdefault('fdr_correction', True)

        # Set user-specified estimation parameters
        self.max_lag_target = max_lag_target
        self.max_lag_sources = max_lag_sources
        self.min_lag_sources = min_lag_sources
        self.tau_sources = tau_sources
        self.tau_target = tau_target

        # Create class attributes for estimation
        self.te_omnibus = None
        self.te_sign_sources = None
        self.sign_omnibus = False
        self.sign_sign_sources = None
        self.pvalue_omnibus = None
        self.pvalues_sign_sources = None
        self._min_stats_surr_table = None
        super().__init__()

    def _initialise(self, data, sources, target):
        """Check input and set everything to initial values."""
        # Check the provided target and sources.
        self._check_target(target, data.n_processes)
        self._check_source_set(sources, data.n_processes)

        # Check provided search depths (lags) for source and target, set the
        # current_value.
        max_lag = max(self.max_lag_sources, self.max_lag_target)

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
            self.te_omnibus = None
            self.sign_sign_sources = None
            self.pvalue_omnibus = None
            self.pvalues_sign_sources = None
            self.te_sign_sources = None
            self._min_stats_surr_table = None

        # Check if the user provided a list of candidates that must go into
        # the conditioning set. These will be added and used for TE estimation,
        # but never tested for significance.
        try:
            cond = self.options['add_conditionals']
            self._force_conditionals(cond, data)
        except KeyError:
            pass

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
        if VERBOSE:
            print('Testing sources {0}'.format(self.source_set))

    def _include_target_candidates(self, data):
        """Test candidates from the target's past."""
        procs = [self.target]
        # Make samples
        samples = np.arange(self.current_value[1] - 1,
                            self.current_value[1] - self.max_lag_target - 1,
                            -self.tau_target).tolist()
        candidates = self._define_candidates(procs, samples)
        sources_found = self._include_candidates(candidates, data)

        # If no candidates were found in the target's past, add at least one
        # sample so we are still calculating a proper TE.
        if not sources_found:  # TODO put a flag in to make this optional
            print(('No informative sources in the target''s past - ' +
                   'adding point at t-1 in the target'))
            idx = (self.current_value[0], self.current_value[1] - 1)
            realisations = data.get_realisations(self.current_value, [idx])[0]
            self._append_selected_vars([idx], realisations)

    def _include_source_candidates(self, data):
        """Test candidates in the source's past."""
        procs = self.source_set
        samples = np.arange(self.current_value[1] - self.min_lag_sources,
                            self.current_value[1] - self.max_lag_sources,
                            -self.tau_sources)
        candidates = self._define_candidates(procs, samples)
        # TODO include non-selected target candidates as further candidates,
        # they may get selected due to synergies
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
            temp_te = self._cmi_estimator.estimate_mult(
                                n_chunks=len(candidate_set),
                                re_use=['var2', 'conditional'],
                                var1=cand_real,
                                var2=self._current_value_realisations,
                                conditional=self._selected_vars_realisations)

            # Test max CMI for significance with maximum statistics.
            te_max_candidate = max(temp_te)
            max_candidate = candidate_set[np.argmax(temp_te)]
            if VERBOSE:
                print('testing {0} from candidate set {1}'.format(
                                    self._idx_to_lag([max_candidate])[0],
                                    self._idx_to_lag(candidate_set)), end='')
            significant = stats.max_statistic(self, data, candidate_set,
                                              te_max_candidate)[0]

            # If the max is significant keep it and test the next candidate. If
            # it is not significant break. There will be no further significant
            # sources b/c they all have lesser TE.
            if significant:
                if VERBOSE:
                    print(' -- significant')
                success = True
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
        self._append_selected_vars(cond,
                                   data.get_realisations(self.current_value,
                                                         cond)[0])
