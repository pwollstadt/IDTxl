"""Parent class for all network inference.

Created on Mon Mar  7 18:13:27 2016

@author: patricia
"""
import numpy as np
from .network_analysis import Network_analysis

VERBOSE = True


class Network_inference(Network_analysis):
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

        # Set user-specified estimation parameters
        if max_lag_target is None:
            self.max_lag_target = max_lag_sources
        else:
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
        self.options = options
        self._min_stats_surr_table = None
        super().__init__()

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
        # Multivariate_te has been used before.
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
        # TODO include non-selected target candidates as further candidates, they may get selected due to synergies
        self._include_candidates(candidates, data)

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
