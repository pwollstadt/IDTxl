"""Parent class for network inference and network comparison.
"""
import copy as cp
import itertools as it
import numpy as np
from .estimator import find_estimator
from . import idtxl_utils as utils


class NetworkAnalysis():
    """Provide an analysis setup for network inference or comparison.

    The class provides routines to check user input and set defaults.
    """

    def __init__(self):
        self.target = None
        self.current_value = None
        self.selected_vars_full = []
        self.selected_vars_sources = []
        self.selected_vars_target = []
        self._current_value_realisations = None
        self._selected_vars_realisations = None
        self._min_stats_surr_table = None

    @property
    def current_value(self):
        """Get index of the current_value."""
        return self._current_value

    @current_value.setter
    def current_value(self, idx):
        if (idx is not None) and (type(idx) is not tuple):
            raise TypeError(('The current value should be a tuple (index ' +
                             'process, index sample).'))
        self._current_value = idx

    @property
    def _current_value_realisations(self):
        """Get realisations of the current_value."""
        if self.__current_value_realisations is None:
            print('Attribute has not been set yet.')
        if type(self.__current_value_realisations) is tuple:
            raise TypeError('something went wrong')
        return self.__current_value_realisations

    @_current_value_realisations.setter
    def _current_value_realisations(self, realisations):
        self.__current_value_realisations = realisations

    @_current_value_realisations.deleter
    def _current_value_realisations(self):
        del self.__current_value_realisations

    @property
    def selected_vars_full(self):
        """List of indices of the full conditional set."""
        if self._selected_vars_full is None:
            print('Attribute has not been set yet.')
        return self._selected_vars_full

    @selected_vars_full.setter
    def selected_vars_full(self, idx_list):
        if (type(idx_list) is not list and (type(idx_list[0]) is not tuple)):
            raise TypeError(('Expected a list of tuples (index process, ' +
                             'index sample).'))
        self._selected_vars_full = idx_list

    @property
    def selected_vars_target(self):
        """List of indices of target samples in the conditional set."""
        if self._selected_vars_target is None:
            print('Attribute has not been set yet.')
        return self._selected_vars_target

    @selected_vars_target.setter
    def selected_vars_target(self, idx_list):
        if (idx_list is not None and type(idx_list) is not list):
            raise TypeError(('Expected a list of tuples (index process, ' +
                             'index sample).'))
        self._selected_vars_target = idx_list

    @property
    def selected_vars_sources(self):
        """List of indices of source samples in the conditional set."""
        if self._selected_vars_sources is None:
            print('Attribute has not been set yet.')
        return self._selected_vars_sources

    @selected_vars_sources.setter
    def selected_vars_sources(self, idx_list):
        if (idx_list is not None and type(idx_list) is not list):
            raise TypeError(('Expected a list of tuples (index process, ' +
                             'index sample).'))
        self._selected_vars_sources = idx_list

    @property
    def _selected_vars_realisations(self):
        """Get realisations of the full conditional set."""
        return self.__selected_vars_realisations

    @_selected_vars_realisations.setter
    def _selected_vars_realisations(self, realisations):
        self.__selected_vars_realisations = realisations

    @property
    def _selected_vars_target_realisations(self):
        """Get realisations of the target samples in the conditional.

        Note:
            Each time this property is called, realisations are actually
            extracted from the array of all realisations, which may be slow!
            Use temporary variables to speed things up.
        """
        if self.selected_vars_target is None:
            return None
        indices = np.zeros(len(self.selected_vars_target)).astype(int)
        for i, idx in enumerate(self.selected_vars_target):
            indices[i] = self.selected_vars_full.index(idx)
        self._selected_vars_target_realisations = (
                                self._selected_vars_realisations[:, indices])
        return self.__selected_vars_target_realisations

    @_selected_vars_target_realisations.setter
    def _selected_vars_target_realisations(self, realisations):
        self.__selected_vars_target_realisations = realisations

    @property
    def _selected_vars_sources_realisations(self):
        """Get realisations of the source samples in the conditional.

        Note:
            Each time this property is called, realisations are actually
            extracted from the array of all realisations, which may be slow!
            Use temporary variables to speed things up.
        """
        indices = np.zeros(len(self.selected_vars_sources)).astype(int)
        for (i, idx) in enumerate(self.selected_vars_sources):
            indices[i] = self.selected_vars_full.index(idx)
        self._selected_vars_sources_realisations = (
                                self._selected_vars_realisations[:, indices])
        return self.__selected_vars_sources_realisations

    @_selected_vars_sources_realisations.setter
    def _selected_vars_sources_realisations(self, realisations):
        self.__selected_vars_sources_realisations = realisations

    def _append_selected_vars_realisations(self, realisations):
        """Append realisations of conditionals to existing realisations.

        Returns:
            realisations: numpy array with dimensions replications x number
                of indices.
        """
        if self._selected_vars_realisations is None or realisations.size == 0:
            self._selected_vars_realisations = realisations
        else:
            self._selected_vars_realisations = np.hstack(
                            (self._selected_vars_realisations, realisations))

    def _idx_to_lag(self, idx_list, current_value_sample=None):
        """Change sample indices to lags for each sample in the list."""
        if current_value_sample is None:
            try:
                current_value_sample = self.current_value[1]
            except (AttributeError, TypeError):
                raise AttributeError('Current value not set.')

        lag_list = cp.copy(idx_list)
        for c in idx_list:
            if c[1] > current_value_sample:
                raise IndexError('Sample time index larger than current '
                                 'value.')
            lag_list[idx_list.index(c)] = (c[0], current_value_sample - c[1])
        return lag_list

    def _lag_to_idx(self, lag_list, current_value_sample=None):
        """Change sample lags to indices for each sample in the list."""
        if current_value_sample is None:
            try:
                current_value_sample = self.current_value[1]
            except (AttributeError, TypeError):
                raise AttributeError('Current value not set.')

        idx_list = cp.copy(lag_list)
        for c in lag_list:
            if c[1] > current_value_sample:
                raise IndexError('Sample lag larger than current value.')
            idx_list[lag_list.index(c)] = (c[0], current_value_sample - c[1])
        return idx_list

    def _set_cmi_estimator(self):
        """Check and set requested CMI estimator."""
        # Set CMI estimator. Check if the user requested the estimation of
        # local values. If so, initialise a local estimator additionally to the
        # average estimator. Internally, the average estimator is used for
        # building the non-uniform embedding, etc. The local estimator is used
        # to estimate single-link MI/TE or single-process AIS in the end.
        try:
            EstimatorClass = find_estimator(self.settings['cmi_estimator'])
        except KeyError:
            raise RuntimeError('Please provide an estimator class or name!')
        if self.settings['local_values']:
            self.settings['local_values'] = False
            self._cmi_estimator = EstimatorClass(self.settings)
            self.settings['local_values'] = True
            self._cmi_estimator_local = EstimatorClass(self.settings)
        else:
            self._cmi_estimator = EstimatorClass(self.settings)

    def _separate_realisations(self, idx_full, idx_single):
        """Separate single index realisations from a set of realisations.

        Return the realisations of a single index and the realisations of the
        remaining set of indices. The function takes realisations from the
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
        # Get indices of the remaining variables.
        idx_remaining = cp.copy(idx_full)
        idx_remaining.pop(idx_remaining.index(idx_single))

        # Find the indices of the columns with the realisations of the
        # requested variables (the single one to be removed and the remaining
        # variables).
        array_col_single = self.selected_vars_full.index(idx_single)
        array_col_remain = np.zeros(len(idx_remaining)).astype(int)
        for (i, idx) in enumerate(idx_remaining):
            array_col_remain[i] = self.selected_vars_full.index(idx)

        # Get realisations of the single and remaining variables.
        real_single = np.expand_dims(
                        self._selected_vars_realisations[:, array_col_single],
                        axis=1)
        if len(idx_full) == 1:
            # If no realiastions remain, set variable to None instead of and
            # empty array so the JIDT estimator doesn't break
            real_remain = None
        else:
            real_remain = self._selected_vars_realisations[:, array_col_remain]

        return real_remain, real_single

    def _define_candidates(self, processes, samples):
        """Build a list of candidate indices.

        Build a list of candidate indices. Note that variables that were
        manually added to the conditioning set via the 'add_conditionals'
        setting are removed from the candidate set if both sets are not
        disjoint.

        Args:
            processes : list of int
                process indices
            samples: list of int
                sample indices

        Returns:
            a list of tuples, where each tuple holds the index of one
            candidate and has the form (process index, sample index), indices
            are absolute values with respect to some data array.
        """
        candidate_set = self._build_variable_list(processes, samples)
        # Remove candidates that were already manullay added to the
        # conditioning set via the 'add_conditionals' setting. Otherwise the
        # candidates get tested in the inclusion step.
        candidate_set = self._remove_forced_conditionals(candidate_set)
        return candidate_set

    def _build_variable_list(self, processes, samples):
        """Build a list of variable tuples with (process index, sample index).

        Args:
            processes : list of int
                process indices
            samples: list of int
                sample indices

        Returns:
            a list of variable tuples
        """
        var_list = []
        for idx in it.product(processes, samples):
            var_list.append(idx)
        return var_list

    def _remove_forced_conditionals(self, candidate_set):
        """Remove enforced conditioning variables from candidate set."""
        if self.settings['add_conditionals'] is not None:
            cond = self.settings['add_conditionals']
            if type(cond) is tuple:  # easily add single variable
                cond = [cond]
            cond_idx = self._lag_to_idx(cond)            
            candidate_set = list(set(candidate_set).difference(set(cond_idx)))
        return candidate_set

    def _append_selected_vars_idx(self, idx):
        """Append indices of conditionals to existing list.

        Args:
            idx : list of tuples
                indices of selected variables, where each entry is a tuple
                (idx process, idx sample), where indices are absolute values
                with respect to entries in a data array
        """
        if self.selected_vars_full is None:
            self.selected_vars_full = idx
        else:
            for i in idx:
                self.selected_vars_full.append(i)
        # separate indices into source and target indices
        for i in idx:
            if i[0] == self.target:
                self.selected_vars_target.append(i)
            else:
                self.selected_vars_sources.append(i)

    def _append_selected_vars(self, idx, realisations):
        """Append indices and realisation of selected variables.

        Args:
            idx : list of tuples
                indices of selected variables, where each entry is a tuple
                (idx process, idx sample), where indices are absolute values
                with respect to entries in a data array
            realisations : numpy array
                realisations of the selected variables
        """
        assert len(idx) == realisations.shape[1], (
            'Dimensionality of realisations array ({0}) and length of index '
            'list ({1}) do not match.'.format(realisations.shape[1], len(idx)))
        self._append_selected_vars_idx(idx)
        self._append_selected_vars_realisations(realisations)

    def _remove_selected_var(self, idx):
        """Remove a single selected variable and its realisations."""
        self._selected_vars_realisations = utils.remove_column(
                                         self._selected_vars_realisations,
                                         self.selected_vars_full.index(idx))
        self.selected_vars_full.pop(self.selected_vars_full.index(idx))
        if idx[0] == self.target:
            self.selected_vars_target.pop(
                                        self.selected_vars_target.index(idx))
        else:
            self.selected_vars_sources.pop(
                                        self.selected_vars_sources.index(idx))

    def _calculate_single_link(
                    self, data, current_value, source_vars, target_vars=None,
                    sources='all', conditioning='full'):
        """Calculate dependency measure for all links into a target.

        Calculate dependency measure for all links into a target. A single link
        may consist of information that multiple past variables in a source
        have about the target. The measure can be transfer entropy or mutual
        information and is estimated as the joint information all selected past
        variables from a single source have about the target.

        The conditioning defines which variables are included in the
        conditioning set when estimating a dependency measure. This can be set
        to

        - 'full' to include all selected variables (for multivariate TE this
          includes the target's past variables and past variables from all
          other inferred sources, for multivariate MI this includes past
          variables from all other inferred sources) from all other inferred
          sources and the target's past,
        - 'target' to include variables from the target's past alone (for
          bivariate TE estimation),
        - 'none' for no conditioning (for bivariate MI estimation).

        For transfer entropy, the information transfer is calculated
        conditional on the target's past. For multivariate TE or MI, the
        information (transfer) is calculated conditionally on selected
        variables from further sources in the network.

        Measures can be estimated either for 'all' sources (determined from the
        selected source variables) or for individual sources. A list of
        estimated values for each link (source-target combination) is returned.

        Args:
            data : Data instance
                raw data for analysis
            current_value : tuple
                index of the current value used for estimation, (idx process,
                idx sample)
            source_vars : np array of tuples
                array of past source variables, where one tuple describes a
                single variable as (idx process, idx sample)
            target_vars : np array of tuples [optional]
                array of past target variables
            sources : list of ints | 'all' [optional]
                return estimates for selected sources or all sources (default)
            conditioning : str [optional]
                set conditioning set, 'full' for all selected variables
                (target's and sources' past), 'target' for variables from the
                target's past only, 'none' for no conditioning

        Returns:
            numpy array
                estimate of dependency measure for each link
                
        Raises:
            ex.AlgorithmExhaustedError
                Raised from estimate() when calculation cannot be made
        """
        # Get realisations of target variables and the current value, constant
        # over sources. Permute current value realisations to generate
        # surrogates if requested.
        target_realisations = data.get_realisations(
            current_value, target_vars)[0]
        current_value_realisations = data.get_realisations(
            current_value, [current_value])[0]

        # Check requested sources.
        if sources == 'all':
            sources = np.unique([s[0] for s in source_vars])
        else:
            if type(sources) is int:  # handle integer inputs
                sources = [sources]
            sources = np.array(sources)
            if any(sources > (data.n_processes - 1)):
                raise RuntimeError('At least one source ({0}) is not in no. '
                                   'nodes in the data ({1}).'.format(
                                       sources, data.n_processes))

        # Allocate memory: either a multidimensional array if local values are
        # required, or a 1D-array for averaged values for each link.
        if self.settings['local_values']:
            # Collect local values in a [sources x samples x replications]
            # matrix.
            links = np.zeros((
                len(sources),
                data.n_realisations_samples(current_value),
                data.n_replications))
        else:
            links = np.zeros(len(sources))

        # Loop over individual sources.
        for (i, s) in enumerate(sources):

            # Separate source variables in variables belonging to the current
            # link and variables belonging to the conditioning set. Get
            # realisations for the current link's selected source variables.
            link_vars = [i for i in source_vars if i[0] == s]
            conditional_vars = [i for i in source_vars if i[0] != s]
            source_realisations, replication_ind = data.get_realisations(
                current_value, link_vars)

            # Determine which type of conditioning is requested.
            if conditioning == 'full':
                if target_realisations is None:
                    # Use sources' pasts only, returns None if conditional vars
                    # is empty.
                    conditional_realisations = data.get_realisations(
                            current_value, conditional_vars)[0]
                else:
                    # Use target's and sources' past, check if conditional vars
                    # is not empty, otherwise np.hstack crashes.
                    if conditional_vars:
                        conditional_realisations = np.hstack((
                            data.get_realisations(
                                current_value, conditional_vars)[0],
                            target_realisations))
                    else:   # use target's past only
                        conditional_realisations = target_realisations

            elif conditioning == 'target':  # use target's past only (biv. TE)
                conditional_realisations = target_realisations
            elif conditioning == 'none':  # no conditioning (bivariate MI)
                conditional_realisations = None
            else:
                raise RuntimeError('Unknown conditioning: {0}.'.format(
                    conditioning))

            if self.settings['local_values']:
                local_values = self._cmi_estimator_local.estimate(
                    current_value_realisations,
                    source_realisations,
                    conditional_realisations)
                links[i] = local_values.reshape(
                    max(replication_ind) + 1, sum(replication_ind == 0)).T
            else:
                links[i] = self._cmi_estimator.estimate(
                    current_value_realisations,
                    source_realisations,
                    conditional_realisations)

        return links
