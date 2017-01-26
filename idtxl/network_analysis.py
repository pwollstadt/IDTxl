"""Parent class for network inference and network comparison.

Created on Mon Mar  7 18:13:27 2016

@author: patricia
"""
import copy as cp
import itertools as it
import numpy as np
from . import idtxl_utils as utils


class Network_analysis():
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
        indices = np.zeros(len(self.selected_vars_target)).astype(int)
        i = 0
        for idx in self.selected_vars_target:
            indices[i] = self.selected_vars_full.index(idx)
            i += 1
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
        i = 0
        for idx in self.selected_vars_sources:
            indices[i] = self.selected_vars_full.index(idx)
            i += 1
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

    def _separate_realisations(self, idx_full, idx_single):
        """Separate a single indexes' realisations from a set of realisations.

        Return the realisations of a single index and the realisations of the
        remaining set of indexes. The function takes realisations from the
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
        # Get indidces of the remaining variables.
        idx_remaining = cp.copy(idx_full)
        idx_remaining.pop(idx_remaining.index(idx_single))

        # Find the indices of the columns with the realisations of the
        # requested variables (the single one to be removed and the remaining
        # variables).
        array_col_single = self.selected_vars_full.index(idx_single)
        array_col_remain = np.zeros(len(idx_remaining)).astype(int)
        i = 0
        for idx in idx_remaining:
            array_col_remain[i] = self.selected_vars_full.index(idx)
            i += 1

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
        candidate_set = []
        for idx in it.product(processes, samples):
            candidate_set.append(idx)
        return candidate_set

    def _append_selected_vars_idx(self, idx):
        """Append indices of conditionals to existing list."""
        if self.selected_vars_full is None:
            self.selected_vars_full = idx
        else:
            for i in idx:
                self.selected_vars_full.append(i)
        # separate indexes into source and target indixes
        for i in idx:
            if i[0] == self.target:
                self.selected_vars_target.append(i)
            else:
                self.selected_vars_sources.append(i)

    def _append_selected_vars(self, idx, realisations):
        """Append indices and realisation of selected variables.

        Args:
            idx : list of tuples
                indeces of selected variables, where each entry is a tuple
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

    def _clean_up(self):
        """Remove temporary data (realisations) at the end of the analysis."""
        self._current_value_realisations = None
        self._selected_vars_sources_realisations = None
        self._selected_vars_target_realisations = None
        self._current_value_realisations = None
        self._min_stats_surr_table = None
