# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:13:27 2016

@author: patricia
"""
import numpy as np
import copy as cp
from . import idtxl_utils as utils

class Network_analysis(): # TODO which 'algorithms' do we want to provide for this? biv TE, mult TE, mult granger, biv granger, ...?
    """Provide an analysis setup for multivariate network inference.

    Hold variables that are relevant for multivariate network inference.
    The class holds

    (1) analysis parameters
    (2) 'analysis pattern', i.e., indices of random variables used for
    network inference (e.g. current value and conditional in transfer entropy
    estimation)
    (3) temporary data for analysis, i.e., realisations of the variables

    The class provide routines to check user input and set defaults. The
    'analysis pattern' is represented by tuples or list of tuples (process
    index, sample index), where a tuple indicates where to find realisations in
    the data.

    Args:
        max_lag : int
            maximum search depth when looking for conditionals
        target : int
            index of the target process in the data

    Attributes:
        current_value : tuple
            index of the current value
        selected_vars_full : list of tuples
            full set of random variables to be conditioned on
        selected_vars_sources : list of tuples
            set of conditionals coming from souce processes
        selected_vars_target : list of tuples
            set of conditionals coming from the target process
    """
    def __init__(self): # TODO a lot of these needs to go into the child class
        self.target = None
        self.current_value = None  # TODO rename this to current_target_variable??
        self.selected_vars_full = []
        self.selected_vars_sources = []
        self.selected_vars_target = []
        self._current_value_realisations = None
        self._selected_vars_realisations = None
        self._replication_index = None

    @property
    def current_value(self):
        """Index of the current_value."""
        return self._current_value

    @current_value.setter
    def current_value(self, idx):
        if (idx is not None) and (type(idx) is not tuple):
            raise TypeError(('The current value should be a tuple (index ' +
                             'process, index sample).'))
        self._current_value = idx

    @property
    def _current_value_realisations(self):
        """Realisations of the current_value."""
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
        """Realisations of the full conditional set."""
        return self.__selected_vars_realisations

    @_selected_vars_realisations.setter
    def _selected_vars_realisations(self, realisations):
        self.__selected_vars_realisations = realisations

    @property
    def _selected_vars_target_realisations(self):
        """Realisations of the target samples in the conditional.

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
        """Realisations of the source samples in the conditional.

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

    def _remove_candidate(self, idx):
        """Remove a single candidate and its realisations from the object."""

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
        # Get realisations for all indices from the class attribute
        # ._selected_vars_realisations. Find the respective columns.
        idx_remaining = cp.copy(idx_full)
        idx_remaining.pop(idx_remaining.index(idx_single))
        array_col_single = self.selected_vars_full.index(idx_single)
        array_col_remain = np.zeros(len(idx_remaining)).astype(int)
        i = 0
        # Find the columns with realisations of the remaining variables
        for idx in idx_remaining:
            array_col_remain[i] = self.selected_vars_full.index(idx)
            i += 1

        real_single = np.expand_dims(
                        self._selected_vars_realisations[:, array_col_single],
                        axis=1)
        if len(idx_full) == 1:
            real_remain = None  # so the JIDT estimator doesn't break
        else:
            real_remain = self._selected_vars_realisations[:, array_col_remain]
        return real_remain, real_single

    def _clean_up(self):
        """Remove temporary data at the end of the analysis."""
        self._current_value_realisations = None
        self._selected_vars_sources_realisations = None
        self._selected_vars_target_realisations = None
        self._current_value_realisations = None
        self.min_stats_surr_table = None

    def _idx_to_lag(self, idx_list):
        """Change sample indices to lags for each index in the list."""
        lag_list = cp.copy(idx_list)
        for c in idx_list:
            lag_list[idx_list.index(c)] = (c[0], self.current_value[1] - c[1])
        return lag_list

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
        self._append_selected_vars_idx(cond)
        self._append_selected_vars_realisations(
                        data.get_realisations(self.current_value, cond)[0])


if __name__ == '__main__':
    max_lag = 5
    target = 0
    n = Network_analysis()
