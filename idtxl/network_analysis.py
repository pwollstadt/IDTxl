# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:13:27 2016

Greedy algorithm for multivariate network inference using transfer entropy.
For details see Lizier ??? and Faes ???.

If executed as standalone, the script applies the algorithm to example data
presented in Montalto, PLOS ONE, 2014, (eq. 14).

Usage:
    python multivariate_te.py

@author: patricia
"""
import numpy as np
import utils as utils

class Network_analyses(): # TODO which 'algorithms' do we want to provide for this? biv TE, mult TE, mult granger, biv granger, ...?
    """Provide an analysis setup for multivariate network inference.

    Hold variables that are relevant for multivariate network inference.
    The class holds
        (1) analysis parameters
        (2) data
        (3) 'analysis pattern', i.e., indices of random variables used for
            network inference (e.g. current value and conditional in transfer
            entropy estimation)
    The class provide routines to check user input and set defaults. The
    'analysis pattern' is represented by tuples or list of tuples (process
    index, sample index), where a tuple indicates where to find realisations in
    the data.

    Args:
        max_lag (int): maximum search depth when looking for conditionals
        target (int): index of the target process in the data

    Attributes:
        current_value: index of the current value
        conditional_full: full set of random variables to be conditioned on
        conditional_sources: set of conditionals coming from souce processes
        conditional_target: set of conditionals coming from the target process
    """
    def __init__(self, max_lag, target): # TODO a lot of these needs to go into the child class
        self.target = target
        self.current_target_variable = (target, max_lag)
        self.conditional_full = None  # TODO this is not consistent with the other two
        self.conditional_sources = []
        self.conditional_target = []
        self._current_value_realisations = None
        self._conditional_realisations = None

    @property
    def current_value(self):
        """Index of the current_value."""
        return self._current_value

    @current_value.setter
    def current_value(self, idx):
        if type(idx) is not tuple:
            raise TypeError(('The current value should be a tuple (index ' +
                             'process, index sample).'))
        self._current_value = idx

    @property
    def _current_value_realisations(self):
        """Realisations of the current_value."""
        if self.__current_value_realisations is None:
            print('Attribute has not been set yet.')
        return self.__current_value_realisations

    @_current_value_realisations.setter
    def _current_value_realisations(self, realisations):
        self.__current_value_realisations = realisations

    @_current_value_realisations.deleter
    def _current_value_realisations(self):
        del self.__current_value_realisations

    @property
    def conditional_full(self):
        """List of indices of the full conditional set."""
        if self._current_value_realisations is None:
            print('Attribute has not been set yet.')
        return self._conditional_full

    @conditional_full.setter
    def conditional_full(self, idx_list):
        if (idx_list is not None) and (type(idx_list) is not list or
                                       (type(idx_list[0]) is not tuple)):
            raise TypeError(('Expected a list of tuples (index process, ' +
                             'index sample).'))
        self._conditional_full = idx_list

    @property
    def conditional_target(self):
        """List of indices of target samples in the conditional set."""
        if self._current_value_realisations is None:
            print('Attribute has not been set yet.')
        return self._conditional_target

    @conditional_target.setter
    def conditional_target(self, idx_list):
        if (idx_list is not None and type(idx_list) is not list):
            raise TypeError(('Expected a list of tuples (index process, ' +
                             'index sample).'))
        self._conditional_target = idx_list

    @property
    def conditional_sources(self):
        """List of indices of source samples in the conditional set."""
        if self._current_value_realisations is None:
            print('Attribute has not been set yet.')
        return self._conditional_sources

    @conditional_sources.setter
    def conditional_sources(self, idx_list):
        if (idx_list is not None and type(idx_list) is not list):
            raise TypeError(('Expected a list of tuples (index process, ' +
                             'index sample).'))
        self._conditional_sources = idx_list

    @property
    def _conditional_realisations(self):
        """Realisations of the full conditional set."""
        return self.__conditional_realisations

    @_conditional_realisations.setter
    def _conditional_realisations(self, realisations):
        self.__conditional_realisations = realisations

    @property
    def _conditional_target_realisations(self): # TODO this may be rather slow, but it is not called often
        """Realisations of the target samples in the conditional."""
        indices = np.zeros(len(self.conditional_target)).astype(int)
        i = 0
        for idx in self.conditional_target:
            indices[i] = self.conditional_full.index(idx)
            i += 1
        self._conditional_target_realisations = (self.
                                        _conditional_realisations[:, indices])
        return self.__conditional_target_realisations

    @_conditional_target_realisations.setter
    def _conditional_target_realisations(self, realisations):
        self.__conditional_target_realisations = realisations

    @property
    def _conditional_sources_realisations(self):  # TODO these are slow, write that into the docstring for both functions
        """Realisations of the source samples in the conditional.

        Get realisations from the whole set of realisations -> this is slow!!
        """
        indices = np.zeros(len(self.conditional_sources)).astype(int)
        i = 0
        for idx in self.conditional_sources:
            indices[i] = self.conditional_full.index(idx)
            i += 1
        self._conditional_sources_realisations = (self.
                                     _conditional_realisations[:, indices])
        return self.__conditional_sources_realisations

    @_conditional_sources_realisations.setter
    def _conditional_sources_realisations(self, realisations):
        self.__conditional_sources_realisations = realisations

    def _append_conditional_idx(self, idx):
        """Append indices of conditionals to existing list."""
        if self.conditional_full is None:
            self.conditional_full = idx
        else:
            for i in idx:
                self.conditional_full.append(i)
        # separate indexes into source and target indixes
        for i in idx:
            if i[0] == self.target:
                self.conditional_target.append(i)
            else:
                self.conditional_sources.append(i)

    def _append_conditional_realisations(self, realisations):
        """Append realisations of conditionals to existing realisations.

        Returns:
            realisations: numpy array with dimensions replications x number
                of indices.
        """
        if self._conditional_realisations is None or realisations.size == 0:
            self._conditional_realisations = realisations
        else:
            self._conditional_realisations = np.hstack(
                                (self._conditional_realisations, realisations))

    def _remove_candidate(self, idx):
        """Remove a candidate and its realisations from the object."""

        self._conditional_realisations = utils.remove_column(
                                         self._conditional_realisations,
                                         self.conditional_full.index(idx))
        self.conditional_full.pop(self.conditional_full.index(idx))
        if idx[0] == self.target:
            self.conditional_target.pop(self.conditional_target.index(idx))
        else:
            self.conditional_sources.pop(self.conditional_sources.index(idx))


if __name__ == '__main__':
    max_lag = 5
    target = 0
    n = Network_analyses(max_lag, target)
