# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 18:13:27 2016

@author: patricia
"""
import numpy as np


class Data():
    """Hold data for information dynamics estimation.

    Data are realisations for processes over time and replications (where a
    replication is a repetition of the process in time or space). Data are
    storend in a 3-dimensional array, where axes represent processes, samples
    over time, and replications.

    Attributes:
        data: 3-dimensional array that holds realizations
        current_value: index of the current value
        source_set: realisations of all source processes
        n_replications: number of replications
        n_samples: number of samples
    """
    def __init__(self, data=None, dim_order="psr"):  # TODO check dimorder in Python
        """Check and assign input to attributes."""

        if data is not None:
            self.data = self._check_dim_order(data, dim_order)
            self._set_data_size()
            self.current_value = None
            self.current_value_realisations = None

    def _check_dim_order(self, data, dim_order):
        """Reshape data array to processes x samples x replications."""
        if dim_order[0] != 'p':
            data = data.swapaxes(0, dim_order.index('p'))
        if dim_order[1] != 's':
            data = data.swapaxes(1, dim_order.index('s'))
        return data

    def _set_data_size(self):
        """Set the data size."""
        self.n_processes = self.data.shape[0]
        self.n_samples = self.data.shape[1]
        self.n_replications = self.data.shape[2]

    def _get_data(self, idx_list, analysis_setup, shuffle=False):
        """Return realisations for a list of indices.

        Return realisations for indices in list. Optionally, realisations can
        be shuffled to create surrogate data for statistical testing. For
        shuffling, data blocks are permuted over replications:

        orig:
            rep.:   1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 ...
            sample: 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 ...

        shuffled:
            rep.:   3 3 3 3 1 1 1 1 4 4 4 4 6 6 6 6 2 2 2 2 5 5 5 5 ...
            sample: 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 ...

        Args:
            idx_list: list of tuples, representing data indices
            shuffle: boolean flag, permute blocks of replications over trials

        Returns:
            realisations: numpy array with dimensions replications x number
                of indices
        """
        n_realisations_time = self.n_samples - analysis_setup.current_value[1]
        n_realisations_replications = self.n_replications
        realisations = np.empty((n_realisations_time *
                                 n_realisations_replications,
                                 len(idx_list)))

        if shuffle:
            replications_order = np.random.permutation(self.n_replications)
        else:
            replications_order = np.arange(self.n_replications)

        i = 0
        for idx in idx_list:  # TODO this should work for single trials!
            r = 0
            for sample in range(n_realisations_time):
                for replication in replications_order:
                    realisations[r, i] = self.data[idx[0], idx[1] + sample,  # TODO change to lags
                                                   replication]
                    r += 1
        i += 1

        return realisations

    def get_realisations(self, analysis_setup, idx):
        """Return realisations over samples and replications.

        Return realisations of random variables represented by a list of
        indices. An index is expected to have the form (process index, sample
        index).

        Args:
            idx: list of indices

        Returns:
            realisations: numpy array with dimensions replications x number
                of indices

        Raises:
            TypeError if idx_realisations is not a list
        """
        if type(idx) is not list:
            e = TypeError('idx_realisations must be a list of tuples.')
            raise(e)
        return self._get_data(idx, analysis_setup, shuffle=False)

    def add_realisations(self, idx_realisations, realisations):
        """Add realisations of (a set of) RV to existing realisations.

        Args:
            idx_realisations: list od tuples, where each tuple represents one
                sample as (process index, sample index)

        Returns:
            realisations: one-dimensional numpy array of realisations.
        """
        new_realisations = self.get_realisations(idx_realisations)
        if realisations is None:
            return realisations
        else:
            return np.vstack((realisations, new_realisations))

    def generate_surrogates(self, idx_candidates, analysis_setup):
        """Return realisations over samples and permuted replications.

        Return realisations for a list of indices, where realisations are
        shuffled over replications to create surrogate data. An index is
        expected to have the form (process index, sample index). Realisations
        are shuffled block-wise by permuting the order of replications:

        orig:
            rep.:   1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 ...
            sample: 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 ...

        shuffled:
            rep.:   3 3 3 3 1 1 1 1 4 4 4 4 6 6 6 6 2 2 2 2 5 5 5 5 ...
            sample: 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 ...

        Returns:
            shuffled realisations: numpy array with dimensions replications x
                number of indices

        Raises:
            TypeError if idx_realisations is not a list
        """
        if type(idx_candidates) is not list:
            e = TypeError('idx_realisations needs to be a list of tuples.')
            raise(e)
        return self._get_data(idx_candidates, analysis_setup, shuffle=True)

    def generate_mute_data(self, n_samples=1000, n_replications=10):
        """Generate example data for a 5-process network.

        Generate example data and overwrite the instance's current data. The
        network is used as an example the paper on the MuTE toolbox (Montalto,
        PLOS ONE, 2014, eq. 14). The network has the following (non-linear)
        couplings:

        1 -> 2
        1 -> 3
        1 -> 4 (non-linear)
        4 -> 5
        5 -> 4

        Args:
            n_samples: number of samples simulated for each process and
                replication
            n_replications: number of replications
        """

        n_processes = 5
        n_samples = n_samples
        n_replications = n_replications

        x = np.zeros((n_processes, n_samples + 3,
                      n_replications))
        x[:, 0:3, :] = np.random.normal(size=(n_processes, 3,
                                              n_replications))
        term_1 = 0.95 * np.sqrt(2)
        term_2 = 0.25 * np.sqrt(2)
        term_3 = -0.25 * np.sqrt(2)
        for r in range(n_replications):
            for n in range(3, n_samples + 3):
                x[0, n, r] = (term_1 * x[0, n - 1, r] -
                              0.9025 * x[0, n - 2, r] + np.random.normal())
                x[1, n, r] = 0.5 * x[0, n - 2, r] ** 2 + np.random.normal()
                x[2, n, r] = -0.4 * x[0, n - 3, r] + np.random.normal()
                x[3, n, r] = (-0.5 * x[0, n - 2, r] ** 2 +
                              term_2 * x[3, n - 1, r] +
                              term_2 * x[4, n - 1, r] +
                              np.random.normal())
                x[4, n, r] = (term_3 * x[3, n - 1, r] +
                              term_2 * x[4, n - 1, r] +
                              np.random.normal())

        self.data = x[:, 3:, :]
        self._set_data_size()


if __name__ == '__main__':
    d_mute = Data()  # initialise an empty data object
    d_mute.generate_mute_data()  # simulate data from the MuTE paper

    dat = np.arange(10000).reshape((2, 1000, 5))  # random data with correct
    d = Data(dat)                                 # order od dimensions

    dat = np.arange(10000).reshape((5, 1000, 2))  # random data with incorrect
    d = Data(dat, 'rsp')                          # order of dimensions
