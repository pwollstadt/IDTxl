"""Provide data structures for IDTxl analysis."""
import numpy as np


from . import idtxl_utils as utils
from idtxl.lazy_array import LazyArray

VERBOSE = False


class Data():
    """Store data for information dynamics estimation.

    Data takes a 1- to 3-dimensional array representing realisations of random
    variables in dimensions: processes, samples (over time), and replications.
    If necessary, data reshapes provided realisations to fit the format
    expected by IDTxl, which is a 3-dimensional array with axes representing
    (process index, sample index, replication index). Indicate the actual order
    of dimensions in the provided array in a three-character string, e.g. 'spr'
    for an array with realisations over (1) samples in time, (2) processes, (3)
    replications.

    Example:

        >>> data_mute = Data()              # initialise empty data object
        >>> data_mute.generate_mute_data()  # simulate data from MuTE paper
        >>>
        >>> # Create data objects with data of various sizes
        >>> d = np.arange(10000).reshape((2, 1000, 5))  # 2 procs.,
        >>> data_1 = Data(d, dim_order='psr')           # 1000 samples, 5 repl.
        >>>
        >>> d = np.arange(3000).reshape((3, 1000))  # 3 procs.,
        >>> data_2 = Data(d, dim_order='ps')        # 1000 samples
        >>>
        >>> # Overwrite data in existing object with random data
        >>> d = np.arange(5000)
        >>> data_2.set_data(data_new, 's')

    Note:
        Realisations are stored as attribute 'data'. This can only be set via
        the 'set_data()' method.

    Args:
        data : numpy array [optional]
            1/2/3-dimensional array with raw data
        dim_order : string [optional]
            order of dimensions, accepts any combination of the characters
            'p', 's', and 'r' for processes, samples, and replications; must
            have the same length as the data dimensionality, e.g., 'ps' for a
            two-dimensional array of data from several processes over time
            (default='psr')
        normalise : bool [optional]
            if True, data gets normalised per process (default=True)
        seed : int [optional]
            can be set to a fixed integer to get repetitive results on the
            same data with multiple runs of analyses. Otherwise a random
            seed is set as default.

    Attributes:
        data : numpy array
            realisations, can only be set via 'set_data' method
        n_processes : int
            number of processes
        n_replications : int
            number of replications
        n_samples : int
            number of samples in time
        normalise : bool
            if true, all data gets z-standardised per process
        initial_state : array
            initial state of the seed for shuffled permutations
    """

    def __init__(self, data=None, dim_order="psr", normalise=True, seed=None):
        np.random.seed(seed)
        self.initial_state = np.random.get_state()
        self.seed = seed
        self._random_bit_generator = np.random.Philox(seed)
        self.normalise = normalise
        self.n_processes = 0
        self.n_samples = 0
        self.n_replications = 0
        if data is not None:
            self.set_data(data, dim_order)

    def get_rgb_key(self):
        return tuple(self._random_bit_generator.state['state']['key'])
    
    def get_rgb_count(self):
        return tuple(self._random_bit_generator.state['state']['counter'])

    def n_realisations(self, current_value=None):
        """Number of realisations over samples and replications.

        Args:
            current_value : tuple [optional]
                reference point for calculation of number of realisations
                (e.g. when using an embedding of length k, we count
                realisations from the k+1th sample because we loose the first k
                samples to the embedding); if no current_value is provided, the
                number of all samples is used
        """
        return self.n_realisations_samples(current_value) * self.n_realisations_repl()

    def n_realisations_samples(self, current_value=None):
        """Number of realisations over samples.

        Args:
            current_value : tuple [optional]
                reference point for calculation of number of realisations
                (e.g. when using an embedding of length k, the current value is
                at sample k + 1; we thus count realisations from the k + 1st
                sample because we loose the first k samples to the embedding)
        """
        if current_value is None:
            return self.n_samples
        if current_value[1] >= self.n_samples:
            raise RuntimeError(
                f"The sample index of the current value ({current_value}) is larger than the"
                f" number of samples in the data set ({self.n_samples})."
            )
        return self.n_samples - current_value[1]

    def n_realisations_repl(self):
        """Number of realisations over replications."""
        return self.n_replications

    @property
    def data(self):
        """Return data array."""
        return self._data

    @data.setter
    def data(self, d):
        if hasattr(self, "data"):
            raise AttributeError(
                "You can not assign a value to this attribute"
                " directly, use the set_data method instead."
            )
        self._data = d

    @data.deleter
    def data(self):
        print("overwriting existing data")
        del self._data

    def set_data(self, data, dim_order):
        """Overwrite data in an existing Data object.

        Args:
            data : numpy array
                1- to 3-dimensional array of realisations
            dim_order : string
                order of dimensions, accepts any combination of the characters
                'p', 's', and 'r' for processes, samples, and replications;
                must have the same length as number of dimensions in data
        """
        if len(dim_order) > 3:
            raise RuntimeError("dim_order can not have more than three entries")
        if len(dim_order) != data.ndim:
            raise RuntimeError(
                f"Data array dimension ({data.ndim}) and length of "
                f"dim_order ({len(dim_order)}) are not equal."
            )

        # Bring data into the order processes x samples x replications and set
        # set data.
        data_ordered = self._reorder_data(data, dim_order)
        self._set_data_size(data_ordered)
        print(
            f"Adding data with properties: {self.n_processes} processes, {self.n_samples} "
            f"samples, {self.n_replications} replications"
        )
        try:
            delattr(self, "data")
        except AttributeError:
            pass
        if self.normalise:
            self.data = self._normalise_data(data_ordered)
        else:
            self.data = data_ordered
        self.data_type = type(self.data[0, 0, 0])

    def _normalise_data(self, d):
        """Z-standardise data separately for each process."""
        d_standardised = np.empty(d.shape)
        for process in range(self.n_processes):
            s = utils.standardise(
                d[process, :, :].reshape(1, self.n_realisations()), dimension=1
            )
            d_standardised[process, :, :] = s.reshape(
                self.n_samples, self.n_replications
            )
        return d_standardised

    def _reorder_data(self, data, dim_order):
        """Reorder data dimensions to processes x samples x replications."""
        # add singletons for missing dimensions
        missing_dims = "psr"
        for dim in dim_order:
            missing_dims = missing_dims.replace(dim, "")
        for dim in missing_dims:
            data = np.expand_dims(data, data.ndim)
            dim_order += dim

        # reorder array dims if necessary
        if dim_order[0] != "p":
            ind_p = dim_order.index("p")
            data = data.swapaxes(0, ind_p)
            dim_order = utils.swap_chars(dim_order, 0, ind_p)
        if dim_order[1] != "s":
            data = data.swapaxes(1, dim_order.index("s"))
        return data

    def _set_data_size(self, data):
        """Set the data size."""
        self.n_processes = data.shape[0]
        self.n_samples = data.shape[1]
        self.n_replications = data.shape[2]

    def get_seed(self):
        """return the initial seed of the data"""
        return self.initial_state

    def get_state(self):
        """return the current state of the random seed"""
        return np.random.get_state()

    def get_realisations(self, current_value, idx_list, shuffle=False, return_replication_idx=False):
        """Return realisations for a list of indices.

        Return realisations for indices in list. Optionally, realisations can
        be shuffled to create surrogate data for statistical testing. For
        shuffling, data blocks are permuted over replications while their
        temporal order stays intact within replications:

        Original data:
            +--------------+---------+---------+---------+---------+---------+-----+
            | repl. ind.   | 1 1 1 1 | 2 2 2 2 | 3 3 3 3 | 4 4 4 4 | 5 5 5 5 | ... |
            +--------------+---------+---------+---------+---------+---------+-----+
            | sample index | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | ... |
            +--------------+---------+---------+---------+---------+---------+-----+

        Shuffled data:
            +--------------+---------+---------+---------+---------+---------+-----+
            | repl. ind.   | 3 3 3 3 | 1 1 1 1 | 4 4 4 4 | 2 2 2 2 | 5 5 5 5 | ... |
            +--------------+---------+---------+---------+---------+---------+-----+
            | sample index | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | ... |
            +--------------+---------+---------+---------+---------+---------+-----+

        Args:
            idx_list: list of tuples
                variable indices
            current_value : tuple
                index of the current value in current analysis, has to have the
                form (idx process, idx sample); if current_value == idx, all
                samples for a process are returned
            shuffle: bool
                if true permute blocks of replications over trials
            return_replication_idx: bool
                additionally return the replication index of each sample

        Returns:
            numpy array
                realisations with dimensions (no. samples * no.replications) x
                number of indices
            numpy array
                replication index for each realisation with dimensions (no.
                samples * no.replications) x number of indices. Only returned
                if return_replication_idx is true.
        """
        if not hasattr(self, "data"):
            raise AttributeError("No data has been added to this Data() instance.")
        # Return None if index list is empty.
        if not idx_list:
            return (None, None) if return_replication_idx else None
        # Check if requested indices are smaller than the current_value.
        if not all(np.array([x[1] for x in idx_list]) <= current_value[1]):
            print('Index list: {0}\ncurrent value: {1}'.format(idx_list,
                                                               current_value))
            raise RuntimeError('All indices for which data is retrieved must '
                               ' be smaller than the current value.')
                
        assert shuffle == False, 'Shuffling is not implemented yet'

        n_real_time = self.n_realisations_samples(current_value)
        n_real_repl = self.n_realisations_repl()
        
        la = LazyArray(self.data)

        la = la.shifted(coords=tuple(idx[0] for idx in idx_list), shifts=tuple(idx[1] for idx in idx_list), length=n_real_time)
        la = la.swapaxes(0, 2)
        la = la.reshape(n_real_repl * n_real_time, len(idx_list))

        return la

    def _get_data_slice(self, process, offset_samples=0, shuffle=False):
        """Return data slice for a single process.

        Return data slice for process. Optionally, an offset can be provided
        such that data are returned from sample 'offset_samples' onwards.
        Also, data
        can be shuffled over replications to create surrogate data for
        statistical testing. For shuffling, data blocks are permuted over
        replications while their temporal order stays intact within
        replications:

        Original data:
            +---------------+---------+---------+---------+---------+---------+-----+
            | repl. index:  | 1 1 1 1 | 2 2 2 2 | 3 3 3 3 | 4 4 4 4 | 5 5 5 5 | ... |
            +---------------+---------+---------+---------+---------+---------+-----+
            | sample index: | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | ... |
            +---------------+---------+---------+---------+---------+---------+-----+

        Shuffled data:
            +---------------+---------+---------+---------+---------+---------+-----+
            | repl. index:  | 3 3 3 3 | 1 1 1 1 | 4 4 4 4 | 2 2 2 2 | 5 5 5 5 | ... |
            +---------------+---------+---------+---------+---------+---------+-----+
            | sample index: | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | ... |
            +---------------+---------+---------+---------+---------+---------+-----+

        If the current_value is provided, data are returned from an offset
        specified by the index wrt the current_value.

        Args:
            process: int
                process index
            offset_samples : int
                offset in samples
            shuffle: bool
                if true permute blocks of data over trials

        Returns:
            numpy array
                data slice with dimensions no. samples - offset x no.
                replications
            numpy array
                list of replications indices
        """
        # Check if requested indices are smaller than the current_value.
        if offset_samples > self.n_samples:
            print(
                "Offset {0} must be smaller than number of samples in the "
                " data ({1})".format(offset_samples, self.n_samples)
            )
            raise RuntimeError("Offset must be smaller than no. samples.")

        # Shuffle the replication order if requested. This creates surrogate
        # data by permuting replications while keeping the order of samples
        # intact.
        if shuffle:
            replication_index = np.random.permutation(self.n_replications)
        else:
            replication_index = np.arange(self.n_replications)

        try:
            data_slice = self.data[process, offset_samples:, replication_index]
        except IndexError as e:
            raise IndexError(
                "You tried to access process {process} with an offset of {offset_samples} in a data set "
                "of {self.n_processes} processes and {self.n_samples} samples."
            ) from e
        assert not np.isnan(
            data_slice
        ).any(), "There are nans in the retrieved data slice."
        return data_slice.T, replication_index

    def slice_permute_replications(self, process):
        """Return data slice with permuted replications (time stays intact).

        Create surrogate data by permuting realisations over replications while
        keeping the temporal structure (order of samples) intact. Return
        realisations for all indices in the list, where an index is expected to
        have the form (process index, sample index). Realisations are permuted
        block-wise by permuting the order of replications
        """
        return self._get_data_slice(process, shuffle=True)

    def slice_permute_samples(self, process, perm_settings):
        """Return slice of data with permuted samples (repl. stays intact).

        Create surrogate data by permuting data in a slice over samples (time)
        while keeping the order of replications intact. Return slice for the
        entry specified by 'process'. Realisations are permuted according to
        the settings specified in perm_settings:

        Original data:
            +--------------+-----------------+-----------------+-----------------+-----+
            | repl. ind.   | 1 1 1 1 1 1 1 1 | 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+
            | sample index | 1 2 3 4 5 6 7 8 | 1 2 3 4 5 6 7 8 | 1 2 3 4 5 6 7 8 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+

        Circular shift by 2, 6, and 4 samples:
            +--------------+-----------------+-----------------+-----------------+-----+
            | repl. ind.   | 1 1 1 1 1 1 1 1 | 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+
            | sample index | 7 8 1 2 3 4 5 6 | 3 4 5 6 7 8 1 2 | 5 6 7 8 1 2 3 4 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+

        Permute blocks of 3 samples:
            +--------------+-----------------+-----------------+-----------------+-----+
            | repl. ind.   | 1 1 1 1 1 1 1 1 | 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+
            | sample index | 4 5 6 7 8 1 2 3 | 1 2 3 7 8 4 5 6 | 7 8 4 5 6 1 2 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+

        Permute data locally within a range of 4 samples:
            +--------------+-----------------+-----------------+-----------------+-----+
            | repl. ind.   | 1 1 1 1 1 1 1 1 | 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+
            | sample index | 1 2 4 3 8 5 6 7 | 4 1 2 3 5 7 8 6 | 3 1 2 4 8 5 6 7 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+

        Random permutation:
            +--------------+-----------------+-----------------+-----------------+-----+
            | repl. ind.   | 1 1 1 1 1 1 1 1 | 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+
            | sample index | 4 2 5 7 1 3 2 6 | 7 5 3 4 2 1 8 5 | 1 2 4 3 6 8 7 5 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+

        Permuting samples is the fall-back option for surrogate creation if the
        number of replications is too small to allow for a sufficient number of
        permutations for the generation of surrogate data.

        Args:
            process : int
                process for which to return data slice
            perm_settings : dict
                settings specifying the allowed permutations:

                - perm_type : str
                  permutation type, can be

                    - 'circular': shifts time series by a random
                      number of samples
                    - 'block': swaps blocks of samples,
                    - 'local': swaps samples within a given range, or
                    - 'random': swaps samples at random,

                - additional settings depending on the perm_type (n is the
                  number of samples):

                    - if perm_type == 'circular':

                      'max_shift' : int
                        the maximum number of samples for shifting
                        (default=n/2)
                    - if perm_type == 'block':

                      'block_size' : int
                        no. samples per block (default=n/10)
                      'perm_range' : int
                          range in which blocks can be swapped (default=max)

                    - if perm_type == 'local':

                      'perm_range' : int
                          range in samples over which realisations can be
                          permuted (default=n/10)

        Returns:
            numpy array
                data slice with data permuted over samples with dimensions
                samples x number of replications
            numpy array
                index of permuted samples

        Note:
            This permutation scheme is the fall-back option if the number of
            replications is too small to allow a sufficient number of
            permutations for the generation of surrogate data.
        """
        data_slice = self._get_data_slice(process, shuffle=True)[0]
        data_slice_perm = np.empty(data_slice.shape).astype(self.data_type)
        perm = self._get_permutation_samples(data_slice.shape[0], perm_settings)
        for r in range(self.n_replications):
            data_slice_perm[:, r] = data_slice[perm, r]
        return data_slice_perm, perm

    def permute_replications(self, current_value, idx_list, return_replication_idx=False):
        """Return realisations with permuted replications (time stays intact).

        Create surrogate data by permuting realisations over replications while
        keeping the temporal structure (order of samples) intact. Return
        realisations for all indices in the list, where an index is expected to
        have the form (process index, sample index). Realisations are permuted
        block-wise by permuting the order of replications:

        Original data:
            +--------------+---------+---------+---------+---------+---------+-----+
            | repl. ind.   | 1 1 1 1 | 2 2 2 2 | 3 3 3 3 | 4 4 4 4 | 5 5 5 5 | ... |
            +--------------+---------+---------+---------+---------+---------+-----+
            | sample index | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | ... |
            +--------------+---------+---------+---------+---------+---------+-----+

        Permuted data:
            +--------------+---------+---------+---------+---------+---------+-----+
            | repl. ind.   | 3 3 3 3 | 1 1 1 1 | 4 4 4 4 | 2 2 2 2 | 5 5 5 5 | ... |
            +--------------+---------+---------+---------+---------+---------+-----+
            | sample index | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | 1 2 3 4 | ... |
            +--------------+---------+---------+---------+---------+---------+-----+


        Args:
            current_value : tuple
                index of the current_value in the data
            idx_list : list of tuples
                indices of variables
            return_replication_idx: bool
                additionally return the replication index of each sample


        Returns:
            numpy array
                permuted realisations with dimensions replications x number of
                indices
            numpy array
                replication index for each realisation. Only returned
                if return_replication_idx is true.

        Raises:
            TypeError if idx_realisations is not a list
        """
        if type(idx_list) is not list:
            raise TypeError('idx needs to be a list of tuples.')
        
        if return_replication_idx:
            raise NotImplementedError('return_replication_idx is not implemented yet')
                
        realisations = self.get_realisations(current_value,
                                                                idx_list)
        n_samples = self.n_realisations_samples(current_value)

        realisations = realisations.reshape(self.n_replications, n_samples, len(idx_list)) # Add replication axis to facilitate permutation.
        realisations = realisations.shuffled(axis=0, philox_key=self.get_rgb_key(), philox_counter=self.get_rgb_count())
        realisations = realisations.reshape(self.n_replications * n_samples, len(idx_list)) # Flatten out the replications axis again.

        # Advance the random bit generator to avoid reusing the same random numbers for the next permutation.
        self._random_bit_generator.advance(2**64)

        return realisations

    def permute_samples(self, current_value, idx_list, perm_settings):
        """Return realisations with permuted samples (repl. stays intact).

        Create surrogate data by permuting realisations over samples (time)
        while keeping the order of replications intact. Surrogates can be
        created for multiple variables in parallel, where variables are
        provided as a list of indices. An index is expected to have the form
        (process index, sample index).

        Permuting samples in time is the fall-back option for surrogate data
        creation. The default method for surrogate data creation is the
        permutation of replications, while keeping the order of samples in time
        intact. If the number of replications is too small to allow for a
        sufficient number of permutations for the generation of surrogate data,
        permutation of samples in time is chosen instead.

        Different permutation strategies can be chosen to permute realisations
        in time. Note that if data consists of multiple replications, within
        each replication, samples are shuffled following the same permutation
        pattern:

        Original data:
            +--------------+-----------------+-----------------+-----------------+-----+
            | repl. ind.   | 1 1 1 1 1 1 1 1 | 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+
            | sample index | 1 2 3 4 5 6 7 8 | 1 2 3 4 5 6 7 8 | 1 2 3 4 5 6 7 8 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+

        Circular shift by a random number of samples, e.g. 4 samples:
            +--------------+-----------------+-----------------+-----------------+-----+
            | repl. ind.   | 1 1 1 1 1 1 1 1 | 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+
            | sample index | 5 6 7 8 1 2 3 4 | 5 6 7 8 1 2 3 4 | 5 6 7 8 1 2 3 4 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+

        Permute blocks of 3 samples:
            +--------------+-----------------+-----------------+-----------------+-----+
            | repl. ind.   | 1 1 1 1 1 1 1 1 | 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+
            | sample index | 4 5 6 7 8 1 2 3 | 4 5 6 7 8 1 2 3 | 4 5 6 7 8 1 2 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+

        Permute data locally within a range of 4 samples:
            +--------------+-----------------+-----------------+-----------------+-----+
            | repl. ind.   | 1 1 1 1 1 1 1 1 | 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+
            | sample index | 1 2 4 3 8 5 6 7 | 1 2 4 3 8 5 6 7 | 1 2 4 3 8 5 6 7 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+

        Random permutation:
            +--------------+-----------------+-----------------+-----------------+-----+
            | repl. ind.   | 1 1 1 1 1 1 1 1 | 2 2 2 2 2 2 2 2 | 3 3 3 3 3 3 3 3 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+
            | sample index | 4 2 5 7 1 3 2 6 | 4 2 5 7 1 3 2 6 | 4 2 5 7 1 3 2 6 | ... |
            +--------------+-----------------+-----------------+-----------------+-----+

        Args:
            current_value : tuple
                index of the current_value in the data
            idx_list : list of tuples
                indices of variables
            perm_settings : dict
                settings specifying the allowed permutations:

                - perm_type : str
                  permutation type, can be

                    - 'random': swaps samples at random,
                    - 'circular': shifts time series by a random number of
                      samples
                    - 'block': swaps blocks of samples,
                    - 'local': swaps samples within a given range, or

                - additional settings depending on the perm_type (n is the
                  number of samples):

                    - if perm_type == 'circular':

                      'max_shift' : int
                        the maximum number of samples for shifting
                        (e.g., number of samples / 2)

                    - if perm_type == 'block':

                      'block_size' : int
                        no. samples per block (e.g., number of samples / 10)
                      'perm_range' : int
                        range in which blocks can be swapped (e.g., number
                        of samples / block_size)

                    - if perm_type == 'local':

                      'perm_range' : int
                        range in samples over which realisations can be
                        permuted (e.g., number of samples / 10)

        Returns:
            numpy array
                permuted realisations with dimensions replications x number of
                indices

        Raises:
            TypeError if idx_realisations is not a list

        Note:
            This permutation scheme is the fall-back option if surrogate data
            can not be created by shuffling replications because the number of
            replications is too small to generate the requested number of
            permutations.
        """
        realisations = self.get_realisations(current_value, idx_list)
        n_samples = self.n_realisations_samples(current_value)

        # Extract random bit number generator state
        philox_key = self.get_rgb_key()
        philox_counter = self.get_rgb_count()

        if perm_settings['perm_type'] == 'random':
            realisations = realisations.reshape(self.n_replications, n_samples, len(idx_list)) # Add replication axis to facilitate permutation.
            realisations = realisations.shuffled(axis=1, philox_key=philox_key, philox_counter=philox_counter)
            realisations = realisations.reshape(self.n_replications * n_samples, len(idx_list)) # Flatten out the replications axis again.
        elif perm_settings['perm_type'] == 'circular':
            max_shift = perm_settings['max_shift']
            shift = np.random.Generator(self._random_bit_generator).integers(1, max_shift + 1)
            realisations = realisations.rolled(shift, axis=0)
        elif perm_settings['perm_type'] == 'block':
            block_size = perm_settings['block_size']
            perm_range = perm_settings['perm_range']
            realisations = realisations.block_shuffled(block_size, perm_range, philox_key=philox_key, philox_counter=philox_counter)
        elif perm_settings['perm_type'] == 'local':
            perm_range = perm_settings['perm_range']
            realisations = realisations.local_shuffled(perm_range, philox_key=philox_key, philox_counter=philox_counter)

        # Advance the random bit generator to avoid reusing the same random numbers for the next permutation.
        self._random_bit_generator.advance(2**64)        

        return realisations

    def generate_mute_data(self, n_samples=1000, n_replications=10):
        """Generate example data for a 5-process network.

        Generate example data and overwrite the instance's current data. The
        network is used as an example the paper on the MuTE toolbox (Montalto,
        PLOS ONE, 2014, eq. 14) and was orginally proposed by Baccala &
        Sameshima (2001). The network consists of five auto-regressive (AR)
        processes with model orders 2 and the following (non-linear) couplings:

        0 -> 1, u = 2 (non-linear)
        0 -> 2, u = 3
        0 -> 3, u = 2 (non-linear)
        3 -> 4, u = 1
        4 -> 3, u = 1

        References:

        - Montalto, A., Faes, L., & Marinazzo, D. (2014) MuTE: A MATLAB toolbox
          to compare established and novel estimators of the multivariate
          transfer entropy. PLoS ONE 9(10): e109462.
          https://doi.org/10.1371/journal.pone.0109462
        - Baccala, L.A. & Sameshima, K. (2001). Partial directed coherence: a
          new concept in neural structure determination. Biol Cybern 84:
          463–474. https://doi.org/10.1007/PL00007990

        Args:
            n_samples : int
                number of samples simulated for each process and replication
            n_replications : int
                number of replications
        """
        n_processes = 5

        x = np.zeros((n_processes, n_samples + 3, n_replications))
        x[:, 0:3, :] = np.random.normal(size=(n_processes, 3, n_replications))
        term_1 = 0.95 * np.sqrt(2)
        term_2 = 0.25 * np.sqrt(2)
        term_3 = -0.25 * np.sqrt(2)
        for r in range(n_replications):
            for n in range(3, n_samples + 3):
                x[0, n, r] = (
                    term_1 * x[0, n - 1, r]
                    - 0.9025 * x[0, n - 2, r]
                    + np.random.normal()
                )
                x[1, n, r] = 0.5 * x[0, n - 2, r] ** 2 + np.random.normal()
                x[2, n, r] = -0.4 * x[0, n - 3, r] + np.random.normal()
                x[3, n, r] = (
                    -0.5 * x[0, n - 2, r] ** 2
                    + term_2 * x[3, n - 1, r]
                    + term_2 * x[4, n - 1, r]
                    + np.random.normal()
                )
                x[4, n, r] = (
                    term_3 * x[3, n - 1, r]
                    + term_2 * x[4, n - 1, r]
                    + np.random.normal()
                )
        self.set_data(x[:, 3:, :], "psr")

    def generate_var_data(
        self,
        n_samples=1000,
        n_replications=10,
        coefficient_matrices=np.array([[[0.5, 0], [0.4, 0.5]]]),
        noise_std=0.1,
    ):
        """Generate discrete-time VAR (vector auto-regressive) time series.

        Generate data and overwrite the instance's current data.

        Args:
            n_samples : int [optional]
                number of samples simulated for each process and replication
            n_replications : int [optional]
                number of replications
            coefficient_matrices : numpy array [optional]
                coefficient matrices: numpy array with dimensions
                (VAR order, number of processes, number of processes). Each
                square coefficient matrix corresponds to a lag, starting from
                lag=1. The total number of provided matrices implicitly
                determines the order of the VAR process.
                (default = np.array([[[0.5, 0], [0.4, 0.5]]]))
            noise_std : float [optional]
                standard deviation of uncorrelated Gaussian noise
                (default = 0.1)
        """
        order = np.shape(coefficient_matrices)[0]
        n_processes = np.shape(coefficient_matrices)[1]
        samples_transient = n_processes * 10

        # Check stability of the VAR process, which is a sufficient condition
        # for stationarity.
        var_reduced_form = np.zeros((n_processes * order, n_processes * order))
        var_reduced_form[0:n_processes, :] = np.reshape(
            np.transpose(coefficient_matrices, (1, 0, 2)),
            [n_processes, n_processes * order],
        )
        var_reduced_form[n_processes:, 0 : n_processes * (order - 1)] = np.eye(
            n_processes * (order - 1)
        )
        # Condition for stability: the absolute values of all the eigenvalues
        # of the reduced-form coefficient matrix are smaller than 1. A stable
        # VAR process is also stationary.
        is_stable = max(np.abs(np.linalg.eigvals(var_reduced_form))) < 1
        if not is_stable:
            raise RuntimeError("VAR process is not stable and may be nonstationary.")

        # Initialise time series matrix. The 3 dimensions represent
        # (processes, samples, replications). Only the last n_samples will be
        # kept, in order to discard transient effects.
        x = np.zeros(
            (n_processes, order + samples_transient + n_samples, n_replications)
        )

        # Generate (different) initial conditions for each replication:
        # Uniformly sample from the [0,1] interval and tile as many times as
        # the VAR process order along the second dimension.
        x[:, 0:order, :] = np.tile(
            np.random.rand(n_processes, 1, n_replications), (1, order, 1)
        )

        # Compute time series
        for i_repl in range(0, n_replications):
            for i_sample in range(order, order + samples_transient + n_samples):
                for i_delay in range(1, order + 1):
                    x[:, i_sample, i_repl] += np.dot(
                        coefficient_matrices[i_delay - 1, :, :],
                        x[:, i_sample - i_delay, i_repl],
                    )
                # Add uncorrelated Gaussian noise vector
                x[:, i_sample, i_repl] += np.random.normal(
                    0, noise_std, x[:, i_sample, i_repl].shape  # mean
                )

        # Discard transient effects (only take end of time series)
        self.set_data(x[:, -(n_samples + 1) : -1, :], "psr")

    def generate_logistic_maps_data(
        self,
        n_samples=1000,
        n_replications=10,
        coefficient_matrices=np.array([[[0.5, 0], [0.4, 0.5]]]),
        noise_std=0.1,
    ):
        """Generate discrete-time coupled-logistic-maps time series.

        Generate data and overwrite the instance's current data.

        The implemented logistic map function is f(x) = 4 * x * (1 - x).

        Args:
            n_samples : int [optional]
                number of samples simulated for each process and replication
            n_replications : int [optional]
                number of replications
            coefficient_matrices : numpy array [optional]
                coefficient matrices: numpy array with dimensions
                (order, number of processes, number of processes). Each
                square coefficient matrix corresponds to a lag, starting from
                lag=1. The total number of provided matrices implicitly
                determines the order of the stochastic process.
                (default = np.array([[[0.5, 0], [0.4, 0.5]]]))
            noise_std : float [optional]
                standard deviation of uncorrelated Gaussian noise
                (default = 0.1)
        """
        order = np.shape(coefficient_matrices)[0]
        n_processes = np.shape(coefficient_matrices)[1]
        samples_transient = n_processes * 10

        # Define activation function (logistic map)
        def f(x):
            return 4 * x * (1 - x)

        # Initialise time series matrix. The 3 dimensions represent
        # (processes, samples, replications). Only the last n_samples will be
        # kept, in order to discard transient effects.
        x = np.zeros(
            (n_processes, order + samples_transient + n_samples, n_replications)
        )

        # Generate (different) initial conditions for each replication:
        # Uniformly sample from the [0,1] interval and tile as many times as
        # the stochastic process order along the second dimension.
        x[:, 0:order, :] = np.tile(
            np.random.rand(n_processes, 1, n_replications), (1, order, 1)
        )

        # Compute time series
        for i_repl in range(0, n_replications):
            for i_sample in range(order, order + samples_transient + n_samples):
                for i_delay in range(1, order + 1):
                    x[:, i_sample, i_repl] += np.dot(
                        coefficient_matrices[i_delay - 1, :, :],
                        x[:, i_sample - i_delay, i_repl],
                    )
                # Compute activation function
                x[:, i_sample, i_repl] = f(x[:, i_sample, i_repl])
                # Add uncorrelated Gaussian noise vector
                x[:, i_sample, i_repl] += np.random.normal(
                    0, noise_std, x[:, i_sample, i_repl].shape  # mean
                )
                # ensure values are in the [0, 1] range
                x[:, i_sample, i_repl] = x[:, i_sample, i_repl] % 1

        # Discard transient effects (only take end of time series)
        self.set_data(x[:, -(n_samples + 1) : -1, :], "psr")
