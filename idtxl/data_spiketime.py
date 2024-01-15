"""Provide spiketime data structures for IDTxl analysis."""

import numpy as np
from sys import stderr
from scipy.optimize import newton
import idtxl.hde_utils as utl
import os

VERBOSE = False

FAST_EMBEDDING_AVAILABLE = True
try:
    import idtxl.hde_fast_embedding as fast_emb
except:
    FAST_EMBEDDING_AVAILABLE = False
    print(
        """
    Error importing Cython fast embedding module for HDE estimator.\n
    When running the HDE estimator, the slow Python implementation for optimizing the HDE embedding will be used,\n
    this may take a long time. Other estimators are not affected.\n
    """,
        file=stderr,
        flush=True,
    )


class Data_spiketime:
    """Store data for Rudelt estimators and optimization.

    Data takes a 1-dimensional numpy array representing the processes.
    The spike times for each process needs to be added as nested numpy arrays.

    Example:

        >>> data_Rudelt = Data()              # initialise empty data object
        >>> data_Rudelt.load_Rudelt_data()    # load example data
        >>>
        >>> # Create a numpy array for all processes
        >>> spiketimedata = np.empty(shape=(2), dtype=np.ndarray) # 2 processes
        >>> # add nested array of spike times, e.g. for first process
        >>> spiketimedata[0] = [0.0032, 0.0043, ........]
        >>> # insert the spiketimedata array into the data object:
        >>> data = Data_spiketime(spiketimedata)

    Note:
        Realisations are stored as attribute 'data'. This can only be set via
        the 'set_data()' method.

    Args:
        data : numpy array [optional]
            1-dimensional array with nested spike time data
        dim_order : string [optional]
            order of dimensions, accepts any combination of the characters
            'p' and 'r' for processes and replications; must
            have the same length as the data dimensionality, e.g., 'pr' for a
            two-dimensional array of data from several processes over time
            (default='pr')
        normalise : bool [optional]
            if True, data gets normalised (rebase spike times to zero) per process (default=True)
        exponent_base: int
            exponent base used for creating the symbols from binary data
        seed : int [optional]
            can be set to a fixed integer to get repetitive results on the
            same data with multiple runs of analyses. Otherwise a random
            seed is set as default.

    Attributes:
        data : numpy array
            realisations, can only be set via 'set_data' method
        n_processes : int
            number of processes
        n_spiketimes(process, replication) : int
            number of spike times of given process and replication

    implemented in idtxl by Michael Lindner, Göttingen 2021
    """

    def __init__(self, data=None, exponent_base=2, seed=None):
        np.random.seed(seed)
        self.initial_state = np.random.get_state()
        self.exponent_base = exponent_base
        if data is not None:
            self.set_data(data)

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
        else:
            self._data = d

    @data.deleter
    def data(self):
        print("overwriting existing data")
        del self._data

    def set_data(self, data):
        """Overwrite data in an existing Data object.

        Args:
            data : numpy array
                1-dimensional array of realisations
        """

        # get number of processes and replications
        self._set_data_size(data)
        # get number of spike times and check arrays
        self.get_nr_spiketimes(data)

        print(
            f"Adding data with properties: {self.n_processes} processes "
            f"with variable number of spike times: {self.n_spiketimes}"
        )

        try:
            delattr(self, "data")
        except AttributeError:
            pass

        # order and rebase timestamps
        self.data = self._sort_rebase_spiketimes(data)

        self.data_type = type(self.data[0])

    # def _reorder_data(self, data, dim_order):
    #    """Reorder data dimensions to processes x samples x replications."""
    #    # add singletons for missing dimensions
    #    missing_dims = 'pr'
    #    for dim in dim_order:
    #        missing_dims = missing_dims.replace(dim, '')
    #    for dim in missing_dims:
    #        data = np.expand_dims(data, data.ndim)
    #        dim_order += dim

    #    # reorder array dims if necessary
    #    if dim_order[0] != 'p':
    #        ind_p = dim_order.index('p')
    #        data = data.swapaxes(0, ind_p)
    #        # dim_order = utils.swap_chars(dim_order, 0, ind_p) # ------------- TODO check

    #    return data

    def get_nr_spiketimes(self, data):
        """Get number of spike times for each process and replication"""
        n_spiketimes = np.empty(shape=(self.n_processes))
        for process in range(self.n_processes):
            slicetimes = data[process]
            if type(slicetimes) is not np.ndarray:
                raise RuntimeError(
                    "Data array must contain numpy array of spike times. "
                    f"Process {str(process)} does not contain a numpy array."
                )
            n_spiketimes[process] = int(len(slicetimes))

        self.n_spiketimes = n_spiketimes

    def _sort_rebase_spiketimes(self, data):
        """Sort spike times ascending and start with 0 by subtracting min"""
        for process in range(self.n_processes):
            slicetimes = data[process]
            slicetimes_sort = np.sort(slicetimes)
            slicetimes_sort = slicetimes_sort - np.min(slicetimes_sort)
            data[process] = slicetimes_sort
        return data

    def _set_data_size(self, data):
        """Set the data size."""
        self.n_processes = data.shape[0]

    def get_seed(self):
        """return the initial seed of the data"""
        return self.initial_state

    def n_spiketimes(self, process):
        """Number of spiketimes."""
        return self.n_spiketimes[process]

    def get_spike_times_single(self, process):
        """get spike times of one process and replication"""
        return self.data[process]

    def get_realisations_symbols(
        self,
        process_list,
        past_range_T,
        number_of_bins_d,
        scaling_k,
        embedding_step_size,
        shuffle=False,
        output_spike_times=False,
    ):
        """
        Return arrays of symbols (joint symbols), past symbols and current symbols of the spike times
        for the given embedding options for the indices in process list.
        Additionally it returns the lengths of spike times and optionally the original spike times.

        Example for exponent_base of 2
             1    0   1   1   0     -->     joint symbol: 10110 --> 22  (1·2^4 + 0·2^3 + 1·2^2 + 1·2^1 + 0·2^0 )
            |-past range T-|  t             past symbol:  1011  --> 11  (1·2^3 + 0·2^2 + 1·2^1 + 1·2^0)
                                            current symbol    0 --> 0


        Args:
            process_list : int or list of int
                list or processes that should be extracted from the data
            past_range_T : float
                The past range T (in seconds) to be used for embeddings
            number_of_bins_d : int
                The number of bins d in the embedding
            scaling_k : float
                The scaling exponent k for the bins in the embedding.
            embedding_step_size : float
                Step size ∆t (in seconds) with which the window is slid through the data
            output_spike_times : bool
                return the original spike times additionally to the symbols
                (default: False)

        Returns:
            symbol_array : numpy array
                array with lengths of no. process
                the raw joint symbols (over time) are included as nested array
            past_symbol_array : numpy array
                array with lengths of no. process
                the raw past symbols (over time) are included as nested array
            current_symbol_array : numpy array
                array with lengths of no. process
                the raw current symbols (over time) are included as nested array
            symbol_length : numpy array
                array with lengths of no. process including the length of joint symbols
            spike_times : numpy array (optional)
                array with lengths of no. process
                original spike times are included as nested array
        """

        if not hasattr(self, "data"):
            raise AttributeError("No data has been added to this Data() " "instance.")

        first_bin_size = self.get_first_bin_size_for_embedding(
            past_range_T, number_of_bins_d, scaling_k
        )

        # create output array
        if isinstance(process_list, list):
            processlen = len(process_list)
        elif isinstance(process_list, int):
            processlen = 1
            process_list = [process_list]
        else:
            raise RuntimeError("Process_list must be list or integer!")

        past_symbol_array = np.empty(processlen, dtype=np.ndarray)
        current_symbol_array = np.empty(processlen, dtype=np.ndarray)
        symbol_array = np.empty(processlen, dtype=np.ndarray)
        symbol_array_lengths = np.empty(processlen, dtype=int)

        if output_spike_times:
            spike_times_array = np.empty(processlen, dtype=np.ndarray)

        # Retrieve data.
        i = 0
        for idx in process_list:
            spike_times = self.data[idx]
            try:
                if FAST_EMBEDDING_AVAILABLE:
                    (
                        int_symbols,
                        int_past_symbols,
                        int_current_symbols,
                    ) = fast_emb.get_symbol_array(
                        spike_times,
                        past_range_T,
                        number_of_bins_d,
                        scaling_k,
                        embedding_step_size,
                        first_bin_size,
                        self.exponent_base,
                    )
                else:
                    raw_symbols = self.get_raw_symbols(
                        spike_times,
                        number_of_bins_d,
                        scaling_k,
                        first_bin_size,
                        embedding_step_size,
                    )

                    median_number_of_spikes_per_bin = (
                        self.get_median_number_of_spikes_per_bin(raw_symbols)
                    )

                    int_symbols = np.empty(shape=len(raw_symbols), dtype=int)
                    int_past_symbols = np.empty(shape=len(raw_symbols), dtype=int)
                    int_current_symbols = np.empty(shape=len(raw_symbols), dtype=int)
                    j = 0
                    for raw_symbol in raw_symbols:
                        symbol = raw_symbol > median_number_of_spikes_per_bin
                        past_symbol = symbol[0:-1]
                        current_symbol = symbol[-1]
                        symbol = symbol.astype(int)
                        past_symbol = past_symbol.astype(int)
                        current_symbol = current_symbol.astype(int)
                        int_symbols[j] = int(self.symbol_array_to_binary(symbol))
                        int_past_symbols[j] = int(
                            self.symbol_array_to_binary(past_symbol)
                        )
                        int_current_symbols[j] = int(current_symbol)
                        j += 1

                symbol_array[i] = int_symbols
                past_symbol_array[i] = int_past_symbols
                current_symbol_array[i] = int_current_symbols
                symbol_array_lengths[i] = len(int_symbols)

                if output_spike_times:
                    spike_times_array[i] = spike_times
            except IndexError:
                raise IndexError(
                    "You tried to access process {0} in a "
                    "data set with {1} processes.".format(idx, self.n_processes)
                )

            i += 1

        if output_spike_times:
            return (
                symbol_array,
                past_symbol_array,
                current_symbol_array,
                symbol_array_lengths,
                spike_times_array,
            )
        else:
            return (
                symbol_array,
                past_symbol_array,
                current_symbol_array,
                symbol_array_lengths,
            )

    def get_first_bin_size_for_embedding(
        self, past_range_T, number_of_bins_d, scaling_k
    ):
        """
        Get size of first bin for the embedding, based on the parameters
        T, d and k.
        """

        return newton(
            lambda first_bin_size: self.get_past_range(
                number_of_bins_d, first_bin_size, scaling_k
            )
            - past_range_T,
            0.005,
            tol=1e-03,
            maxiter=100,
        )

    @staticmethod
    def get_past_range(number_of_bins_d, first_bin_size, scaling_k):
        """
        Get the past range T of the embedding, based on the parameters d, tau_1 and k.
        """

        return np.sum(
            [
                first_bin_size * 10 ** ((number_of_bins_d - i) * scaling_k)
                for i in range(1, number_of_bins_d + 1)
            ]
        )

    def get_raw_symbols(
        self,
        spike_times,
        number_of_bins_d,
        scaling_k,
        first_bin_size,
        embedding_step_size,
    ):
        """
        Get the raw symbols (in which the number of spikes per bin are counted,
        ie not necessarily binary quantity), as obtained by applying the
        embedding.
        """

        # the window is the embedding plus the response,
        # ie the embedding and one additional bin of size embedding_step_size
        window_delimiters = self.get_window_delimiters(
            number_of_bins_d, scaling_k, first_bin_size, embedding_step_size
        )
        window_length = window_delimiters[-1]
        num_spike_times = len(spike_times)
        last_spike_time = spike_times[-1]

        num_symbols = int((last_spike_time - window_length) / embedding_step_size)

        raw_symbols = []

        time = 0
        spike_index_lo = 0

        for symbol_num in range(num_symbols):
            while (
                spike_index_lo < num_spike_times and spike_times[spike_index_lo] < time
            ):
                spike_index_lo += 1
            spike_index_hi = spike_index_lo
            while (
                spike_index_hi < num_spike_times
                and spike_times[spike_index_hi] < time + window_length
            ):
                spike_index_hi += 1

            spikes_in_window = np.zeros(number_of_bins_d + 1)

            embedding_bin_index = 0
            for spike_index in range(spike_index_lo, spike_index_hi):
                while (
                    spike_times[spike_index]
                    > time + window_delimiters[embedding_bin_index]
                ):
                    embedding_bin_index += 1
                spikes_in_window[embedding_bin_index] += 1

            raw_symbols += [spikes_in_window]

            time += embedding_step_size

        return raw_symbols

    @staticmethod
    def get_window_delimiters(
        number_of_bins_d, scaling_k, first_bin_size, embedding_step_size
    ):
        """
        Get delimiters of the window, used to describe the embedding. The
        window includes both the past embedding and the response.

        The delimiters are times, relative to the first bin, that separate
        two consequent bins.
        """

        bin_sizes = [
            first_bin_size * 10 ** ((number_of_bins_d - i) * scaling_k)
            for i in range(1, number_of_bins_d + 1)
        ]
        window_delimiters = [
            sum([bin_sizes[j] for j in range(i)])
            for i in range(1, number_of_bins_d + 1)
        ]
        window_delimiters.append(
            window_delimiters[number_of_bins_d - 1] + embedding_step_size
        )
        return window_delimiters

    @staticmethod
    def get_median_number_of_spikes_per_bin(raw_symbols):
        """
        Given raw symbols (in which the number of spikes per bin are counted,
        ie not necessarily binary quantity), get the median number of spikes
        for each bin, among all symbols obtained by the embedding.
        """

        # number_of_bins here is number_of_bins_d + 1,
        # as it here includes not only the bins of the embedding but also the response
        number_of_bins = len(raw_symbols[0])

        spike_counts_per_bin = [[] for i in range(number_of_bins)]

        for raw_symbol in raw_symbols:
            for i in range(number_of_bins):
                spike_counts_per_bin[i] += [raw_symbol[i]]

        return [np.median(spike_counts_per_bin[i]) for i in range(number_of_bins)]

    def symbol_array_to_binary(self, symbol_array):
        """
        Given an array of 1s and 0s, representing spikes and the absence
        thereof, read the array as a binary number to obtain a
        (base 10) integer.
        """

        return sum(
            [
                self.exponent_base ** (len(symbol_array) - i - 1) * symbol_array[i]
                for i in range(0, len(symbol_array))
            ]
        )

    def get_bootstrap_realisations_symbols(
        self,
        process_list,
        past_range_T,
        number_of_bins_d,
        scaling_k,
        embedding_step_size,
        symbol_block_length=None,
    ):
        """
        Get symbols using get_realisation_symbols of the processes and replications and then
        resample the symbols using boostrap method for each process and replication separately.

            Args:
            process_list : int or list of int
                list or processes that should be extracted from the data
            past_range_T : float
                The past range T (in seconds) to be used for embeddings
            number_of_bins_d : int
                The number of bins d in the embedding
            scaling_k : float
                The scaling exponent k for the bins in the embedding.
            embedding_step_size : float
                Step size ∆t (in seconds) with which the window is slid through the data
            symbol_block_length : int
                The number of symbols that should be drawn in each block for bootstrap resampling
                If it is set to None (recommended), the length is automatically chosen, based
                on heuristics

        Returns:
            bs_symbol_array : numpy array
                array with length no. process
                the raw joint symbols (over time) are included as nested array
            bs_past_symbol_array : numpy array
                array with length no. process
                the raw past symbols (over time) are included as nested array
            bs_current_symbol_array : numpy array
                array with length no. process
                the raw current symbols (over time) are included as nested array
        """

        # create output array
        if isinstance(process_list, list):
            nr_processes = len(process_list)
        elif isinstance(process_list, int):
            nr_processes = 1
            process_list = [process_list]
        else:
            raise RuntimeError("Process_list must be list or integer!")

        (
            symbol_array,
            past_symbol_array,
            current_symbol_array,
            sl,
        ) = self.get_realisations_symbols(
            process_list,
            past_range_T,
            number_of_bins_d,
            scaling_k,
            embedding_step_size,
            output_spike_times=False,
        )

        # create output array
        bs_symbol_array = np.empty(nr_processes, dtype=np.ndarray)
        bs_past_symbol_array = np.empty(nr_processes, dtype=np.ndarray)
        bs_current_symbol_array = np.empty(nr_processes, dtype=np.ndarray)

        # get bootstrap realisations - loop over processes
        for i in range(nr_processes):
            spike_times = self.data[process_list[i]]
            firing_rate = self.get_firingrate(process_list[i], embedding_step_size)
            if symbol_block_length is None:
                symbol_block_length = max(
                    1, int(1 / (firing_rate * embedding_step_size))
                )

            else:
                min_num_symbols = 1 + int(
                    (
                        spike_times[-1]
                        - spike_times[0]
                        - (past_range_T + embedding_step_size)
                    )
                    / embedding_step_size
                )
                if symbol_block_length >= min_num_symbols:
                    print(
                        "Warning. Block length too large given number of symbols. Skipping."
                    )
                    return []

            try:
                if FAST_EMBEDDING_AVAILABLE:
                    (
                        bs_symbol,
                        bs_past_symbol,
                        bs_current_symbol,
                    ) = fast_emb.get_bootstrap_arrays(
                        symbol_array[i],
                        past_symbol_array[i],
                        current_symbol_array[i],
                        symbol_block_length,
                    )
                else:
                    bs_symbol = np.empty(len(symbol_array[i]), dtype=int)
                    bs_past_symbol = np.empty(len(symbol_array[i]), dtype=int)
                    bs_current_symbol = np.empty(len(symbol_array[i]), dtype=int)

                    on = 0
                    for rep in range(
                        int(np.floor(len(symbol_array[i]) / symbol_block_length))
                    ):
                        randidx = np.random.randint(
                            0, len(symbol_array[i]) - (symbol_block_length - 1)
                        )
                        bs_symbol[on : on + symbol_block_length] = symbol_array[i][
                            randidx : randidx + symbol_block_length
                        ]
                        bs_past_symbol[
                            on : on + symbol_block_length
                        ] = past_symbol_array[i][
                            randidx : randidx + symbol_block_length
                        ]
                        bs_current_symbol[
                            on : on + symbol_block_length
                        ] = current_symbol_array[i][
                            randidx : randidx + symbol_block_length
                        ]

                        on += symbol_block_length

                    res = int(len(symbol_array[i]) % symbol_block_length)
                    randidx = np.random.randint(0, len(symbol_array[i]) - (res - 1))
                    bs_symbol[on : on + res] = symbol_array[i][randidx : randidx + res]
                    bs_past_symbol[on : on + res] = past_symbol_array[i][
                        randidx : randidx + res
                    ]
                    bs_current_symbol[on : on + res] = current_symbol_array[i][
                        randidx : randidx + res
                    ]

                bs_symbol_array[i] = bs_symbol
                bs_past_symbol_array[i] = bs_past_symbol
                bs_current_symbol_array[i] = bs_current_symbol

            except IndexError:
                raise IndexError(
                    "You tried to access process {0} in a "
                    "data set with {1} processes.".format(i, self.n_processes)
                )

        return bs_symbol_array, bs_past_symbol_array, bs_current_symbol_array

    def load_Rudelt_data(self):
        """
        Load the Rudelt data into the data_spiketime object for testing the estimators and optimization algorithm
            References:
                [1]: L. Rudelt, D. G. Marx, M. Wibral, V. Priesemann: Embedding
                    optimization reveals long-lasting history dependence in
                    neural spiking activity, 2021, PLOS Computational Biology, 17(6)

                [2]: https://github.com/Priesemann-Group/hdestimator

        """

        currentpath = os.path.dirname(__file__)
        currentpath = os.path.split(currentpath)[0]
        datafile = os.path.join(currentpath, "test/data/spike_times.dat")
        spiketimes = np.loadtxt(datafile, dtype=float)

        spiketimedata = np.empty(shape=(1), dtype=np.ndarray)
        spiketimedata[0] = spiketimes

        np.random.seed(None)
        self.initial_state = np.random.get_state()
        self.exponent_base = 2

        self.set_data(spiketimedata)

    def get_recording_length(self, process):
        """get recording length of spike times"""
        spike_times = self.data[process]
        return spike_times[-1] - spike_times[0]

    def get_firingrate(self, process, embedding_step_size):
        """get firing rate of spike times"""
        spike_times = self.data[process]
        recording_lengths = spike_times[-1] - spike_times[0]
        firing_rates = utl.get_binned_firing_rate(spike_times, embedding_step_size)
        return np.average(firing_rates, weights=recording_lengths)

    def get_H_spiking(self, process, embedding_step_size):
        """get entropy of spike times"""
        firing_rate = self.get_firingrate(process, embedding_step_size)
        return utl.get_shannon_entropy(
            [firing_rate * embedding_step_size, 1 - firing_rate * embedding_step_size]
        )

    def get_realisations(self, process_list):
        """
        Return arrays  original spike times for the given indices in process list.

        Args:
            process_list : int or list of int
                list or processes that should be extracted from the data

        Returns:
            spike_times : numpy array (optional)
                array with lengths of no. process
                original spike times are included as nested array
        """

        # create output array
        if isinstance(process_list, list):
            processlen = len(process_list)
        elif isinstance(process_list, int):
            processlen = 1
            process_list = [process_list]
        else:
            raise RuntimeError("Process_list must be list or integer!")

        spike_times_array = np.empty(processlen, dtype=np.ndarray)

        i = 0
        for idx in process_list:
            try:
                spike_times = self.data[idx]
                spike_times_array[i] = spike_times
            except IndexError:
                raise IndexError(
                    "You tried to access process {0} in a "
                    "data set with {1} processes.".format(idx, self.n_processes)
                )

            i += 1

        return spike_times_array
