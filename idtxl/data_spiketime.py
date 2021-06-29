"""Provide spiketime data structures for IDTxl analysis."""

import numpy as np
from . import idtxl_utils as utils
from sys import stderr, exit
from scipy.optimize import newton
import idtxl.hde_utils as utl
import os

VERBOSE = False

FAST_EMBEDDING_AVAILABLE = True
try:
    import idtxl.hde_fast_embedding as fast_emb
except:
    FAST_EMBEDDING_AVAILABLE = False
    print("""
    Error importing Cython fast embedding module. Continuing with slow Python implementation.\n
    This may take a long time.\n
    """, file=stderr, flush=True)

class Data_spiketime():
    """

        # ------------------------------------------------------------------------------------------ TODO

        INPUT NESTED NUMPY ARRAY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        data
        dim_order
        exponent_base
        seed
    """

    def __init__(self, data=None, dim_order='pr', exponent_base=2, seed=None):
        np.random.seed(seed)
        self.initial_state = np.random.get_state()
        self.exponent_base = exponent_base
        if data is not None:
            self.set_data(data, dim_order)

    @property
    def data(self):
        """Return data array."""
        return self._data

    @data.setter
    def data(self, d):
        if hasattr(self, 'data'):
            raise AttributeError('You can not assign a value to this attribute'
                                 ' directly, use the set_data method instead.')
        else:
            self._data = d

    @data.deleter
    def data(self):
        print('overwriting existing data')
        del(self._data)

    def set_data(self, data, dim_order):
        """Overwrite data in an existing Data object.

        Args:
            data : numpy array
                1- to 2-dimensional array of realisations
            dim_order : string
                order of dimensions, accepts any combination of the characters
                'p' and 'r' for processes and replications;
                must have the same length as number of dimensions in data
        """
        if len(dim_order) > 2:
            raise RuntimeError('dim_order can not have more than two '
                               'entries')
        if len(dim_order) != data.ndim:
            raise RuntimeError('Data array dimension ({0}) and length of '
                               'dim_order ({1}) are not equal.'.format(
                                           data.ndim, len(dim_order)))

        # Bring data into the order processes x replications
        data_ordered = self._reorder_data(data, dim_order)

        # get number of processes and replications
        self._set_data_size(data_ordered)
        # get number of spike times and check arrays
        self._get_nr_spiketimes(data)

        print('Adding data with properties: {0} processes and {1} '
              'replications with variable number of spike times: {2}'.format(self.n_processes,
                                                                             self.n_replications,
                                                                             self.n_spiketimes))

        try:
            delattr(self, 'data')
        except AttributeError:
            pass

        # order and rebase timestamps
        self.data = self._sort_rebase_spiketimes(data_ordered)

        self.data_type = type(self.data[0, 0])


    def _reorder_data(self, data, dim_order):
        """Reorder data dimensions to processes x samples x replications."""
        # add singletons for missing dimensions
        missing_dims = 'pr'
        for dim in dim_order:
            missing_dims = missing_dims.replace(dim, '')
        for dim in missing_dims:
            data = np.expand_dims(data, data.ndim)
            dim_order += dim

        # reorder array dims if necessary
        if dim_order[0] != 'p':
            ind_p = dim_order.index('p')
            data = data.swapaxes(0, ind_p)
            dim_order = utils.swap_chars(dim_order, 0, ind_p) # ------------------------------------------------------------- TODO check
        #if dim_order[1] != 's':
        #    data = data.swapaxes(1, dim_order.index('s'))
        return data

    def _get_nr_spiketimes(self, data):
        """Get number of spike times for each process and replication"""
        n_spiketimes = np.empty(shape=(self.n_processes, self.n_replications))
        for process in range(self.n_processes):
            for replication in range(self.n_replications):
                slicetimes = data[process, replication]
                if type(slicetimes) is not np.ndarray:
                    raise RuntimeError('Data array must contain numpy array of spike times. '
                                       'Process {0}''s replication {1} does not contain a numpy array.'.format(
                                        str(process), str(replication)))
                n_spiketimes[process, replication] = int(len(slicetimes))

        self.n_spiketimes = n_spiketimes

    def _sort_rebase_spiketimes(self, data):
        """Sort spike times ascending and start with 0 by subtracting min """
        for process in range(self.n_processes):
            for replication in range(self.n_replications):
                slicetimes = data[process, replication]
                slicetimes_sort = np.sort(slicetimes)
                slicetimes_sort = slicetimes_sort - np.min(slicetimes_sort)
                data[process, replication] = slicetimes_sort

        return data

    def _set_data_size(self, data):
        """Set the data size."""
        self.n_processes = data.shape[0]
        self.n_replications = data.shape[1]

    def get_seed(self):
        """return the initial seed of the data"""
        return self.initial_state

    def n_realisations_repl(self):
        """Number of realisations over replications."""
        return self.n_replications

    def n_spiketimes(self, process, replication):
        """Number of spiketimes."""
        return self.n_spiketimes[process, replication]

    def get_realisations_symbols(self,
                         process_list,
                         past_range_T,
                         number_of_bins_d,
                         scaling_k,
                         embedding_step_size,
                         shuffle=False,
                         output_spike_times=False):

        """
         # ------------------------------------------------------------------------------------------ TODO
        Return realisations for a list of indices.

        Return symbols, symbols lengths of spike times and their realisation
        for indices in process list and given embedding options.

        Optionally, realisations can
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

        Args:# ------------------------------------------------------------------------------------------ TODO describe inputs
            process_list:

            past_range_T:

            number_of_bins_d:

            scaling_k:

            embedding_step_size:

            shuffle: bool
                if true permute blocks of replications over trials

        Returns:
            numpy array # ------------------------------------------------------------------------------------------ TODO check output dimensions
                raw symbols with dimensions no. samples x no.replications
                (number of indices)
            numpy array
                median number of spikes per bin each realisation with dimensions (no.
                samples x no.replications (number of indices)
            numpy array
                original spiketimes (optional) (default = False)

        """

        if not hasattr(self, 'data'):
            raise AttributeError('No data has been added to this Data() '
                                         'instance.')

        first_bin_size = self.get_first_bin_size_for_embedding(past_range_T, number_of_bins_d, scaling_k)

        # Shuffle the replication order if requested. This creates surrogate
        # data by permuting replications while keeping the order of samples
        # intact.
        if shuffle:
            replications_order = np.random.permutation(self.n_replications)
        else:
            replications_order = np.arange(self.n_replications)

        # create output array
        past_symbol_array = np.empty((len(process_list), len(replications_order)), dtype=np.ndarray)
        current_symbol_array = np.empty((len(process_list), len(replications_order)), dtype=np.ndarray)
        symbol_array = np.empty((len(process_list), len(replications_order)), dtype=np.ndarray)
        symbol_array_lengths = np.empty((len(process_list), len(replications_order)), dtype=int)

        if output_spike_times:
            spike_times_array = np.empty((len(process_list), len(replications_order)), dtype=np.ndarray)

        # Retrieve data.
        i = 0
        for idx in process_list:
            r = 0
            for replication in replications_order:
                try:
                    spike_times = self.data[idx, replication]
                    if FAST_EMBEDDING_AVAILABLE:
                        int_symbols, int_past_symbols, int_current_symbols = \
                            fast_emb.get_symbol_array(spike_times,
                                                      past_range_T,
                                                      number_of_bins_d,
                                                      scaling_k,
                                                      embedding_step_size,
                                                      first_bin_size,
                                                      self.exponent_base)
                    else:
                        raw_symbols = self.get_raw_symbols(spike_times,
                                                           number_of_bins_d,
                                                           scaling_k,
                                                           first_bin_size,
                                                           embedding_step_size)

                        median_number_of_spikes_per_bin = self.get_median_number_of_spikes_per_bin(raw_symbols)

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
                            int_past_symbols[j] = int(self.symbol_array_to_binary(past_symbol))
                            int_current_symbols[j] = int(current_symbol)
                            j += 1

                    symbol_array[i, r] = int_symbols
                    past_symbol_array[i, r] = int_past_symbols
                    current_symbol_array[i, r] = int_current_symbols
                    symbol_array_lengths[i, r] = len(int_symbols)

                    if output_spike_times:
                        spike_times_array[i, r] = spike_times

                except IndexError:
                    raise IndexError('You tried to access process {0} in a '
                                     'data set with {1} processes.'.format(idx, self.n_processes))

                r += 1
            i += 1

        if output_spike_times:
            return symbol_array, past_symbol_array, current_symbol_array, symbol_array_lengths, spike_times_array
        else:
            return symbol_array, past_symbol_array, current_symbol_array, symbol_array_lengths

    def get_first_bin_size_for_embedding(self, past_range_T, number_of_bins_d, scaling_k):
        """
        Get size of first bin for the embedding, based on the parameters
        T, d and k.
        """

        return newton(lambda first_bin_size: self.get_past_range(number_of_bins_d,
                                                                 first_bin_size,
                                                                 scaling_k) - past_range_T,
                      0.005, tol=1e-03, maxiter=100)

    @staticmethod
    def get_past_range(number_of_bins_d, first_bin_size, scaling_k):
        """
        Get the past range T of the embedding, based on the parameters d, tau_1 and k.
        """

        return np.sum([first_bin_size * 10 ** ((number_of_bins_d - i) * scaling_k)
                       for i in range(1, number_of_bins_d + 1)])

    def get_raw_symbols(self,
                        spike_times,
                        number_of_bins_d,
                        scaling_k,
                        first_bin_size,
                        embedding_step_size):
        """
        Get the raw symbols (in which the number of spikes per bin are counted,
        ie not necessarily binary quantity), as obtained by applying the
        embedding.
        """

        # the window is the embedding plus the response,
        # ie the embedding and one additional bin of size embedding_step_size
        window_delimiters = self.get_window_delimiters(number_of_bins_d,
                                                       scaling_k,
                                                       first_bin_size,
                                                       embedding_step_size)
        window_length = window_delimiters[-1]
        num_spike_times = len(spike_times)
        last_spike_time = spike_times[-1]

        num_symbols = int((last_spike_time - window_length) / embedding_step_size)

        raw_symbols = []

        time = 0
        spike_index_lo = 0

        for symbol_num in range(num_symbols):
            while spike_index_lo < num_spike_times and spike_times[spike_index_lo] < time:
                spike_index_lo += 1
            spike_index_hi = spike_index_lo
            while (spike_index_hi < num_spike_times and
                   spike_times[spike_index_hi] < time + window_length):
                spike_index_hi += 1

            spikes_in_window = np.zeros(number_of_bins_d + 1)

            embedding_bin_index = 0
            for spike_index in range(spike_index_lo, spike_index_hi):
                while spike_times[spike_index] > time + window_delimiters[embedding_bin_index]:
                    embedding_bin_index += 1
                spikes_in_window[embedding_bin_index] += 1

            raw_symbols += [spikes_in_window]

            time += embedding_step_size

        return raw_symbols

    @staticmethod
    def get_window_delimiters(number_of_bins_d, scaling_k, first_bin_size, embedding_step_size):
        """
        Get delimiters of the window, used to describe the embedding. The
        window includes both the past embedding and the response.

        The delimiters are times, relative to the first bin, that separate
        two consequent bins.
        """

        bin_sizes = [first_bin_size * 10 ** ((number_of_bins_d - i) * scaling_k)
                     for i in range(1, number_of_bins_d + 1)]
        window_delimiters = [sum([bin_sizes[j] for j in range(i)])
                             for i in range(1, number_of_bins_d + 1)]
        window_delimiters.append(window_delimiters[number_of_bins_d - 1] + embedding_step_size)
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

        return sum([self.exponent_base ** (len(symbol_array) - i - 1) * symbol_array[i]
                    for i in range(0, len(symbol_array))])

    def load_Rudelt_data(self):

        """
                                                        # ---------------------------------------------------------------------TODO

        """

        currentpath = os.path.dirname(__file__)
        currentpath = os.path.split(currentpath)[0]
        datafile = os.path.join(currentpath, 'test/data/spike_times.dat')
        spiketimes = np.loadtxt(datafile, dtype=float)

        spiketimedata = np.empty(shape=(1, 1), dtype=np.ndarray)
        spiketimedata[0, 0] = spiketimes

        np.random.seed(None)
        self.initial_state = np.random.get_state()
        self.exponent_base = 2

        dim_order = 'pr'
        self.set_data(spiketimedata, dim_order)


    # ----------------------------------------------------------------------------- TODO spiketime stats  - add check for data

    def _spike_times_stats(self,
                              spike_times,
                              embedding_step_size):
        """
        Save some statistics about the spike times.
        """

        self.recording_lengths = [spt[-1] - spt[0] for spt in spike_times]
        self.recording_length = sum(self.recording_lengths)
        self.recording_length_sd = np.std(self.recording_lengths)

        firing_rates = [utl.get_binned_firing_rate(spt, embedding_step_size)
                        for spt in spike_times]

        self.firing_rate = np.average(firing_rates, weights=self.recording_lengths)
        self.firing_rate_sd = np.sqrt(np.average((firing_rates - self.firing_rate) ** 2,
                                                             weights=self.recording_lengths))

        self.H_spiking = utl.get_shannon_entropy([self.firing_rate * embedding_step_size,
                                                  1 - self.firing_rate * embedding_step_size])

    def get_recording_length(self):
        """get recording length of spike times"""
        return self.recording_length

    def get_spiketime_firingrate(self):
        """get firing rate of spike times"""
        return self.firing_rate

    def get_H_spiking(self):
        """get entropy of spike times"""
        return self.H_spiking


