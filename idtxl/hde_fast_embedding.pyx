# cython: profile=True

import numpy as np
import idtxl.hde_fast_embedding_utils as emb
from sys import stderr, exit

cimport cython
cimport numpy as np
DTYPE = np.uint64
ctypedef np.uint64_t DTYPE_t

def get_median_number_of_spikes_per_bin(raw_symbols):
    return np.median(raw_symbols, axis=0)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray[DTYPE_t, ndim=2] get_raw_symbols(np.ndarray[np.double_t, ndim= 1] spike_times,
                                                 past_range_T,
                                                 number_of_bins_d,
                                                 scaling_k,
                                                 first_bin_size,
                                                 embedding_step_size):

    # the window is the embedding plus the response,
    # ie the embedding and one additional bin of size embedding_step_size
    cdef np.ndarray[np.double_t, ndim= 1] \
        window_delimiters = np.array(emb.get_window_delimiters(number_of_bins_d,
                                                               scaling_k,
                                                               first_bin_size,
                                                               embedding_step_size))
    cdef double window_length = window_delimiters[len(window_delimiters) - 1]
    cdef long num_spike_times = len(spike_times)
    cdef double last_spike_time = spike_times[num_spike_times - 1]

    cdef long num_symbols = int((last_spike_time - window_length) / embedding_step_size)

    cdef np.ndarray[DTYPE_t, ndim=2] raw_symbols = np.zeros(num_symbols * (number_of_bins_d+1),
                                                            dtype=DTYPE).reshape(num_symbols,
                                                                                 number_of_bins_d+1)

    cdef DTYPE_t[:,:] raw_symbols_view = raw_symbols

    cdef long symbol_num = 0
    cdef double time = 0
    
    cdef long spike_index_lo = 0
    cdef long spike_index_hi, spike_index
    
    cdef int embedding_bin_index
    cdef np.ndarray[DTYPE_t, ndim=1] spikes_in_window = np.zeros(number_of_bins_d+1, dtype=DTYPE)
    
    for symbol_num in range(num_symbols):
        while(spike_index_lo < num_spike_times and spike_times[spike_index_lo] < time):
            spike_index_lo += 1
        spike_index_hi = spike_index_lo
        while(spike_index_hi < num_spike_times and
              spike_times[spike_index_hi] < time + window_length):
            spike_index_hi += 1

        for embedding_bin_index in range(number_of_bins_d + 1):
            spikes_in_window[embedding_bin_index] = 0
            
        embedding_bin_index = 0        
        for spike_index in range(spike_index_lo, spike_index_hi):
            while(spike_times[spike_index] > time + window_delimiters[embedding_bin_index]):
                embedding_bin_index += 1
            spikes_in_window[embedding_bin_index] += 1

        for embedding_bin_index in range(number_of_bins_d + 1):
            raw_symbols_view[symbol_num][embedding_bin_index] = spikes_in_window[embedding_bin_index]

        time += embedding_step_size
        
    return raw_symbols

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray[DTYPE_t, ndim=1] get_symbols(DTYPE_t[:,:] raw_symbols, exponent_base, mode=None):
    cdef long num_symbols
    cdef int symbol_length
    num_symbols, symbol_length = np.shape(raw_symbols) # symbol_length: number_of_bins_d + 1
    cdef np.ndarray[np.float64_t, ndim=1] median_number_of_spikes_per_bin
    if mode == 'median':
        median_number_of_spikes_per_bin = get_median_number_of_spikes_per_bin(raw_symbols)
    else:
        median_number_of_spikes_per_bin = np.zeros(symbol_length)

    cdef np.ndarray[DTYPE_t, ndim=1] raw_symbol
    cdef np.ndarray[DTYPE_t, ndim=1] symbols = np.zeros(num_symbols, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] past_symbols = np.zeros(num_symbols, dtype=DTYPE)
    cdef int symbol, i, past_symbol
    cdef int symbol_num = 0
    
    for symbol_num in range(num_symbols):
        symbol = 0
        for i in range(symbol_length):
            if raw_symbols[symbol_num][i] > median_number_of_spikes_per_bin[i]:
                symbol += exponent_base ** (symbol_length - i - 1)
        symbols[symbol_num] = symbol

    return symbols


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray[DTYPE_t, ndim=1] get_past_symbols(DTYPE_t[:,:] raw_symbols, exponent_base, mode=None):
    cdef long num_symbols
    cdef int symbol_length
    num_symbols, symbol_length = np.shape(raw_symbols) # symbol_length: number_of_bins_d + 1
    cdef np.ndarray[np.float64_t, ndim=1] median_number_of_spikes_per_bin
    if mode == 'median':
        median_number_of_spikes_per_bin = get_median_number_of_spikes_per_bin(raw_symbols)
    else:
        median_number_of_spikes_per_bin = np.zeros(symbol_length)

    cdef np.ndarray[DTYPE_t, ndim=1] raw_symbol
    cdef np.ndarray[DTYPE_t, ndim=1] symbols = np.zeros(num_symbols, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] past_symbols = np.zeros(num_symbols, dtype=DTYPE)
    cdef int symbol, i, past_symbol
    cdef int symbol_num = 0

    for symbol_num in range(num_symbols):
        symbol = 0
        for i in range(symbol_length-1):
            if raw_symbols[symbol_num][i] > median_number_of_spikes_per_bin[i]:
                symbol += exponent_base ** (symbol_length -1 - i - 1)
        symbols[symbol_num] = symbol

    return symbols


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray[DTYPE_t, ndim=1] get_current_symbols(DTYPE_t[:,:] raw_symbols,
                                             mode=None):
    cdef long num_symbols
    cdef int symbol_length
    num_symbols, symbol_length = np.shape(raw_symbols) # symbol_length: number_of_bins_d + 1
    cdef np.ndarray[np.float64_t, ndim=1] median_number_of_spikes_per_bin
    if mode == 'median':
        median_number_of_spikes_per_bin = get_median_number_of_spikes_per_bin(raw_symbols)
    else:
        median_number_of_spikes_per_bin = np.zeros(symbol_length)

    cdef np.ndarray[DTYPE_t, ndim=1] raw_symbol
    cdef np.ndarray[DTYPE_t, ndim=1] symbols = np.zeros(num_symbols, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] past_symbols = np.zeros(num_symbols, dtype=DTYPE)
    cdef int symbol, i, past_symbol
    cdef int symbol_num = 0

    for symbol_num in range(num_symbols):
        # symbol = raw_symbols[symbol_num][symbol_length]
        # if raw_symbols[symbol_num][symbol_length] > median_number_of_spikes_per_bin[symbol_length]:
        #    symbol += 1
            # symbol += exponent_base ** (symbol_length -1 - i - 1)
        symbols[symbol_num] = raw_symbols[symbol_num][symbol_length]

    return symbols


def count_symbols(DTYPE_t[:] symbols):
    symbols = np.sort(symbols)
    cdef np.ndarray[DTYPE_t, ndim=1] unq_symbols = np.unique(symbols)

    cdef dict symbol_counts = {}

    cdef DTYPE_t symbol
    cdef long num_symbols = len(symbols)
    cdef long symbols_index = 0
    cdef DTYPE_t symbol_num
    cdef int symbol_count
    
    cdef long num_unq_symbols = len(unq_symbols)
    for symbol_num in range(num_unq_symbols):
        symbol = unq_symbols[symbol_num]
        
        symbol_count = 0
        while(symbols_index < num_symbols and symbols[symbols_index] == symbol):
            symbol_count += 1
            symbols_index += 1
        
        symbol_counts[symbol] = symbol_count
    return symbol_counts


def get_symbol_array(spike_times, past_range_T, number_of_bins_d, scaling_k,
                     embedding_step_size, first_bin_size, exponent_base):

    cdef np.ndarray[DTYPE_t, ndim=2] raw_symbols = get_raw_symbols(spike_times, past_range_T,
                                                                   number_of_bins_d,
                                                                   scaling_k,
                                                                   first_bin_size,
                                                                   embedding_step_size)

    cdef np.ndarray[DTYPE_t, ndim=1] symbols = get_symbols(raw_symbols, exponent_base, mode='median')

    cdef np.ndarray[DTYPE_t, ndim=1] past_symbols = get_past_symbols(raw_symbols, exponent_base, mode='median')

    cdef np.ndarray[DTYPE_t, ndim=1] current_symbols = get_current_symbols(raw_symbols, mode='median')

    return symbols, past_symbols, current_symbols


def get_bootstrap_arrays(symbol_array, past_symbol_array, current_symbol_array, symbol_block_length):

    cdef int num_sym = 0
    cdef np.ndarray[DTYPE_t, ndim=1] bs_symbol_array = np.empty(len(symbol_array), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] bs_past_symbol_array = np.empty(len(symbol_array), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] bs_current_symbol_array = np.empty(len(symbol_array), dtype=DTYPE)
    cdef int onset, r


    onset = 0
    for i in range(int(np.floor(len(symbol_array) / symbol_block_length))):
        r = np.random.randint(0, len(symbol_array) - (symbol_block_length - 1))
        bs_symbol_array[onset:onset+symbol_block_length] = symbol_array[r:r + symbol_block_length]
        bs_past_symbol_array[onset:onset+symbol_block_length] = past_symbol_array[r:r + symbol_block_length]
        bs_current_symbol_array[onset:onset+symbol_block_length] = current_symbol_array[r:r + symbol_block_length]

        onset += symbol_block_length

    res = int(len(symbol_array) % symbol_block_length)
    r = np.random.randint(0, len(symbol_array) - (res - 1))
    bs_symbol_array[onset:onset + res] = symbol_array[r:r + res]
    bs_past_symbol_array[onset:onset + res] = past_symbol_array[r:r + res]
    bs_current_symbol_array[onset:onset + res] = current_symbol_array[r:r + res]


    return bs_symbol_array, bs_past_symbol_array, bs_current_symbol_array

