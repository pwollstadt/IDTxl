import numpy as np
from scipy.optimize import newton
from collections import Counter
from sys import stderr, exit

FAST_EMBEDDING_AVAILABLE = True
try:
    #import pathlib
    #import sys
    #idtxl_path = pathlib.Path(__file__).parent.absolute()
    #sys.path.append(str(idtxl_path))
    #print(sys.path)

    #path = pathlib.Path('.')
    #full_path = path.absolute()
    #my_path = full_path.as_posix()
    # or
    #my_path = path.resolve()

    import idtxl.hde_fast_embedding as fast_emb
except:
    FAST_EMBEDDING_AVAILABLE = False
    print("""
    Error importing Cython fast embedding module. Continuing with slow Python implementation.\n
    This may take a long time.\n
    """, file=stderr, flush=True)

def get_set_of_scalings(past_range_T,
                        number_of_bins_d,
                        number_of_scalings,
                        min_first_bin_size,
                        min_step_for_scaling):
    """
    Get scaling exponents such that the uniform embedding as well as
    the embedding for which the first bin has a length of
    min_first_bin_size (in seconds), as well as linearly spaced 
    scaling factors in between, such that in total
    number_of_scalings scalings are obtained.
    """

    min_scaling = 0
    if past_range_T / number_of_bins_d <= min_first_bin_size or number_of_bins_d == 1:
        max_scaling = 0
    else:
        # for the initial guess assume the largest bin dominates, so k is approx. log(T) / d
            
        max_scaling = newton(lambda scaling: get_past_range(number_of_bins_d,
                                                            min_first_bin_size,
                                                            scaling)
                             - past_range_T,
                             np.log10(past_range_T
                                      / min_first_bin_size) / (number_of_bins_d - 1),
                             tol = 1e-04, maxiter = 500)

    while np.linspace(min_scaling, max_scaling,
                      number_of_scalings, retstep = True)[1] < min_step_for_scaling:
        number_of_scalings -= 1
        
    return np.linspace(min_scaling, max_scaling, number_of_scalings)


def get_embeddings(embedding_past_range_set,
                   embedding_number_of_bins_set,
                   embedding_scaling_exponent_set):
    """
    Get all combinations of parameters T, d, k, based on the
    sets of selected parameters.
    """

    embeddings = []
    for past_range_T in embedding_past_range_set:
        for number_of_bins_d in embedding_number_of_bins_set:
            if not isinstance(number_of_bins_d, int) or number_of_bins_d < 1:
                print("Error: numer of bins {} is not a positive integer. Skipping.".format(number_of_bins_d),
                      file=stderr, flush=True)
                continue
                    
            if type(embedding_scaling_exponent_set) == dict:
                scaling_set_given_T_and_d = get_set_of_scalings(past_range_T,
                                                                number_of_bins_d,
                                                                **embedding_scaling_exponent_set)
            else:
                scaling_set_given_T_and_d = embedding_scaling_exponent_set
                    
            for scaling_k in scaling_set_given_T_and_d:
                embeddings += [(past_range_T, number_of_bins_d, scaling_k)]

    return embeddings

def get_fist_bin_size_for_embedding(embedding):
    """
    Get size of first bin for the embedding, based on the parameters
    T, d and k.
    """

    past_range_T, number_of_bins_d, scaling_k = embedding
    return newton(lambda first_bin_size: get_past_range(number_of_bins_d,
                                                        first_bin_size,
                                                        scaling_k) - past_range_T,
                  0.005, tol = 1e-03, maxiter = 100)


def get_past_range(number_of_bins_d, first_bin_size, scaling_k):
    """
    Get the past range T of the embedding, based on the parameters d, tau_1 and k.
    """

    return np.sum([first_bin_size * 10**((number_of_bins_d - i) * scaling_k)
                   for i in range(1, number_of_bins_d + 1)])

def get_window_delimiters(number_of_bins_d, scaling_k, first_bin_size, embedding_step_size):
    """
    Get delimiters of the window, used to describe the embedding. The
    window includes both the past embedding and the response.

    The delimiters are times, relative to the first bin, that separate 
    two consequent bins.
    """

    bin_sizes = [first_bin_size * 10**((number_of_bins_d - i) * scaling_k)
                 for i in range(1, number_of_bins_d + 1)]
    window_delimiters = [sum([bin_sizes[j] for j in range(i)])
                         for i in range(1, number_of_bins_d + 1)]
    window_delimiters.append(window_delimiters[number_of_bins_d - 1] + embedding_step_size)
    return window_delimiters

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


def symbol_binary_to_array(symbol_binary, number_of_bins_d):
    """
    Given a binary representation of a symbol (cf symbol_array_to_binary),
    convert it back into its array-representation.
    """

    # assert 2 ** number_of_bins_d > symbol_binary
    
    spikes_in_window = np.zeros(number_of_bins_d)
    for i in range(0, number_of_bins_d):
        b = 2 ** (number_of_bins_d - 1 - i)
        if b <= symbol_binary:
            spikes_in_window[i] = 1
            symbol_binary -= b
    return spikes_in_window

def symbol_array_to_binary(spikes_in_window, number_of_bins_d):
    """
    Given an array of 1s and 0s, representing spikes and the absence
    thereof, read the array as a binary number to obtain a 
    (base 10) integer.
    """

    # assert len(spikes_in_window) == number_of_bins_d

    # TODO check if it makes sense to use len(spikes_in_window)
    # directly, to avoid mismatch as well as confusion
    # as number_of_bins_d here can also be number_of_bins
    # as in get_median_number_of_spikes_per_bin, ie
    # including the response
    
    return sum([2 ** (number_of_bins_d - i - 1) * spikes_in_window[i]
                for i in range(0, number_of_bins_d)])

def get_raw_symbols(spike_times,
                    embedding,
                    first_bin_size,
                    embedding_step_size):
    """
    Get the raw symbols (in which the number of spikes per bin are counted,
    ie not necessarily binary quantity), as obtained by applying the
    embedding.
    """
    
    past_range_T, number_of_bins_d, scaling_k = embedding

    # the window is the embedding plus the response,
    # ie the embedding and one additional bin of size embedding_step_size
    window_delimiters = get_window_delimiters(number_of_bins_d,
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
        while(spike_index_lo < num_spike_times and spike_times[spike_index_lo] < time):
            spike_index_lo += 1
        spike_index_hi = spike_index_lo
        while(spike_index_hi < num_spike_times and
              spike_times[spike_index_hi] < time + window_length):
            spike_index_hi += 1

        spikes_in_window = np.zeros(number_of_bins_d + 1)


        embedding_bin_index = 0
        for spike_index in range(spike_index_lo, spike_index_hi):
            while(spike_times[spike_index] > time + window_delimiters[embedding_bin_index]):
                embedding_bin_index += 1
            spikes_in_window[embedding_bin_index] += 1

        raw_symbols += [spikes_in_window]


        time += embedding_step_size

    return raw_symbols

def get_symbol_counts(spike_times, embedding, embedding_step_size):
    """
    Apply embedding to the spike times to obtain the symbol counts.
    """

    if FAST_EMBEDDING_AVAILABLE:
        return Counter(fast_emb.get_symbol_counts(spike_times, embedding, embedding_step_size))
    
    past_range_T, number_of_bins_d, scaling_k = embedding
    first_bin_size = get_fist_bin_size_for_embedding(embedding)

    raw_symbols = get_raw_symbols(spike_times,
                                  embedding,
                                  first_bin_size,
                                  embedding_step_size)

    median_number_of_spikes_per_bin = get_median_number_of_spikes_per_bin(raw_symbols)

    symbol_counts = Counter()
    
    for raw_symbol in raw_symbols:
        symbol_array = [int(raw_symbol[i] > median_number_of_spikes_per_bin[i])
                        for i in range(number_of_bins_d + 1)]

        symbol = symbol_array_to_binary(symbol_array, number_of_bins_d + 1)

        symbol_counts[symbol] += 1

    return symbol_counts
