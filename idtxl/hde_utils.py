"""Utils for the history dependence estimators

from:
    [1]: L. Rudelt, D. G. Marx, M. Wibral, V. Priesemann: Embedding
        optimization reveals long-lasting history dependence in
        neural spiking activity (in prep.)

    [2]: https://github.com/Priesemann-Group/hdestimator

implemented in idtxl by Michael Lindner, Göttingen 2021

"""

import numpy as np
from collections import Counter


def get_CI_bounds(R,
                  bs_Rs,
                  bootstrap_CI_use_sd=True,
                  bootstrap_CI_percentile_lo=2.5,
                  bootstrap_CI_percentile_hi=97.5):
    """
    Given bootstrap replications bs_Rs of the estimate for R,
    obtain the lower and upper bound of a 95% confidence 
    interval based on the standard deviation; or an arbitrary
    confidence interval based on percentiles of the replications.
    """
    
    if bootstrap_CI_use_sd:
        sigma_R = np.std(bs_Rs)
        CI_lo = R - 2 * sigma_R
        CI_hi = R + 2 * sigma_R
    else:
        CI_lo = np.percentile(bs_Rs, bootstrap_CI_percentile_lo)
        CI_hi = np.percentile(bs_Rs, bootstrap_CI_percentile_hi)
    return (CI_lo, CI_hi)


def add_up_dicts(dicts):
    return sum((Counter(dict(d)) for d in dicts),
               Counter())


def get_min_key_for_max_value(d):
    """
    For a dictionary d, get the key for the largest value.
    If largest value is attained several times, get the 
    smallest respective key.
    """
    
    sorted_keys = sorted([key for key in d.keys()])
    values = [d[key] for key in sorted_keys]
    max_value = max(values)
    for key, value in zip(sorted_keys, values):
        if value == max_value:
            return key 


def get_max_R_T(max_Rs):
    """
    Get R and T for which R is maximised. 

    If R is maximised at several T, get the 
    smallest respective T.
    """

    max_R_T = get_min_key_for_max_value(max_Rs)
    max_R = max_Rs[max_R_T]
    return max_R, max_R_T


def get_shannon_entropy(probabilities):
    """
    Get the entropy of a random variable based on the probabilities
    of its outcomes.
    """

    return - sum((p * np.log(p) for p in probabilities if not p == 0))


def get_H_spiking(symbol_counts):
    """
    Get the (unconditional) entropy of a spike train, based
    on its outcomes, as stored in the symbol_counts dictionary.

    For each symbol, determine what the response was (spike/ no spike),
    and obtain the spiking probability. From that compute the entropy.
    """
    
    number_of_spikes = 0
    number_of_symbols = 0
    for symbol, counts in symbol_counts.items():
        number_of_symbols += counts
        if symbol % 2 == 1:
            number_of_spikes += counts            
            
    p_spike = number_of_spikes / number_of_symbols
    return get_shannon_entropy([p_spike,
                                1 - p_spike])


def get_binned_neuron_activity(spike_times, bin_size, relative_to_median_activity=False):
    """
    Get an array of 0s and 1s representing the spike train.
    """

    number_of_bins = int(spike_times[-1] / bin_size) + 1
    binned_neuron_activity = np.zeros(number_of_bins, dtype=int)
    if relative_to_median_activity:
        for spike_time in spike_times:
            binned_neuron_activity[int(spike_time / bin_size)] += 1
        median_activity = np.median(binned_neuron_activity)
        spike_indices = np.where(binned_neuron_activity > median_activity)
        binned_neuron_activity = np.zeros(number_of_bins, dtype=int)
        binned_neuron_activity[spike_indices] = 1
    else:
        binned_neuron_activity[[int(spike_time / bin_size) for spike_time in spike_times]] = 1
    return binned_neuron_activity


def get_binned_firing_rate(spike_times, bin_size):
    """
    Get the firing rate of a spike train, as obtained after binning the activity.
    """

    binned_neuron_activity = get_binned_neuron_activity(spike_times, bin_size)
    number_of_bins = int(spike_times[-1] / bin_size) + 1
    return sum(binned_neuron_activity) / (number_of_bins * bin_size)


def remove_key(d, key):
    """    
    Remove an entry from a dictionary .
    """

    r = d.copy()
    del r[key]
    return r


def get_past_symbol_counts(symbol_counts, merge=True):
    """
    Get symbol_counts for the symbols excluding the response.
    """

    past_symbol_counts = [Counter(), Counter()]
    for symbol in symbol_counts:
        response = int(symbol % 2)
        past_symbol = symbol // 2
        past_symbol_counts[response][past_symbol] = symbol_counts[symbol]

    if merge:
        merged_past_symbol_counts = Counter()
        for response in [0, 1]:
            for symbol in past_symbol_counts[response]:
                merged_past_symbol_counts[symbol] += past_symbol_counts[response][symbol]
        return merged_past_symbol_counts
    else:
        return past_symbol_counts

