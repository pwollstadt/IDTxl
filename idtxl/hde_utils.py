"""Utils for the history dependence estimators

from:
    [1]: L. Rudelt, D. G. Marx, M. Wibral, V. Priesemann: Embedding
        optimization reveals long-lasting history dependence in
        neural spiking activity (in prep.)

    [2]: https://github.com/Priesemann-Group/hdestimator

implemented in idtxl by Michael Lindner, Göttingen 2021

"""

import numpy as np
import h5py
import ast
import tempfile
from os import listdir, mkdir, replace
from os.path import isfile, abspath
import io
from sys import stderr
import hashlib
from collections import Counter
from idtxl.hde_estimators import hde_api as hapi
import idtxl.hde_embedding as emb
from idtxl.hde_estimators import hde_bbc_estimator as bbc
from idtxl.hde_estimators import hde_shuffling_estimator as sh

# FAST_UTILS_AVAILABLE = True
# try:
#     import hde_fast_utils as fast_utl
# except:
#     FAST_UTILS_AVAILABLE = False
#     print("""
#     Error importing Cython fast utils module. Continuing with slow Python implementation.\n
#     This may take a long time.\n
#     """, file=stderr, flush=True)

#
# main routines
#
    
def save_history_dependence_for_embeddings(f, spike_times, estimation_method,
                                           embedding_past_range_set,
                                           embedding_number_of_bins_set,
                                           embedding_scaling_exponent_set,
                                           embedding_step_size,
                                           **kwargs):
    """
    Apply embeddings to spike times to obtain symbol counts.  Estimate
    the history dependence for each embedding.  Save results to file.
    """

    if kwargs['cross_val'] == None or kwargs['cross_val'] == 'h1':
        embeddings = emb.get_embeddings(embedding_past_range_set,
                                        embedding_number_of_bins_set,
                                        embedding_scaling_exponent_set)
    elif kwargs['cross_val'] == 'h2':
        # here we set cross_val to h1, because we load the
        # embeddings that maximise R from the optimisation step
        embeddings = get_embeddings_that_maximise_R(f,
                                                    estimation_method,
                                                    embedding_step_size,
                                                    bbc_tolerance=kwargs['bbc_tolerance'],
                                                    get_as_list=True,
                                                    cross_val='h1')
        
    for embedding in embeddings:
        past_range_T = embedding[0]
        number_of_bins_d = embedding[1]
        first_bin_size = emb.get_fist_bin_size_for_embedding(embedding)

        symbol_counts = load_from_analysis_file(f,
                                                "symbol_counts",
                                                embedding_step_size=embedding_step_size,
                                                embedding=embedding,
                                                cross_val=kwargs['cross_val'])
        if symbol_counts == None:
            symbol_counts = add_up_dicts([emb.get_symbol_counts(spt,
                                                                embedding,
                                                                embedding_step_size)
                                          for spt in spike_times])
            save_to_analysis_file(f,
                                  "symbol_counts",
                                  embedding_step_size=embedding_step_size,
                                  embedding=embedding,
                                  symbol_counts=symbol_counts,
                                  cross_val=kwargs['cross_val'])

        if estimation_method == 'bbc':
            history_dependence = load_from_analysis_file(f,
                                                         "history_dependence",
                                                         embedding_step_size=embedding_step_size,
                                                         embedding=embedding,
                                                         estimation_method="bbc",
                                                         cross_val=kwargs['cross_val'])

            if history_dependence == None:
                history_dependence, bbc_term = hapi.get_history_dependence(estimation_method,
                                                                           symbol_counts,
                                                                           number_of_bins_d)
                save_to_analysis_file(f,
                                      "history_dependence",
                                      embedding_step_size=embedding_step_size,
                                      embedding=embedding,
                                      first_bin_size=first_bin_size,
                                      estimation_method="bbc",
                                      history_dependence=history_dependence,
                                      bbc_term=bbc_term,
                                      cross_val=kwargs['cross_val'])
      
        elif estimation_method == 'shuffling':
            history_dependence = load_from_analysis_file(f,
                                                         "history_dependence",
                                                         embedding_step_size=embedding_step_size,
                                                         embedding=embedding,
                                                         estimation_method="shuffling",
                                                         cross_val=kwargs['cross_val'])
            if history_dependence == None:
                history_dependence = hapi.get_history_dependence(estimation_method,
                                                                 symbol_counts,
                                                                 number_of_bins_d)
                save_to_analysis_file(f,
                                      "history_dependence",
                                      embedding_step_size=embedding_step_size,
                                      embedding=embedding,
                                      first_bin_size=first_bin_size,
                                      estimation_method="shuffling",
                                      history_dependence=history_dependence,
                                      cross_val=kwargs['cross_val'])

def save_spike_times_stats(f, spike_times,
                           embedding_step_size,
                           **kwargs):
    """
    Save some statistics about the spike times.
    """

    recording_length = load_from_analysis_file(f,
                                               "recording_length")
    if recording_length == None:
        recording_lengths = [spt[-1] - spt[0] for spt in spike_times]
        recording_length = sum(recording_lengths)
        recording_length_sd = np.std(recording_lengths)

        save_to_analysis_file(f,
                              "recording_length",
                              recording_length=recording_length)
        save_to_analysis_file(f,
                              "recording_length_sd",
                              recording_length_sd=recording_length_sd)

    
    firing_rate = load_from_analysis_file(f,
                                          "firing_rate")
    if firing_rate == None:
        firing_rates = [get_binned_firing_rate(spt, embedding_step_size)
                        for spt in spike_times]
        recording_lengths = [spt[-1] - spt[0] for spt in spike_times]
        recording_length = sum(recording_lengths)

        firing_rate = np.average(firing_rates, weights=recording_lengths)
        firing_rate_sd = np.sqrt(np.average((firing_rates - firing_rate) ** 2,
                                            weights=recording_lengths))

        save_to_analysis_file(f,
                              "firing_rate",
                              firing_rate=firing_rate)
        save_to_analysis_file(f,
                              "firing_rate_sd",
                              firing_rate_sd=firing_rate_sd)

    H_spiking = load_from_analysis_file(f,
                                       "H_spiking")
    
    if H_spiking == None:
        H_spiking = get_shannon_entropy([firing_rate * embedding_step_size,
                                        1 - firing_rate * embedding_step_size])
        
        save_to_analysis_file(f,
                              "H_spiking",
                              H_spiking=H_spiking)

                
def get_embeddings_that_maximise_R(f,
                                   estimation_method,
                                   embedding_step_size,
                                   bbc_tolerance=None,
                                   dependent_var="T",
                                   get_as_list=False,
                                   cross_val=None,
                                   **kwargs):
    """
    For each T (or d), get the embedding for which R is maximised.

    For the bbc estimator, here the bbc_tolerance is applied, ie 
    get the unbiased embeddings that maximise R.
    """

    assert dependent_var in ["T", "d"]
    assert cross_val in [None, "h1", "h2"]

    if bbc_tolerance == None \
       or cross_val == "h2": # apply bbc only for optimization
        bbc_tolerance = np.inf

    if cross_val == None:
        root_dir = 'embeddings'
    else:
        root_dir = '{}_embeddings'.format(cross_val)
        
    max_Rs = {}
    embeddings_that_maximise_R = {}

    embedding_step_size_label = get_parameter_label(embedding_step_size)
    
    for past_range_T_label in f["{}/{}".format(root_dir, embedding_step_size_label)].keys():
        for number_of_bins_d_label in f["{}/{}/{}".format(root_dir,
                                                          embedding_step_size_label,
                                                          past_range_T_label)].keys():
            for scaling_k_label in f["{}/{}/{}/{}".format(root_dir,
                                                          embedding_step_size_label,
                                                          past_range_T_label,
                                                          number_of_bins_d_label)].keys():
                past_range_T = float(past_range_T_label)
                number_of_bins_d = int(float(number_of_bins_d_label))
                scaling_k = float(scaling_k_label)
                embedding = (past_range_T,
                             number_of_bins_d,
                             scaling_k)
                history_dependence = load_from_analysis_file(f,
                                                             "history_dependence",
                                                             embedding_step_size=embedding_step_size,
                                                             embedding=embedding,
                                                             estimation_method=estimation_method,
                                                             cross_val=cross_val)
                # if it has been estimated for one estimator, but not the other
                # it might be None. skip if this is the case
                if history_dependence == None:
                    continue
                
                if estimation_method == "bbc":
                    bbc_term = load_from_analysis_file(f,
                                                       "bbc_term",
                                                       embedding_step_size=embedding_step_size,
                                                       embedding=embedding,
                                                       estimation_method=estimation_method,
                                                       cross_val=cross_val)
                    if bbc_term >= bbc_tolerance:
                        continue

                if dependent_var == "T":
                    if not past_range_T in embeddings_that_maximise_R \
                       or history_dependence > max_Rs[past_range_T]:
                        max_Rs[past_range_T] = history_dependence
                        embeddings_that_maximise_R[past_range_T] = (number_of_bins_d,
                                                                    scaling_k)
                elif dependent_var == "d":
                    if not number_of_bins_d in embeddings_that_maximise_R \
                       or history_dependence > max_Rs[number_of_bins_d]:
                        max_Rs[number_of_bins_d] = history_dependence
                        embeddings_that_maximise_R[number_of_bins_d] = (past_range_T,
                                                                        scaling_k)

    if get_as_list:
        embeddings = []
        if dependent_var == "T":
            for past_range_T in embeddings_that_maximise_R:
                number_of_bins_d, scaling_k = embeddings_that_maximise_R[past_range_T]
                embeddings += [(past_range_T, number_of_bins_d, scaling_k)]
        elif dependent_var == "d":
            for number_of_bins_d in embeddings_that_maximise_R:
                past_range_T, scaling_k = embeddings_that_maximise_R[number_of_bins_d]
                embeddings += [(past_range_T, number_of_bins_d, scaling_k)]
        return embeddings
    else:
        return embeddings_that_maximise_R, max_Rs

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

def get_temporal_depth_T_D(f,
                           estimation_method,
                           bootstrap_CI_use_sd=True,
                           bootstrap_CI_percentile_lo=2.5,
                           bootstrap_CI_percentile_hi=97.5,
                           get_R_thresh=False,
                           **kwargs):
    """
    Get the temporal depth T_D, the past range for the 
    'optimal' embedding parameters.

    Given the maximal history dependence R at each past range T,
    (cf get_embeddings_that_maximise_R), first find the smallest T at 
    which R is maximised (cf get_max_R_T).  If bootstrap replications
    for this R are available, get the smallest T at which this R minus
    one standard deviation of the bootstrap estimates is attained.
    """

    # load data
    embedding_maximising_R_at_T, max_Rs \
        = get_embeddings_that_maximise_R(f,
                                         estimation_method=estimation_method,
                                         **kwargs)

    Ts = sorted([key for key in max_Rs.keys()])
    Rs = [max_Rs[T] for T in Ts]

    # first get the max history dependence, and if available its bootstrap replications
    max_R, max_R_T = get_max_R_T(max_Rs)

    number_of_bins_d, scaling_k = embedding_maximising_R_at_T[max_R_T]
    bs_Rs = load_from_analysis_file(f,
                                    "bs_history_dependence",
                                    embedding_step_size=kwargs["embedding_step_size"],
                                    embedding=(max_R_T,
                                               number_of_bins_d,
                                               scaling_k),
                                    estimation_method=estimation_method,
                                    cross_val=kwargs['cross_val'])

    if isinstance(bs_Rs, np.ndarray):
        max_R_sd = np.std(bs_Rs)
    else:
        max_R_sd = 0

    R_tot_thresh = max_R - max_R_sd

    T_D = min(Ts)
    for R, T in zip(Rs, Ts):
        if R >= R_tot_thresh:
            T_D = T
            break

    if not get_R_thresh:
        return T_D
    else:
        return T_D, R_tot_thresh

def get_information_timescale_tau_R(f,
                                    estimation_method,
                                    **kwargs):
    """
    Get the information timescale tau_R, a characteristic
    timescale of history dependence similar to an autocorrelation 
    time.
    """

    # load data
    embedding_maximising_R_at_T, max_Rs \
        = get_embeddings_that_maximise_R(f,
                                         estimation_method=estimation_method,
                                         **kwargs)

    Ts = np.array(sorted([key for key in max_Rs.keys()]))
    Rs = np.array([max_Rs[T] for T in Ts])

    R_tot = get_R_tot(f,
                      estimation_method=estimation_method,
                      **kwargs)

    T_0 = kwargs["timescale_minimum_past_range"]

    # get dRs
    dRs = []
    R_prev = 0.

    # No values higher than R_tot are allowed,
    # otherwise the information timescale might be
    # misestimated because of spurious contributions
    # at large T
    for R, T in zip(Rs[Rs <= R_tot], Ts[Rs <= R_tot]):

        # No negative increments are allowed
        dRs += [np.amax([0.0, R - R_prev])]
        
        # The increment is taken with respect to the highest previous value of R
        if R > R_prev:
            R_prev = R

    dRs = np.pad(dRs, (0, len(Rs) - len(dRs)),
                 mode='constant', constant_values=0)


    # compute tau_R
    Ts_0 = np.append([0], Ts)
    dRs_0 = dRs[Ts_0[:-1] >= T_0]
    
    # Only take into considerations contributions beyond T_0
    Ts_0 = Ts_0[Ts_0 >= T_0]
    norm = np.sum(dRs_0)
    
    if norm == 0.:
        tau = 0.0
    else:
        Ts_0 -= Ts_0[0]
        tau = np.dot(((Ts_0[:-1] + Ts_0[1:]) / 2), dRs_0) / norm
    return tau

    
def get_R_tot(f,
              estimation_method,
              return_averaged_R=False,
              **kwargs):
    embedding_maximising_R_at_T, max_Rs \
        = get_embeddings_that_maximise_R(f,
                                         estimation_method=estimation_method,
                                         **kwargs)
    
    if return_averaged_R:
        T_D, R_tot_thresh = get_temporal_depth_T_D(f,
                                                   estimation_method=estimation_method,
                                                   get_R_thresh=True,
                                                   **kwargs)

        Ts = sorted([key for key in max_Rs.keys()])
        Rs = [max_Rs[T] for T in Ts]

        T_max = T_D
        for R, T in zip(Rs, Ts):
            if T < T_D:
                continue
            T_max = T
            if R < R_tot_thresh:
                break
            
        return np.average([R for R, T in zip(Rs, Ts) if T >= T_D and T < T_max])

    else:
        temporal_depth_T_D = get_temporal_depth_T_D(f,
                                                    estimation_method=estimation_method,
                                                    **kwargs)
        
        return max_Rs[temporal_depth_T_D]


# # FIXME make this more general, pass embedding and do not
# # automatically apply it to the opt embedding
# # and think about making is flexible for AIS...
# def estimate_bootstrap_bias(f,
#                             embedding_step_size,
#                             estimation_method,
#                             **kwargs):
#     embedding_maximising_R_at_T, max_Rs \
#         = get_embeddings_that_maximise_R(f,
#                                          embedding_step_size=embedding_step_size,
#                                          estimation_method=estimation_method,
#                                          **kwargs)

#     T_D = get_temporal_depth_T_D(f,
#                                  estimation_method,
#                                  embedding_step_size=embedding_step_size,
#                                  **kwargs)

#     number_of_bins_d, scaling_k = embedding_maximising_R_at_T[T_D]
#     embedding = (T_D, number_of_bins_d, scaling_k)


#     # first get the bootstrapped Rs
#     bs_Rs = load_from_analysis_file(f,
#                                     "bs_history_dependence",
#                                     embedding_step_size=embedding_step_size,
#                                     embedding=embedding,
#                                     estimation_method=estimation_method,
#                                     cross_val=kwargs['cross_val'])

#     # then do the plugin estimate
#     alphabet_size_past = 2 ** int(number_of_bins_d)
#     alphabet_size = alphabet_size_past * 2

#     symbol_counts = load_from_analysis_file(f,
#                                             "symbol_counts",
#                                             embedding_step_size=embedding_step_size,
#                                             embedding=embedding,
#                                             cross_val=kwargs['cross_val'])
#     past_symbol_counts = get_past_symbol_counts(symbol_counts)

#     mk = bbc.get_multiplicities(symbol_counts,
#                                 alphabet_size)
#     mk_past = bbc.get_multiplicities(past_symbol_counts,
#                                      alphabet_size_past)

#     N = np.sum((mk[n] * n for n in mk.keys()))

#     H_plugin_joint = bbc.plugin_entropy(mk, N)
#     H_plugin_past = bbc.plugin_entropy(mk_past, N)
#     H_plugin_cond = H_plugin_joint - H_plugin_past
#     H_spiking = load_from_analysis_file(f,
#                                        "H_spiking")
#     I_plugin = H_spiking - H_plugin_cond
#     history_dependence_plugin = I_plugin / H_spiking

#     return np.average(bs_Rs) - history_dependence_plugin


def compute_CIs(f,
                spike_times,
                estimation_method,
                embedding_step_size,
                block_length_l=None,
                target_R='R_max',
                **kwargs):
    """ 
    Compute bootstrap replications of the history dependence estimate
    which can be used to obtain confidence intervals.

    Load symbol counts, resample, then estimate entropy for each sample
    and save to file.

    :param target_R: One of 'R_max', 'R_tot' or 'nonessential'.
    If set to R_max, replications of R are produced for the T at which
    R is maximised.
    If set to R_tot, replications of R are produced for T = T_D (cf 
    get_temporal_depth_T_D).  
    If set to nonessential, replications of R are produced for each T
    (one embedding per T, cf get_embeddings_that_maximise_R).  These 
    are not otherwise used in the analysis and are probably only useful
    if the resulting plot is visually inspected, so in most cases it can
    be set to zero.
    """

    assert target_R in ['nonessential', 'R_max', 'R_tot']

    number_of_bootstraps = kwargs['number_of_bootstraps_{}'.format(target_R)]

    if number_of_bootstraps == 0:
        return
    
    embedding_maximising_R_at_T, max_Rs \
        = get_embeddings_that_maximise_R(f,
                                         embedding_step_size=embedding_step_size,
                                         estimation_method=estimation_method,
                                         **kwargs)

    recording_length = load_from_analysis_file(f,
                                               "recording_length")

    firing_rate = load_from_analysis_file(f,
                                          "firing_rate")

    if block_length_l == None:
        # eg firing rate is 4 Hz, ie there is 1 spikes per 1/4 seconds, 
        # for every second the number of symbols is 1/ embedding_step_size
        # so we observe on average one spike every 1 / (firing_rate * embedding_step_size) symbols
        # (in the reponse, ignoring the past activity)
        block_length_l = max(1, int(1 / (firing_rate * embedding_step_size)))

    if target_R == 'nonessential':
        # bootstrap R for unessential Ts (not required for the main analysis)
        embeddings = []

        for past_range_T in embedding_maximising_R_at_T:
            number_of_bins_d, scaling_k = embedding_maximising_R_at_T[past_range_T]
            embeddings += [(past_range_T, number_of_bins_d, scaling_k)]

    elif target_R == 'R_max':
        # bootstrap R for the max R, to get a good estimate for the standard deviation
        # which is used to determine R_tot
        max_R, max_R_T = get_max_R_T(max_Rs)
        number_of_bins_d, scaling_k = embedding_maximising_R_at_T[max_R_T]
        
        embeddings = [(max_R_T, number_of_bins_d, scaling_k)]
    elif target_R == 'R_tot':
        T_D = get_temporal_depth_T_D(f,
                                     estimation_method,
                                     embedding_step_size=embedding_step_size,
                                     **kwargs)
        number_of_bins_d, scaling_k = embedding_maximising_R_at_T[T_D]
        
        embeddings = [(T_D, number_of_bins_d, scaling_k)]

    for embedding in embeddings:
        stored_bs_Rs = load_from_analysis_file(f,
                                               "bs_history_dependence",
                                               embedding_step_size=embedding_step_size,
                                               embedding=embedding,
                                               estimation_method=estimation_method,
                                               cross_val=kwargs['cross_val'])
        if isinstance(stored_bs_Rs, np.ndarray):
            number_of_stored_bootstraps = len(stored_bs_Rs)
        else:
            number_of_stored_bootstraps = 0

        if not number_of_bootstraps > number_of_stored_bootstraps:
            continue

        bs_history_dependence \
            = get_bootstrap_history_dependence(spike_times,
                                               embedding,
                                               embedding_step_size,
                                               estimation_method,
                                               number_of_bootstraps - number_of_stored_bootstraps,
                                               block_length_l)

        save_to_analysis_file(f,
                              "bs_history_dependence",
                              embedding_step_size=embedding_step_size,
                              embedding=embedding,
                              estimation_method=estimation_method,
                              bs_history_dependence=bs_history_dependence,
                              cross_val=kwargs['cross_val'])


def get_bootstrap_history_dependence(spike_times,
                                     embedding,
                                     embedding_step_size,
                                     estimation_method,
                                     number_of_bootstraps,
                                     block_length_l):
    """
    For a given embedding, return bootstrap replications for R.
    """
    past_range_T, number_of_bins_d, scaling_k = embedding

    # compute total number of symbols in original data:
    # this is the amount of symbols we want to replicate
    min_num_symbols = 1 + int((min([spt[-1] - spt[0] for spt in spike_times])
                               - (past_range_T + embedding_step_size))
                              / embedding_step_size)
    
    symbol_block_length = int(block_length_l)
    
    if symbol_block_length >= min_num_symbols:
        print("Warning. Block length too large given number of symbols. Skipping.")
        return []

    # compute the bootstrap replications

    bs_Rs = np.zeros(number_of_bootstraps)

    symbols_array \
        = [get_symbols_array(spt, embedding, embedding_step_size)
           for spt in spike_times]

    for rep in range(number_of_bootstraps):
        bs_symbol_counts \
            = add_up_dicts([get_bootstrap_symbol_counts_from_symbols_array(symbols_array[i],
                                                                           symbol_block_length)
                            for i in range(len(symbols_array))])

        bs_history_dependence = hapi.get_history_dependence(estimation_method,
                                                            bs_symbol_counts,
                                                            number_of_bins_d,
                                                            bbc_tolerance=np.inf)

        bs_Rs[rep] = bs_history_dependence

    return bs_Rs

def get_symbols_array(spike_times, embedding, embedding_step_size):
    """
    Apply an embedding to a spike train and get the resulting symbols.
    """
    
    past_range_T, number_of_bins_d, scaling_k = embedding
    first_bin_size = emb.get_fist_bin_size_for_embedding(embedding)

    raw_symbols = emb.get_raw_symbols(spike_times,
                                      embedding,
                                      first_bin_size,
                                      embedding_step_size)

    median_number_of_spikes_per_bin = emb.get_median_number_of_spikes_per_bin(raw_symbols)

    # symbols_array: array containing symbols
    # symbol_array: array of spikes representing symbol
    symbols_array = np.zeros(len(raw_symbols))
    
    for symbol_index, raw_symbol in enumerate(raw_symbols):
        symbol_array = [int(raw_symbol[i] > median_number_of_spikes_per_bin[i])
                        for i in range(number_of_bins_d + 1)]

        symbol = emb.symbol_array_to_binary(symbol_array, number_of_bins_d + 1)

        symbols_array[symbol_index] = symbol

    return symbols_array

def get_bootstrap_symbol_counts_from_symbols_array(symbols_array,
                                                   symbol_block_length):
    """
    Given an array of symbols (cf get_symbols_array), get bootstrap 
    replications of the symbol counts.
    """

    num_symbols = len(symbols_array)

    rand_indices = np.random.randint(0, num_symbols - (symbol_block_length - 1),
                                     size=int(num_symbols/ symbol_block_length))

    symbol_counts = Counter()
    
    for rand_index in rand_indices:
        for symbol in symbols_array[rand_index:rand_index + symbol_block_length]:
            symbol_counts[symbol] += 1

    residual_block_length = num_symbols - sum(symbol_counts.values())

    if residual_block_length > 0:
        rand_index_residual = np.random.randint(0, num_symbols - (residual_block_length - 1))

        for symbol in symbols_array[rand_index_residual:rand_index_residual + residual_block_length]:
            symbol_counts[symbol] += 1

    return symbol_counts

# def perform_permutation_test(f,
#                              number_of_permutations,
#                              embedding_step_size,
#                              estimation_method,
#                              **kwargs):
#     """
#     Perform a permutation test to check whether th history dependece 
#     in the target neuron is significantly different from zero.
    
#     This is performed for R_tot, the R for which T = T_D (cf 
#     get_temporal_depth_T_D).
#     """
    
#     embedding_maximising_R_at_T, max_Rs = get_embeddings_that_maximise_R(f,
#                                                                           embedding_step_size=embedding_step_size,
#                                                                           estimation_method=estimation_method,
#                                                                           **kwargs)

#     temporal_depth_T_D = get_temporal_depth_T_D(f,
#                                                 estimation_method=estimation_method,
#                                                 embedding_step_size=embedding_step_size,
#                                                 **kwargs)

#     R_tot = max_Rs[temporal_depth_T_D]
#     opt_number_of_bins_d, opt_scaling_k \
#         = embedding_maximising_R_at_T[temporal_depth_T_D]
    
#     opt_embedding = (temporal_depth_T_D, opt_number_of_bins_d, opt_scaling_k)

#     symbol_counts = load_from_analysis_file(f,
#                                             "symbol_counts",
#                                             embedding_step_size=embedding_step_size,
#                                             embedding=opt_embedding,
#                                             cross_val=kwargs['cross_val'])

#     stored_pt_Rs = load_from_analysis_file(f,
#                                            "pt_history_dependence",
#                                            embedding_step_size=embedding_step_size,
#                                            embedding=opt_embedding,
#                                            estimation_method=estimation_method,
#                                            cross_val=kwargs['cross_val'])
    
#     if isinstance(stored_pt_Rs, np.ndarray):
#         number_of_stored_permutations = len(stored_pt_Rs)
#     else:
#         number_of_stored_permutations = 0

#     if not number_of_permutations - number_of_stored_permutations > 0:
#         return

#     pt_history_dependence = np.zeros(number_of_permutations - number_of_stored_permutations)

#     for rep in range(number_of_permutations - number_of_stored_permutations):
#         if FAST_UTILS_AVAILABLE:
#             pt_symbol_counts, pt_number_of_spikes \
#                 = fast_utl.get_permutation_symbol_counts(symbol_counts)
#         else:
#             pt_symbol_counts, pt_number_of_spikes = get_permutation_symbol_counts(symbol_counts)

#         history_dependence = hapi.get_history_dependence(estimation_method,
#                                                          pt_symbol_counts,
#                                                          opt_number_of_bins_d,
#                                                          bbc_tolerance=np.inf)
        
#         pt_history_dependence[rep] = history_dependence

#     save_to_analysis_file(f,
#                           "pt_history_dependence",
#                           embedding_step_size=embedding_step_size,
#                           embedding=opt_embedding,
#                           estimation_method=estimation_method,
#                           pt_history_dependence=pt_history_dependence)


# def get_permutation_symbol_counts(symbol_counts):
#     """
#     Get symbols counts with permutated responses.

#     input:  dictionary with counts as computed from the data
#     output: dictionary with counts as produced by permutation re-draw
#     """

#     total_number_of_symbols = sum((number_of_occurrences
#                                    for number_of_occurrences in symbol_counts.values()))
#     number_of_spikes = sum((symbol_counts[symbol]
#                             for symbol in symbol_counts if symbol % 2 == 1))

#     past_symbol_counts = get_past_symbol_counts(symbol_counts)

#     remaining_number_of_symbols = total_number_of_symbols
#     remaining_past_symbol_counts = past_symbol_counts.copy()

#     permutated_symbol_counts = {}

#     for draw in range(number_of_spikes):
#         past_symbol = get_random_symbol(remaining_past_symbol_counts,
#                                         remaining_number_of_symbols)
#         symbol = past_symbol * 2 + 1
#         if symbol in permutated_symbol_counts:
#             permutated_symbol_counts[symbol] += 1
#         else:
#             permutated_symbol_counts[symbol] = 1
#         remaining_number_of_symbols -= 1
#         remaining_past_symbol_counts[past_symbol] -= 1
        
#     for past_symbol in remaining_past_symbol_counts:
#         symbol = past_symbol * 2
#         permutated_symbol_counts[symbol] = remaining_past_symbol_counts[past_symbol]

#     return permutated_symbol_counts, number_of_spikes

# def get_random_symbol(symbol_counts, number_of_symbols):
#     """
#     Get a random symbol from the symbol_counts dictionary.
#     """

#     randint = np.random.randint(number_of_symbols)
#     count_index = 0
#     for symbol in symbol_counts:
#         count_index += symbol_counts[symbol]
#         if count_index > randint:
#             return symbol

#
# information theoretic measure of entropy
#

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

def get_smoothed_neuron_activity(spt,
                                 averaging_time,
                                 binning_time=0.005):
    """
    Get a smoothed version of the neuron activity, for visualization.

    cf https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    binned_neuron_activity \
        = get_binned_neuron_activity(spt,
                                     binning_time)
    smoothing_window_len = int(averaging_time / binning_time)

    s = np.r_[binned_neuron_activity[smoothing_window_len-1:0:-1],
              binned_neuron_activity,
              binned_neuron_activity[-2:-smoothing_window_len-1:-1]]

    return np.convolve(np.hanning(smoothing_window_len) /
                       np.hanning(smoothing_window_len).sum(), s,
                       mode='valid') / binning_time


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


def create_default_settings_file(ESTIMATOR_DIR="."):
    """
    Create the  default settings/parameters file, in case the one 
    shipped with the tool is missing. 
    """
    
    settings = {'embedding_step_size' : 0.005,
                'embedding_past_range_set' : [float("{:.5f}".format(np.exp(x))) for x in np.arange(np.log(0.005), np.log(5.001), 0.05 * np.log(10))],
                'embedding_number_of_bins_set' : [int(x) for x in np.linspace(1,5,5)],
                'embedding_scaling_exponent_set' : {'number_of_scalings': 10,
                                                    'min_first_bin_size' : 0.005,
                                                    'min_step_for_scaling': 0.01},
                'estimation_method' : "shuffling",
                'bbc_tolerance' : 0.05,
                'cross_validated_optimization' : False,
                'return_averaged_R' : True,
                'timescale_minimum_past_range' : 0.01,
                'number_of_bootstraps_R_max' : 250,
                'number_of_bootstraps_R_tot' : 250,
                'number_of_bootstraps_nonessential' : 0,
                'block_length_l' : None,
                'bootstrap_CI_use_sd' : True,
                'bootstrap_CI_percentile_lo' : 2.5,
                'bootstrap_CI_percentile_hi' : 97.5,
                # 'number_of_permutations' : 100,
                'auto_MI_bin_size_set' : [0.005, 0.01, 0.025, 0.05, 0.25, 0.5],
                'auto_MI_max_delay' : 5,
                'label' : '""',
                'ANALYSIS_DIR' : "./analysis",
                'persistent_analysis' : True,
                # 'verbose_output' : False,
                'plot_AIS' : False,
                'plot_settings' : {'figure.figsize' : [6.3, 5.5],
                                   'axes.labelsize': 9,
                                   'font.size': 9,
                                   'legend.fontsize': 8,
                                   'xtick.labelsize': 8,
                                   'ytick.labelsize': 8,
                                   'savefig.format': 'pdf'},
                'plot_color' : "'#4da2e2'"}
    
    with open('{}/settings/default.yaml'.format(ESTIMATOR_DIR), 'w') as settings_file:
        for setting_name in settings:
            if isinstance(settings[setting_name], dict):
                settings_file.write("{} :\n".format(setting_name))
                for s in settings[setting_name]:
                    if isinstance(settings[setting_name][s], str):
                        settings_file.write("    '{}' : '{}'\n".format(s, settings[setting_name][s]))
                    else:
                        settings_file.write("    '{}' : {}\n".format(s, settings[setting_name][s]))
            else:
                settings_file.write("{} : {}\n".format(setting_name, settings[setting_name]))
    settings_file.close()

def get_analysis_stats(f,
                       analysis_num,
                       estimation_method=None,
                       **kwargs):
    """
    Get statistics of the analysis, to export them to a csv file.
    """

    stats = {
        "analysis_num" : str(analysis_num),
        "label" : kwargs["label"],
        "tau_R_bbc" : "-",
        "T_D_bbc" : "-",
        "R_tot_bbc" : "-",
        "R_tot_bbc_CI_lo" : "-",
        "R_tot_bbc_CI_hi" : "-",
        # "R_tot_bbc_RMSE" : "-",
        # "R_tot_bbc_bias" : "-",
        "AIS_tot_bbc" : "-",
        "AIS_tot_bbc_CI_lo" : "-",
        "AIS_tot_bbc_CI_hi" : "-",
        # "AIS_tot_bbc_RMSE" : "-",
        # "AIS_tot_bbc_bias" : "-",
        "opt_number_of_bins_d_bbc" : "-",
        "opt_scaling_k_bbc" : "-",
        "opt_first_bin_size_bbc" : "-",
        # "asl_permutation_test_bbc" : "-",
        "tau_R_shuffling" : "-",
        "T_D_shuffling" : "-",
        "R_tot_shuffling" : "-",
        "R_tot_shuffling_CI_lo" : "-",
        "R_tot_shuffling_CI_hi" : "-",
        # "R_tot_shuffling_RMSE" : "-",
        # "R_tot_shuffling_bias" : "-",
        "AIS_tot_shuffling" : "-",
        "AIS_tot_shuffling_CI_lo" : "-",
        "AIS_tot_shuffling_CI_hi" : "-",
        # "AIS_tot_shuffling_RMSE" : "-",
        # "AIS_tot_shuffling_bias" : "-",
        "opt_number_of_bins_d_shuffling" : "-",
        "opt_scaling_k_shuffling" : "-",
        "opt_first_bin_size_shuffling" : "-",
        # "asl_permutation_test_shuffling" : "-",
        "embedding_step_size" : get_parameter_label(kwargs["embedding_step_size"]),
        "bbc_tolerance" : get_parameter_label(kwargs["bbc_tolerance"]),
        "timescale_minimum_past_range" : get_parameter_label(kwargs["timescale_minimum_past_range"]),
        "number_of_bootstraps_bbc" : "-",
        "number_of_bootstraps_shuffling" : "-",
        "bs_CI_percentile_lo" : "-",
        "bs_CI_percentile_hi" : "-",
        # "number_of_permutations_bbc" : "-",
        # "number_of_permutations_shuffling" : "-",
        "firing_rate" : get_parameter_label(load_from_analysis_file(f, "firing_rate")),
        "firing_rate_sd" : get_parameter_label(load_from_analysis_file(f, "firing_rate_sd")),
        "recording_length" : get_parameter_label(load_from_analysis_file(f, "recording_length")),
        "recording_length_sd" : get_parameter_label(load_from_analysis_file(f, "recording_length_sd")),
        "H_spiking" : "-",
    }
    
    if stats["label"] == "":
        stats["label"] = "-"

    H_spiking = load_from_analysis_file(f, "H_spiking")
    stats["H_spiking"] = get_parameter_label(H_spiking)

    for estimation_method in ["bbc", "shuffling"]:

        embedding_maximising_R_at_T, max_Rs \
            = get_embeddings_that_maximise_R(f,
                                             estimation_method=estimation_method,
                                             **kwargs)

        if len(embedding_maximising_R_at_T) == 0:
            continue

        tau_R = get_information_timescale_tau_R(f,
                                                estimation_method=estimation_method,
                                                **kwargs)
        
        temporal_depth_T_D = get_temporal_depth_T_D(f,
                                                    estimation_method=estimation_method,
                                                    **kwargs)

        R_tot = get_R_tot(f,
                          estimation_method=estimation_method,
                          **kwargs)
        opt_number_of_bins_d, opt_scaling_k \
            = embedding_maximising_R_at_T[temporal_depth_T_D]

        stats["tau_R_{}".format(estimation_method)] = get_parameter_label(tau_R)
        stats["T_D_{}".format(estimation_method)] = get_parameter_label(temporal_depth_T_D)
        stats["R_tot_{}".format(estimation_method)] = get_parameter_label(R_tot)
        stats["AIS_tot_{}".format(estimation_method)] = get_parameter_label(R_tot * H_spiking)
        stats["opt_number_of_bins_d_{}".format(estimation_method)] \
            = get_parameter_label(opt_number_of_bins_d)
        stats["opt_scaling_k_{}".format(estimation_method)] \
            = get_parameter_label(opt_scaling_k)

        stats["opt_first_bin_size_{}".format(estimation_method)] \
            = get_parameter_label(load_from_analysis_file(f,
                                                          "first_bin_size",
                                                          embedding_step_size\
                                                          =kwargs["embedding_step_size"],
                                                          embedding=(temporal_depth_T_D,
                                                                     opt_number_of_bins_d,
                                                                     opt_scaling_k),
                                                          estimation_method=estimation_method,
                                                          cross_val=kwargs['cross_val']))

        if not kwargs['return_averaged_R']:
            bs_Rs = load_from_analysis_file(f,
                                            "bs_history_dependence",
                                            embedding_step_size=kwargs["embedding_step_size"],
                                            embedding=(temporal_depth_T_D,
                                                       opt_number_of_bins_d,
                                                       opt_scaling_k),
                                            estimation_method=estimation_method,
                                            cross_val=kwargs['cross_val'])
            if isinstance(bs_Rs, np.ndarray):
                stats["number_of_bootstraps_{}".format(estimation_method)] \
                    = str(len(bs_Rs))

        if not stats["number_of_bootstraps_{}".format(estimation_method)] == "-":
            R_tot_CI_lo, R_tot_CI_hi = get_CI_bounds(R_tot,
                                                     bs_Rs,
                                                     kwargs["bootstrap_CI_use_sd"],
                                                     kwargs["bootstrap_CI_percentile_lo"],
                                                     kwargs["bootstrap_CI_percentile_hi"])
            stats["R_tot_{}_CI_lo".format(estimation_method)] \
                = get_parameter_label(R_tot_CI_lo)
            stats["R_tot_{}_CI_hi".format(estimation_method)] \
                = get_parameter_label(R_tot_CI_hi)
            stats["AIS_tot_{}_CI_lo".format(estimation_method)] \
                = get_parameter_label(R_tot_CI_lo * H_spiking)
            stats["AIS_tot_{}_CI_hi".format(estimation_method)] \
                = get_parameter_label(R_tot_CI_hi * H_spiking)

            # bias = estimate_bootstrap_bias(f,
            #                                estimation_method=estimation_method,
            #                                **kwargs)

            # variance = np.var(bs_Rs)

            # stats["R_tot_{}_RMSE".format(estimation_method)] \
            #     = get_parameter_label(np.sqrt(variance + bias**2))

            # stats["R_tot_{}_bias".format(estimation_method)] \
            #     = get_parameter_label(bias)

            # TODO RMSE, bias for AIS

        if kwargs["bootstrap_CI_use_sd"]:
            stats["bs_CI_percentile_lo"] = get_parameter_label(2.5)
            stats["bs_CI_percentile_hi"] = get_parameter_label(97.5)
        else:
            stats["bs_CI_percentile_lo"] = get_parameter_label(kwargs["bootstrap_CI_percentile_lo"])
            stats["bs_CI_percentile_hi"] = get_parameter_label(kwargs["bootstrap_CI_percentile_hi"])

        # pt_Rs = load_from_analysis_file(f,
        #                                 "pt_history_dependence",
        #                                 embedding_step_size=kwargs["embedding_step_size"],
        #                                 embedding=(temporal_depth_T_D,
        #                                            opt_number_of_bins_d,
        #                                            opt_scaling_k),
        #                                 estimation_method=estimation_method,
        #                                 cross_val=kwargs['cross_val'])

        # if isinstance(pt_Rs, np.ndarray):
        #     stats["asl_permutation_test_{}".format(estimation_method)] \
        #         = get_parameter_label(get_asl_permutation_test(pt_Rs, R_tot))

        #     stats["number_of_permutations_{}".format(estimation_method)] \
        #         = str(len(pt_Rs))

    return stats

def get_histdep_data(f,
                     analysis_num,
                     estimation_method,
                     **kwargs):
    """
    Get R values for each T, as needed for the plots, to export them 
    to a csv file.
    """
    
    histdep_data = {
        "T" : [],
        "max_R_bbc" : [],
        "max_R_bbc_CI_lo" : [],
        "max_R_bbc_CI_hi" : [],
        # "max_R_bbc_CI_med" : [],
        "max_AIS_bbc" : [],
        "max_AIS_bbc_CI_lo" : [],
        "max_AIS_bbc_CI_hi" : [],
        # "max_AIS_bbc_CI_med" : [],
        "number_of_bins_d_bbc" : [],
        "scaling_k_bbc" : [],
        "first_bin_size_bbc" : [],
        "max_R_shuffling" : [],
        "max_R_shuffling_CI_lo" : [],
        "max_R_shuffling_CI_hi" : [],
        # "max_R_shuffling_CI_med" : [],
        "max_AIS_shuffling" : [],
        "max_AIS_shuffling_CI_lo" : [],
        "max_AIS_shuffling_CI_hi" : [],
        # "max_AIS_shuffling_CI_med" : [],
        "number_of_bins_d_shuffling" : [],
        "scaling_k_shuffling" : [],
        "first_bin_size_shuffling" : [],
    }
    
    for estimation_method in ['bbc', 'shuffling']:
        # kwargs["estimation_method"] = estimation_method
        
        embedding_maximising_R_at_T, max_Rs \
            = get_embeddings_that_maximise_R(f,
                                             estimation_method=estimation_method,
                                             **kwargs)

        if len(embedding_maximising_R_at_T) == 0:
            if estimation_method == 'bbc':
                embedding_maximising_R_at_T_bbc = {}
                max_Rs_bbc = []
                max_R_bbc_CI_lo = {}
                max_R_bbc_CI_hi = {}
                # max_R_bbc_CI_med = {}
            elif estimation_method == 'shuffling':
                embedding_maximising_R_at_T_shuffling = {}
                max_Rs_shuffling = []
                max_R_shuffling_CI_lo = {}
                max_R_shuffling_CI_hi = {}
                # max_R_shuffling_CI_med = {}
            continue

        max_R_CI_lo = {}
        max_R_CI_hi = {}
        # max_R_CI_med = {}
        
        for past_range_T in embedding_maximising_R_at_T:
            number_of_bins_d, scaling_k = embedding_maximising_R_at_T[past_range_T]
            embedding = (past_range_T, number_of_bins_d, scaling_k)

            bs_Rs = load_from_analysis_file(f,
                                            "bs_history_dependence",
                                            embedding_step_size=kwargs["embedding_step_size"],
                                            embedding=embedding,
                                            estimation_method=estimation_method,
                                            cross_val=kwargs['cross_val'])

            if isinstance(bs_Rs, np.ndarray):
                max_R_CI_lo[past_range_T], max_R_CI_hi[past_range_T] \
                    = get_CI_bounds(max_Rs[past_range_T],
                                    bs_Rs,
                                    kwargs["bootstrap_CI_use_sd"],
                                    kwargs["bootstrap_CI_percentile_lo"],
                                    kwargs["bootstrap_CI_percentile_hi"])
                # max_R_CI_med[past_range_T] \
                #     = np.median(bs_Rs)
            else:
                max_R_CI_lo[past_range_T] \
                    = max_Rs[past_range_T]

                max_R_CI_hi[past_range_T] \
                    = max_Rs[past_range_T]

                # max_R_CI_med[past_range_T] \
                #     = max_Rs[past_range_T]
                

        if estimation_method == 'bbc':
            embedding_maximising_R_at_T_bbc = embedding_maximising_R_at_T.copy()
            max_Rs_bbc = max_Rs.copy()
            max_R_bbc_CI_lo = max_R_CI_lo.copy()
            max_R_bbc_CI_hi = max_R_CI_hi.copy()
            # max_R_bbc_CI_med = max_R_CI_med.copy()
        elif estimation_method == 'shuffling':
            embedding_maximising_R_at_T_shuffling = embedding_maximising_R_at_T.copy()
            max_Rs_shuffling = max_Rs.copy()
            max_R_shuffling_CI_lo = max_R_CI_lo.copy()
            max_R_shuffling_CI_hi = max_R_CI_hi.copy()
            # max_R_shuffling_CI_med = max_R_CI_med.copy()

    Ts = sorted(np.unique(np.hstack(([R for R in max_Rs_bbc],
                                      [R for R in max_Rs_shuffling]))))
    H_spiking = load_from_analysis_file(f,
                                       "H_spiking")

    for T in Ts:
        histdep_data["T"] += [get_parameter_label(T)]
        if T in max_Rs_bbc:
            number_of_bins_d = embedding_maximising_R_at_T_bbc[T][0]
            scaling_k = embedding_maximising_R_at_T_bbc[T][1]
            first_bin_size = emb.get_fist_bin_size_for_embedding((T,
                                                                  number_of_bins_d,
                                                                  scaling_k))
            
            histdep_data["max_R_bbc"] \
                += [get_parameter_label(max_Rs_bbc[T])]
            histdep_data["max_R_bbc_CI_lo"] \
                += [get_parameter_label(max_R_bbc_CI_lo[T])]
            histdep_data["max_R_bbc_CI_hi"] \
                += [get_parameter_label(max_R_bbc_CI_hi[T])]
            # histdep_data["max_R_bbc_CI_med"] \
            #     += [get_parameter_label(max_R_bbc_CI_med[T])]
            histdep_data["max_AIS_bbc"] \
                += [get_parameter_label(max_Rs_bbc[T] * H_spiking)]
            histdep_data["max_AIS_bbc_CI_lo"] \
                += [get_parameter_label(max_R_bbc_CI_lo[T] * H_spiking)]
            histdep_data["max_AIS_bbc_CI_hi"] \
                += [get_parameter_label(max_R_bbc_CI_hi[T] * H_spiking)]
            # histdep_data["max_AIS_bbc_CI_med"] \
            #     += [get_parameter_label(max_R_bbc_CI_med[T] * H_spiking)]
            histdep_data["number_of_bins_d_bbc"] \
                += [get_parameter_label(number_of_bins_d)]
            histdep_data["scaling_k_bbc"] \
                += [get_parameter_label(scaling_k)]
            histdep_data["first_bin_size_bbc"] \
                += [get_parameter_label(first_bin_size)]
        else:
            for key in histdep_data:
                if 'bbc' in key:
                    histdep_data[key] += ['-']
        if T in max_Rs_shuffling:
            number_of_bins_d = embedding_maximising_R_at_T_shuffling[T][0]
            scaling_k = embedding_maximising_R_at_T_shuffling[T][1]
            first_bin_size = emb.get_fist_bin_size_for_embedding((T,
                                                                  number_of_bins_d,
                                                                  scaling_k))
            histdep_data["max_R_shuffling"] \
                += [get_parameter_label(max_Rs_shuffling[T])]
            histdep_data["max_R_shuffling_CI_lo"] \
                += [get_parameter_label(max_R_shuffling_CI_lo[T])]
            histdep_data["max_R_shuffling_CI_hi"] \
                += [get_parameter_label(max_R_shuffling_CI_hi[T])]
            # histdep_data["max_R_shuffling_CI_med"] \
            #     += [get_parameter_label(max_R_shuffling_CI_med[T])]
            histdep_data["max_AIS_shuffling"] \
                += [get_parameter_label(max_Rs_shuffling[T] * H_spiking)]
            histdep_data["max_AIS_shuffling_CI_lo"] \
                += [get_parameter_label(max_R_shuffling_CI_lo[T] * H_spiking)]
            histdep_data["max_AIS_shuffling_CI_hi"] \
                += [get_parameter_label(max_R_shuffling_CI_hi[T] * H_spiking)]
            # histdep_data["max_AIS_shuffling_CI_med"] \
            #     += [get_parameter_label(max_R_shuffling_CI_med[T] * H_spiking)]
            histdep_data["number_of_bins_d_shuffling"] \
                += [get_parameter_label(number_of_bins_d)]
            histdep_data["scaling_k_shuffling"] \
                += [get_parameter_label(scaling_k)]
            histdep_data["first_bin_size_shuffling"] \
                += [get_parameter_label(first_bin_size)]
        else:
            for key in histdep_data:
                if 'shuffling' in key:
                    histdep_data[key] += ['-']
    return histdep_data

def get_data_index_from_CSV_header(header,
                                   data_label):
    """
    Get column index to reference data within a csv file.
    """
    
    header = header.strip()
    if header.startswith('#'):
        header = header[1:]
    labels = header.split(',')
    for index, label in enumerate(labels):
        if label == data_label:
            return index
    return np.float('nan')

def is_float(x):
  try:
    float(x)
    return True
  except ValueError:
    return False

def load_from_CSV_file(csv_file,
                       data_label):
    """
    Get all data of a column in a csv file.
    """
    
    csv_file.seek(0) # jump to start of file

    lines = (line for line in csv_file.readlines())
    header = next(lines)

    data_index = get_data_index_from_CSV_header(header,
                                                data_label)

    data = []
    for line in lines:
        datum = line.split(',')[data_index]
        if is_float(datum):
            data += [float(datum)]
        elif data_label == 'label':
            data += [datum]
        else:
            data += [np.float('nan')]

    if len(data) == 1:
        return data[0]
    else:
        return data

def load_auto_MI_data(f_csv_auto_MI_data):
    """
    Load the data from the auto MI csv file as needed for the plot.
    """
    
    auto_MI_bin_sizes = load_from_CSV_file(f_csv_auto_MI_data,
                                           "auto_MI_bin_size")
    delays = np.array(load_from_CSV_file(f_csv_auto_MI_data,
                                         "delay"))
    auto_MIs = np.array(load_from_CSV_file(f_csv_auto_MI_data,
                                           "auto_MI"))

    auto_MI_data = {}
    for auto_MI_bin_size in np.unique(auto_MI_bin_sizes):
        indices = np.where(auto_MI_bin_sizes == auto_MI_bin_size)
        auto_MI_data[auto_MI_bin_size] = ([float(delay) for delay in delays[indices]],
                                          [float(auto_MI) for auto_MI in auto_MIs[indices]])

    return auto_MI_data
    



# def get_asl_permutation_test(pt_history_dependence, R_tot):
#     """
#     Compute the permutation test statistic (the achieved significance
#     level, ASL_perm, eg. eq. 15.18 in Efron and Tibshirani: An
#     Introduction to the Bootstrap).  The ASL states the probability
#     under the null hypothesis of obtaining the given value for the
#     history dependence.  Here the null hypothesis is that there is no
#     history dependence.  Typically, if ASL < 0.05, the null hypothesis
#     is rejected, and it is infered that there was history dependence
#     in the data.
#     """
#     return sum([1 for R in pt_history_dependence
#                 if R > R_tot]) / len(pt_history_dependence)

def analyse_auto_MI(f,
                    spike_times,
                    auto_MI_bin_size_set,
                    auto_MI_max_delay,
                    **settings):
    """
    Get the auto MI for the spike times.  If it is available from file, load
    it, else compute it.
    """
    
    for auto_MI_bin_size in auto_MI_bin_size_set:
        number_of_delays = int(auto_MI_max_delay / auto_MI_bin_size) + 1
        
        auto_MI = load_from_analysis_file(f,
                                          "auto_MI",
                                          auto_MI_bin_size=auto_MI_bin_size)
        
        if isinstance(auto_MI, np.ndarray) and len(auto_MI) >= number_of_delays:
            continue

        # if no analysis found or new analysis includes more delays:
        # perform the analysis

        auto_MI = get_auto_MI(spike_times, auto_MI_bin_size, number_of_delays)

        save_to_analysis_file(f,
                              "auto_MI",
                              auto_MI_bin_size=auto_MI_bin_size,
                              auto_MI=auto_MI)


def get_auto_MI(spike_times, bin_size, number_of_delays):
    """
    Compute the auto mutual information in the neuron's activity, a 
    measure closely related to history dependence.
    """

    binned_neuron_activity = []

    for spt in spike_times:
        # represent the neural activity as an array of 0s (no spike) and 1s (spike)
        binned_neuron_activity += [get_binned_neuron_activity(spt,
                                                              bin_size,
                                                              relative_to_median_activity=True)]

    p_spike = sum([sum(bna)
                  for bna in binned_neuron_activity]) / sum([len(bna)
                                                             for bna in binned_neuron_activity])
    H_spiking = get_shannon_entropy([p_spike,
                                    1 - p_spike])

    auto_MIs = []
    
    # compute auto MI
    for delay in range(number_of_delays):

        symbol_counts = []
        for bna in binned_neuron_activity:
            number_of_symbols = len(bna) - delay - 1
            
            symbols = np.array([2 * bna[i] + bna[i + delay + 1]
                                for i in range(number_of_symbols)])

            symbol_counts += [dict([(unq_symbol, len(np.where(symbols==unq_symbol)[0]))
                                    for unq_symbol in np.unique(symbols)])]

        symbol_counts = add_up_dicts(symbol_counts)
        number_of_symbols = sum(symbol_counts.values())
        # number_of_symbols = sum([len(bna) - delay - 1 for bna in binned_neuron_activity])

        H_joint = get_shannon_entropy([number_of_occurrences / number_of_symbols
                                       for number_of_occurrences in symbol_counts.values()])

        # I(X : Y) = H(X) - H(X|Y) = H(X) - (H(X,Y) - H(Y)) = H(X) + H(Y) - H(X,Y)
        # auto_MI = 2 * H_spiking - H_joint
        auto_MI = 2 - H_joint/ H_spiking # normalized auto MI = auto MI / H_spiking

        auto_MIs += [auto_MI]

    return auto_MIs

def get_auto_MI_data(f_analysis,
                     analysis_num,
                     **kwargs):
    """
    Get auto MI values for each delay, as needed for the plots, to export 
    them to a csv file.
    """

    auto_MI_data = {
        "auto_MI_bin_size" : [],
        "delay" : [],
        "auto_MI" : []
    }

    for auto_MI_bin_size in kwargs["auto_MI_bin_size_set"]:
        auto_MIs = load_from_analysis_file(f_analysis,
                                           "auto_MI",
                                           auto_MI_bin_size=auto_MI_bin_size)

        if isinstance(auto_MIs, np.ndarray):
            for delay, auto_MI in enumerate(auto_MIs):
                    auto_MI_data["auto_MI_bin_size"] += [get_parameter_label(auto_MI_bin_size)]
                    auto_MI_data["delay"] += [get_parameter_label(delay * auto_MI_bin_size)]
                    auto_MI_data["auto_MI"] += [get_parameter_label(auto_MI)]

    return auto_MI_data
    

#
# export the data to CSV files for further processing
#

def create_CSV_files(f_analysis,
                     f_csv_stats,
                     f_csv_histdep_data,
                     f_csv_auto_MI_data,
                     analysis_num,
                     **kwargs):
    """
    Create three files per neuron (one for summary stats, one for
    detailed data for the history dependence plots and one for the
    auto mutual information plot), and write the respective data.
    """
    
    stats = get_analysis_stats(f_analysis,
                               analysis_num,
                               **kwargs)

    f_csv_stats.write("#{}\n".format(",".join(stats.keys())))
    f_csv_stats.write("{}\n".format(",".join(stats.values())))


    
    histdep_data = get_histdep_data(f_analysis,
                                    analysis_num,
                                    **kwargs)

    f_csv_histdep_data.write("#{}\n".format(",".join(histdep_data.keys())))
    histdep_data_m = np.array([vals for vals in histdep_data.values()])
    for line_num in range(np.size(histdep_data_m, axis=1)):
        f_csv_histdep_data.write("{}\n".format(",".join(histdep_data_m[:,line_num])))


    auto_MI_data = get_auto_MI_data(f_analysis,
                                    analysis_num,
                                    **kwargs)

    f_csv_auto_MI_data.write("#{}\n".format(",".join(auto_MI_data.keys())))
    auto_MI_data_m = np.array([vals for vals in auto_MI_data.values()])
    for line_num in range(np.size(auto_MI_data_m, axis=1)):
        f_csv_auto_MI_data.write("{}\n".format(",".join(auto_MI_data_m[:,line_num])))


def get_CSV_files(task,
                  persistent_analysis,
                  analysis_dir):
    """
    Create csv files for create_CSV_files, back up existing ones.
    """

    if not persistent_analysis:
        f_csv_stats = tempfile.NamedTemporaryFile(mode='w+',
                                                  suffix='.csv', prefix='statistics')
        f_csv_histdep_data = tempfile.NamedTemporaryFile(mode='w+',
                                                         suffix='.csv', prefix='histdep_data')
        f_csv_auto_MI_data = tempfile.NamedTemporaryFile(mode='w+',
                                                         suffix='.csv', prefix='auto_MI_data')
        return f_csv_stats, f_csv_histdep_data, f_csv_auto_MI_data

    csv_stats_file_name = "statistics.csv"
    csv_histdep_data_file_name = "histdep_data.csv"
    csv_auto_MI_data_file_name = "auto_MI_data.csv"

    if task == "csv-files" or task == "full-analysis":
        file_mode = 'w+'
        # backup existing files (overwrites old backups)
        for f_csv in [csv_stats_file_name,
                      csv_histdep_data_file_name,
                      csv_auto_MI_data_file_name]:
            if isfile("{}/{}".format(analysis_dir,
                                     f_csv)):
                replace("{}/{}".format(analysis_dir, f_csv),
                        "{}/{}.old".format(analysis_dir, f_csv))
    elif task == 'plots':
        file_mode = 'r'

        files_missing = False
        for f_csv in [csv_stats_file_name,
                      csv_histdep_data_file_name,
                      csv_auto_MI_data_file_name]:
            if not isfile("{}/{}".format(analysis_dir,
                                         f_csv)):
                files_missing = True
        if files_missing:
            return None, None, None
    else:
        return None, None, None
    
    f_csv_stats = open("{}/{}".format(analysis_dir,
                                      csv_stats_file_name), file_mode)
    f_csv_histdep_data = open("{}/{}".format(analysis_dir,
                                             csv_histdep_data_file_name), file_mode)
    f_csv_auto_MI_data = open("{}/{}".format(analysis_dir,
                                             csv_auto_MI_data_file_name), file_mode)
    
    return f_csv_stats, f_csv_histdep_data, f_csv_auto_MI_data
        

#
# read the spike times from file
#

def get_spike_times_from_file(file_names,
                              hdf5_datasets=None):
    """
    Get spike times from a file (either one spike time per line, or
    a dataset in a hdf5 file.).

    Ignore lines that don't represent times, sort the spikes 
    chronologically, shift spike times to start at 0 (remove any silent
    time at the beginning..).

    It is also possible to import spike times from several non-contiguous
    parts.  These can be located either from many file names or many
    hdf5 datasets within one file.  It is also possible to provide them
    from one file only by using '----------' as a delimiter.
    """

    parts_delimiter = '----------'
    spike_times_raw = []

    if not hdf5_datasets == None:
        if type(file_names) == list or type(hdf5_datasets) == list:
            if not type(file_names) == list:
                file_names = [file_names] * len(hdf5_datasets)
            if not type(hdf5_datasets) == list:
                hdf5_datasets = [hdf5_datasets] * len(file_names)
            if not len(file_names) == len(hdf5_datasets):
                print('Error. Number of hdf filenames and datasets do not match. Please provide them in a 1:n, n:1 or n:n relation.',
                      file=stderr, flush=True)
                return None
        else:
            file_names = [file_names]
            hdf5_datasets = [hdf5_datasets]

        for file_name, hdf5_dataset in zip(file_names, hdf5_datasets):
            f = h5py.File(file_name, 'r')

            if not hdf5_dataset in f:
                print("Error: Dataset {} not found in file {}.".format(hdf5_dataset,
                                                                       file_name),
                      file=stderr, flush=True)
                return None

            spike_times_part = f[hdf5_dataset][()]
            if len(spike_times_part) > 0:
                spike_times_raw += [spike_times_part]

            f.close()

    else:
        if not type(file_names) == list:
            file_names = [file_names]

        for file_name in file_names:
            spike_times_part = []

            with open(file_name, 'r') as f:
                for line in f.readlines():
                    try:
                        spike_times_part += [float(line)]
                    except:
                        if line.strip() == parts_delimiter:
                            if len(spike_times_part) > 0:
                                spike_times_raw += [spike_times_part]
                                spike_times_part = []
                        continue
            
            if len(spike_times_part) > 0:
                spike_times_raw += [spike_times_part]
                spike_times_part = []

            f.close()

    spike_times = []
    if len(spike_times_raw) > 0:
        for spike_times_part in spike_times_raw:
            spike_times += [np.array(sorted(spike_times_part)) - min(spike_times_part)]

        # if len(spike_times) == 1:
        #     return spike_times[0]
        # else:
        return np.array(spike_times)
    else:
        return np.array([])

#
# functions related to the storage and retrival to/ from
# the analysis file (in hdf5 format)
#

def get_parameter_label(parameter):
    """
    Get a number in a unified format as label for the hdf5 file. 
    """
    
    return "{:.5f}".format(parameter)

def find_existing_parameter(new_parameter, existing_parameters, tolerance=1e-5):
    """
    Search for a parameter value in a list, return label for
    the hdf5 file and whether an existing one was found.

    Tolerance should be no lower than precision in get_parameter_label.
    """
    
    new_parameter = float(new_parameter)
    if not isinstance(existing_parameters, list):
        existing_parameters = [existing_parameters]
    for existing_parameter in existing_parameters:
        if np.abs(float(existing_parameter) - new_parameter) <= tolerance:
            return existing_parameter, True
    return get_parameter_label(new_parameter), False

def get_or_create_data_directory_in_file(f,
                                         data_label,
                                         embedding_step_size=None,
                                         embedding=None,
                                         estimation_method=None,
                                         auto_MI_bin_size=None,
                                         get_only=False,
                                         cross_val=None,
                                         **kwargs):
    """
    Search for directory in hdf5, optionally create it if nonexistent 
    and return it.
    """
    
    if data_label in ["firing_rate",
                      "firing_rate_sd",
                      "H_spiking",
                      "recording_length",
                      "recording_length_sd"]:
        root_dir = "other"
    elif data_label == "auto_MI":
        root_dir = "auto_MI"
    elif not cross_val == None:
        root_dir = "{}_embeddings".format(cross_val)
    else:
        root_dir = "embeddings"

    if not root_dir in f.keys():
        if get_only:
            return None
        else:
            f.create_group(root_dir)

    data_dir = f[root_dir]
    
    if data_label in ["firing_rate",
                      "firing_rate_sd",
                      "H_spiking",
                      "recording_length",
                      "recording_length_sd"]:
        return data_dir
    elif data_label == "auto_MI":
        bin_size_label, found = find_existing_parameter(auto_MI_bin_size,
                                                        [key for key in data_dir.keys()])
        if found:
            data_dir = data_dir[bin_size_label]
        else:
            if get_only:
                return None
            else:
                data_dir = data_dir.create_group(bin_size_label)
        return data_dir
    else:
        past_range_T = embedding[0]
        number_of_bins_d = embedding[1]
        scaling_k = embedding[2]
        for parameter in [embedding_step_size,
                          past_range_T,
                          number_of_bins_d,
                          scaling_k]:
            parameter_label, found = find_existing_parameter(parameter,
                                                             [key for key in data_dir.keys()])

            if found:
                data_dir = data_dir[parameter_label]
            else:
                if get_only:
                    return None
                else:
                    data_dir = data_dir.create_group(parameter_label)

        if data_label == "symbol_counts":
            return data_dir
        else:
            if not estimation_method in data_dir:
                if get_only:
                    return None
                else:
                    data_dir.create_group(estimation_method)
            return data_dir[estimation_method]
                
def save_to_analysis_file(f,
                          data_label,
                          estimation_method=None,
                          **data):
    """
    Sava data to hdf5 file, overwrite or expand as necessary.
    """

    data_dir = get_or_create_data_directory_in_file(f,
                                                    data_label,
                                                    estimation_method=estimation_method,
                                                    **data)

    if data_label in ["firing_rate",
                      "firing_rate_sd",
                      "H_spiking",
                      "recording_length",
                      "recording_length_sd"]:
        if not data_label in data_dir:
            data_dir.create_dataset(data_label, data=data[data_label])

    # we might want to update the auto mutual information
    # so if already stored, delete it first
    elif data_label == "auto_MI":
        if not data_label in data_dir:
            data_dir.create_dataset(data_label, data=data[data_label])
        else:
            del data_dir[data_label]
            data_dir.create_dataset(data_label, data=data[data_label])
                
    elif data_label == "symbol_counts":
        if not data_label in data_dir:
            data_dir.create_dataset(data_label, data=str(dict(data[data_label])))
            
    elif data_label == "history_dependence":
        if not data_label in data_dir:
            data_dir.create_dataset(data_label, data=data[data_label])
        if estimation_method == "bbc" and not "bbc_term" in data_dir:
            data_dir.create_dataset("bbc_term", data=data["bbc_term"])
        if not "first_bin_size" in data_dir:
            data_dir.create_dataset("first_bin_size", data=data["first_bin_size"])
            
    # bs: bootstrap, pt: permutation test
    # each value is stored, so that addition re-draws can be made and
    # the median/ CIs re-computed
    elif data_label in ["bs_history_dependence",
                        "pt_history_dependence"]:
        if not data_label in data_dir:
            data_dir.create_dataset(data_label, data=data[data_label])
        else:
            new_and_old_data_joint = np.hstack((data_dir[data_label][()],
                                               data[data_label]))
            del data_dir[data_label]
            data_dir.create_dataset(data_label, data=new_and_old_data_joint)


def load_from_analysis_file(f,
                            data_label,
                            **data):
    """
    Load data from hdf5 file if it exists.
    """

    data_dir = get_or_create_data_directory_in_file(f,
                                                    data_label,
                                                    get_only=True,
                                                    **data)
    
    if data_dir == None or data_label not in data_dir:
        return None
    elif data_label == "symbol_counts":
        symbol_counts = data_dir[data_label][()]
        if type(symbol_counts) == bytes:
            return Counter(ast.literal_eval(symbol_counts.decode('utf-8')))
        else:
            return Counter(ast.literal_eval(symbol_counts))
    else:
        return data_dir[data_label][()]

def check_version(version, required_version):
    """
    Check version (of h5py module).
    """

    for i, j in zip(version.split('.'),
                    required_version.split('.')):
        try:
            i = int(i)
            j = int(j)
        except:
            print("Warning: Could not check version {} against {}".format(version,
                                                                          required_version),
                  file=stderr, flush=True)
            return False
        
        if i > j:
            return True
        elif i == j:
            continue
        elif i < j:
            return False
    return True

def get_hash(spike_times):
    """
    Get hash representing the spike times, for bookkeeping.
    """

    if len(spike_times) == 1:
        m = hashlib.sha256()
        m.update(str(spike_times[0]).encode('utf-8'))
        return m.hexdigest()
    else:
        ms = []
        for spt in spike_times:
            m = hashlib.sha256()
            m.update(str(spt).encode('utf-8'))
            ms += [m.hexdigest()]
        m = hashlib.sha256()
        m.update(str(sorted(ms)).encode('utf-8'))
        return m.hexdigest()


def Gile(persistent_analysis, analysis_dir):
    """
    Get the hdf5 file to store the analysis in (either
    temporarily or persistently.)
    """
    
    analysis_file_name = "analysis_data.h5"
    
    if not persistent_analysis:
        if check_version(h5py.__version__, '2.9.0'):
            return h5py.File(io.BytesIO(), 'a')
        else:
            tf = tempfile.NamedTemporaryFile(suffix='.h5', prefix='analysis_')
            return h5py.File(tf.name, 'a')

    analysis_file = h5py.File("{}/{}".format(analysis_dir,
                                             analysis_file_name), 'a')
        
    return analysis_file

def get_or_create_analysis_dir(spike_times,
                               spike_times_file_names,
                               root_analysis_dir):
    """
    Search for existing folder containing associated analysis.
    """

    analysis_num = -1
    analysis_dir_prefix = 'ANALYSIS'
    prefix_len = len(analysis_dir_prefix)
    analysis_id_file_name = '.associated_spike_times_file'
    analysis_id = {'path': '\n'.join([abspath(spike_times_file_name).strip()
                                      for spike_times_file_name in
                                      spike_times_file_names]),
                   'hash': get_hash(spike_times)}
    existing_analysis_found = False
    
    for d in sorted(listdir(root_analysis_dir)):
        if not d.startswith(analysis_dir_prefix):
            continue

        try:
            analysis_num = int(d[prefix_len:])
        except:
            continue

        if isfile("{}/{}/{}".format(root_analysis_dir,
                                    d,
                                    analysis_id_file_name)):

            with open("{}/{}/{}".format(root_analysis_dir,
                                        d,
                                        analysis_id_file_name), 'r') as analysis_id_file:
                lines = analysis_id_file.readlines()
                for line in lines:
                    if line.strip() == analysis_id['hash']:
                        existing_analysis_found = True
            analysis_id_file.close()
        else:
            continue

        if existing_analysis_found:
            break

    if not existing_analysis_found:
        analysis_num += 1

    # if several dirs are attempted to be created in parallel
    # this might create a race condition -> test for success
    successful = False

    while not successful:
        analysis_num_label = str(analysis_num)
        if len(analysis_num_label) < 4:
            analysis_num_label = (4 - len(analysis_num_label)) * "0" + analysis_num_label

        analysis_dir = "{}/ANALYSIS{}".format(root_analysis_dir, analysis_num_label)

        if not existing_analysis_found:
            try:
                mkdir(analysis_dir)
            except:
                analysis_num += 1
                continue
            with open("{}/{}".format(analysis_dir,
                                     analysis_id_file_name), 'w') as analysis_id_file:
                analysis_id_file.write("{}\n{}\n".format(analysis_id['path'],
                                                         analysis_id['hash']))
            analysis_id_file.close()
            successful = True
        else:
            successful = True

    return analysis_dir, analysis_num, existing_analysis_found
