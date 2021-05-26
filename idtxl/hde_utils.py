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
#import idtxl.hde_embedding as emb
#from idtxl.hde_estimators import hde_bbc_estimator as bbc
#from idtxl.hde_estimators import hde_shuffling_estimator as sh

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


def get_analysis_file(persistent_analysis, analysis_dir):
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
