"""Provide HDE estimators."""

import sys
import logging
from pkg_resources import resource_filename
from scipy.special import digamma
import numpy as np
from scipy.optimize import newton, minimize
import types
from os.path import isfile, isdir, realpath, dirname, exists
import ast
from sys import exit, stderr, argv, path
from idtxl.estimator import Estimator
import idtxl.hde_utils as utl
from . import idtxl_exceptions as ex
from collections import Counter
import mpmath as mp
# import multiprocessing

FAST_EMBEDDING_AVAILABLE = True
try:
    import idtxl.hde_fast_embedding as fast_emb
except:
    FAST_EMBEDDING_AVAILABLE = False
    print("""
        Error importing Cython fast embedding module. Continuing with slow Python implementation.\n
        This may take a long time.\n
        """, file=stderr, flush=True)


logger = logging.getLogger(__name__)

class hdAbstractEstimator(Estimator):

    """
    ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

    """

    def __init__(self, settings=None):

        # Get defaults for estimator settings
        settings = self._check_settings(settings)
        self.settings = settings.copy()
        #self.settings.setdefault('noise_level', np.float32(1e-8))
        self.settings.setdefault('debug', False)

        self.settings.setdefault('task', 'full-analysis')
        self.settings.setdefault('embedding_step_size', 0.005)
        self.settings.setdefault('embedding_past_range_set', [float("{:.5f}".format(np.exp(x))) for x in
                                                              np.arange(np.log(0.005), np.log(5.001),
                                                                        0.05 * np.log(10))])
        self.settings.setdefault('embedding_number_of_bins_set', [int(x) for x in np.linspace(1, 5, 5)])
        self.settings.setdefault('embedding_scaling_exponent_set', {'number_of_scalings': 10,
                                                                    'min_first_bin_size': 0.005,
                                                                    'min_step_for_scaling': 0.01})
        self.settings.setdefault('cross_validated_optimization', False)
        self.settings.setdefault('return_averaged_R', True)
        self.settings.setdefault('timescale_minimum_past_range', 0.01)
        self.settings.setdefault('number_of_bootstraps_R_max', 250)
        self.settings.setdefault('number_of_bootstraps_R_tot', 250)
        self.settings.setdefault('number_of_bootstraps_nonessential', 0)
        self.settings.setdefault('block_length_l', None)
        self.settings.setdefault('bootstrap_CI_use_sd', True)
        self.settings.setdefault('bootstrap_CI_percentile_lo', 2.5)
        self.settings.setdefault('bootstrap_CI_percentile_hi', 97.5)
        # self.settings.setdefault('number_of_permutations', 100)
        self.settings.setdefault('auto_MI_bin_size_set', [0.005, 0.01, 0.025, 0.05, 0.25, 0.5])
        self.settings.setdefault('auto_MI_max_delay', 5)

        ## ---------------------------------------------------------------------------------------------------------------------- to remove
        #self.settings.setdefault('label', '""')
        #self.settings.setdefault('ANALYSIS_DIR', "./analysis")
        #self.settings.setdefault('analysis_file_name', None)
        #self.settings.setdefault('persistent_analysis', True)
        #self.settings.setdefault('verbose_output', False)
        #self.settings.setdefault('createPDF', False)
        #self.settings.setdefault('plot_AIS', False)
        #self.settings.setdefault('output_image', None)
        #self.settings.setdefault('plot_settings', {'figure.figsize': [6.3, 5.5],
        #                                           'axes.labelsize': 9,
        #                                            'font.size': 9,
        #                                            'legend.fontsize': 8,
        #                                            'xtick.labelsize': 8,
        #                                            'ytick.labelsize': 8,
        #                                            'savefig.format': 'pdf'})
        #self.settings.setdefault('plot_color', '#4da2e2')
        ## ---------------------------------------------------------------------------------------------------------------------- to remove


        #if self.settings['return_counts'] and not self.settings['debug']:
        #    raise RuntimeError(
        #        'Set debug option to True to return neighbor counts.'

    def is_parallel(self):
        return False

    def is_analytic_null_estimator(self):
        return False

    def get_set_of_scalings(self, past_range_T,
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

            max_scaling = newton(lambda scaling: self.get_past_range(number_of_bins_d,
                                                                min_first_bin_size,
                                                                scaling)
                                                 - past_range_T,
                                 np.log10(past_range_T
                                          / min_first_bin_size) / (number_of_bins_d - 1),
                                 tol=1e-04, maxiter=500)

        while np.linspace(min_scaling, max_scaling,
                          number_of_scalings, retstep=True)[1] < min_step_for_scaling:
            number_of_scalings -= 1

        return np.linspace(min_scaling, max_scaling, number_of_scalings)

    def get_embeddings(self,
                       embedding_past_range_set,
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
                    scaling_set_given_T_and_d = self.get_set_of_scalings(past_range_T,
                                                                    number_of_bins_d,
                                                                    **embedding_scaling_exponent_set)
                else:
                    scaling_set_given_T_and_d = embedding_scaling_exponent_set

                for scaling_k in scaling_set_given_T_and_d:
                    embeddings += [(past_range_T, number_of_bins_d, scaling_k)]

        return embeddings

    def get_history_dependence(self,
                               symbol_counts,
                               number_of_bins_d,
                               past_symbol_counts=None,
                               bbc_tolerance=None,
                               H_uncond=None,
                               return_ais=False,
                               **kwargs):
        """
        Get history dependence for binary random variable that takes
        into account outcomes with dimension d into the past, and dim 1
        at response, based on symbol counts.

        If no past_symbol_counts are provided, uses representation for
        symbols as given by emb.symbol_array_to_binary to obtain them.
        """

        # if no (unconditional) entropy of the response is provided,
        # assume it is a one-dimensional binary outcome (as in
        # a spike train) and compute it based on that assumption
        if H_uncond is None:
            H_uncond = utl.get_H_spiking(symbol_counts)

        if past_symbol_counts is None:
            past_symbol_counts = utl.get_past_symbol_counts(symbol_counts)

        alphabet_size_past = 2 ** int(number_of_bins_d)  # K for past activity
        alphabet_size = alphabet_size_past * 2  # K

        if self.settings['estimation_method'] == "bbc":
            return self.bbc_estimator(symbol_counts,
                                     past_symbol_counts,
                                     alphabet_size,
                                     alphabet_size_past,
                                     H_uncond,
                                     bbc_tolerance=bbc_tolerance,
                                     return_ais=return_ais)

        elif self.settings['estimation_method'] == "shuffling":
            return self.shuffling_estimator(symbol_counts,
                                          number_of_bins_d,
                                          H_uncond,
                                          return_ais=return_ais)

    def get_first_bin_size_for_embedding(self, embedding):
        """
        Get size of first bin for the embedding, based on the parameters
        T, d and k.
        """

        past_range_T, number_of_bins_d, scaling_k = embedding
        return newton(lambda first_bin_size: self.get_past_range(number_of_bins_d,
                                                            first_bin_size,
                                                            scaling_k) - past_range_T,
                      0.005, tol=1e-03, maxiter=100)

    def get_past_range(self, number_of_bins_d, first_bin_size, scaling_k):
        """
        Get the past range T of the embedding, based on the parameters d, tau_1 and k.
        """

        return np.sum([first_bin_size * 10 ** ((number_of_bins_d - i) * scaling_k)
                       for i in range(1, number_of_bins_d + 1)])

    def get_window_delimiters(self, number_of_bins_d, scaling_k, first_bin_size):
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
        window_delimiters.append(window_delimiters[number_of_bins_d - 1] + self.settings['embedding_step_size'])
        return window_delimiters

    def get_median_number_of_spikes_per_bin(self, raw_symbols):
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

    def symbol_binary_to_array(self, symbol_binary, number_of_bins_d):
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

    def symbol_array_to_binary(self, spikes_in_window, number_of_bins_d):
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

    def get_raw_symbols(self,
                        spike_times,
                        embedding,
                        first_bin_size):
        """
        Get the raw symbols (in which the number of spikes per bin are counted,
        ie not necessarily binary quantity), as obtained by applying the
        embedding.
        """

        past_range_T, number_of_bins_d, scaling_k = embedding

        # the window is the embedding plus the response,
        # ie the embedding and one additional bin of size embedding_step_size
        window_delimiters = self.get_window_delimiters(number_of_bins_d,
                                                       scaling_k,
                                                       first_bin_size)
        window_length = window_delimiters[-1]
        num_spike_times = len(spike_times)
        last_spike_time = spike_times[-1]

        num_symbols = int((last_spike_time - window_length) / self.settings['embedding_step_size'])

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
                while (spike_times[spike_index] > time + window_delimiters[embedding_bin_index]):
                    embedding_bin_index += 1
                spikes_in_window[embedding_bin_index] += 1

            raw_symbols += [spikes_in_window]

            time += self.settings['embedding_step_size']

        return raw_symbols

    def get_symbol_counts(self, spike_times, embedding, embedding_step_size):
        """
        Apply embedding to the spike times to obtain the symbol counts.
        """

        if FAST_EMBEDDING_AVAILABLE:
            return Counter(fast_emb.get_symbol_counts(spike_times, embedding, embedding_step_size))

        past_range_T, number_of_bins_d, scaling_k = embedding
        first_bin_size = self.get_first_bin_size_for_embedding(embedding)

        raw_symbols = self.get_raw_symbols(spike_times,
                                      embedding,
                                      first_bin_size)

        median_number_of_spikes_per_bin = self.get_median_number_of_spikes_per_bin(raw_symbols)

        symbol_counts = Counter()

        for raw_symbol in raw_symbols:
            symbol_array = [int(raw_symbol[i] > median_number_of_spikes_per_bin[i])
                            for i in range(number_of_bins_d + 1)]

            symbol = self.symbol_array_to_binary(symbol_array, number_of_bins_d + 1)

            symbol_counts[symbol] += 1

        return symbol_counts

    def analyse_auto_MI(self, spike_times):
        """
        Get the auto MI for the spike times.  If it is available from file, load
        it, else compute it.
        """

        auto_MI_data = {
            "auto_MI_bin_size": [],
            "delay": [],
            "auto_MI": []
        }
        auto_MI_dict = {}
        for auto_MI_bin_size in self.settings['auto_MI_bin_size_set']:
            number_of_delays = int(self.settings['auto_MI_max_delay'] / auto_MI_bin_size) + 1

            # auto_MI = self.settings['auto_MI']
            # if isinstance(auto_MI, np.ndarray) and len(auto_MI) >= number_of_delays:
            #    continue

            # if no analysis found or new analysis includes more delays:
            # perform the analysis
            auto_MI = self.get_auto_MI(spike_times, auto_MI_bin_size, number_of_delays)

            auto_MI_data["auto_MI_bin_size"] += [str(auto_MI_bin_size)]
            auto_MI_data["delay"] += [str(number_of_delays)]
            auto_MI_dict[str(auto_MI_bin_size)] = auto_MI

        auto_MI_data['auto_MI'] = auto_MI_dict
        self.settings['auto_MI'] = auto_MI_data

    def get_auto_MI(self, spike_times, bin_size, number_of_delays):
        """
        Compute the auto mutual information in the neuron's activity, a
        measure closely related to history dependence.
        """

        binned_neuron_activity = []

        for spt in spike_times:
            # represent the neural activity as an array of 0s (no spike) and 1s (spike)
            binned_neuron_activity += [utl.get_binned_neuron_activity(spt,
                                                                      bin_size,
                                                                      relative_to_median_activity=True)]

        p_spike = sum([sum(bna)
                       for bna in binned_neuron_activity]) / sum([len(bna)
                                                                  for bna in binned_neuron_activity])
        H_spiking = utl.get_shannon_entropy([p_spike,
                                         1 - p_spike])

        auto_MIs = []

        # compute auto MI
        for delay in range(number_of_delays):

            symbol_counts = []
            for bna in binned_neuron_activity:
                number_of_symbols = len(bna) - delay - 1

                symbols = np.array([2 * bna[i] + bna[i + delay + 1]
                                    for i in range(number_of_symbols)])

                symbol_counts += [dict([(unq_symbol, len(np.where(symbols == unq_symbol)[0]))
                                        for unq_symbol in np.unique(symbols)])]

            symbol_counts = utl.add_up_dicts(symbol_counts)
            number_of_symbols = sum(symbol_counts.values())
            # number_of_symbols = sum([len(bna) - delay - 1 for bna in binned_neuron_activity])

            H_joint = utl.get_shannon_entropy([number_of_occurrences / number_of_symbols
                                           for number_of_occurrences in symbol_counts.values()])

            # I(X : Y) = H(X) - H(X|Y) = H(X) - (H(X,Y) - H(Y)) = H(X) + H(Y) - H(X,Y)
            # auto_MI = 2 * H_spiking - H_joint
            auto_MI = 2 - H_joint / H_spiking  # normalized auto MI = auto MI / H_spiking

            auto_MIs += [auto_MI]

        return auto_MIs

    def get_information_timescale_tau_R(self):
        """
        Get the information timescale tau_R, a characteristic
        timescale of history dependence similar to an autocorrelation
        time.
        """

        max_Rs = self.settings['max_Rs']

        Ts = np.array(sorted([key for key in max_Rs.keys()]))
        Rs = np.array([max_Rs[T] for T in Ts])

        R_tot = self.get_R_tot()

        T_0 = self.settings["timescale_minimum_past_range"]

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

    def get_R_tot(self,
                  return_averaged_R=False,
                  **kwargs):

        max_Rs = self.settings['max_Rs']

        if return_averaged_R:
            T_D, R_tot_thresh = self.get_temporal_depth_T_D(get_R_thresh=True)

            Ts = sorted([key for key in max_Rs.keys()])
            Rs = [max_Rs[T] for T in Ts]

            T_max = T_D
            for R, T in zip(Rs, Ts):
                if T < T_D:
                    continue
                T_max = T
                if R < R_tot_thresh:
                    break

            return np.average([R for R, T in zip(Rs, Ts) if T_D <= T < T_max])

        else:

            temporal_depth_T_D = self.get_temporal_depth_T_D()

            return max_Rs[temporal_depth_T_D]

    def _check_input(self, data):

        EXIT_SUCCESS = 0
        EXIT_FAILURE = 1

        if self.settings['debug']:
            print("Check inputs")

        # check that required settings are defined
        required_parameters = ['embedding_past_range_set', 'embedding_number_of_bins_set',
                               'embedding_scaling_exponent_set', 'embedding_step_size',
                               'bbc_tolerance', 'timescale_minimum_past_range',
                               'number_of_bootstraps_R_max', 'number_of_bootstraps_R_tot',
                               'number_of_bootstraps_nonessential',
                               'block_length_l',
                               'bootstrap_CI_percentile_lo',
                               'bootstrap_CI_percentile_hi',
                               # 'number_of_permutations',
                               'auto_MI_bin_size_set',
                               'auto_MI_max_delay']

        required_settings = ['cross_validated_optimization',
                             'return_averaged_R',
                             'bootstrap_CI_use_sd'] + required_parameters

        # check if all settings are defined
        for required_setting in required_settings:
            if not required_setting in self.settings:
                print("Error in settings file: {} is not defined. Aborting.".format(required_setting),
                      file=stderr, flush=True)
                exit(EXIT_FAILURE)

        # evaluate settings (turn strings into booleans etc if applicable)
        for setting_key in ['cross_validated_optimization',
                            'return_averaged_R',
                            'bootstrap_CI_use_sd']:
            try:
                if not type(self.settings[setting_key]) == bool:
                    self.settings[setting_key] = ast.literal_eval(self.settings[setting_key])
            except:
                print("Error: setting {0} needs to be boolean but is defined as {1}. Aborting.".format(setting_key, self.settings[setting_key]),
                      file=stderr, flush=True)
                exit(EXIT_FAILURE)

        for parameter_key in required_parameters:
            if isinstance(self.settings[parameter_key], list):
                # self.settings[parameter_key] = [ast.literal_eval(element) for element in self.settings[parameter_key]]
                try:
                    all(isinstance(x, float) for x in self.settings[parameter_key])
                except:
                    print("Error: list in {} must contain only float values! Aborted.".format(self.settings[parameter_key]))
                    exit(EXIT_FAILURE)
            elif parameter_key == 'embedding_scaling_exponent_set' \
                    and isinstance(self.settings['embedding_scaling_exponent_set'], dict):
                # embedding_scaling_exponent_set can be passed either as a
                # list, in which case it is evaluated as such or it can be
                # passed by specifying three parameters that determine how
                # many scaling exponents should be used.  In the latter case, the
                # uniform embedding as well as the embedding for which
                # the first bin has a length of min_first_bin_size (in
                # seconds) are used, as well as linearly spaced scaling
                # factors in between, such that in total
                # number_of_scalings scalings are used

                for key in self.settings['embedding_scaling_exponent_set']:
                    if not isinstance(self.settings['embedding_scaling_exponent_set'][key], float):
                        if not isinstance(self.settings['embedding_scaling_exponent_set'][key], int):
                            print("Error: {} must contain only float or integer values! Aborted.".format(
                                self.settings['embedding_scaling_exponent_set']))
                            exit(EXIT_FAILURE)
            else:
                # self.settings[parameter_key] = ast.literal_eval(self.settings[parameter_key])       ??????????????????????????????????????????????????
                a=1  # ------------------------------------------------------------------------------------------------------to remove / for debugging

        # Cython implementation uses 64bit unsigned integers for the symbols,
        # we allow up to 62 bins (window has 1 bin more..)
        if max(self.settings['embedding_number_of_bins_set']) > 62:
            print("Error: Max number of bins too large; use less than 63. Aborting.",
                  file=stderr, flush=True)
            exit(EXIT_FAILURE)

        # If R_tot is computed as an average over Rs, no confidence interval can be estimated
        if self.settings['return_averaged_R']:
            self.settings['number_of_bootstraps_R_tot'] = 0

        """
        if 'parallel' in self.settings:
            if 'numberCPUs' not in self.settings:
                self.settings['numberCPUs'] = multiprocessing.cpu_count()
        else:
            self.settings['parallel'] = False
        """
    # end of def _check_input

    def get_symbols_array(self, spike_times, embedding, embedding_step_size):
        """
        Apply an embedding to a spike train and get the resulting symbols.
        """

        past_range_T, number_of_bins_d, scaling_k = embedding
        first_bin_size = self.get_first_bin_size_for_embedding(embedding)

        raw_symbols = self.get_raw_symbols(spike_times,
                                          embedding,
                                          first_bin_size)

        median_number_of_spikes_per_bin = self.get_median_number_of_spikes_per_bin(raw_symbols)

        # symbols_array: array containing symbols
        # symbol_array: array of spikes representing symbol
        symbols_array = np.zeros(len(raw_symbols))

        for symbol_index, raw_symbol in enumerate(raw_symbols):
            symbol_array = [int(raw_symbol[i] > median_number_of_spikes_per_bin[i])
                            for i in range(number_of_bins_d + 1)]

            symbol = self.symbol_array_to_binary(symbol_array, number_of_bins_d + 1)

            symbols_array[symbol_index] = symbol

        return symbols_array

    def get_temporal_depth_T_D(self, get_R_thresh=False,
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
            = self.get_embeddings_that_maximise_R()

        Ts = sorted([key for key in max_Rs.keys()])
        Rs = [max_Rs[T] for T in Ts]

        # first get the max history dependence, and if available its bootstrap replications
        max_R, max_R_T = utl.get_max_R_T(max_Rs)

        # number_of_bins_d, scaling_k = embedding_maximising_R_at_T[max_R_T]

        bs_Rs = self.settings['bs_history_dependence']

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

    def get_embeddings_that_maximise_R(self,
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

        if bbc_tolerance is None \
                or cross_val == "h2":  # apply bbc only for optimization
            bbc_tolerance = np.inf

        if cross_val is None:
            root_dir = 'embeddings'
        else:
            root_dir = '{}_embeddings'.format(cross_val)

        max_Rs = {}
        embeddings_that_maximise_R = {}

        for i in range(len(self.settings['embeddings'])):
            embedding = self.settings['embeddings'][i]
            past_range_T = float(embedding[0])
            number_of_bins_d = int(float(embedding[1]))
            scaling_k = float(embedding[2])
            history_dependence = self.settings['history_dependence'][i]

            if self.settings['estimation_method'] == "bbc":
                if self.settings['bbc_term'] >= self.settings['bbc_tolerance']:
                    continue

            if dependent_var == "T":
                if not past_range_T in embeddings_that_maximise_R \
                                or history_dependence > max_Rs[past_range_T]:
                    max_Rs[past_range_T] = history_dependence
                    embeddings_that_maximise_R[past_range_T] = (number_of_bins_d, scaling_k)
            elif dependent_var == "d":
                if not number_of_bins_d in embeddings_that_maximise_R \
                                or history_dependence > max_Rs[number_of_bins_d]:
                    max_Rs[number_of_bins_d] = history_dependence
                    embeddings_that_maximise_R[number_of_bins_d] = (past_range_T, scaling_k)

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

    def get_history_dependence_for_embeddings_bbc(self, data):
        """
        Apply embeddings to spike times to obtain symbol counts.  Estimate
        the history dependence for each embedding.  Save results to file.
        """
        if self.settings['cross_val'] is None or self.settings['cross_val'] == 'h1':
            embeddings = self.get_embeddings(self.settings['embedding_past_range_set'],
                                             self.settings['embedding_number_of_bins_set'],
                                             self.settings['embedding_scaling_exponent_set'])
        elif self.settings['cross_val'] == 'h2':
            # here we set cross_val to h1, because we load the
            # embeddings that maximise R from the optimisation step
            embeddings = self.get_embeddings_that_maximise_R(bbc_tolerance=self.settings['bbc_tolerance'],
                                                             get_as_list=True,
                                                             cross_val='h1')

        history_dependence = np.empty(len(embeddings))
        history_dependence = []
        bbc_term = []
        count = 0
        for embedding in embeddings:
            past_range_T = embedding[0]
            number_of_bins_d = embedding[1]
            first_bin_size = self.get_first_bin_size_for_embedding(embedding)

            symbol_counts = utl.add_up_dicts([self.get_symbol_counts(spt,
                                                                     embedding,
                                                                     self.settings['embedding_step_size'])
                                              for spt in data])

            history_dependence, bbc_term = self.get_history_dependence(symbol_counts, number_of_bins_d)

            history_dependence[count] = history_dependence
            bbc_term[count] = bbc_term

            count += 1

        self.settings['embeddings'] = embeddings
        self.settings['history_dependence'] = history_dependence

        return history_dependence, bbc_term


    def get_history_dependence_for_embeddings_shuffling(self, data):
        """
        Apply embeddings to spike times to obtain symbol counts.  Estimate
        the history dependence for each embedding.  Save results to file.
        """

        if self.settings['cross_val'] is None or self.settings['cross_val'] == 'h1':
            embeddings = self.get_embeddings(self.settings['embedding_past_range_set'],
                                             self.settings['embedding_number_of_bins_set'],
                                             self.settings['embedding_scaling_exponent_set'])
        elif self.settings['cross_val'] == 'h2':
            # here we set cross_val to h1, because we load the
            # embeddings that maximise R from the optimisation step
            embeddings = self.get_embeddings_that_maximise_R(bbc_tolerance=self.settings['bbc_tolerance'],
                                                             get_as_list=True,
                                                             cross_val='h1')

        history_dependence = np.empty(len(embeddings))
        count = 0
        for embedding in embeddings:
            past_range_T = embedding[0]
            number_of_bins_d = embedding[1]
            first_bin_size = self.get_first_bin_size_for_embedding(embedding)
            symbol_counts = utl.add_up_dicts([self.get_symbol_counts(spt,
                                                                 embedding,
                                                                 self.settings['embedding_step_size'])
                                              for spt in data])

            history_dependence[count] = self.get_history_dependence(symbol_counts, number_of_bins_d)

            count += 1

        self.settings['embeddings'] = embeddings
        self.settings['history_dependence'] = history_dependence

        return history_dependence

    def get_bootstrap_history_dependence(self,
                                         spike_times,
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
            = [self.get_symbols_array(spt, embedding, self.settings['embedding_step_size'])
               for spt in spike_times]

        for rep in range(number_of_bootstraps):
            bs_symbol_counts \
                = utl.add_up_dicts([self.get_bootstrap_symbol_counts_from_symbols_array(symbols_array[i],
                                                                                        symbol_block_length)
                                for i in range(len(symbols_array))])

            bs_history_dependence = self.get_history_dependence(bs_symbol_counts,
                                                                number_of_bins_d,
                                                                bbc_tolerance=np.inf)

            bs_Rs[rep] = bs_history_dependence

        return bs_Rs

    def get_bootstrap_symbol_counts_from_symbols_array(self,
                                                       symbols_array,
                                                       symbol_block_length):
        """
        Given an array of symbols (cf get_symbols_array), get bootstrap
        replications of the symbol counts.
        """

        num_symbols = len(symbols_array)

        rand_indices = np.random.randint(0, num_symbols - (symbol_block_length - 1),
                                         size=int(num_symbols / symbol_block_length))

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

    def compute_CIs(self,
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

        number_of_bootstraps = self.settings['number_of_bootstraps_{}'.format(target_R)]

        if number_of_bootstraps == 0:
            return

        embedding_maximising_R_at_T, max_Rs \
            = self.get_embeddings_that_maximise_R()
        self.settings['embedding_maximising_R_at_T'] = embedding_maximising_R_at_T
        self.settings['max_Rs'] = max_Rs

        if block_length_l is None:
            # eg firing rate is 4 Hz, ie there is 1 spikes per 1/4 seconds,
            # for every second the number of symbols is 1/ embedding_step_size
            # so we observe on average one spike every 1 / (firing_rate * embedding_step_size) symbols
            # (in the reponse, ignoring the past activity)
            block_length_l = max(1, int(1 / (self.settings['firing_rate'] * self.settings['embedding_step_size'])))

        if target_R == 'nonessential':
            # bootstrap R for unessential Ts (not required for the main analysis)
            embeddings = []

            for past_range_T in embedding_maximising_R_at_T:
                number_of_bins_d, scaling_k = embedding_maximising_R_at_T[past_range_T]
                embeddings += [(past_range_T, number_of_bins_d, scaling_k)]

        elif target_R == 'R_max':
            # bootstrap R for the max R, to get a good estimate for the standard deviation
            # which is used to determine R_tot
            max_R, max_R_T = utl.get_max_R_T(max_Rs)
            self.settings['max_R'] = max_R
            self.settings['max_R_T'] = max_R_T
            number_of_bins_d, scaling_k = embedding_maximising_R_at_T[max_R_T]

            embeddings = [(max_R_T, number_of_bins_d, scaling_k)]
        elif target_R == 'R_tot':
            T_D = self.get_temporal_depth_T_D()
            self.settings['T_D'] = T_D
            number_of_bins_d, scaling_k = embedding_maximising_R_at_T[T_D]

            embeddings = [(T_D, number_of_bins_d, scaling_k)]

        for embedding in embeddings:

            if 'bs_history_dependence' in self.settings:
                stored_bs_Rs = self.settings['bs_history_dependence']
            else:
                stored_bs_Rs = None

            if isinstance(stored_bs_Rs, np.ndarray):
                number_of_stored_bootstraps = len(stored_bs_Rs)
            else:
                number_of_stored_bootstraps = 0

            if not number_of_bootstraps > number_of_stored_bootstraps:
                continue

            self.settings['bs_history_dependence'] \
                = self.get_bootstrap_history_dependence(spike_times,
                                                   embedding,
                                                   embedding_step_size,
                                                   estimation_method,
                                                   number_of_bootstraps - number_of_stored_bootstraps,
                                                   block_length_l)

            a=1 # ------------------------------------------------------------------------------------------------------to remove / for debugging


    def get_spike_times_stats(self,
                              spike_times,
                              embedding_step_size):
        """
        Save some statistics about the spike times.
        """

        self.settings['recording_lengths'] = [spt[-1] - spt[0] for spt in spike_times]
        self.settings['recording_length'] = sum(self.settings['recording_lengths'])
        self.settings['recording_length_sd'] = np.std(self.settings['recording_lengths'])

        firing_rates = [utl.get_binned_firing_rate(spt, embedding_step_size)
                        for spt in spike_times]

        self.settings['firing_rate'] = np.average(firing_rates, weights=self.settings['recording_lengths'])
        self.settings['firing_rate_sd'] = np.sqrt(np.average((firing_rates - self.settings['firing_rate']) ** 2,
                                                             weights=self.settings['recording_lengths']))

        self.settings['H_spiking'] = utl.get_shannon_entropy([self.settings['firing_rate'] * embedding_step_size,
                                                          1 - self.settings['firing_rate'] * embedding_step_size])

class hdEstimatorBBC(hdAbstractEstimator):
    """BBC History dependence estimator

    ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

    implemented in idtxl by Michael Lindner, Göttingen 2021


    settings??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    """

    """
    Estimate the history dependence and temporal depth of a single
    neuron, based on information-theoretical measures for spike time
    data

    References:

        [1]: L. Rudelt, D. G. Marx, M. Wibral, V. Priesemann: Embedding
            optimization reveals long-lasting history dependence in
            neural spiking activity (in prep.)

        [2]: https://github.com/Priesemann-Group/hdestimator



    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)
        self.settings.setdefault('estimation_method', "bbc")
        self.settings.setdefault('bbc_tolerance', 0.05)

    def d_xi(self, beta, K):
        """
        First derivative of xi(beta).

        xi(beta) is the entropy of the system when no data has been observed.
        d_xi is the prior for the nsb estimator
        """

        return K * mp.psi(1, K * beta + 1.) - mp.psi(1, beta + 1.)

    def d2_xi(self, beta, K):
        """
        Second derivative of xi(beta) (cf d_xi).
        """

        return K ** 2 * mp.psi(2, K * beta + 1) - mp.psi(2, beta + 1)

    def d3_xi(self, beta, K):
        """
        Third derivative of xi(beta) (cf d_xi).
        """

        return K ** 3 * mp.psi(3, K * beta + 1) - mp.psi(3, beta + 1)

    def rho(self, beta, mk, K, N):
        """
        rho(beta, data) is the Dirichlet multinomial likelihood.

        rho(beta, data) together with the d_xi(beta) make up
        the posterior for the nsb estimator
        """

        return np.prod([mp.power(mp.rf(beta, np.double(n)), mk[n]) for n in mk]) / mp.rf(K * beta,
                                                                                         np.double(N))

    def unnormalized_posterior(self, beta, mk, K, N):
        """
        The (unnormalized) posterior in the nsb estimator.

        Product of the likelihood rho and the prior d_xi;
        the normalizing factor is given by the marginal likelihood
        """

        return self.rho(beta, mk, K, N) * self.d_xi(beta, K)

    def d_log_rho(self, beta, mk, K, N):
        """
        First derivate of the logarithm of the Dirichlet multinomial likelihood.
        """

        return K * (mp.psi(0, K * beta) - mp.psi(0, K * beta + N)) - K * mp.psi(0, beta) \
               + np.sum((mk[n] * mp.psi(0, n + beta) for n in mk))

    def d2_log_rho(self, beta, mk, K, N):
        """
        Second derivate of the logarithm of the Dirichlet multinomial likelihood.
        """

        return K ** 2 * (mp.psi(1, K * beta) - mp.psi(1, K * beta + N)) - K * mp.psi(1, beta) \
               + np.sum((mk[n] * mp.psi(1, n + beta) for n in mk))

    def d_log_rho_xi(self, beta, mk, K, N):
        """
        First derivative of the logarithm of the nsb (unnormalized) posterior.
        """

        return self.d_log_rho(beta, mk, K, N) + self.d2_xi(beta, K) / self.d_xi(beta, K)

    def d2_log_rho_xi(self, beta, mk, K, N):
        """
        Second derivative of the logarithm of the nsb (unnormalized) posterior.
        """

        return self.d2_log_rho(beta, mk, K, N) \
               + (self.d3_xi(beta, K) * self.d_xi(beta, K) - self.d2_xi(beta, K) ** 2) / self.d_xi(beta, K) ** 2

    def log_likelihood_DP_alpha(self, a, K1, N):
        """
        Alpha-dependent terms of the log-likelihood of a Dirichlet Process.
        """

        return (K1 - 1.) * mp.log(a) - mp.log(mp.rf(a + 1., N - 1.))

    def get_beta_MAP(self, mk, K, N):
        """
        Get the maximum a posteriori (MAP) value for beta.

        Provides the location of the peak, around which we integrate.

        beta_MAP is the value for beta for which the posterior of the
        NSB estimator is maximised (or, equivalently, of the logarithm
        thereof, as computed here).
        """

        K1 = K - mk[0]

        if self.d_log_rho(10 ** 1, mk, K, N) > 0:
            print("Warning: No ML parameter was found.", file=stderr, flush=True)
            beta_MAP = np.float('nan')
        else:
            try:
                # first guess computed via posterior of Dirichlet process
                DP_est = self.alpha_ML(mk, K1, N) / K
                beta_MAP = newton(lambda beta: float(self.d_log_rho_xi(beta, mk, K, N)), DP_est,
                                  lambda beta: float(self.d2_log_rho_xi(beta, mk, K, N)),
                                  tol=5e-08, maxiter=500)
            except:
                print("Warning: No ML parameter was found. (Exception caught.)", file=stderr, flush=True)
                beta_MAP = np.float('nan')
        return beta_MAP

    def alpha_ML(self, mk, K1, N):
        """
        Compute first guess for the beta_MAP (cf get_beta_MAP) parameter
        via the posterior of a Dirichlet process.
        """

        mk = utl.remove_key(mk, 0)
        # rnsum      = np.array([_logvarrhoi_DP(n, mk[n]) for n in mk]).sum()
        estlist = [N * (K1 - 1.) / r / (N - K1) for r in np.arange(6., 1.5, -0.5)]
        varrholist = {}
        for a in estlist:
            # varrholist[_logvarrho_DP(a, rnsum, K1, N)] = a
            varrholist[self.log_likelihood_DP_alpha(a, K1, N)] = a
        a_est = varrholist[max(varrholist.keys())]
        res = minimize(lambda a: -self.log_likelihood_DP_alpha(a[0], K1, N),
                       a_est, method='Nelder-Mead')
        return res.x[0]

    def get_integration_bounds(self, mk, K, N):
        """
        Find the integration bounds for the estimator.

        Typically it is a delta-like distribution so it is sufficient
        to integrate around this peak. (If not this function is not
        called.)
        """

        beta_MAP = self.get_beta_MAP(mk, K, N)
        if np.isnan(beta_MAP):
            intbounds = np.float('nan')
        else:
            std = np.sqrt(- self.d2_log_rho_xi(beta_MAP, mk, K, N) ** (-1))
            intbounds = [np.float(np.amax([10 ** (-50), beta_MAP - 8 * std])),
                         np.float(beta_MAP + 8 * std)]

        return intbounds

    def H1(self, beta, mk, K, N):
        """
        Compute the first moment (expectation value) of the entropy H.

        H is the entropy one obtains with a symmetric Dirichlet prior
        with concentration parameter beta and a multinomial likelihood.
        """

        norm = N + beta * K
        return mp.psi(0, norm + 1) - np.sum((mk[n] * (n + beta) *
                                             mp.psi(0, n + beta + 1) for n in mk)) / norm

    def nsb_entropy(self, mk, K, N):
        """
        Estimate the entropy of a system using the NSB estimator.

        :param mk: multiplicities
        :param K:  number of possible symbols/ state space of the system
        :param N:  total number of observed symbols
        """

        mp.pretty = True

        # find the concentration parameter beta
        # for which the posterior is maximised
        # to integrate around this peak
        integration_bounds = self.get_integration_bounds(mk, K, N)

        if np.any(np.isnan(integration_bounds)):
            # if no peak was found, integrate over the whole range
            # by reformulating beta into w so that the range goes from 0 to 1
            # instead of from 1 to infinity

            integration_bounds = [0, 1]

            def unnormalized_posterior_w(w, mk, K, N):
                sbeta = w / (1 - w)
                beta = sbeta * sbeta
                return self.unnormalized_posterior(beta, mk, K, N) * 2 * sbeta / (1 - w) / (1 - w)

            def H1_w(w, mk, K, N):
                sbeta = w / (1 - w)
                beta = sbeta * sbeta
                return self.H1(w, mk, K, N)

            marginal_likelihood = mp.quadgl(lambda w: unnormalized_posterior_w(w, mk, K, N),
                                            integration_bounds)
            H_nsb = mp.quadgl(lambda w: H1_w(w, mk, K, N) * unnormalized_posterior_w(w, mk, K, N),
                              integration_bounds) / marginal_likelihood

        else:
            # integrate over the possible entropies, weighted such that every entropy is equally likely
            # and normalize with the marginal likelihood
            marginal_likelihood = mp.quadgl(lambda beta: self.unnormalized_posterior(beta, mk, K, N),
                                            integration_bounds)
            H_nsb = mp.quadgl(lambda beta: self.H1(beta, mk, K, N) * self.unnormalized_posterior(beta, mk, K, N),
                              integration_bounds) / marginal_likelihood

        return H_nsb

    def plugin_entropy(self, mk, N):
        """
        Estimate the entropy of a system using the Plugin estimator.

        (In principle this is the same function as utl.get_shannon_entropy,
        only here it is a function of the multiplicities, not the probabilities.)

        :param mk: multiplicities
        :param N:  total number of observed symbols
        """

        mk = utl.remove_key(mk, 0)
        return - sum((mk[n] * (n / N) * np.log(n / N) for n in mk))

    def get_multiplicities(self, symbol_counts, alphabet_size):
        """
        Get the multiplicities of some given symbol counts.

        To estimate the entropy of a system, it is only important how
        often a symbol/ event occurs (the probability that it occurs), not
        what it represents. Therefore, computations can be simplified by
        summarizing symbols by their frequency, as represented by the
        multiplicities.
        """

        mk = dict(((value, 0) for value in symbol_counts.values()))
        number_of_observed_symbols = np.count_nonzero([value for value in symbol_counts.values()])

        for symbol in symbol_counts.keys():
            mk[symbol_counts[symbol]] += 1

        # the number of symbols that have not been observed in the data
        mk[0] = alphabet_size - number_of_observed_symbols

        return mk

    def bayesian_bias_criterion(self, R_nsb, R_plugin, bbc_tolerance):
        """
        Get whether the Bayesian bias criterion (bbc) is passed.

        :param R_nsb: history dependence computed with NSB estimator
        :param R_plugin: history dependence computed with plugin estimator
        :param bbc_tolerance: tolerance for the Bayesian bias criterion
        """

        if self.get_bbc_term(R_nsb, R_plugin) < bbc_tolerance:
            return 1
        else:
            return 0

    def get_bbc_term(self, R_nsb, R_plugin):
        """
        Get the bbc tolerance-independent term of the Bayesian bias
        criterion (bbc).

        :param R_nsb: history dependence computed with NSB estimator
        :param R_plugin: history dependence computed with plugin estimator
        """

        if R_nsb > 0:
            return np.abs(R_nsb - R_plugin) / R_nsb
        else:
            return np.inf

    def bbc_estimator(self,
                      symbol_counts,
                      past_symbol_counts,
                      alphabet_size,
                      alphabet_size_past,
                      H_uncond,
                      bbc_tolerance=None,
                      return_ais=False):
        """
        Estimate the entropy of a system using the BBC estimator.
        """

        mk = self.get_multiplicities(symbol_counts, alphabet_size)
        mk_past = self.get_multiplicities(past_symbol_counts, alphabet_size_past)

        N = sum((mk[n] * n for n in mk.keys()))

        H_nsb_joint = self.nsb_entropy(mk, alphabet_size, N)
        H_nsb_past = self.nsb_entropy(mk_past, alphabet_size_past, N)

        H_nsb_cond = H_nsb_joint - H_nsb_past
        I_nsb = H_uncond - H_nsb_cond
        R_nsb = I_nsb / H_uncond

        H_plugin_joint = self.plugin_entropy(mk, N)
        H_plugin_past = self.plugin_entropy(mk_past, N)

        H_plugin_cond = H_plugin_joint - H_plugin_past
        I_plugin = H_uncond - H_plugin_cond
        R_plugin = I_plugin / H_uncond

        if return_ais:
            ret_val = np.float(I_nsb)
        else:
            ret_val = np.float(R_nsb)

        if bbc_tolerance is not None:
            if self.bayesian_bias_criterion(R_nsb, R_plugin, bbc_tolerance):
                return ret_val
            else:
                return None
        else:
            return ret_val, np.float(self.get_bbc_term(R_nsb, R_plugin))


    def get_history_dependence_for_single_embedding(self,
                                                    spike_times,
                                                    recording_length,
                                                    estimation_method,
                                                    embedding,
                                                    embedding_step_size,
                                                    bbc_tolerance=None,
                                                    **kwargs):
        """
        Apply embedding to spike_times to obtain symbol counts.
        Get history dependence from symbol counts.
        """

        past_range_T, number_of_bins_d, scaling_k = embedding

        symbol_counts = self.get_symbol_counts(spike_times, embedding, embedding_step_size)

        history_dependence, bbc_term = self.get_history_dependence(symbol_counts,
                                                                  number_of_bins_d,
                                                                  bbc_tolerance=None,
                                                                  **kwargs)

        if bbc_tolerance is None:
            return history_dependence, bbc_term

        if bbc_term >= bbc_tolerance:
            return None

    def estimate(self, data):
        a=1


class hdEstimatorShuffling(hdAbstractEstimator):
    """Shuffling History dependence estimator

    ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

    implemented in idtxl by Michael Lindner, Göttingen 2021


    settings??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)
        self.settings.setdefault('estimation_method', "shuffling")
        self.settings.setdefault('bbc_tolerance', 0.05)

    def get_P_X_uncond(self, number_of_symbols):
        """
        Compute P(X), the probability of the current activity using
        the plug-in estimator.
        """

        return [number_of_symbols[0] / sum(number_of_symbols),
                number_of_symbols[1] / sum(number_of_symbols)]

    def get_P_X_past_uncond(self, past_symbol_counts, number_of_symbols):
        """
        Compute P(X_past), the probability of the past activity using
        the plug-in estimator.
        """

        P_X_past_uncond = {}
        for response in [0, 1]:
            for symbol in past_symbol_counts[response]:
                if symbol in P_X_past_uncond:
                    P_X_past_uncond[symbol] += past_symbol_counts[response][symbol]
                else:
                    P_X_past_uncond[symbol] = past_symbol_counts[response][symbol]
        number_of_symbols_uncond = sum(number_of_symbols)

        for symbol in P_X_past_uncond:
            P_X_past_uncond[symbol] /= number_of_symbols_uncond
        return P_X_past_uncond

    def get_P_X_past_cond_X(self, past_symbol_counts, number_of_symbols):
        """
        Compute P(X_past | X), the probability of the past activity conditioned
        on the response X using the plug-in estimator.
        """

        P_X_past_cond_X = [{}, {}]
        for response in [0, 1]:
            for symbol in past_symbol_counts[response]:
                P_X_past_cond_X[response][symbol] \
                    = past_symbol_counts[response][symbol] / number_of_symbols[response]
        return P_X_past_cond_X

    def get_H0_X_past_cond_X_eq_x(self, marginal_probabilities, number_of_bins_d):
        """
        Compute H_0(X_past | X = x), cf get_H0_X_past_cond_X.
        """
        return utl.get_shannon_entropy(marginal_probabilities) \
               + utl.get_shannon_entropy(1 - marginal_probabilities)

    def get_H0_X_past_cond_X(self, marginal_probabilities, number_of_bins_d, P_X_uncond):
        """
        Compute H_0(X_past | X), the estimate of the entropy for the past
        symbols given a response, under the assumption that activity in
        the past contributes independently towards the response.
        """
        H0_X_past_cond_X_eq_x = [0, 0]
        for response in [0, 1]:
            H0_X_past_cond_X_eq_x[response] \
                = self.get_H0_X_past_cond_X_eq_x(marginal_probabilities[response],
                                            number_of_bins_d)
        return sum([P_X_uncond[response] * H0_X_past_cond_X_eq_x[response] for response in [0, 1]])

    def get_H_X_past_uncond(self, P_X_past_uncond):
        """
        Compute H(X_past), the plug-in estimate of the entropy for the past symbols, given
        their probabilities.
        """

        return utl.get_shannon_entropy(P_X_past_uncond.values())

    def get_H_X_past_cond_X(self, P_X_uncond, P_X_past_cond_X):
        """
        Compute H(X_past | X), the plug-in estimate of the conditional entropy for the past
        symbols, conditioned on the response X,  given their probabilities.
        """

        return sum((P_X_uncond[response] * self.get_H_X_past_uncond(P_X_past_cond_X[response])
                    for response in [0, 1]))

    def get_marginal_frequencies_of_spikes_in_bins(self, symbol_counts, number_of_bins_d):
        """
        Compute for each past bin 1...d the sum of spikes found in that bin across all
        observed symbols.
        """
        return np.array(sum((self.symbol_binary_to_array(symbol, number_of_bins_d)
                             * symbol_counts[symbol]
                             for symbol in symbol_counts)), dtype=int)

    def get_shuffled_symbol_counts(self, symbol_counts, past_symbol_counts, number_of_bins_d,
                                   number_of_symbols):
        """
        Simulate new data by, for each past bin 1...d, permutating the activity
        across all observed past_symbols (for a given response X). The marginal
        probability of observing a spike given the response is thus preserved for
        each past bin.
        """
        number_of_spikes = sum(past_symbol_counts[1].values())

        marginal_frequencies = [self.get_marginal_frequencies_of_spikes_in_bins(past_symbol_counts[response],
                                                                           number_of_bins_d)
                                for response in [0, 1]]

        shuffled_past_symbols = [np.zeros(number_of_symbols[response]) for response in [0, 1]]

        for i in range(0, number_of_bins_d):
            for response in [0, 1]:
                shuffled_past_symbols[response] \
                    += 2 ** (number_of_bins_d - i - 1) \
                       * np.random.permutation(np.hstack((np.ones(marginal_frequencies[response][i]),
                                                          np.zeros(number_of_symbols[response] \
                                                                   - marginal_frequencies[response][i]))))

        for response in [0, 1]:
            shuffled_past_symbols[response] = np.array(shuffled_past_symbols[response], dtype=int)

        shuffled_past_symbol_counts = [Counter(), Counter()]

        for response in [0, 1]:
            for past_symbol in shuffled_past_symbols[response]:
                shuffled_past_symbol_counts[response][past_symbol] += 1

        marginal_probabilities = [marginal_frequencies[response] / number_of_symbols[response]
                                  for response in [0, 1]]

        return shuffled_past_symbol_counts, marginal_probabilities

    def shuffling_MI(self, symbol_counts, number_of_bins_d):
        """
        Estimate the mutual information between current and past activity
        in a spike train using the shuffling estimator.

        To obtain the shuffling estimate, compute the plug-in estimate and
        a correction term to reduce its bias.

        For the plug-in estimate:
        - Extract the past_symbol_counts from the symbol_counts.
        - I_plugin = H(X_past) - H(X_past | X)

        Notation:
        X: current activity, aka response
        X_past: past activity

        P_X_uncond: P(X)
        P_X_past_uncond: P(X_past)
        P_X_past_cond_X: P(X_past | X)

        H_X_past_uncond: H(X_past)
        H_X_past_cond_X: H(X_past | X)

        I_plugin: plugin estimate of I(X_past; X)


        For the correction term:
        - Simulate additional data under the assumption that activity
        in the past contributes independently towards the current activity.
        - Compute the entropy under the assumptions of the model, which
        due to its simplicity is easy to sample and the estimate unbiased
        - Compute the entropy using the plug-in estimate, whose bias is
        similar to that of the plug-in estimate on the original data
        - Compute the correction term as the difference between the
        unbiased and biased terms

        Notation:
        P0_sh_X_past_cond_X: P_0,sh(X_past | X), equiv. to P(X_past | X)
                             on the shuffled data

        H0_X_past_cond_X: H_0(X_past | X), based on the model of independent
        contributions
        H0_sh_X_past_cond_X: H_0,sh(X_past | X), based on
        P0_sh_X_past_cond_X, ie the plug-in estimate

        I_corr: the correction term to reduce the bias of I_plugin


        :param symbol_counts: the activity of a spike train is embedded into symbols,
        whose occurences are counted (cf emb.get_symbol_counts)
        :param number_of_bins_d: the number of bins of the embedding
        """

        # plug-in estimate
        past_symbol_counts = utl.get_past_symbol_counts(symbol_counts, merge=False)
        number_of_symbols = [sum(past_symbol_counts[response].values()) for response in [0, 1]]

        P_X_uncond = self.get_P_X_uncond(number_of_symbols)
        P_X_past_uncond = self.get_P_X_past_uncond(past_symbol_counts, number_of_symbols)
        P_X_past_cond_X = self.get_P_X_past_cond_X(past_symbol_counts, number_of_symbols)

        H_X_past_uncond = self.get_H_X_past_uncond(P_X_past_uncond)
        H_X_past_cond_X = self.get_H_X_past_cond_X(P_X_uncond, P_X_past_cond_X)

        I_plugin = H_X_past_uncond - H_X_past_cond_X

        # correction term
        shuffled_past_symbol_counts, marginal_probabilities \
            = self.get_shuffled_symbol_counts(symbol_counts, past_symbol_counts, number_of_bins_d,
                                         number_of_symbols)

        P0_sh_X_past_cond_X = self.get_P_X_past_cond_X(shuffled_past_symbol_counts, number_of_symbols)

        H0_X_past_cond_X = self.get_H0_X_past_cond_X(marginal_probabilities, number_of_bins_d, P_X_uncond)
        H0_sh_X_past_cond_X = self.get_H_X_past_cond_X(P_X_uncond, P0_sh_X_past_cond_X)

        I_corr = H0_X_past_cond_X - H0_sh_X_past_cond_X

        # shuffling estimate
        return I_plugin - I_corr

    def shuffling_estimator(self, symbol_counts, number_of_bins_d, H_uncond,
                            return_ais=False):
        """
        Estimate the history dependence in a spike train using the shuffling estimator.

        :param symbol_counts: the activity of a spike train is embedded into symbols,
        whose occurences are counted (cf emb.get_symbol_counts)
        :param number_of_bins_d: the number of bins of the embedding
        :param H_uncond: the (unconditional) spiking entropy of the spike train
        :param return_ais: define whether to return the unnormalized mutual information,
        aka active information storage (ais), instead of the history dependence
        """

        I_sh = self.shuffling_MI(symbol_counts,
                            number_of_bins_d)

        if return_ais:
            return I_sh
        else:
            return I_sh / H_uncond


    def get_history_dependence_for_single_embedding(self,
                                                    spike_times,
                                                    recording_length,
                                                    estimation_method,
                                                    embedding,
                                                    embedding_step_size,
                                                    bbc_tolerance=None,
                                                    **kwargs):
        """
        Apply embedding to spike_times to obtain symbol counts.
        Get history dependence from symbol counts.
        """

        past_range_T, number_of_bins_d, scaling_k = embedding

        symbol_counts = self.get_symbol_counts(spike_times, embedding, embedding_step_size)

        history_dependence = self.get_history_dependence(symbol_counts,
                                                        number_of_bins_d,
                                                        **kwargs)

        return history_dependence

    def estimate(self, data):

        """Estimate HDE shuffling

            Args:
                data : numpy array
                    realisations of first variable,

                Returns:
                ???????????????????????????????????????????????????????????????????????????????????????????????????????????
        """

        if self.settings['debug']:
            import pprint
            pprint.pprint(self.settings, width=1)

        # get spike times
        spike_times = np.sort(data) - min(data)

        # check inputs
        self._check_input(spike_times)

        # check data
        data = self._ensure_one_dim_input(spike_times)

        if len(data.shape) == 1:
            data = spike_times[np.newaxis, :]
        elif data.shape[0] > data.shape[1]:
            data = spike_times.T

        self.get_spike_times_stats(data, self.settings['embedding_step_size'])
        
        self.settings['estimation_method'] = 'shuffling'

        if self.settings['cross_validated_optimization']:
            spike_times_optimization, spike_times_validation = np.split(data, 2, axis=1)
            self.settings['cross_val'] = 'h1'  # first half of the data
            self.get_history_dependence_for_embeddings_shuffling(spike_times_optimization)

            self.settings['cross_val'] = 'h2'  # second half of the data
            self.get_history_dependence_for_embeddings_shuffling(spike_times_validation)
            self.compute_CIs(data, target_R='R_max', **self.settings)
        else:
            self.settings['cross_val'] = None
            self.get_history_dependence_for_embeddings_shuffling(data)
            self.compute_CIs(data, target_R='R_max', **self.settings)

        self.compute_CIs(data, target_R='R_tot', **self.settings)
        self.compute_CIs(data, target_R='nonessential', **self.settings)

        self.analyse_auto_MI(data)

        T_D = self.get_temporal_depth_T_D()
        tau_R = self.get_information_timescale_tau_R()
        R_tot = self.get_R_tot()
        AIS_tot = R_tot * self.settings['H_spiking']
        opt_number_of_bins_d, opt_scaling_k = self.settings['embedding_maximising_R_at_T'][T_D]

        max_Rs = self.settings['max_Rs']
        mr = np.array(list(max_Rs.items()), dtype=float)
        HD_max_R = mr[:,1]

        results = {'estimation_method': 'shuffling',
                   'T_D': T_D,
                   'tau_R': tau_R,
                   'R_tot': R_tot,
                   'AIS_tot': AIS_tot,
                   'opt_number_of_bins_d': opt_number_of_bins_d,
                   'opt_scaling_k': opt_scaling_k,
                   'history_dependence': self.settings['history_dependence'],
                   'embedding_maximising_R_at_T': self.settings['embedding_maximising_R_at_T'],
                   'auto_MI': self.settings['auto_MI'],
                   'max_Rs': self.settings['max_Rs'],
                   'max_R_T': self.settings['max_R_T'],
                   'HD_max_R': HD_max_R,
                   'settings': self.settings}

        return results