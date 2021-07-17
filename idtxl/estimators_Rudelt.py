"""Provide HDE estimators."""

import logging
import numpy as np
from scipy.optimize import newton, minimize
import ast
from sys import exit, stderr
from idtxl.estimator import Estimator
import idtxl.hde_utils as utl
from collections import Counter
import mpmath as mp

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


class RudeltAbstractEstimator(Estimator):
    """
    Abstract class for implementation of nsb and plugin estimators from Rudelt.

    Abstract class for implementation of nsb and plugin estimators, child classes
    implement estimators for mutual information (MI) .

    References:

        [1]: L. Rudelt, D. G. Marx, M. Wibral, V. Priesemann: Embedding
            optimization reveals long-lasting history dependence in
            neural spiking activity (in prep.)

        [2]: https://github.com/Priesemann-Group/hdestimator

    implemented in idtxl by Michael Lindner, Göttingen 2021

    Args:
        settings : dict
            - embedding_step_size : float [optional]
                Step size delta t (in seconds) with which the window is slid through the data
                (default = 0.005).
            - normalise : bool [optional]
                rebase spike times to zero
                (default=True)
            - return_averaged_R : bool [optional]
                If set to True, compute R̂tot as the average over R̂(T ) for T ∈ [T̂D, Tmax ] instead of
                R̂tot = R(T̂D ). If set to True, the setting for number_of_bootstraps_R_tot is ignored and
                set to 0
                (default=True)
    """

    def __init__(self, settings=None):
                # ---------------------------------------------------------------------------------------------------------- TODO check input and chekc inf settings are given from optimization
        # Get defaults for estimator settings
        settings = self._check_settings(settings)
        self.settings = settings.copy()
        self.settings.setdefault('normalize', True)
        self.settings.setdefault('embedding_step_size', 0.005)
        self.settings.setdefault('return_averaged_R', True)


    def is_parallel(self):
        return False

    def is_analytic_null_estimator(self):
        return False


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

    def get_symbol_counts(self, symbol_array):
        """
        Count how often symbols occur
        """

        symbol_counts = Counter()
        for symbol in np.unique(symbol_array):
            symbol_counts[symbol] += len(np.where(symbol_array == symbol)[0])

        return symbol_counts

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

    def _check_input(self, data):       # -------------------------------------------------------------------------------- TODO

        EXIT_SUCCESS = 0
        EXIT_FAILURE = 1

        # check that required settings are defined
        required_parameters = ['embedding_past_range_set', 'embedding_number_of_bins_set',
                               'embedding_scaling_exponent_set', 'embedding_step_size',
                               'bbc_tolerance', 'timescale_minimum_past_range',
                               'number_of_bootstraps_R_max', 'number_of_bootstraps_R_tot',
                               'number_of_bootstraps_nonessential',
                               'block_length_l',
                               'bootstrap_CI_percentile_lo',
                               'bootstrap_CI_percentile_hi']

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


class RudeltAbstractNSBEstimator(RudeltAbstractEstimator):
    """Abstract class for implementation of nsb estimators from Rudelt.

    Abstract class for implementation of nsb estimators, child classes
    implement nsb estimators for mutual information (MI) .

    implemented in idtxl by Michael Lindner, Göttingen 2021

    Args:
        settings : dict
            - embedding_step_size : float [optional]
                Step size delta t (in seconds) with which the window is slid through the data
                (default = 0.005).
            - normalise : bool [optional]
                rebase spike times to zero
                (default=True)
            - return_averaged_R : bool [optional]
                If set to True, compute R̂tot as the average over R̂(T ) for T ∈ [T̂D, Tmax ] instead of
                R̂tot = R(T̂D ). If set to True, the setting for number_of_bootstraps_R_tot is ignored and
                set to 0
                (default=True)
    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)

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


class RudeltNSBEstimatorSymbolsMI(RudeltAbstractNSBEstimator):
    """History dependence NSB estimator

    Calculate the mutual information (MI) of one variable depending on its past
    using NSB estimator. See parent class for references.

    implemented in idtxl by Michael Lindner, Göttingen 2021

    Args:
        settings : dict
            - embedding_step_size : float [optional]
                Step size delta t (in seconds) with which the window is slid through the data
                (default = 0.005).
            - normalise : bool [optional]
                rebase spike times to zero
                (default=True)
            - return_averaged_R : bool [optional]
                If set to True, compute R̂tot as the average over R̂(T ) for T ∈ [T̂D, Tmax ] instead of
                R̂tot = R(T̂D ). If set to True, the setting for number_of_bootstraps_R_tot is ignored and
                set to 0
                (default=True)
    """

    def __init__(self, settings=None):
        # Set default estimator settings.
        super().__init__(settings)


    def nsb_estimator(self,
                      symbol_counts,
                      past_symbol_counts,
                      alphabet_size,
                      alphabet_size_past,
                      H_uncond):
        """
        Estimate the entropy of a system using the NSB estimator.
        """

        mk = self.get_multiplicities(symbol_counts, alphabet_size)
        mk_past = self.get_multiplicities(past_symbol_counts, alphabet_size_past)

        N = sum((mk[n] * n for n in mk.keys()))

        H_nsb_joint = self.nsb_entropy(mk, alphabet_size, N)
        H_nsb_past = self.nsb_entropy(mk_past, alphabet_size_past, N)

        H_nsb_cond = H_nsb_joint - H_nsb_past
        I_nsb = H_uncond - H_nsb_cond
        R_nsb = I_nsb / H_uncond

        return I_nsb, R_nsb

    def estimate(self, symbol_array, past_symbol_array, current_symbol_array):
        """Estimate mutual information using NSB estimator.

        Args:
            symbol_array : 1D numpy array
                realisations of symbols based on current and past states.
                (first output of get_realisations_symbol from data_spiketimes object)
            past_symbol_array : numpy array
                realisations of symbols based on current and past states.
                (first output of get_realisations_symbol from data_spiketimes object)
            current_symbol_array : numpy array
                realisations of symbols based on current and past states.
                (first output of get_realisations_symbol from data_spiketimes object)

        Returns:
            float
                MI (AIS)
            float
                MI / H_uncond (History dependence)
        """

        symbol_counts = self.get_symbol_counts(symbol_array)

        current_symbol_counts = self.get_symbol_counts(current_symbol_array)
        H_uncond = utl.get_H_spiking(symbol_counts)

        past_symbol_counts = self.get_symbol_counts(past_symbol_array)

        number_of_bins_d_join = len(list(np.binary_repr(np.max(symbol_array))))

        alphabet_size_past = 2 ** int(number_of_bins_d_join - 1)  # K for past activity
        alphabet_size = alphabet_size_past * 2  # K

        I, R = self.nsb_estimator(symbol_counts,
                                  past_symbol_counts,
                                  alphabet_size,
                                  alphabet_size_past,
                                  H_uncond)

        return np.float(I), np.float(R)


class RudeltPluginEstimatorSymbolsMI(RudeltAbstractEstimator):
    """Plugin History dependence estimator

    Calculate the mutual information (MI) of one variable depending on its past
    using plugin estimator. See parent class for references.

    implemented in idtxl by Michael Lindner, Göttingen 2021

    Args:
        settings : dict
            - embedding_step_size : float [optional] - Step size delta t (in seconds) with which the window is slid
                            through the data (default = 0.005).
            - normalise : bool [optional] - rebase spike times to zero (default=True)
            - return_averaged_R : bool [optional] - rebase spike times to zero (default=True)

    """

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

    def plugin_estimator(self,
                         symbol_counts,
                         past_symbol_counts,
                         alphabet_size,
                         alphabet_size_past,
                         H_uncond):
        """
        Estimate the entropy of a system using the BBC estimator.
        """

        mk = self.get_multiplicities(symbol_counts, alphabet_size)
        mk_past = self.get_multiplicities(past_symbol_counts, alphabet_size_past)

        N = sum((mk[n] * n for n in mk.keys()))

        H_plugin_joint = self.plugin_entropy(mk, N)
        H_plugin_past = self.plugin_entropy(mk_past, N)

        H_plugin_cond = H_plugin_joint - H_plugin_past
        I_plugin = H_uncond - H_plugin_cond
        R_plugin = I_plugin / H_uncond

        return I_plugin, R_plugin

    def estimate(self, symbol_array, past_symbol_array, current_symbol_array):
        """Estimate mutual information using plugin estimator.

        Args:
            symbol_array : 1D numpy array
                realisations of symbols based on current and past states.
                (first output of get_realisations_symbol from data_spiketimes object)
            past_symbol_array : numpy array
                realisations of symbols based on current and past states.
                (first output of get_realisations_symbol from data_spiketimes object)
            current_symbol_array : numpy array
                realisations of symbols based on current and past states.
                (first output of get_realisations_symbol from data_spiketimes object)

        Returns:
            float
                MI (AIS)
            float
                MI / H_uncond (History dependence)
        """

        symbol_counts = self.get_symbol_counts(symbol_array)

        current_symbol_counts = self.get_symbol_counts(current_symbol_array)
        # H_uncond_orig = utl.get_H_spiking(symbol_counts)
        H_uncond = utl.get_H_spiking(symbol_counts)

        # past_symbol_counts = utl.get_past_symbol_counts(symbol_counts)
        past_symbol_counts = self.get_symbol_counts(past_symbol_array)

        # number_of_bins_d_join = np.array(list(np.binary_repr(np.max(past_symbol_array)))).astype(np.int8)
        number_of_bins_d_join = len(list(np.binary_repr(np.max(symbol_array))))

        alphabet_size_past = 2 ** int(number_of_bins_d_join - 1)  # K for past activity
        alphabet_size = alphabet_size_past * 2  # K

        I, R = self.plugin_estimator(symbol_counts,
                                     past_symbol_counts,
                                     alphabet_size,
                                     alphabet_size_past,
                                     H_uncond)

        return np.float(I), np.float(R)


class RudeltBBCEstimator(RudeltAbstractEstimator):
    """
    BBC Estimator using NSB and Plugin estimator

    Calculate the mutual information (MI) of one variable depending on its past
    using nsb and plugin estimator and check if bias criterion is passed/
    See parent class for references.

    implemented in idtxl by Michael Lindner, Göttingen 2021

    Args:
        settings : dict
            - embedding_step_size : float [optional]
                Step size delta t (in seconds) with which the window is slid through the data
                (default = 0.005).
            - normalise : bool [optional]
                rebase spike times to zero
                (default=True)
            - return_averaged_R : bool [optional]
                If set to True, compute R̂tot as the average over R̂(T ) for T ∈ [T̂D, Tmax ] instead of
                R̂tot = R(T̂D ). If set to True, the setting for number_of_bootstraps_R_tot is ignored and
                set to 0
                (default=True)
    """

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

    def estimate(self, symbol_array, past_symbol_array, current_symbol_array, bbc_tolerance=None):
        """
        Calculate the mutual information (MI) of one variable depending on its past
        using nsb and plugin estimator and check if bias criterion is passed/

        Args:
            symbol_array : 1D numpy array
                realisations of symbols based on current and past states.
                (first output of get_realisations_symbol from data_spiketimes object)
            past_symbol_array : numpy array
                realisations of symbols based on current and past states.
                (first output of get_realisations_symbol from data_spiketimes object)
            current_symbol_array : numpy array
                realisations of symbols based on current and past states.
                (first output of get_realisations_symbol from data_spiketimes object)

        Returns:
            float
                MI (AIS)
            float
                MI / H_uncond (History dependence)
            float
                bbc_term
        """
        estnsb = RudeltNSBEstimatorSymbolsMI()
        I_nsb, R_nsb = estnsb.estimate(symbol_array, past_symbol_array, current_symbol_array)

        estplugin = RudeltPluginEstimatorSymbolsMI()
        I_plugin, R_plugin = estplugin.estimate(symbol_array, past_symbol_array, current_symbol_array)

        if not bbc_tolerance == None:
            if self.bayesian_bias_criterion(R_nsb, R_plugin, bbc_tolerance):
                return np.float(I_nsb), np.float(R_nsb)
            else:
                return None
        else:
            return np.float(I_nsb), np.float(R_nsb), np.float(self.get_bbc_term(R_nsb,
                                                  R_plugin))


class RudeltShufflingEstimator(RudeltAbstractEstimator):
    """
    Estimate the history dependence in a spike train using the shuffling estimator.

    See parent class for references.

    implemented in idtxl by Michael Lindner, Göttingen 2021
    """

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

    def estimate(self, symbol_array):
        """
        Estimate the history dependence in a spike train using the shuffling estimator.

         Args:
            symbol_array : 1D numpy array
                realisations of symbols based on current and past states.
                (first output of get_realisations_symbol from data_spiketimes object)

        Returns:
            float
                MI (AIS)
            float
                MI / H_uncond (History dependence)
        """
        symbol_counts = self.get_symbol_counts(symbol_array)

        # number_of_bins_d_join = np.array(list(np.binary_repr(np.max(symbol_array)))).astype(np.int8)
        number_of_bins_d_join = len(list(np.binary_repr(np.max(symbol_array))))

        H_uncond = utl.get_H_spiking(symbol_counts)

        I_sh = self.shuffling_MI(symbol_counts,
                            number_of_bins_d_join -1)

        R_sh = I_sh / H_uncond

        return I_sh, R_sh