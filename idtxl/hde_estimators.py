

from sys import exit, stderr
import numpy as np
import mpmath as mp
from scipy.optimize import newton, minimize
import idtxl.hde_utils as utl
from collections import Counter
import idtxl.hde_embedding as emb


class hde_api():

    def __init__(self):
        pass

    def get_history_dependence(self,
                               estimation_method,
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
        if H_uncond == None:
            H_uncond = utl.get_H_spiking(symbol_counts)

        if past_symbol_counts == None:
            past_symbol_counts = utl.get_past_symbol_counts(symbol_counts)

        alphabet_size_past = 2 ** int(number_of_bins_d)  # K for past activity
        alphabet_size = alphabet_size_past * 2  # K

        if estimation_method == "bbc":
            return hde_bbc_estimator.bbc_estimator(symbol_counts,
                                     past_symbol_counts,
                                     alphabet_size,
                                     alphabet_size_past,
                                     H_uncond,
                                     bbc_tolerance=bbc_tolerance,
                                     return_ais=return_ais)

        elif estimation_method == "shuffling":
            return hde_shuffling_estimator.shuffling_estimator(symbol_counts,
                                          number_of_bins_d,
                                          H_uncond,
                                          return_ais=return_ais)

    ## below are functions for estimates on spike trains

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

        symbol_counts = emb.get_symbol_counts(spike_times, embedding, embedding_step_size)

        if estimation_method == 'bbc':
            history_dependence, bbc_term = self.get_history_dependence(estimation_method,
                                                                  symbol_counts,
                                                                  number_of_bins_d,
                                                                  bbc_tolerance=None,
                                                                  **kwargs)

            if bbc_tolerance == None:
                return history_dependence, bbc_term

            if bbc_term >= bbc_tolerance:
                return None

        elif estimation_method == 'shuffling':
            history_dependence = self.get_history_dependence(estimation_method,
                                                        symbol_counts,
                                                        number_of_bins_d,
                                                        **kwargs)

        return history_dependence

    def get_history_dependence_for_embedding_set(self, spike_times,
                                                 recording_length,
                                                 estimation_method,
                                                 embedding_past_range_set,
                                                 embedding_number_of_bins_set,
                                                 embedding_scaling_exponent_set,
                                                 embedding_step_size,
                                                 bbc_tolerance=None,
                                                 dependent_var="T",
                                                 **kwargs):
        """
        Apply embeddings to spike_times to obtain symbol counts.
        For each T (or d), get history dependence R for the embedding for which
        R is maximised.
        """

        assert dependent_var in ["T", "d"]

        if bbc_tolerance == None:
            bbc_tolerance = np.inf

        max_Rs = {}
        embeddings_that_maximise_R = {}

        for embedding in emb.get_embeddings(embedding_past_range_set,
                                            embedding_number_of_bins_set,
                                            embedding_scaling_exponent_set):
            past_range_T, number_of_bins_d, scaling_k = embedding

            history_dependence = self.get_history_dependence_for_single_embedding(spike_times,
                                                                             recording_length,
                                                                             estimation_method,
                                                                             embedding,
                                                                             embedding_step_size,
                                                                             bbc_tolerance=bbc_tolerance,
                                                                             **kwargs)
            if history_dependence == None:
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

        return embeddings_that_maximise_R, max_Rs

    def get_CI_for_embedding(self,
                             history_dependence,
                             spike_times,
                             estimation_method,
                             embedding,
                             embedding_step_size,
                             number_of_bootstraps,
                             block_length_l=None,
                             bootstrap_CI_use_sd=True,
                             bootstrap_CI_percentile_lo=2.5,
                             bootstrap_CI_percentile_hi=97.5):
        """
        Compute confidence intervals for the history dependence estimate
        based on either the standard deviation or percentiles of
        bootstrap replications of R.
        """

        if block_length_l == None:
            # eg firing rate is 4 Hz, ie there is 1 spikes per 1/4 seconds,
            # for every second the number of symbols is 1/ embedding_step_size
            # so we observe on average one spike every 1 / (firing_rate * embedding_step_size) symbols
            # (in the reponse, ignoring the past activity)
            firing_rate = utl.get_binned_firing_rate(spike_times, embedding_step_size)
            block_length_l = max(1, int(1 / (firing_rate * embedding_step_size)))

        bs_history_dependence \
            = utl.get_bootstrap_history_dependence([spike_times],
                                                   embedding,
                                                   embedding_step_size,
                                                   estimation_method,
                                                   number_of_bootstraps,
                                                   block_length_l)

        return utl.get_CI_bounds(history_dependence,
                                 bs_history_dependence,
                                 bootstrap_CI_use_sd=bootstrap_CI_use_sd,
                                 bootstrap_CI_percentile_lo=bootstrap_CI_percentile_lo,
                                 bootstrap_CI_percentile_hi=bootstrap_CI_percentile_hi)


class hde_bbc_estimator():

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

    def __init__(self):
        pass

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


class hde_shuffling_estimator():

    def __init__(self):
        pass

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
        return np.array(sum((emb.symbol_binary_to_array(symbol, number_of_bins_d)
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
