
import numpy as np
import hde_utils as utl







class optimization_Rudelt():

    def __init__(self, settings=None):
        settings = self._check_settings(settings)
        self.settings = settings.copy()

        self.settings.setdefault('auto_MI_bin_size_set', [0.005, 0.01, 0.025, 0.05, 0.25, 0.5])
        self.settings.setdefault('auto_MI_max_delay', 5)


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
            auto_MI_d = {}
            auto_MI_d[0] = np.linspace(0, self.settings['auto_MI_max_delay'], len(auto_MI))
            auto_MI_d[1] = auto_MI

            auto_MI_dict[str(auto_MI_bin_size)] = auto_MI_d

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

    def get_history_dependence_for_embeddings(self, data):
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
        if self.settings['estimation_method'] == 'bbc':
            bbc_term = np.empty(len(embeddings))

        count = 0
        for embedding in embeddings:
            past_range_T = embedding[0]
            number_of_bins_d = embedding[1]
            first_bin_size = self.get_first_bin_size_for_embedding(embedding)
            symbol_counts = utl.add_up_dicts([self.get_symbol_counts(spt,
                                                                 embedding,
                                                                 self.settings['embedding_step_size'])
                                              for spt in data])

            if self.settings['estimation_method'] == 'shuffling':
                history_dependence[count] = self.get_history_dependence(symbol_counts, number_of_bins_d)
            elif self.settings['estimation_method'] == 'bbc':
                history_dependence[count], bbc_term[count] = self.get_history_dependence(symbol_counts, number_of_bins_d)

            count += 1

        self.settings['embeddings'] = embeddings
        self.settings['history_dependence'] = history_dependence
        if self.settings['estimation_method'] == 'bbc':
            self.settings['bbc_term'] = bbc_term

        # return history_dependence


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
            return self.b_estimator(symbol_counts,
                                     past_symbol_counts,
                                     alphabet_size,
                                     alphabet_size_past,
                                     H_uncond,
                                     bbc_tolerance=bbc_tolerance,
                                     return_ais=return_ais)

        elif self.settings['estimation_method'] == "shuffling":
            return self.p_estimator(symbol_counts,
                                          number_of_bins_d,
                                          H_uncond,
                                          return_ais=return_ais)



    def get_history_dependence_for_embeddings(self, data):
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
        if self.settings['estimation_method'] == 'bbc':
            bbc_term = np.empty(len(embeddings))

        count = 0
        for embedding in embeddings:
            past_range_T = embedding[0]
            number_of_bins_d = embedding[1]
            first_bin_size = self.get_first_bin_size_for_embedding(embedding)
            symbol_counts = utl.add_up_dicts([self.get_symbol_counts(spt,
                                                                 embedding,
                                                                 self.settings['embedding_step_size'])
                                              for spt in data])

            if self.settings['estimation_method'] == 'shuffling':
                history_dependence[count] = self.get_history_dependence(symbol_counts, number_of_bins_d)
            elif self.settings['estimation_method'] == 'bbc':
                history_dependence[count], bbc_term[count] = self.get_history_dependence(symbol_counts, number_of_bins_d)

            count += 1

        self.settings['embeddings'] = embeddings
        self.settings['history_dependence'] = history_dependence
        if self.settings['estimation_method'] == 'bbc':
            self.settings['bbc_term'] = bbc_term

        # return history_dependence

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
                if self.settings['bbc_term'][i] >= self.settings['bbc_tolerance']:
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







    def optimize(self, data, settings):
        """
        Estimate the history dependence and temporal depth of a single
        neuron, based on information-theoretical measures for spike time
        data

        Optimization of
        using Rudelt bbc estimator

        ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????




        settings??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

        References:

            [1]: L. Rudelt, D. G. Marx, M. Wibral, V. Priesemann: Embedding
                optimization reveals long-lasting history dependence in
                neural spiking activity (in prep.)

            [2]: https://github.com/Priesemann-Group/hdestimator

        implemented in idtxl by Michael Lindner, Göttingen 2021

        Args:
            data : numpy array
                realisations of first variable,

            settings
        ???????????????????????????????????????????????????????????????????????????????????????????????????????????
        Returns:
            T_D
            tau_R
            R_tot
            AIS_tot
            opt_number_of_bins_d
            opt_scaling_k
            history_dependence
            embedding_maximising_R_at_T
            max_Rs
            max_R_T
            HD_max_R

            auto_MI  (optional
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

        if self.settings['cross_validated_optimization']:
            spike_times_optimization, spike_times_validation = np.split(data, 2, axis=1)
            self.settings['cross_val'] = 'h1'  # first half of the data
            self.get_history_dependence_for_embeddings(spike_times_optimization)

            self.settings['cross_val'] = 'h2'  # second half of the data
            self.get_history_dependence_for_embeddings(spike_times_validation)
            self.compute_CIs(data, target_R='R_max', **self.settings)
        else:
            self.settings['cross_val'] = None
            self.get_history_dependence_for_embeddings(data)
            self.compute_CIs(data, target_R='R_max', **self.settings)

        self.compute_CIs(data, target_R='R_tot', **self.settings)
        self.compute_CIs(data, target_R='nonessential', **self.settings)

        # ??????????????????????????????????????????????????????????????????????????????????????????????????????????? make optional
        self.analyse_auto_MI(data)

        T_D = self.get_temporal_depth_T_D()
        tau_R = self.get_information_timescale_tau_R()
        R_tot = self.get_R_tot()
        AIS_tot = R_tot * self.settings['H_spiking']
        opt_number_of_bins_d, opt_scaling_k = self.settings['embedding_maximising_R_at_T'][T_D]

        max_Rs = self.settings['max_Rs']
        mr = np.array(list(max_Rs.items()), dtype=float)
        HD_max_R = mr[:, 1]

        results = {'T_D': T_D,
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
                   'auto_MI': self.settings['auto_MI']['auto_MI'],
                   'settings': self.settings}

        return results
