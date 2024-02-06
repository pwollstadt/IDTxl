""" Optimization of embedding parameters of spike times using the history dependence estimators """

import numpy as np
from scipy.optimize import newton
from sys import stderr
from idtxl.estimators_Rudelt import RudeltBBCEstimator, RudeltShufflingEstimator
import idtxl.hde_utils as utl
from idtxl.results import DotDict, ResultsSingleProcessRudelt
from pathlib import Path
from idtxl.data_spiketime import Data_spiketime


# noinspection PyAttributeOutsideInit
class OptimizationRudelt:
    """
    Optimization of embedding parameters of spike times using the history dependence estimators

    References:

        [1]: L. Rudelt, D. G. Marx, M. Wibral, V. Priesemann: Embedding
            optimization reveals long-lasting history dependence in
            neural spiking activity, 2021, PLOS Computational Biology, 17(6)

        [2]: https://github.com/Priesemann-Group/hdestimator

    implemented in idtxl by Michael Lindner, Göttingen 2021

    Args:
        settings : dict
            - estimation_method : string
                The method to be used to estimate the history dependence 'bbc' or 'shuffling'.
            - embedding_step_size : float
                Step size delta t (in seconds) with which the window is slid through the data.
                (default: 0.005)
            - embedding_number_of_bins_set : list of integer values
                Set of values for d, the number of bins in the embedding.
                (default: [1, 2, 3, 4, 5])
            - embedding_past_range_set : list of floating-point values
                Set of values for T, the past range (in seconds) to be used for embeddings.
                (default: [0.005, 0.00561, 0.00629, 0.00706, 0.00792, 0.00889,
                0.00998, 0.01119, 0.01256, 0.01409, 0.01581, 0.01774, 0.01991,
                0.02233, 0.02506, 0.02812, 0.03155, 0.0354, 0.03972, 0.04456,
                0.05, 0.0561, 0.06295, 0.07063, 0.07924, 0.08891, 0.09976,
                0.11194, 0.12559, 0.14092, 0.15811, 0.17741, 0.19905, 0.22334,
                0.25059, 0.28117, 0.31548, 0.35397, 0.39716, 0.44563, 0.5,
                0.56101, 0.62946, 0.70627, 0.79245, 0.88914, 0.99763, 1.11936,
                1.25594, 1.40919, 1.58114, 1.77407, 1.99054, 2.23342, 2.50594,
                2.81171, 3.15479, 3.53973, 3.97164, 4.45625, 5.0])
            - embedding_scaling_exponent_set : dict
                Set of values for kappa, the scaling exponent for the bins in the embedding.
                Should be a python-dictionary with the three entries 'number_of_scalings', 'min_first_bin_size' and
                'min_step_for_scaling'.
                defaults: {'number_of_scalings': 10, 'min_first_bin_size': 0.005, 'min_step_for_scaling': 0.01})
            - bbc_tolerance : float
                The tolerance for the Bayesian Bias Criterion. Influences which embeddings are
                discarded from the analysis.
                (default: 0.05)
            - return_averaged_R : bool
                Return R_tot as the average over R(T) for T in [T_D, T_max], instead of R_tot = R(T_D).
                If set to True, the setting for number_of_bootstraps_R_tot (see below) is ignored and set to 0
                and CI bounds are not calculated.
                (default: True)
            - timescale_minimum_past_range : float
                Minimum past range T_0 (in seconds) to take into consideration for the estimation of the
                information timescale tau_R.
                (default: 0.01)
            - number_of_bootstraps_R_max : int
                The number of bootstrap re-shuffles that should be used to determine the optimal
                embedding. (Bootstrap the estimates of R_max to determine R_tot.)
                These are computed during the 'history-dependence' task because they are essential
                to obtain R_tot.
                (default: 250)
            - number_of_bootstraps_R_tot : int
                The number of bootstrap re-shuffles that should be used to estimate the confidence
                interval of the optimal embedding. (Bootstrap the estimates of R_tot = R(T_D) to
                obtain a confidence interval for R_tot.).
                These are computed during the 'confidence-intervals' task.
                The setting return_averaged_R (see above) needs to be set to False for this setting
                to take effect.
                (default: 250)
            - number_of_bootstraps_nonessential : int
                The number of bootstrap re-shuffles that should be used to estimate the confidence
                intervals for embeddings other than the optimal one. (Bootstrap the estimates of
                R(T) for all other T.)
                (These are not necessary for the main analysis and therefore default to 0.)
            - symbol_block_length : int
                The number of symbols that should be drawn in each block for bootstrap resampling
                If it is set to None (recommended), the length is automatically chosen, based
                on heuristics
                (default: None)
            - bootstrap_CI_use_sd : bool
                Most of the time we observed normally-distributed bootstrap replications,
                so it is sufficient (and more efficient) to compute confidence intervals
                based on the standard deviation
                (default: True)
            - bootstrap_CI_percentile_lo : float
                The lower percentile for the confidence interval.
                This has no effect if bootstrap_CI_use_sd is set to True
                (default: 2.5)
            - bootstrap_CI_percentile_hi : float
                The upper percentiles for the confidence interval.
                This has no effect if bootstrap_CI_use_sd is set to True
                (default: 97.5)
            - analyse_auto_MI : bool
                perform calculation of auto mutual information of the spike train
                (default: True)
                If set to True:

                - auto_MI_bin_size_set : list of floating-point values
                    Set of values for the sizes of the bins (in seconds).
                    (default: [0.005, 0.01, 0.025, 0.05, 0.25, 0.5])
                - auto_MI_max_delay : int
                    The maximum delay (in seconds) between the past bin and the response.
                    (default: 5)

            - visualization : bool
                create .eps output image showing the optimization values and graphs for
                the history dependence and the auto mutual information
                (default: False)
                if set to True:

                - output_path : String
                    Path where the .eps images should be saved
                - output_prefix : String
                    Prefix of the output images
                    e.g. <output_prefix>_process0.eps

            - debug: bool
                show values while calculating
                (default: False)
    """

    def __init__(self, settings=None):
        settings = self._check_settings(settings)
        self.settings = settings.copy()

        self.settings.setdefault("embedding_step_size", 0.005)
        self.settings.setdefault(
            "embedding_past_range_set",
            [
                0.005,
                0.00561,
                0.00629,
                0.00706,
                0.00792,
                0.00889,
                0.00998,
                0.01119,
                0.01256,
                0.01409,
                0.01581,
                0.01774,
                0.01991,
                0.02233,
                0.02506,
                0.02812,
                0.03155,
                0.0354,
                0.03972,
                0.04456,
                0.05,
                0.0561,
                0.06295,
                0.07063,
                0.07924,
                0.08891,
                0.09976,
                0.11194,
                0.12559,
                0.14092,
                0.15811,
                0.17741,
                0.19905,
                0.22334,
                0.25059,
                0.28117,
                0.31548,
                0.35397,
                0.39716,
                0.44563,
                0.5,
                0.56101,
                0.62946,
                0.70627,
                0.79245,
                0.88914,
                0.99763,
                1.11936,
                1.25594,
                1.40919,
                1.58114,
                1.77407,
                1.99054,
                2.23342,
                2.50594,
                2.81171,
                3.15479,
                3.53973,
                3.97164,
                4.45625,
                5.0,
            ],
        )
        self.settings.setdefault("embedding_number_of_bins_set", [1, 2, 3, 4, 5])
        self.settings.setdefault(
            "embedding_scaling_exponent_set",
            {
                "number_of_scalings": 10,
                "min_first_bin_size": 0.005,
                "min_step_for_scaling": 0.01,
            },
        )

        self.settings.setdefault("bbc_tolerance", 0.05)
        self.settings.setdefault("return_averaged_R", True)
        self.settings.setdefault("timescale_minimum_past_range", 0.01)

        self.settings.setdefault("analyse_auto_MI", True)
        self.settings.setdefault(
            "auto_MI_bin_size_set", [0.005, 0.01, 0.025, 0.05, 0.25, 0.5]
        )
        self.settings.setdefault("auto_MI_max_delay", 5)

        self.settings.setdefault("number_of_bootstraps_R_max", 250)
        self.settings.setdefault("number_of_bootstraps_R_tot", 250)
        self.settings.setdefault("number_of_bootstraps_nonessential", 0)
        self.settings.setdefault("symbol_block_length", None)
        self.settings.setdefault("bootstrap_CI_use_sd", True)
        self.settings.setdefault("bootstrap_CI_percentile_lo", 2.5)
        self.settings.setdefault("bootstrap_CI_percentile_hi", 97.5)
        self.settings.setdefault("visualization", False)
        self.settings.setdefault("debug", False)

        self.check_inputs()  # ------------------------------------------------------------- TODO CHECK INPUTS

        self.embeddings = self.get_embeddings(
            self.settings["embedding_past_range_set"],
            self.settings["embedding_number_of_bins_set"],
            self.settings["embedding_scaling_exponent_set"],
        )

    @staticmethod
    def _check_settings(settings=None):
        """Set default for settings dictionary.

        Check if settings dictionary is None. If None, initialise an empty
        dictionary. If not None check if type is dictionary. Function should be
        called before setting default values.
        """
        if settings is None:
            return {}
        elif type(settings) is not dict:
            raise TypeError("settings should be a dictionary.")
        else:
            return settings

    def check_inputs(self):
        """
        Check input settings for completeness

        """

        args_float = [
            "embedding_step_size",
            "timescale_minimum_past_range",
            "bootstrap_CI_percentile_lo",
            "bootstrap_CI_percentile_hi",
            "bbc_tolerance",
        ]
        for key in args_float:
            assert key in self.settings, key + " has to be specified (see help)!"
            assert isinstance(self.settings[key], float), (
                "Error: setting '"
                + key
                + "' needs to an floating point value (see help!!). Aborting."
            )

        args_int = [
            "number_of_bootstraps_R_max",
            "number_of_bootstraps_R_tot",
            "number_of_bootstraps_nonessential",
        ]
        for key in args_int:
            assert key in self.settings, key + " has to be specified (see help)!"
            assert isinstance(self.settings[key], int), (
                "Error: setting '"
                + key
                + "' needs to an integer value (see help!!). Aborting."
            )

        args_bool = ["return_averaged_R", "bootstrap_CI_use_sd", "debug"]
        for key in args_bool:
            assert key in self.settings, key + " has to be specified (see help)!"
            assert isinstance(self.settings[key], bool), (
                "Error: setting '"
                + key
                + "' needs to a boolean expression (see help!!). Aborting."
            )

        assert (
            "embedding_past_range_set" in self.settings
        ), "embedding_past_range_set has to be specified (see help)!"
        assert isinstance(self.settings["embedding_past_range_set"], list), (
            "Error: setting 'embedding_past_range_set' needs to be a list but is defined as {0}. "
            "Aborting.".format(type(self.settings["embedding_past_range_set"]))
        )

        assert (
            "embedding_number_of_bins_set" in self.settings
        ), "embedding_number_of_bins_set has to be specified (see help)!"
        assert isinstance(self.settings["embedding_number_of_bins_set"], list), (
            "Error: setting 'embedding_number_of_bins_set' needs to a list but is defined as {0}. "
            "Aborting.".format(type(self.settings["embedding_number_of_bins_set"]))
        )

        assert (
            "embedding_scaling_exponent_set" in self.settings
        ), "embedding_scaling_exponent_set has to be specified (see help)!"

        assert (
            "number_of_scalings" in self.settings["embedding_scaling_exponent_set"]
        ), "'number_of_scalings' have to be specified in self.settings['embedding_scaling_exponent_set'] (see help)!"
        assert isinstance(
            self.settings["embedding_scaling_exponent_set"]["number_of_scalings"], int
        ), (
            "Error: setting 'number_of_scalings' needs to be an integer value but is defined as {0}. "
            "Aborting.".format(type(self.settings["number_of_scalings"]))
        )
        assert (
            "min_first_bin_size" in self.settings["embedding_scaling_exponent_set"]
        ), "'min_first_bin_size' have to be specified in self.settings['embedding_scaling_exponent_set'] (see help)!"
        assert isinstance(
            self.settings["embedding_scaling_exponent_set"]["min_first_bin_size"], float
        ), (
            "Error: setting 'min_first_bin_sizes' needs to be a floating point value but is defined as {0}. "
            "Aborting.".format(type(self.settings["min_first_bin_sizes"]))
        )
        assert (
            "min_step_for_scaling" in self.settings["embedding_scaling_exponent_set"]
        ), "'min_step_for_scaling' have to be specified in self.settings['embedding_scaling_exponent_set'] (see help)!"
        assert isinstance(
            self.settings["embedding_scaling_exponent_set"]["min_step_for_scaling"],
            float,
        ), (
            "Error: setting 'min_step_for_scaling' needs to be a floating point value but is defined as {0}. "
            "Aborting.".format(type(self.settings["min_step_for_scaling"]))
        )

        assert (
            "symbol_block_length" in self.settings
        ), "symbol_block_length has to be specified (see help)!"
        if self.settings["symbol_block_length"] is not None:
            assert isinstance(self.settings["symbol_block_length"], int), (
                "Error: setting 'symbol_block_length' needs to an integer value but is defined as {0}. "
                "Aborting.".format(type(self.settings["symbol_block_length"]))
            )

        if self.settings["analyse_auto_MI"]:
            assert isinstance(self.settings["analyse_auto_MI"], bool), (
                "Error: setting 'analyse_auto_MI' needs to be a boolean expression but is defined as {0}. "
                "Aborting.".format(type(self.settings["analyse_auto_MI"]))
            )
            assert (
                "auto_MI_bin_size_set" in self.settings
            ), "If analyse_auto_MI is set to True, auto_MI_bin_size_set has to be specified (see help)!"
            assert isinstance(self.settings["auto_MI_bin_size_set"], list), (
                "Error: setting 'auto_MI_bin_size_set' needs to a list but is defined as {0}. "
                "Aborting.".format(type(self.settings["auto_MI_bin_size_set"]))
            )
            assert (
                "auto_MI_max_delay" in self.settings
            ), "If analyse_auto_MI is set to True, auto_MI_max_delay has to be specified (see help)!"
            assert isinstance(self.settings["auto_MI_max_delay"], int), (
                "Error: setting 'auto_MI_max_delay' needs to be an integer value but is defined as {0}. "
                "Aborting.".format(type(self.settings["auto_MI_max_delay"]))
            )

        if self.settings["visualization"]:
            assert isinstance(self.settings["visualization"], bool), (
                "Error: setting 'visualization' needs to be boolean but is defined as {0}. "
                "Aborting.".format(type(self.settings["visualization"]))
            )
            assert (
                "output_path" in self.settings
            ), "If visualization is set to True an output path has to be specified (see help)!"
            assert (
                "output_prefix" in self.settings
            ), "If visualization is set to True an output prefix has to be specified (see help)!"

        # Cython implementation uses 64bit unsigned integers for the symbols,
        # we allow up to 62 bins (window has 1 bin more..)
        assert (
            max(self.settings["embedding_number_of_bins_set"]) <= 62
        ), "Error: Max number of bins too large; use less than 63. Aborting."

        # If R_tot is computed as an average over Rs, no confidence interval can be estimated
        if self.settings["return_averaged_R"]:
            self.settings["number_of_bootstraps_R_tot"] = 0

    def get_embeddings(
        self,
        embedding_past_range_set,
        embedding_number_of_bins_set,
        embedding_scaling_exponent_set,
    ):
        """
        Get all combinations of parameters T, d, k, based on the
        sets of selected parameters.
        """

        embeddings = []
        for past_range_T in embedding_past_range_set:
            for number_of_bins_d in embedding_number_of_bins_set:
                if not isinstance(number_of_bins_d, int) or number_of_bins_d < 1:
                    print(
                        "Error: number of bins {} is not a positive integer. Skipping.".format(
                            number_of_bins_d
                        ),
                        file=stderr,
                        flush=True,
                    )
                    continue

                if type(embedding_scaling_exponent_set) == dict:
                    scaling_set_given_T_and_d = self.get_set_of_scalings(
                        past_range_T, number_of_bins_d, **embedding_scaling_exponent_set
                    )
                else:
                    scaling_set_given_T_and_d = embedding_scaling_exponent_set

                for scaling_k in scaling_set_given_T_and_d:
                    embeddings += [(past_range_T, number_of_bins_d, scaling_k)]

        return embeddings

    def get_set_of_scalings(
        self,
        past_range_T,
        number_of_bins_d,
        number_of_scalings,
        min_first_bin_size,
        min_step_for_scaling,
    ):
        """
        Get scaling exponents such that the uniform embedding as well as
        the embedding for which the first bin has a length of
        min_first_bin_size (in seconds), as well as linearly spaced
        scaling factors in between, such that in total
        number_of_scalings scalings are obtained.
        """

        min_scaling = 0
        if (
            past_range_T / number_of_bins_d <= min_first_bin_size
            or number_of_bins_d == 1
        ):
            max_scaling = 0
        else:
            # for the initial guess assume the largest bin dominates, so k is approx. log(T) / d

            max_scaling = newton(
                lambda scaling: self.get_past_range(
                    number_of_bins_d, min_first_bin_size, scaling
                )
                - past_range_T,
                np.log10(past_range_T / min_first_bin_size) / (number_of_bins_d - 1),
                tol=1e-04,
                maxiter=500,
            )

        while (
            np.linspace(min_scaling, max_scaling, number_of_scalings, retstep=True)[1]
            < min_step_for_scaling
        ):
            number_of_scalings -= 1

        return np.linspace(min_scaling, max_scaling, number_of_scalings)

    def get_past_range(self, number_of_bins_d, first_bin_size, scaling_k):
        """
        Get the past range T of the embedding, based on the parameters d, tau_1 and k.
        """

        return np.sum(
            [
                first_bin_size * 10 ** ((number_of_bins_d - i) * scaling_k)
                for i in range(1, number_of_bins_d + 1)
            ]
        )

    def get_history_dependence(self, data, process):
        """
        Estimate the history dependence for each embedding to all given processes.
        """

        # load estimators
        if self.settings["estimation_method"] == "bbc":
            estbbc = RudeltBBCEstimator()
        elif self.settings["estimation_method"] == "shuffling":
            estshu = RudeltShufflingEstimator()

        # get history dependence
        history_dependence = np.empty(shape=(len(self.embeddings)))
        if self.settings["estimation_method"] == "bbc":
            bbc_term = np.empty(shape=(len(self.embeddings)))

        embedding_count = 0
        for embedding in self.embeddings:
            if self.settings["debug"]:
                print(
                    "Embedding: "
                    + str(embedding[0])
                    + ", "
                    + str(embedding[1])
                    + ", "
                    + str(embedding[2])
                )

            (
                symbol_array,
                past_symbol_array,
                current_symbol_array,
                symbol_array_length,
            ) = data.get_realisations_symbols(
                process,
                embedding[0],
                embedding[1],
                embedding[2],
                self.settings["embedding_step_size"],
                output_spike_times=False,
            )

            if self.settings["estimation_method"] == "bbc":
                I_bbc, R_bbc, bbc_t = estbbc.estimate(
                    symbol_array[0], past_symbol_array[0], current_symbol_array[0]
                )
                history_dependence[embedding_count] = R_bbc
                bbc_term[embedding_count] = bbc_t

                if self.settings["debug"]:
                    print("\tHD: " + str(R_bbc) + " BBC: " + str(bbc_t))

            elif self.settings["estimation_method"] == "shuffling":
                I_sh, R_sh = estshu.estimate(symbol_array[0])
                history_dependence[embedding_count] = R_sh

                if self.settings["debug"]:
                    print("\tHD: " + str(R_sh))

            embedding_count += 1

        if self.settings["estimation_method"] == "bbc":
            return history_dependence, bbc_term
        elif self.settings["estimation_method"] == "shuffling":
            return history_dependence

    def get_bootstrap_history_dependence(
        self, data, embedding, number_of_bootstraps, symbol_block_length=None
    ):
        """
        For a given embedding, return bootstrap replications for R.
        """
        estbbc = RudeltBBCEstimator()
        estshu = RudeltShufflingEstimator()

        if symbol_block_length is not None:
            symbol_block_length = int(symbol_block_length)

        # compute the bootstrap replications
        bs_Rs = np.zeros(number_of_bootstraps)

        for rep in range(number_of_bootstraps):
            (
                bs_symbol_array,
                bs_past_symbol_array,
                bs_current_symbol_array,
            ) = data.get_bootstrap_realisations_symbols(
                self.process,
                embedding[0],
                embedding[1],
                embedding[2],
                self.settings["embedding_step_size"],
                symbol_block_length=symbol_block_length,
            )

            if self.settings["estimation_method"] == "bbc":
                I_bbc, R_bbc, bbc_t = estbbc.estimate(
                    bs_symbol_array[0],
                    bs_past_symbol_array[0],
                    bs_current_symbol_array[0],
                )
                bs_Rs[rep] = R_bbc

                if self.settings["debug"]:
                    print("\tHD: " + str(R_bbc) + " BBC: " + str(bbc_t))

            elif self.settings["estimation_method"] == "shuffling":
                I_sh, R_sh = estshu.estimate(bs_symbol_array[0])
                bs_Rs[rep] = R_sh

                if self.settings["debug"]:
                    print("\tHD: " + str(R_sh))

        return bs_Rs

    def get_temporal_depth_T_D(self, get_R_thresh=False):
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
        embedding_maximising_R_at_T, max_Rs = self.get_embeddings_that_maximise_R()

        Ts = sorted([key for key in max_Rs.keys()])
        Rs = [max_Rs[T] for T in Ts]

        # first get the max history dependence, and if available its bootstrap replications
        max_R, max_R_T = utl.get_max_R_T(max_Rs)

        number_of_bins_d, scaling_k = embedding_maximising_R_at_T[max_R_T]

        embindex = self.embeddings.index((max_R_T, number_of_bins_d, scaling_k))

        bs_Rs = self.bs_history_dependence[embindex]

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

    def get_embeddings_that_maximise_R(
        self, bbc_tolerance=None, dependent_var="T", get_as_list=False
    ):
        """
        For each T (or d), get the embedding for which R is maximised.

        For the bbc estimator, here the bbc_tolerance is applied, ie
        get the unbiased embeddings that maximise R.
        """

        assert dependent_var in ["T", "d"]

        if bbc_tolerance is None:
            bbc_tolerance = np.inf

        max_Rs = {}
        embeddings_that_maximise_R = {}

        for i in range(len(self.embeddings)):
            embedding = self.embeddings[i]
            past_range_T = float(embedding[0])
            number_of_bins_d = int(float(embedding[1]))
            scaling_k = float(embedding[2])
            history_dependence = self.history_dependence[i]

            if self.settings["estimation_method"] == "bbc":
                if (
                    self.bbc_term[i] >= self.settings["bbc_tolerance"]
                ):  # ----------------------- TODO check
                    continue

            if dependent_var == "T":
                if (
                    not past_range_T in embeddings_that_maximise_R
                    or history_dependence > max_Rs[past_range_T]
                ):
                    max_Rs[past_range_T] = history_dependence
                    embeddings_that_maximise_R[past_range_T] = (
                        number_of_bins_d,
                        scaling_k,
                    )
            elif dependent_var == "d":
                if (
                    not number_of_bins_d in embeddings_that_maximise_R
                    or history_dependence > max_Rs[number_of_bins_d]
                ):
                    max_Rs[number_of_bins_d] = history_dependence
                    embeddings_that_maximise_R[number_of_bins_d] = (
                        past_range_T,
                        scaling_k,
                    )

        if get_as_list:
            embeddings = []
            if dependent_var == "T":
                for past_range_T in embeddings_that_maximise_R:
                    number_of_bins_d, scaling_k = embeddings_that_maximise_R[
                        past_range_T
                    ]
                    embeddings += [(past_range_T, number_of_bins_d, scaling_k)]
            elif dependent_var == "d":
                for number_of_bins_d in embeddings_that_maximise_R:
                    past_range_T, scaling_k = embeddings_that_maximise_R[
                        number_of_bins_d
                    ]
                    embeddings += [(past_range_T, number_of_bins_d, scaling_k)]
            return embeddings
        else:
            return embeddings_that_maximise_R, max_Rs

    def get_information_timescale_tau_R(self):
        """
        Get the information timescale tau_R, a characteristic
        timescale of history dependence similar to an autocorrelation
        time.
        """

        max_Rs = self.max_Rs

        Ts = np.array(sorted([key for key in max_Rs.keys()]))
        Rs = np.array([max_Rs[T] for T in Ts])

        R_tot = self.get_R_tot()

        T_0 = self.settings["timescale_minimum_past_range"]

        # get dRs
        dRs = []
        R_prev = 0.0

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

        dRs = np.pad(dRs, (0, len(Rs) - len(dRs)), mode="constant", constant_values=0)

        # compute tau_R
        Ts_0 = np.append([0], Ts)
        dRs_0 = dRs[Ts_0[:-1] >= T_0]

        # Only take into considerations contributions beyond T_0
        Ts_0 = Ts_0[Ts_0 >= T_0]
        norm = np.sum(dRs_0)

        if norm == 0.0:
            tau = 0.0
        else:
            Ts_0 -= Ts_0[0]
            tau = np.dot(((Ts_0[:-1] + Ts_0[1:]) / 2), dRs_0) / norm
        return tau

    def get_R_tot(self, return_averaged_R=False, **kwargs):
        max_Rs = self.max_Rs

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

    def compute_CIs(self, data, target_R="R_max", symbol_block_length=None):
        """
        Compute bootstrap replications of the history dependence estimate
        which can be used to obtain confidence intervals.

        Args:
            data : data_spiketime object
                Input data
            target_R : String
                One of 'R_max', 'R_tot' or 'nonessential'.
                If set to R_max, replications of R are produced for the T at which
                R is maximised.
                If set to R_tot, replications of R are produced for T = T_D (cf
                get_temporal_depth_T_D).
                If set to nonessential, replications of R are produced for each T
                (one embedding per T, cf get_embeddings_that_maximise_R).  These
                are not otherwise used in the analysis and are probably only useful
                if the resulting plot is visually inspected, so in most cases it can
                be set to zero.
            symbol_block_length : int
                The number of symbols that should be drawn in each block for bootstrap resampling
                If it is set to None (recommended), the length is automatically chosen, based
                on heuristics
        """

        assert target_R in ["nonessential", "R_max", "R_tot"]

        number_of_bootstraps = self.settings["number_of_bootstraps_{}".format(target_R)]

        if number_of_bootstraps == 0:
            return

        embedding_maximising_R_at_T, max_Rs = self.get_embeddings_that_maximise_R()
        self.embedding_maximising_R_at_T = embedding_maximising_R_at_T
        self.max_Rs = max_Rs

        if target_R == "nonessential":
            # bootstrap R for unessential Ts (not required for the main analysis)
            embeddings = []
            for past_range_T in embedding_maximising_R_at_T:
                number_of_bins_d, scaling_k = embedding_maximising_R_at_T[past_range_T]
                embeddings += [(past_range_T, number_of_bins_d, scaling_k)]

        elif target_R == "R_max":
            # bootstrap R for the max R, to get a good estimate for the standard deviation
            # which is used to determine R_tot
            max_R, max_R_T = utl.get_max_R_T(max_Rs)
            self.max_R = max_R
            self.max_R_T = max_R_T
            number_of_bins_d, scaling_k = embedding_maximising_R_at_T[max_R_T]
            embeddings = [(max_R_T, number_of_bins_d, scaling_k)]

        elif target_R == "R_tot":
            T_D = self.get_temporal_depth_T_D()
            number_of_bins_d, scaling_k = embedding_maximising_R_at_T[T_D]

            embeddings = [(T_D, number_of_bins_d, scaling_k)]

        for embedding in embeddings:
            embindex = self.embeddings.index(embedding)

            if hasattr(self, "bs_history_dependence"):
                stored_bs_Rs = self.bs_history_dependence[embindex]
            else:
                self.bs_history_dependence = dict()
                stored_bs_Rs = None

            if isinstance(stored_bs_Rs, np.ndarray):
                number_of_stored_bootstraps = len(stored_bs_Rs)
            else:
                number_of_stored_bootstraps = 0

            if not number_of_bootstraps > number_of_stored_bootstraps:
                continue

            bs_R = self.get_bootstrap_history_dependence(
                data,
                embedding,
                number_of_bootstraps - number_of_stored_bootstraps,
                symbol_block_length=symbol_block_length,
            )

            if stored_bs_Rs is not None:
                bs_R = np.concatenate([stored_bs_Rs, bs_R])

            self.bs_history_dependence[embindex] = bs_R

    def analyse_auto_MI(self, spike_times):
        """
        Get the auto MI for the spike times.  If it is available from file, load
        it, else compute it.
        """

        auto_MI_data = {"delay": [], "auto_MI": []}
        auto_MI_dict = {}
        for auto_MI_bin_size in self.settings["auto_MI_bin_size_set"]:
            number_of_delays = (
                int(self.settings["auto_MI_max_delay"] / auto_MI_bin_size) + 1
            )

            # perform the MI analysis
            auto_MI = self.get_auto_MI(spike_times, auto_MI_bin_size, number_of_delays)

            auto_MI_data["delay"] += [number_of_delays]

            auto_MI_dict[auto_MI_bin_size] = auto_MI

        auto_MI_data["auto_MI"] = auto_MI_dict
        self.auto_MI = auto_MI_data

    def get_auto_MI(self, spike_times, bin_size, number_of_delays):
        """
        Compute the auto mutual information in the neuron's activity, a
        measure closely related to history dependence.
        """

        binned_neuron_activity = utl.get_binned_neuron_activity(
            spike_times, bin_size, relative_to_median_activity=True
        )

        p_spike = sum(binned_neuron_activity) / len(binned_neuron_activity)

        self.H_spiking = utl.get_shannon_entropy([p_spike, 1 - p_spike])

        auto_MIs = np.empty(number_of_delays)

        # compute auto MI
        for delay in range(number_of_delays):
            symbol_counts = []
            number_of_symbols = len(binned_neuron_activity) - delay - 1
            symbols = np.array(
                [
                    2 * binned_neuron_activity[i]
                    + binned_neuron_activity[i + delay + 1]
                    for i in range(number_of_symbols)
                ]
            )
            symbol_counts += [
                dict(
                    [
                        (unq_symbol, len(np.where(symbols == unq_symbol)[0]))
                        for unq_symbol in np.unique(symbols)
                    ]
                )
            ]

            symbol_counts = utl.add_up_dicts(symbol_counts)
            number_of_symbols = sum(symbol_counts.values())

            H_joint = utl.get_shannon_entropy(
                [
                    number_of_occurrences / number_of_symbols
                    for number_of_occurrences in symbol_counts.values()
                ]
            )

            # I(X : Y) = H(X) - H(X|Y) = H(X) - (H(X,Y) - H(Y)) = H(X) + H(Y) - H(X,Y)
            # auto_MI = 2 * H_spiking - H_joint
            auto_MIs[delay] = (
                2 - H_joint / self.H_spiking
            )  # normalized auto MI = auto MI / H_spiking

        return auto_MIs

    def optimize(self, data, processes="all"):
        """
        Optimize the embedding parameters of spike time data using the Rudelt history dependence estimator.

        References:

            [1]: L. Rudelt, D. G. Marx, M. Wibral, V. Priesemann: Embedding
                optimization reveals long-lasting history dependence in
                neural spiking activity, 2021, PLOS Computational Biology, 17(6)

            [2]: https://github.com/Priesemann-Group/hdestimator

        implemented in idtxl by Michael Lindner, Göttingen 2021

        Args:
            data : Data_spiketime instance
                raw data for analysis
            processes : list of int
                index of processes;
                spike times are optimized all processes specified in the list separately.

        Returns: # -------------------------------------------------------------------------------------------------------- TODO
            ResultsSingleProcessRudelt instance
                results of Rudelt optimization, see documentation of
                ResultsSingleProcessRudelt()
            if visulization in settings was set True (see class OptimizationRudelt):
                .eps images are created for each optimized process containing:
                    - optimized values for the process
                    - graph for the history dependence
                    - graph for auto mutual information (if calculated)

        """

        # check input data
        if type(data) != Data_spiketime:
            raise ValueError(
                "Input Data nneds to be Data_spiketime object but is defined as: "
                "{0}.".format(type(data))
            )

        # check input process list
        if processes == "all":
            processes = [t for t in range(data.n_processes)]
        elif (type(processes) is list) and (type(processes[0]) is int):
            pass
        else:
            raise ValueError(
                "Processes were not specified correctly: " "{0}.".format(processes)
            )

        if self.settings["debug"]:
            import pprint

            pprint.pprint(self.settings, width=1)

        # open result dict
        results = ResultsSingleProcessRudelt(processes=processes)

        # start optimizing given processes
        process_count = 0
        for process in processes:
            # optimize single process
            single_result = self.optimize_single_run(data, process)

            # add results of single process to result object
            results._add_single_result(
                process_count=process_count,
                process=process,
                settings=self.settings,
                results=single_result,
            )

            process_count += 1

            if self.settings["visualization"] == True:
                filename = Path(self.settings["output_path"]).joinpath(
                    "{}_process{}.svg".format(self.settings["output_prefix"], process)
                )
                utl.hde_visualize_results(results, process, filename)

            # remove results from single process from self
            self.remove_subresults_single_process()

        return results

    def optimize_single_run(self, data, process):
        """
        optimizes a single realisation of spike time data given the process number

        Args:
            data : Data_spiketime instance
                raw data for analysis
            process : int
                index of process;

        Returns:
            DotDict
                with the following keys

                Process : int
                    Process that was optimized
                estimation_method : String
                    Estimation method that was used for optimization
                T_D : float
                    Estimated optimal value for the temporal depth TD
                tau_R :
                    Information timescale tau_R, a characteristic timescale of history
                    dependence similar to an autocorrelation time.
                R_tot : float
                    Estimated value for the total history dependence Rtot,
                AIS_tot : float
                    Estimated value for the total active information storage
                opt_number_of_bins_d : int
                    Number of bins d for the embedding that yields (R̂tot ,T̂D)
                opt_scaling_k : int
                    Scaling exponent κ for the embedding that yields (R̂tot , T̂D)
                opt_first_bin_size : int
                    Size of the first bin τ1 for the embedding that yields (R̂tot , T̂D ),
                history_dependence : array with floating-point values
                    Estimated history dependence for each embedding
                firing_rate : float
                    Firing rate of the neuron/ spike train
                recording_length : float
                    Length of the recording (in seconds)
                H_spiking : float
                    Entropy of the spike times

        if analyse_auto_MI was set to True additionally:
            auto_MI : dict
                numpy array of MI values for each delay
            auto_MI_delays : list of int
                list of delays depending on the given auto_MI_bin_sizes and auto_MI_max_delay
        """

        if type(process) is int:
            pass
        else:
            raise ValueError(
                "Process is not specified correctly: " "{0}.".format(process)
            )

        self.process = process

        # get history dependence
        if self.settings["debug"]:
            print("\n\nGet History dependence\n")

        if self.settings["estimation_method"] == "bbc":
            self.history_dependence, self.bbc_term = self.get_history_dependence(
                data, process
            )
        elif self.settings["estimation_method"] == "shuffling":
            self.history_dependence = self.get_history_dependence(data, process)

        if self.settings["debug"]:
            print("\n\nCompute CI\n")

        # get bootstrap history dependence (CI)
        if self.settings["debug"]:
            print("R_max")
        self.compute_CIs(
            data,
            target_R="R_max",
            symbol_block_length=self.settings["symbol_block_length"],
        )
        if self.settings["debug"]:
            print("R_tot")
        self.compute_CIs(
            data,
            target_R="R_tot",
            symbol_block_length=self.settings["symbol_block_length"],
        )
        if self.settings["debug"]:
            print("R_nonessential")
        self.compute_CIs(
            data,
            target_R="nonessential",
            symbol_block_length=self.settings["symbol_block_length"],
        )

        # analyse auto MI
        if self.settings["analyse_auto_MI"]:
            if self.settings["debug"]:
                print("\nAnalyse auto MI\n")
            spike_times = data.get_spike_times_single(process)
            self.analyse_auto_MI(spike_times)

        # get output values
        T_D = self.get_temporal_depth_T_D()
        tau_R = self.get_information_timescale_tau_R()
        R_tot = self.get_R_tot()
        opt_number_of_bins_d, opt_scaling_k = self.embedding_maximising_R_at_T[T_D]
        max_Rs = self.max_Rs
        mr = np.array(list(max_Rs.items()), dtype=float)
        HD_max_R = mr[:, 1]
        opt_first_bin_size = newton(
            lambda first_bin_size: self.get_past_range(
                opt_number_of_bins_d, first_bin_size, opt_scaling_k
            )
            - T_D,
            0.005,
            tol=1e-03,
            maxiter=100,
        )
        firing_rate = data.get_firingrate(process, self.settings["embedding_step_size"])
        recording_length = data.get_recording_length(process)
        H_spiking = data.get_H_spiking(process, self.settings["embedding_step_size"])

        # get CI bounds
        if not self.settings["return_averaged_R"]:
            embedding = (T_D, opt_number_of_bins_d, opt_scaling_k)
            emb_ind = self.embeddings.index(embedding)
            R_tot_CI_lo, R_tot_CI_hi = utl.get_CI_bounds(
                R_tot,
                self.bs_history_dependence[emb_ind],
                self.settings["bootstrap_CI_use_sd"],
                self.settings["bootstrap_CI_percentile_lo"],
                self.settings["bootstrap_CI_percentile_hi"],
            )
        else:
            R_tot_CI_lo = None
            R_tot_CI_hi = None
        R_tot_CI = [R_tot_CI_lo, R_tot_CI_hi]

        if self.settings["debug"]:
            print("Process: " + str(process))
            print("T_D: " + str(T_D))
            print("tau_R: " + str(tau_R))
            print("R_tot: " + str(R_tot))
            print("R_tot_CI: " + str(R_tot_CI))
            print("opt_number_of_bins_d: " + str(opt_number_of_bins_d))
            print("opt_scaling_k: " + str(opt_scaling_k))
            print("opt_first_bin_size: " + str(opt_first_bin_size))
            print("firing_rate: " + str(firing_rate))
            print("recording_length: " + str(recording_length))
            print("H_spiking: " + str(H_spiking))

        # create output dict
        results = {
            "Process": process,
            "estimation_method": self.settings["estimation_method"],
            "T_D": T_D,
            "tau_R": tau_R,
            "R_tot": R_tot,
            "R_tot_CI": R_tot_CI,
            "AIS_tot": R_tot * H_spiking,
            "opt_number_of_bins_d": opt_number_of_bins_d,
            "opt_scaling_k": opt_scaling_k,
            "opt_first_bin_size": opt_first_bin_size,
            "history_dependence": self.history_dependence,
            "firing_rate": firing_rate,
            "recording_length": recording_length,
            "H_spiking": H_spiking,
            "max_R": max_Rs,
        }

        if self.settings["analyse_auto_MI"]:
            results["auto_MI"] = self.auto_MI.get("auto_MI")
            results["auto_MI_delays"] = self.auto_MI.get("delay")

        results_d = DotDict(results)

        return results_d

    def remove_subresults_single_process(self):
        """delete results from self from single process"""

        del self.bs_history_dependence
        del self.embedding_maximising_R_at_T
        del self.history_dependence
        del self.max_R
        del self.max_Rs
        del self.max_R_T
        del self.process
        # del self.T_D
        if self.settings["analyse_auto_MI"]:
            del self.auto_MI
            del self.H_spiking
        if self.settings["estimation_method"] == "bbc":
            del self.bbc_term
