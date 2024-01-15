"""Estimate partial information decomposition (PID).

Estimate PID for multiple sources (up to 4 sources) and one target process
using SxPID estimator.

Note:
    Written for Python 3.4+
"""
import numpy as np
from .single_process_analysis import SingleProcessAnalysis
from .estimator import get_estimator
from .results import ResultsMultivariatePID


class MultivariatePID(SingleProcessAnalysis):
    """Perform partial information decomposition for individual processes.

    Perform partial information decomposition (PID) for multiple source
    processes (up to 4 sources) and a target process in the network.
    Estimate unique, shared, and synergistic information in the multiple
    sources about the target. Call analyse_network() on the whole network
    or a set of nodes or call analyse_single_target() to estimate PID for
    a single process. See docstrings of the two functions for more information.

    References:

    - Williams, P. L., & Beer, R. D. (2010). Nonnegative Decomposition of
      Multivariate Information, 1–14. Retrieved from
      http://arxiv.org/abs/1004.2515
    - Makkeh, A. & Gutknecht, A. & Wibral, M. (2020). A Differentiable measure
      for shared information. 1- 27 Retrieved from
      http://arxiv.org/abs/2002.03356

    Attributes:
        target : int
            index of target process
        sources : array type
            multiple of indices of source processes
        settings : dict
            analysis settings
        results : dict
            estimated PID
    """

    def __init__(self):
        super().__init__()

    def analyse_network(self, settings, data, targets, sources):
        """Estimate partial information decomposition for network nodes.

        Estimate, for multiple nodes (target processes), the partial
        information decomposition (PID) for multiple source processes
        (up to 4 sources) and each of these target processes
        in the network.

        Note:
            For a detailed description of the algorithm and settings see
            documentation of the analyse_single_target() method and
            references in the class docstring.

        Example:

            >>> n = 20
            >>> alph = 2
            >>> s1 = np.random.randint(0, alph, n)
            >>> s2 = np.random.randint(0, alph, n)
            >>> s3 = np.random.randint(0, alph, n)
            >>> target1 = np.logical_xor(s1, s2).astype(int)
            >>> target  = np.logical_xor(target1, s3).astype(int)
            >>> data = Data(np.vstack((s1, s2, s3, target)), 'ps',
            >>> normalise=False)
            >>> settings = {
            >>>     'lags_pid': [[1, 1, 1], [3, 2, 7]],
            >>>     'verbose': False,
            >>>     'pid_estimator': 'SxPID'}
            >>> targets = [0, 1]
            >>> sources = [[1, 2, 3], [0, 2, 3]]
            >>> pid_analysis = MultivariatePID()
            >>> results = pid_analysis.analyse_network(settings, data, targets,
            >>>                                        sources)

        Args:
            settings : dict
                parameters for estimation and statistical testing, see
                documentation of analyse_single_target() for details, can
                contain

                - lags_pid : list of lists of ints [optional] - lags in samples
                  between sources and target
                  (default=[[1, 1, ..., 1], [1, 1, ..., 1], ...])

            data : Data instance
                raw data for analysis
            targets : list of int
                index of target processes
            sources : list of lists
                indices of the multiple source processes for each target, e.g.,
                [[0, 1, 2], [1, 0, 3]], all must lists be of the same lenght and
                list of lists must have the same length as targets

        Returns:
            ResultsMultivariatePID instance
                results of network inference, see documentation of
                ResultsMultivariatePID()
        """
        # Set defaults for PID estimation.
        settings.setdefault("verbose", True)
        settings.setdefault(
            "lags_pid", np.array([[1 for i in range(len(sources[0]))]] * len(targets))
        )

        # Check inputs.
        if not len(targets) == len(sources) == len(settings["lags_pid"]):
            raise RuntimeError(
                "Lists of targets, sources, and lags must have" "the same lengths."
            )
        for lis_1 in sources:
            for lis_2 in sources:
                if not len(lis_1) == len(lis_2):
                    raise RuntimeError(
                        "Lists in the list sources must have" "the same lengths."
                    )
                # ^ if
            # ^ for
        # ^ for

        list_of_lags = settings["lags_pid"]

        # Perform PID estimation for each target individually
        results = ResultsMultivariatePID(
            n_nodes=data.n_processes,
            n_realisations=data.n_realisations(),
            normalised=data.normalise,
        )
        for t in range(len(targets)):
            if settings["verbose"]:
                print(
                    "\n####### analysing target with index {0} from list {1}".format(
                        t, targets
                    )
                )
            settings["lags_pid"] = list_of_lags[t]
            res_single = self.analyse_single_target(
                settings, data, targets[t], sources[t]
            )
            results.combine_results(res_single)
        # Get no. realisations actually used for estimation from single target
        # analysis.
        results.data_properties.n_realisations = (
            res_single.data_properties.n_realisations
        )
        return results

    def analyse_single_target(self, settings, data, target, sources):
        """Estimate partial information decomposition for a network node.

        Estimate partial information decomposition (PID) for multiple source
        processes (up to 4 sources) and a target process in the network.

        Note:
            For a description of the algorithm and the method see references in
            the class and estimator docstrings.

        Example:

            >>> n = 20
            >>> alph = 2
            >>> s1 = np.random.randint(0, alph, n)
            >>> s2 = np.random.randint(0, alph, n)
            >>> s3 = np.random.randint(0, alph, n)
            >>> target1 = np.logical_xor(s1, s2).astype(int)
            >>> target  = np.logical_xor(target1, s3).astype(int)
            >>> data = Data(np.vstack((s1, s2, s3, target)), 'ps',
            >>> normalise=False)
            >>> settings = {
            >>>     'verbose' : false,
            >>>     'pid_estimator': 'SxPID',
            >>>     'lags_pid': [2, 3, 1]}
            >>> pid_analysis = MultivariatePID()
            >>> results = pid_analysis.analyse_single_target(settings=settings,
            >>>                                              data=data,
            >>>                                              target=0,
            >>>                                              sources=[1, 2, 3])

        Args: settings : dict parameters for estimator use and statistics:

                - pid_estimator : str - estimator to be used for PID estimation
                  (for estimator settings see the documentation in the
                  estimators_pid modules)
                - lags_pid : list of ints [optional] - lags in samples between
                  sources and target (default=[1, 1, ..., 1])
                - verbose : bool [optional] - toggle console output
                  (default=True)

            data : Data instance
                raw data for analysis
            target : int
                index of target processes
            sources : list of ints
                indices of the multiple source processes for the target

        Returns: ResultsMultivariatePID instance results of
            network inference, see documentation of
            ResultsPID()
        """
        # Check input and initialise values for analysis.
        self._initialise(settings, data, target, sources)

        # Estimate PID and significance.
        self._calculate_pid(data)

        # Add analyis info.
        results = ResultsMultivariatePID(
            n_nodes=data.n_processes,
            n_realisations=data.n_realisations(self.current_value),
            normalised=data.normalise,
        )
        results._add_single_result(
            settings=self.settings, target=self.target, results=self.results
        )
        self._reset()
        return results

    def _initialise(self, settings, data, target, sources):
        """Check input, set initial or default values for analysis settings."""
        # Check requested PID estimator.
        assert "pid_estimator" in settings, "Estimator was not specified!"
        self._pid_estimator = get_estimator(settings["pid_estimator"], settings)

        self.settings = settings.copy()
        self.settings.setdefault("lags_pid", [1 for i in range(len(sources))])
        self.settings.setdefault("verbose", True)

        # Check if provided lags are correct and work with the number of
        # samples in the data.
        if len(self.settings["lags_pid"]) not in [2, 3, 4]:
            raise RuntimeError("List of lags must have length 2 or 3 or 4.")
        # number of lags is equal to number of sources
        if not len(self.settings["lags_pid"]) == len(sources):
            raise RuntimeError(
                "List of lags must have same length as the list sources."
            )
        for i in range(len(self.settings["lags_pid"])):
            if self.settings["lags_pid"][0] >= data.n_samples:
                raise RuntimeError(
                    "Lag {0} ({1}) is larger than the number of samples in the data "
                    "set ({2}).".format(i, self.settings["lags_pid"][i], data.n_samples)
                )

        # Check if target and sources are provided correctly.
        if type(target) is not int:
            raise RuntimeError("Target must be an integer.")
        if len(sources) not in [2, 3, 4]:
            raise RuntimeError("List of sources must have length 2 or 3 or 4.")
        if target in sources:
            raise RuntimeError(
                "The target ({0}) should not be in the list "
                "of sources ({1}).".format(target, sources)
            )

        self.current_value = (target, max(self.settings["lags_pid"]))
        self.target = target
        # TODO works for single vars only, change to multivariate?
        self.sources = self._lag_to_idx(
            [(sources[i], self.settings["lags_pid"][i]) for i in range(len(sources))]
        )

    def _calculate_pid(self, data):
        # TODO Discuss how and if the following statistical testing should be
        # included included. Remove dummy results.
        # [orig_pid, sign_1, p_val_1,
        #  sign_2, p_val_2] = stats.unq_against_surrogates(self, data)
        # [orig_pid, sign_shd,
        #  p_val_shd, sign_syn, p_val_syn] = stats.syn_shd_against_surrogates(
        #                                                                 self,
        # sign_1 = sign_2 = sign_shd = sign_syn = False
        # p_val_1 = p_val_2 = p_val_shd = p_val_syn = 1.0

        target_realisations = data.get_realisations(
            self.current_value, [self.current_value]
        )[0]

        # CHECK! make sure self.source has the same idx as sources
        data.get_realisations(self.current_value, [self.sources[0]])[0]
        list_sources_var_realisations = [
            data.get_realisations(self.current_value, [self.sources[i]])[0]
            for i in range(len(self.sources))
        ]

        orig_pid = self._pid_estimator.estimate(
            s=list_sources_var_realisations, t=target_realisations
        )

        self.results = orig_pid
        for i in range(len(self.sources)):
            self.results["source_" + str(i + 1)] = self._idx_to_lag([self.sources[i]])
        # ^ for
        self.results["selected_vars_sources"] = [
            self.results["source_" + str(i + 1)][0] for i in range(len(self.sources))
        ]
        self.results["current_value"] = self.current_value
        # self.results['unq_s1_sign'] = sign_1
        # self.results['unq_s2_sign'] = sign_2
        # self.results['unq_s1_p_val'] = p_val_1
        # self.results['unq_s2_p_val'] = p_val_2
        # self.results['syn_sign'] = sign_syn
        # self.results['syn_p_val'] = p_val_syn
        # self.results['shd_sign'] = sign_shd
        # self.results['shd_p_val'] = p_val_shd

        # TODO make mi_against_surrogates in stats more generic, such that
        # it becomes an arbitrary permutation test where one arguemnt gets
        # shuffled and then all arguents are passed to the provided estimator

    def _reset(self):
        """Reset instance after analysis."""
        self.__init__()
        del self.results
        del self.settings
        del self._pid_estimator
