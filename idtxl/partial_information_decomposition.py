"""Estimate partial information decomposition (PID).

Estimate PID for two source and one target process using different estimators.

Note:
    Written for Python 3.4+

@author: patricia
"""
from .single_process_analysis import SingleProcessAnalysis
from .estimator import find_estimator

VERBOSE = True


class PartialInformationDecomposition(SingleProcessAnalysis):
    """Set up network analysis using partial information decomposition.

    Set parameters necessary to infer partial information decomposition (PID)
    for two source and one target process. Estimate unique, shared, and
    synergistic information in the two sources about the target.

    Args:
        options : dict
            parameters for estimator use and statistics:

            - 'n_perm' - number of permutations for statistical testing
              (default=500)
            - 'alpha' - critical alpha level for statistical significance
              (default=0.05)
            - 'tail' - 'one' or 'two' for one- or two-sided statistical
              testing (default='one')
            - 'pid_calc_name' - estimator to be used for PID calculation
              (for estimator options see the respective documentation)
    """

    def __init__(self, options):
        try:
            EstimatorClass = find_estimator(options['pid_estimator'])
        except KeyError:
            raise KeyError('Estimator was not specified!')
        self._pid_estimator = EstimatorClass(options)
        self.options = options
        super().__init__()

    def analyse_network(self, data, targets, sources, lags):
        """Estimate partial information decomposition for network nodes.

        Estimate partial information decomposition (PID) for multiple nodes in
        the network.

        Example:

            >>> n = 20
            >>> alph = 2
            >>> x = np.random.randint(0, alph, n)
            >>> y = np.random.randint(0, alph, n)
            >>> z = np.logical_xor(x, y).astype(int)
            >>> dat = Data(np.vstack((x, y, z)), 'ps', normalise=False)
            >>> analysis_opts = {
            >>>     'alpha': 0.1,
            >>>     'alph_s1': alph,
            >>>     'alph_s2': alph,
            >>>     'alph_t': alph,
            >>>     'max_unsuc_swaps_row_parm': 60,
            >>>     'num_reps': 63,
            >>>     'max_iters': 1000,
            >>>     'pid_calc_name': 'pid_sydney'}
            >>> targets = [0, 1, 2]
            >>> sources = [[1, 2], [0, 2], [0, 1]]
            >>> lags = [[1, 1], [3, 2], [0, 0]]
            >>> pid_analysis = Partial_information_decomposition(analysis_opts)
            >>> res = pid_analysis.analyse_network(dat, targets, sources, lags)

        Note:
            For more details on the estimation of PID see documentation of
            class method 'analyse_single_target'.

        Args:
            data : Data instance
                raw data for analysis
            targets : list of int
                index of target processes
            sources : list of lists
                indices of the two source processes for each target, e.g.,
                [[0, 2], [1, 0]], must have the same length as targets
            lags : list of lists
                lags in samples between sources and target, e.g.,
                [[1, 2], [3, 1]], must have the same length as targets

        Returns:
            dict
                results for each target, see documentation of
                'analyse_single_target' for details
        """
        if not len(targets) == len(sources) == len(lags):
            raise RuntimeError('Lists of targets, sources, and lags must have'
                               'the same lengths.')
        results = {}
        for t in range(len(targets)):
            if VERBOSE:
                print('\n####### analysing target with index {0} from list {1}'
                      .format(t, targets))
            r = self.analyse_single_target(data, targets[t], sources[t],
                                           lags[t])
            results[targets[t]] = r
        return results

    def analyse_single_target(self, data, target, sources, lags=None):
        """Estimate partial information decomposition for a network node.

        Estimate partial information decomposition (PID) for a target node in
        the network.

        Example:

            >>> n = 20
            >>> alph = 2
            >>> x = np.random.randint(0, alph, n)
            >>> y = np.random.randint(0, alph, n)
            >>> z = np.logical_xor(x, y).astype(int)
            >>> dat = Data(np.vstack((x, y, z)), 'ps', normalise=False)
            >>> analysis_opts = {
            >>>     'alpha': 0.1,
            >>>     'alph_s1': alph,
            >>>     'alph_s2': alph,
            >>>     'alph_t': alph,
            >>>     'max_unsuc_swaps_row_parm': 60,
            >>>     'num_reps': 63,
            >>>     'max_iters': 1000,
            >>>     'pid_calc_name': 'pid_sydney'}
            >>> pid_analysis = Partial_information_decomposition(analysis_opts)
            >>> res = pid_analysis.analyse_single_target(data=dat,
            >>>                                          target=0,
            >>>                                          sources=[1, 2],
            >>>                                          lags=[2, 3])

        Args:
            data : Data instance
                raw data for analysis
            target : int
                index of target processes
            sources : list of ints
                indices of the two source processes for the target
            lags : list of ints [optional]
                lags in samples between sources and target (default=[1, 1])

        Returns:
            dict
                unique, shared and synergistic information from both sources,
                statistical significance and p-values, indices of source
                variables, and options used for estimation
        """
        # Check input and initialise values for analysis.
        self._initialise(data, target, sources, lags)

        # Estimate PID and significance.
        self._calculate_pid(data)

        # Add analyis info.
        results = self.results
        results['options'] = self.options
        results['target'] = self.target
        results['source_1'] = self._idx_to_lag([self.sources[0]])
        results['source_2'] = self._idx_to_lag([self.sources[1]])
        return results

    def _initialise(self, data, target, sources, lags):
        if type(target) is not int:
            raise RuntimeError('Target must be an integer.')
        if len(sources) != 2:
            raise RuntimeError('List of sources must have length 2.')
        if lags is None:
            self.lags = [1, 1]
        else:
            if len(lags) != 2:
                raise RuntimeError('List of lags must have length 2.')
            if lags[0] >= data.n_samples:
                raise RuntimeError('Lag 1 ({0}) is larger than the number of '
                                   'samples in the data set ({1}).'.format(
                                                   lags[0], data.n_samples))
            if lags[1] >= data.n_samples:
                raise RuntimeError('Lag 2 ({0}) is larger than the number of '
                                   'samples in the data set ({1}).'.format(
                                                   lags[1], data.n_samples))
        if target in sources:
            raise RuntimeError('The target ({0}) should not be in the list '
                               'of sources ({1}).'.format(target, sources))
        self.lags = lags
        self.max_lag = max(self.lags)
        self.current_value = (target, self.max_lag)
        self.target = target
        # TODO works for single vars only, change to multivariate?
        self.sources = self._lag_to_idx([(sources[0], lags[0]),
                                         (sources[1], lags[1])])

    def _calculate_pid(self, data):

        # TODO Discuss how and if the following statistical testing should be
        # included included. Remove dummy results.
        # [orig_pid, sign_1, p_val_1,
        #  sign_2, p_val_2] = stats.unq_against_surrogates(self, data)
        # [orig_pid, sign_shd,
        #  p_val_shd, sign_syn, p_val_syn] = stats.syn_shd_against_surrogates(
        #                                                                 self,
        sign_1 = sign_2 = sign_shd = sign_syn = False
        p_val_1 = p_val_2 = p_val_shd = p_val_syn = 1.0

        target_realisations = data.get_realisations(
                                            self.current_value,
                                            [self.current_value])[0]
        source_1_realisations = data.get_realisations(
                                            self.current_value,
                                            [self.sources[0]])[0]
        source_2_realisations = data.get_realisations(
                                            self.current_value,
                                            [self.sources[1]])[0]
        orig_pid = self._pid_estimator.estimate(
                                s1=source_1_realisations,
                                s2=source_2_realisations,
                                t=target_realisations)

        if VERBOSE:
            print('\nunq information s1: {0:.8f}, s2: {1:.8f}'.format(
                                                           orig_pid['unq_s1'],
                                                           orig_pid['unq_s2']))
            print('shd information: {0:.8f}, syn information: {1:.8f}'.format(
                                                        orig_pid['shd_s1_s2'],
                                                        orig_pid['syn_s1_s2']))
        self.results = orig_pid
        self.results['s1_unq_sign'] = sign_1
        self.results['s2_unq_sign'] = sign_2
        self.results['s1_unq_p_val'] = p_val_1
        self.results['s2_unq_p_val'] = p_val_2
        self.results['syn_sign'] = sign_syn
        self.results['syn_p_val'] = p_val_syn
        self.results['shd_sign'] = sign_shd
        self.results['shd_p_val'] = p_val_shd

        # TODO make mi_against_surrogates in stats more generic, such that
        # it becomes an arbitrary permutation test where one arguemnt gets
        # shuffled and then all arguents are passed to the provided estimator
