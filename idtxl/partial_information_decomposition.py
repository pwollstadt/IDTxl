"""Estimate partial information decomposition (PID).

Estimate PID for two source and one target process using different estimators.

Note:
    Written for Python 3.4+

@author: patricia
"""
import numpy as np
from .single_process_analysis import SingleProcessAnalysis
from .estimator import find_estimator


class PartialInformationDecomposition(SingleProcessAnalysis):
    """Set up partial information decomposition for individual processes.

    Set parameters necessary to infer partial information decomposition (PID)
    for two source and one target process. Estimate unique, shared, and
    synergistic information in the two sources about the target. Call
    analyse_network() on the whole network or a set of nodes or call
    analyse_single_target() to estimate PID for a single process. See
    docstrings of the two functions for more information.

    Args:
        settings : dict
            parameters for estimator use and statistics:

            - 'n_perm' - number of permutations for statistical testing
              (default=500)
            - 'alpha' - critical alpha level for statistical significance
              (default=0.05)
            - 'tail' - 'one' or 'two' for one- or two-sided statistical
              testing (default='one')
            - 'pid_calc_name' - estimator to be used for PID calculation
              (for estimator settings see the respective documentation)
    """

    def __init__(self):
        super().__init__()

    def analyse_network(self, settings, data, targets, sources):
        """Estimate partial information decomposition for network nodes.

        Estimate partial information decomposition (PID) for multiple nodes in
        the network.

        Note:
            For a detailed description and references see the documentation of
            the analyse_single_target() method of this class.

        Example:

            >>> n = 20
            >>> alph = 2
            >>> x = np.random.randint(0, alph, n)
            >>> y = np.random.randint(0, alph, n)
            >>> z = np.logical_xor(x, y).astype(int)
            >>> dat = Data(np.vstack((x, y, z)), 'ps', normalise=False)
            >>> settings = {
            >>>     'alpha': 0.1,
            >>>     'alph_s1': alph,
            >>>     'alph_s2': alph,
            >>>     'alph_t': alph,
            >>>     'max_unsuc_swaps_row_parm': 60,
            >>>     'num_reps': 63,
            >>>     'max_iters': 1000,
            >>>     'pid_calc_name': 'SydneyPID'}
            >>> targets = [0, 1, 2]
            >>> sources = [[1, 2], [0, 2], [0, 1]]
            >>> lags = [[1, 1], [3, 2], [0, 0]]
            >>> pid_analysis = PartialInformationDecomposition()
            >>> res = pid_analysis.analyse_network(settings, dat, targets,
            >>>                                    sources)

        Args:
            settings : dict
                parameters for estimation and statistical testing, see
                documentation of analyse_single_target() for details, can
                contain

                - lags : list of lists of ints [optional] - lags in samples
                  between sources and target (default=[[1, 1], [1, 1] ...])

            data : Data instance
                raw data for analysis
            targets : list of int
                index of target processes
            sources : list of lists
                indices of the two source processes for each target, e.g.,
                [[0, 2], [1, 0]], must have the same length as targets

        Returns:
            dict
                results for each process, see documentation of
                analyse_single_target()
        """
        settings.setdefault('verbose', True)
        settings.setdefault('lags', np.array([[1, 1]] * len(targets)))
        if not len(targets) == len(sources) == len(settings['lags']):
            raise RuntimeError('Lists of targets, sources, and lags must have'
                               'the same lengths.')
        list_of_lags = settings['lags']
        results = {}
        for t in range(len(targets)):
            if settings['verbose']:
                print('\n####### analysing target with index {0} from list {1}'
                      .format(t, targets))
            settings['lags'] = list_of_lags[t]
            r = self.analyse_single_target(settings, data, targets[t],
                                           sources[t])
            results[targets[t]] = r
        return results

    def analyse_single_target(self, settings, data, target, sources):
        """Estimate partial information decomposition for a network node.

        Estimate partial information decomposition (PID) for a target node in
        the network.

        # TODO add references

        Example:

            >>> n = 20
            >>> alph = 2
            >>> x = np.random.randint(0, alph, n)
            >>> y = np.random.randint(0, alph, n)
            >>> z = np.logical_xor(x, y).astype(int)
            >>> dat = Data(np.vstack((x, y, z)), 'ps', normalise=False)
            >>> settings = {
            >>>     'alpha': 0.1,
            >>>     'alph_s1': alph,
            >>>     'alph_s2': alph,
            >>>     'alph_t': alph,
            >>>     'max_unsuc_swaps_row_parm': 60,
            >>>     'num_reps': 63,
            >>>     'max_iters': 1000,
            >>>     'pid_calc_name': 'SydneyPID',
            >>>     'lags': [2, 3]}
            >>> pid_analysis = PartialInformationDecomposition()
            >>> res = pid_analysis.analyse_single_target(settings=settings,
            >>>                                          data=dat,
            >>>                                          target=0,
            >>>                                          sources=[1, 2])

        Args:
            settings : dict
                parameters for estimator use and statistics:

                - lags : list of ints [optional] - lags in samples between
                  sources and target (default=[1, 1])
                - for estimator settings see the documentation in the
                  estimators_pid modules

            data : Data instance
                raw data for analysis
            target : int
                index of target processes
            sources : list of ints
                indices of the two source processes for the target

        Returns:
            dict
                unique, shared and synergistic information from both sources,
                statistical significance and p-values, indices of source
                variables, and settings used for estimation
        """
        # Check input and initialise values for analysis.
        self._initialise(settings, data, target, sources)

        # Estimate PID and significance.
        self._calculate_pid(data)

        # Add analyis info.
        results = self.results
        results['settings'] = self.settings
        results['target'] = self.target
        results['source_1'] = self._idx_to_lag([self.sources[0]])
        results['source_2'] = self._idx_to_lag([self.sources[1]])
        self._reset()
        return results

    def _initialise(self, settings, data, target, sources):
        """Check input, set initial or default values for analysis settings."""
        # Check requested PID estimator.
        try:
            EstimatorClass = find_estimator(settings['pid_estimator'])
        except KeyError:
            raise RuntimeError('Estimator was not specified!')
        self._pid_estimator = EstimatorClass(settings)

        settings.setdefault('lags', [1, 1])
        settings.setdefault('verbose', True)
        self.settings = settings

        # Check if provided lags are correct and work with the number of
        # samples in the data.
        if len(self.settings['lags']) != 2:
            raise RuntimeError('List of lags must have length 2.')
        if self.settings['lags'][0] >= data.n_samples:
            raise RuntimeError('Lag 1 ({0}) is larger than the number of '
                               'samples in the data set ({1}).'.format(
                                    self.settings['lags'][0], data.n_samples))
        if self.settings['lags'][1] >= data.n_samples:
            raise RuntimeError('Lag 2 ({0}) is larger than the number of '
                               'samples in the data set ({1}).'.format(
                                    self.settings['lags'][1], data.n_samples))

        # Check if target and sources are provided correctly.
        if type(target) is not int:
            raise RuntimeError('Target must be an integer.')
        if len(sources) != 2:
            raise RuntimeError('List of sources must have length 2.')
        if target in sources:
            raise RuntimeError('The target ({0}) should not be in the list '
                               'of sources ({1}).'.format(target, sources))

        self.current_value = (target, max(self.settings['lags']))
        self.target = target
        # TODO works for single vars only, change to multivariate?
        self.sources = self._lag_to_idx([
                                    (sources[0], self.settings['lags'][0]),
                                    (sources[1], self.settings['lags'][1])])

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

        if self.settings['verbose']:
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

    def _reset(self):
        """Reset instance after analysis."""
        self.__init__()
        del self.results
        del self.settings
        del self._pid_estimator
