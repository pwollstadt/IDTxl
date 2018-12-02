"""Perform inference statistics on groups of data."""
import copy as cp
import numpy as np
from scipy.special import binom
from .estimator import find_estimator
from . import stats
from . import idtxl_utils as utils
from .network_analysis import NetworkAnalysis
from .results import ResultsNetworkComparison, DotDict


class NetworkComparison(NetworkAnalysis):
    """Set up network comparison between two experimental conditions.

    The class provides methods for the comparison of networks inferred from
    data recorded under two experimental conditions A and B. Four statistical
    tests are implemented:

    +----------------------+------------+-------------------------------------+
    |units of observation/ |stats_type  |example                              |
    |comparison type       |            |                                     |
    +======================+============+=====================================+
    | replications/        |dependent   |base line (A) vs. task (B)           |
    | **within** a subject +------------+-------------------------------------+
    |                      |independent |detect house (A) vs. face (B)        |
    +----------------------+------------+-------------------------------------+
    | sets of data/        |dependent   |patients (A) vs. matched controls (B)|
    | **between** subjects +------------+-------------------------------------+
    |                      |independent |male (A) vs. female (B) participants |
    +----------------------+------------+-------------------------------------+

    Depending on the units of observations, one of two statistics methods can
    be used: compare_within() and compare_between(). The stats_type is passed
    as an analysis setting, see the documentation of the two methods for
    details.

    Note that for network inference methods that use an embedding, i.e., a
    collection of variables in the source, the joint information in all
    variables about the target is used as a test statistic.
    """

    def __init__(self):
        super().__init__()

    def compare_links_within(self, settings, link_a, link_b, network, data):
        """Compare two links within the same network.

        Compare two links within the same network. Check if information
        transfer is different from information transfer in a second link.

        Note that both links have to be part of the inferred network, i.e.,
        there has to be significant effective connectivity for both links.

        Args:
            settings : dict
                parameters for estimation and statistical testing

                - stats_type : str - 'dependent' or 'independent' for
                  dependent or independent units of observation
                - cmi_estimator : str - estimator to be used for CMI
                  calculation (for estimator settings see the documentation in
                  the estimators_* modules)
                - tail_comp : str [optional] - test tail, 'one' for one-sided
                  test A > B, 'two' for two-sided test (default='two')
                - n_perm_comp : int [optional] - number of permutations
                  (default=500)
                - alpha_comp : float - critical alpha level for statistical
                  significance (default=0.05)
                - permute_in_time : bool [optional] - if True, create
                  surrogates by shuffling data over time. See
                  Data.permute_samples() for settings for further options for
                  surrogate creation
                - verbose : bool [optional] - toggle console output
                  (default=True)

            link_a : array type
                first link, array type with two entries [source target]
            link_b : array type
                second link, array type with two entries [source target]
            network : dict
                results from network inference
            data : Data object
                data from which network was inferred

        Returns
            ResultsNetworkComparison object
                results of network inference, see documentation of
                ResultsNetworkComparison()
        """
        # Check input and analysis parameters.
        self._initialise(settings)
        self._check_n_replications(data, data)
        self._create_union(network)
        if not self._link_exists(link_a):
            raise RuntimeError('Link A is not part of the network.')
        if not self._link_exists(link_b):
            raise RuntimeError('Link B is not part of the network.')

        # Calculate the test statistic as the difference of information
        # transfer between link A and link B.
        te_a = self.calculate_link_te(data, link_a[1], sources=link_a[0])
        te_b = self.calculate_link_te(data, link_b[1], sources=link_b[0])
        self.cmi_diff = te_a - te_b

        # Create surrogate distribution as differences of information transfer
        # estimates of shuffled data. Determine significance of test statistic
        # against surrogate distribution.
        surrogates_a = self._get_surrogates_target(
            data, target=link_a[1], sources=link_a[0])
        surrogates_b = self._get_surrogates_target(
            data, target=link_b[1], sources=link_b[0])
        self.cmi_surr = surrogates_a[0] - surrogates_b[0]
        [sig, pvalue] = stats._find_pvalue(statistic=self.cmi_diff,
                                           distribution=self.cmi_surr,
                                           alpha=self.settings['alpha_comp'],
                                           tail=self.settings['tail_comp'])

        # Create results object.
        results = ResultsNetworkComparison(
            n_nodes=data.n_processes,
            n_realisations=data.n_realisations(self.current_value),
            normalised=data.normalise)
        # Return both the absolute difference and the direction of the
        # effect. Returning just the difference and evalutating the sign
        # does not give the direction of the effect if one or both values
        # are negative (which may happen due to estimator bias).
        self._union_indices_to_lags()
        results._add_results(
            settings=self.settings,
            union_network=self.union,
            results={
                'cmi_diff_abs': {link_a[1]: np.abs(self.cmi_diff),
                                 link_b[1]: np.abs(self.cmi_diff)},
                'a>b': {link_a[1]: [te_a > te_b], link_b[1]: [te_a > te_b]},
                'pval': {link_a[1]: [pvalue], link_b[1]: [pvalue]},
                'cmi_surr': self.cmi_surr,
            })
        self._reset()  # remove attributes
        return results

    def compare_within(self, settings, network_a, network_b, data_a, data_b):
        """Compare networks inferred under two conditions within one subject.

        Compare two networks inferred from data recorded under two different
        experimental conditions within one subject (units of observations are
        replications of one experimental condition within one subject).

        Args:
            settings : dict
                parameters for estimation and statistical testing

                - stats_type : str - 'dependent' or 'independent' for
                  dependent or independent units of observation
                - cmi_estimator : str - estimator to be used for CMI
                  calculation (for estimator settings see the documentation in
                  the estimators_* modules)
                - tail_comp : str [optional] - test tail, 'one' for one-sided
                  test A > B, 'two' for two-sided test (default='two')
                - n_perm_comp : int [optional] - number of permutations
                  (default=500)
                - alpha_comp : float - critical alpha level for statistical
                  significance (default=0.05)
                - permute_in_time : bool [optional] - if True, create
                  surrogates by shuffling data over time. See
                  Data.permute_samples() for settings for further options for
                  surrogate creation
                - verbose : bool [optional] - toggle console output
                  (default=True)

            network_a : dict
                results from network inference, condition a
            network_b : dict
                results from network inference, condition b
            data_a : Data object
                data from which network_a was inferred
            data_b : Data object
                data from which network_b was inferred

        Returns
            ResultsNetworkComparison object
                results of network inference, see documentation of
                ResultsNetworkComparison()
        """
        # Check input and analysis parameters.
        self._initialise(settings)
        self._check_n_replications(data_a, data_b)
        self._check_equal_realisations(data_a, data_b)

        # Main comparison.
        print('\n-------------------------- (1) create union of networks')
        self._create_union(network_a, network_b)
        print('\n-------------------------- (2) calculate differences in TE '
              'values')
        self._calculate_cmi_diff_within(data_a, data_b)
        print('\n-------------------------- (3) create surrogate distribution')
        self._create_surrogate_distribution_within(data_a, data_b)
        print('\n-------------------------- (4) determine p-value')
        self._p_value_union()

        self._union_indices_to_lags()
        results = ResultsNetworkComparison(
            n_nodes=data_a.n_processes,
            n_realisations=data_a.n_realisations(self.current_value),
            normalised=data_a.normalise)
        # Return both the absolute difference and the direction of the
        # effect. Returning just the difference and evalutating the sign
        # does not give the direction of the effect if one or both values
        # are negative (which may happen due to estimator bias).
        results._add_results(
            settings=self.settings,
            union_network=self.union,
            results={
                'cmi_diff_abs': self._get_abs_diff(self.cmi_diff),
                'a>b': self.cmi_comp,
                'pval': self.pvalue,
                'cmi_surr': self.cmi_surr,
            })
        self._reset()  # remove attributes
        return results

    def compare_between(self, settings, network_set_a, network_set_b,
                        data_set_a, data_set_b):
        """Compare networks inferred under two conditions between subjects.

        Compare two sets of networks inferred from two sets of data recorded
        under different experimental conditions within multiple subjects, i.e.,
        data have been recorded from subjects assigned to one of two
        experimental conditions (units of observations are subjects).

        Args:
            settings : dict
                parameters for estimation and statistical testing, see
                documentation of compare_within() for details
            network_set_a : numpy array of dicts
                results from network inference for multiple subjects observed
                under condition a
            network_set_b : numpy array of dicts
                results from network inference for multiple subjects observed
                under condition b
            data_a : numpy array of Data objects
                set of data from which network_set_a was inferred
            data_b : numpy array of Data objects
                set of data from which network_set_b was inferred

        Returns
            ResultsNetworkComparison object
                results of network inference, see documentation of
                ResultsNetworkComparison()
        """
        # Check input and analysis parameters.
        self._initialise(settings)
        self._check_n_subjects(data_set_a, data_set_b)
        data_all = np.hstack((data_set_a, data_set_b))
        self._check_equal_realisations(*data_all)
        network_all = np.hstack((network_set_a, network_set_b))

        # Main comparison.
        print('\n-------------------------- (1) create union of networks')
        self._create_union(*network_all)
        print('\n-------------------------- (2) calculate differences in TE '
              'values')
        self._calculate_cmi_diff_between(data_set_a, data_set_b)
        print('\n-------------------------- (3) create surrogate distribution')
        self._create_surrogate_distribution_between()
        print('\n-------------------------- (4) determine p-value')
        self._p_value_union()

        self._union_indices_to_lags()
        results = ResultsNetworkComparison(
            n_nodes=data_all[0].n_processes,
            n_realisations=data_all[0].n_realisations(self.current_value),
            normalised=data_all[0].normalise)
        # Return both the absolute difference and the direction of the
        # effect. Returning just the difference and evalutating the sign
        # does not give the direction of the effect if one or both values
        # are negative (which may happen due to estimator bias).
        results._add_results(
            settings=self.settings,
            union_network=self.union,
            results={
                'cmi_diff_abs': self._get_abs_diff(self.cmi_diff),
                'a>b': self.cmi_comp,
                'pval': self.pvalue,
                'cmi_surr': self.cmi_surr,
            })
        self._reset()  # remove attributes
        return results

    def calculate_link_te(self, data, target, sources='all'):
        """Calculate the information transfer for whole links into a target.

        Calculate the information transfer for whole links as the joint
        information transfer from all variables selected for a single source
        process into the target. The information transfer is calculated
        conditional on the target's past and, for multivariate TE, conditional
        on selected variables from further sources in the network.

        If sources is set to 'all', a list of information transfer values is
        returned. If sources is set to a single source index, the information
        transfer from this source to the target is returned.

        Args:
            data : Data instance
                raw data for analysis
            target : int
                index of target process
            sources : list of ints | 'all' [optional]
                return estimates for links from selected or all sources into
                the target (default='all')

        Returns:
            numpy array
                information transfer estimate for each link
        """
        # Get lists of source and target variables. Return empty array if there
        # is no significant source variable for requested target.
        source_vars = self.union._single_target[target]['selected_vars_sources']
        target_vars = self.union._single_target[target]['selected_vars_target']
        if not source_vars:
            return np.array([])
        current_value = (target, self.union['max_lag'])
        return self._calculate_single_link(
            data, current_value, source_vars, target_vars, sources=sources)

    def _check_n_replications(self, data_a, data_b):
        """Check if no. replications is sufficient request no. permutations."""
        assert data_a.n_replications == data_b.n_replications, (
                            'Unequal no. replications in the two data sets.')
        n_replications = data_a.n_replications
        if self.settings['stats_type'] == 'dependent':
            if 2**n_replications < self.settings['n_perm_comp']:
                raise RuntimeError('The number of replications {0} in the data'
                                   ' is not sufficient to allow for the '
                                   'requested no. permutations {1}'.format(
                                            n_replications,
                                            self.settings['n_perm_comp']))
        elif self.settings['stats_type'] == 'independent':
            if (binom(2*n_replications, n_replications) <
                    self.settings['n_perm_comp']):
                raise RuntimeError('The number of replications {0} in the data'
                                   ' is not sufficient to allow for the '
                                   'requested no. permutations {1}'.format(
                                                n_replications,
                                                self.settings['n_perm_comp']))
        else:
            raise RuntimeError('Unknown ''stats_type''!')

    def _check_n_subjects(self, data_set_a, data_set_b):
        """Check if no. subjects is sufficient request no. permutations."""
        if self.settings['stats_type'] == 'dependent':
            assert len(data_set_a) == len(data_set_b), (
                    'The number of data sets is not equal between conditions.')
            n_data_sets = len(data_set_a)
            if 2**n_data_sets < self.settings['n_perm_comp']:
                raise RuntimeError('The number of data sets per condition {0} '
                                   'is not sufficient to enable the '
                                   'requested no. permutations {1}'.format(
                                                n_data_sets,
                                                self.settings['n_perm_comp']))
        elif self.settings['stats_type'] == 'independent':
            max_len = max(len(data_set_a), len(data_set_b))
            total_len = len(data_set_a) + len(data_set_b)
            if binom(total_len, max_len) < self.settings['n_perm_comp']:
                raise RuntimeError('The total number of data sets {0} is not '
                                   'sufficient to enable the requested no. '
                                   'permutations {1}'.format(
                                                total_len,
                                                self.settings['n_perm_comp']))
        else:
            raise RuntimeError('Unknown ''stats_type''!')
        self.settings['n_subjects'] = [len(data_set_a), len(data_set_b)]

    def _check_equal_realisations(self, *data_sets):
        """Check if all data sets have an equal no. realisations."""
        n_realisations = data_sets[0].n_realisations()
        for d in data_sets:
            if d.n_realisations() != n_realisations:
                raise RuntimeError('Unequal no. realisations between data '
                                   'sets.')

    def _create_union(self, *networks):
        """Create the union from a set of individual networks."""
        # Collect targets over all networks
        targets_union = []
        for nw in networks:
            targets_union = targets_union + nw.targets_analysed
        targets_union = np.unique(np.array(targets_union))

        # Get the maximum lags from the networks, we need this to get
        # realisations of variables later on.
        self.union = DotDict({})
        self.union['targets_analysed'] = targets_union
        self.union['max_lag'] = (
            networks[0]._single_target[targets_union[0]].current_value[1])

        # Get the union of sources for each target in the union network.
        self.union._single_target = DotDict({})
        for t in targets_union:
            self.union._single_target[t] = DotDict({})
            self.union._single_target[t]['selected_vars_sources'] = []
            self.union._single_target[t]['selected_vars_target'] = []
            for nw in networks:
                # Check if the max_lag is the same for each network going into
                # the comparison.
                try:
                    lag = nw._single_target[t].current_value[1]
                    if self.union['max_lag'] != lag:
                        raise ValueError('Networks seem to have been analyzed '
                                         'using different lags.')
                except KeyError:
                    pass

                # Get the conditionals from source and target for the current
                # network and target (convert them from sample lags to indices
                # before adding them). Use an empty array if no sources were
                # selected for that target.
                try:
                    cond_src = self._lag_to_idx(
                        nw._single_target[t].selected_vars_sources,
                        self.union['max_lag'])
                except KeyError:
                    cond_src = []
                try:
                    cond_tgt = self._lag_to_idx(
                        nw._single_target[t]['selected_vars_target'],
                        self.union['max_lag'])
                except KeyError:
                    cond_tgt = []

                # Add conditional if it isn't already in the union network.
                for c in cond_src:
                    if c not in self.union._single_target[t]['selected_vars_sources']:
                        self.union._single_target[t]['selected_vars_sources'].append(c)
                for c in cond_tgt:
                    if c not in self.union._single_target[t]['selected_vars_target']:
                        self.union._single_target[t]['selected_vars_target'].append(c)
            self.union._single_target[t].sources = np.unique(
                np.array([s[0] for s in (
                    self.union._single_target[t]['selected_vars_sources'])]))

    def _calculate_cmi_diff_within(self, data_a, data_b):
        """Calculate the difference in CMI between conditions within a subject.

        Calculate the difference in the conditional mutual information (CMI)
        for each source > target combination in the union network between data
        recorded under two experimental conditions within one subject. Compare
        the absolute mean TE values between the two groups.

        Args:
            data_a : Data instance
                raw data recorded in condition A
            data_a : Data instance
                raw data recorded in condition B
        """
        # Re-calculate CMI for both data objects using the union network mask.
        cmi_a = self._calculate_cmi_all_links(data_a)
        cmi_b = self._calculate_cmi_all_links(data_b)
        self.cmi_diff = self._calculate_diff(cmi_a, cmi_b)
        # Compare raw TE values between conditions.
        self.cmi_comp = self._compare_union_cmi_within(cmi_a, cmi_b)

    def _calculate_cmi_diff_between(self, data_set_a, data_set_b):
        """Calculate the difference in CMI between two groups of subjects.

        Calculate the difference in the conditional mutual information (CMI)
        for each source > target combination in the union network between data
        sets recorded from subjects measured under one of two experimental
        conditions. Compare the absolute mean TE values between the two groups.

        Returns:
            numpy array
                CMI differences
        """
        # Re-alculate CMI for each data object using the union network mask.
        cmi_set_a = []
        for d in data_set_a:
            cmi_set_a.append(self._calculate_cmi_all_links(d))
        cmi_set_b = []
        for d in data_set_b:
            cmi_set_b.append(self._calculate_cmi_all_links(d))
        self.cmi_diff = self._calculate_diff_of_mean(cmi_set_a, cmi_set_b)
        # Compare raw TE values between conditions.
        self.cmi_comp = self._compare_union_cmi_between(cmi_set_a, cmi_set_b)
        # Keep sets of union CMI, these are later reused to create surrogates.
        self.cmi_set_a = cmi_set_a
        self.cmi_set_b = cmi_set_b
        # TODO Idea: loop over pairs of data in data_set_a and *_b and feed it
        # to the within function? BUT: such an implementation doesn't allow for
        # unbalanced designs, which is a problem and needs to be changed in the
        # within function as well

    def _calculate_diff_of_mean(self, cmi_set_a, cmi_set_b, permute=False):
        """Calculate the difference of the means of CMI for two data sets.

        Calculate the difference of the mean conditional mutual information
        (CMI) of each source > target combination in the union network for a
        set of data recorded under experimental condition a and a set of data
        recorded under experimental condition b. The mean is taken once over
        all data objects in data_set_a and once over data in data_set_b. If
        permute is set to True, data objects are permuted between condition a
        and b before the difference of the mean is calculated to create
        surrogate data sets. These surrogate data can be used as test
        distribution when testing the orginal difference of the means.
        """
        if permute:
            # Permute data obejcts between conditions a and b before
            # calculating the CMI.
            cmi_set_all = np.array(cmi_set_a + cmi_set_b)
            new_partition_a = np.random.choice(range(len(cmi_set_all)),
                                               size=len(cmi_set_a),
                                               replace=False)
            new_partition_b = np.array(list(set(range(0, len(cmi_set_all))) -
                                            set(new_partition_a)))
            cmi_set_a_perm = cmi_set_all[new_partition_a]
            cmi_set_b_perm = cmi_set_all[new_partition_b]

            return self._calculate_diff(self._calculate_mean(cmi_set_a_perm),
                                        self._calculate_mean(cmi_set_b_perm))

        else:
            return self._calculate_diff(self._calculate_mean(cmi_set_a),
                                        self._calculate_mean(cmi_set_b))

    def _calculate_cmi_all_links(self, data, permuted=False):
        """Calculate CMI for each source>target combi in the union network."""
        cmi = {}
        for t in self.union.targets_analysed:
            cmi_temp = []

            # if there are no sources for the current target, continue to next
            if not self.union._single_target[t]['selected_vars_sources']:
                cmi[t] = np.array(cmi_temp)
                continue

            # Calculate the TE for a link, i.e., the joint information all
            # variables in the source have about the target.
            cmi[t] = self.calculate_link_te(data=data, target=t)
        return cmi

    def _compare_union_cmi_between(self, cmi_set_a, cmi_set_b):
        """Compare mean TE between conditions to get direction of effect."""
        cmi_comp = {}
        cmi_set_a_mean = self._calculate_mean(cmi_set_a)
        cmi_set_b_mean = self._calculate_mean(cmi_set_b)
        for t in self.union.targets_analysed:
            cmi_comp[t] = cmi_set_a_mean[t] > cmi_set_b_mean[t]
        return cmi_comp

    def _compare_union_cmi_within(self, cmi_a, cmi_b):
        """Compare TE between conditions to get direction of effect."""
        cmi_comp = {}
        for t in self.union.targets_analysed:
            cmi_comp[t] = cmi_a[t] > cmi_b[t]
        return cmi_comp

    def _get_abs_diff(self, cmi_diff):
        """Get the absolute value for each difference in the union network."""
        cmi_diff_abs = {}
        for t in self.union.targets_analysed:
            cmi_diff_abs[t] = np.abs(cmi_diff[t])
        return cmi_diff_abs

    def _calculate_cmi_all_links_permuted(self, data_a, data_b):
        """Calculate surrogate CMI for union network.

        Calculate conditional mutual information (CMI) for each source > target
        combination in the union network after permuting realisations of
        sources between the two data sets (coming from two conditions).
        Results can be used in a surrogate permutation test of the original CMI
        in the two data sets.
        """
        cmi_a = {}
        cmi_b = {}
        for t in self.union.targets_analysed:
            cmi_temp_a = []
            cmi_temp_b = []
            # If there are no sources for current target, continue  to the next
            if not self.union._single_target[t]['selected_vars_sources']:
                cmi_a[t] = np.array(cmi_temp_a)
                cmi_b[t] = np.array(cmi_temp_b)
                continue

            # Get full conditioning set for current target.
            idx_cond_full = (
                self.union._single_target[t]['selected_vars_target'] +
                self.union._single_target[t]['selected_vars_sources'])
            # Get realisations, where realisations are permuted/swapped
            # replication-wise between two data sets (e.g., from different
            # conditions)
            [cond_full_perm_a,
             cur_val_perm_a,
             cond_full_perm_b,
             cur_val_perm_b] = self._get_permuted_replications(
                 data_a, data_b, t)
            # Calculate CMI from each source to current target t from permuted
            # data
            n_sources = len(
                self.union._single_target[t]['selected_vars_sources'])
            cmi_temp_a = np.zeros(n_sources)
            cmi_temp_b = np.zeros(n_sources)
            for (i, idx_source) in enumerate(
                        self.union._single_target[t]['selected_vars_sources']):
                # Get realisations of current source from the set of all
                # surrogate conditionals and calculate the CMI. Do this for
                # both conditions.
                [temp_cond_real_a, source_real_a] = utils.separate_arrays(
                    idx_cond_full, idx_source, cond_full_perm_a)
                cmi_temp_a[i] = self._cmi_estimator.estimate(
                    cur_val_perm_a, source_real_a, temp_cond_real_a)
                [temp_cond_real_b, source_real_b] = utils.separate_arrays(
                    idx_cond_full, idx_source, cond_full_perm_b)
                cmi_temp_b[i] = self._cmi_estimator.estimate(
                    cur_val_perm_b, source_real_b, temp_cond_real_b)

            cmi_a[t] = cmi_temp_a
            cmi_b[t] = cmi_temp_b

        return cmi_a, cmi_b

    def _calculate_mean(self, cmi_set):
        """Calculate the mean CMI over multiple data sets for all targets."""
        if type(cmi_set) is dict:
            raise TypeError('Input needs to be 1-D array-like of dicts.')
        # The output is a dictionary with targets as keys. The entry for each
        # target is a list of mean values for each source. The order of sources
        # corresponds to the list of sources in
        # self.union._single_target[t].sources.
        cmi_mean = {}
        for t in self.union.targets_analysed:
            n_sources = len(self.union._single_target[t].sources)
            n_datasets = len(cmi_set)
            cmi_mean[t] = np.zeros(n_sources)
            for i_source in range(n_sources):
                temp = np.zeros(n_datasets)
                for (i_data, c) in enumerate(cmi_set):
                    temp[i_data] = c[t][i_source]
                cmi_mean[t][i_source] = np.mean(temp)
        return cmi_mean

    def _calculate_diff(self, cmi_a, cmi_b):
        """Calculate the difference between two CMI estimates for all targets.

        Calculate the differene in CMI for each source > target combination in
        cmi_a/_b. The inputs are assumed to be dictionaries with one entry for
        each target and for each target, cmi values are given as a numpy
        array.
        """
        cmi_diff = {}
        for t in self.union.targets_analysed:
            cmi_diff[t] = cmi_a[t] - cmi_b[t]
        return cmi_diff

    def _create_surrogate_distribution_within(self, data_a, data_b):
        """Create the surrogate distribution for network inference.

        Create distribution by permuting realisations between conditions and
        re-calculating the conditional mutual information (CMI). Realisations
        are shuffled as whole replications, the permutation strategy depends on
        the stats type set in the class instance (dependent or independent):

        For a dependent test, realisations are assumed to be ordered and
        related, i.e., the first replication in condition A is related to the
        first replication in condition B, thus, replications are swapped or
        exchanged between conditions without changing their rank in the data
        set:

        A_1    B_1        ->        B_1    A_1
        A_2    B_2        ->        A_2    B_2
        A_3    B_3        ->        A_3    B_3
        A_4    B_4        ->        B_4    A_4
        A_5    B_5        ->        B_5    A_5
        ...                         ...

        ; for an independent test, replications are not assumed to depend on
        each other, thus, replications are randomply permuted between groups:

        A_1    B_1        ->        A_3    A_1
        A_2    B_2        ->        B_2    B_4
        A_3    B_3        ->        B_3    B_3
        A_4    B_4        ->        A_5    A_4
        A_5    B_5        ->        B_1    A_2
        ...                         ...

        Args:
            data_a : Data instance
                first set of raw data
            data_a : Data instance
                second set of raw data
        """
        self.cmi_surr = {}
        for t in self.union.targets_analysed:
            surrogates_a = self._get_surrogates_target(data_a, t)
            surrogates_b = self._get_surrogates_target(data_b, t)
            self.cmi_surr[t] = np.zeros((  # save surrogates as 2D-array
                len(self.union._single_target[t].sources),
                self.settings['n_perm_comp']))
            for (i, s) in enumerate(self.union._single_target[t].sources):
                self.cmi_surr[t][i, :] = surrogates_a[s] - surrogates_b[s]

    def _get_surrogates_target(self, data, target, sources='all'):
        # Get lists of source and target variables, and the list of significant
        # sources for current target
        source_vars = self.union._single_target[target]['selected_vars_sources']
        target_vars = self.union._single_target[target]['selected_vars_target']
        if sources == 'all':
            sources = np.unique(np.array([s[0] for s in source_vars]))
        else:
            sources = np.array([sources])

        # Get realisations of target variables and the current value, constant
        # over sources. Permute current value realisations to generate
        # surrogates if requested.
        current_value = (target, self.union['max_lag'])
        target_realisations = data.get_realisations(
            current_value, target_vars)[0]
        current_value_surrogates = stats._get_surrogates(
            data, current_value, [current_value],
            n_perm=self.settings['n_perm_comp'], perm_settings=self.settings)

        # Calculate TE for each link, i.e., for a single source and the target
        te_surrogates = {}
        for s in sources:

            # Separate selected source variables in variables belonging to the
            # current link and variables belonging to the conditioning set
            link_vars = [i for i in source_vars if i[0] == s]
            conditional_vars = [i for i in source_vars if i[0] != s]
            # Get realisations for the current link's source variables
            source_realisations = data.get_realisations(
                current_value, link_vars)[0]
            # Get realisations for the conditioning set, consisting of
            # remaining source variables and target realisations. Handle empty
            # sets: these may occur if network comparison is carried out for
            # results from MI network inference.
            if not conditional_vars and not target_vars:
                conditional_realisations = None
            elif not conditional_vars and target_vars:
                conditional_realisations = target_realisations
            elif conditional_vars and not target_vars:
                conditional_realisations = data.get_realisations(
                    current_value, conditional_vars)[0]
            elif conditional_vars and target_vars:
                conditional_realisations = np.hstack((
                    data.get_realisations(current_value, conditional_vars)[0],
                    target_realisations))

            te_surrogates[s] = self._cmi_estimator.estimate_parallel(
                n_chunks=self.settings['n_perm_comp'],
                re_use=['var2', 'conditional'],
                var1=current_value_surrogates,
                var2=source_realisations,
                conditional=conditional_realisations)
        return te_surrogates

    def _create_surrogate_distribution_between(self):
        """Create the surrogate distribution for network inference.

        Create distribution by permuting data sets between conditions, re-
        estimating CMI, and re-calculating the mean of differences. The
        permutation strategy depends on the stats type set in the class
        instance (dependent or independent):

        For a dependent test, subjects are assumed to be ordered and related,
        i.e., the first subject in condition A is related to the first subject
        in condition B, thus, subjects are swapped or exchanged between
        conditions without changing their rank in the data set:

        A_1    B_1        ->        B_1    A_1
        A_2    B_2        ->        A_2    B_2
        A_3    B_3        ->        A_3    B_3
        A_4    B_4        ->        B_4    A_4
        A_5    B_5        ->        B_5    A_5
        ...                         ...

        ; for an independent test, subjects are not assumed to depend on each
        other, thus, subjects are randomply permuted between groups:

        A_1    B_1        ->        A_3    A_1
        A_2    B_2        ->        B_2    B_4
        A_3    B_3        ->        B_3    B_3
        A_4    B_4        ->        A_5    A_4
        A_5    B_5        ->        B_1    A_2
        ...                         ...
        """
        self.cmi_surr = {}
        for t in self.union.targets_analysed:
            n_sources = len(self.union._single_target[t].sources)
            self.cmi_surr[t] = np.zeros(
                (n_sources, self.settings['n_perm_comp']))
        for p in range(self.settings['n_perm_comp']):
            surrogate = self._calculate_diff_of_mean(
                self.cmi_set_a, self.cmi_set_b, permute=True)
            for t in self.union.targets_analysed:
                self.cmi_surr[t][:, p] = surrogate[t]

    def _p_value_union(self):
        """Calculate the p-value for the CMI between each source and target."""
        # Test each original difference against its surrogate distribution.
        self.significance = {}
        self.pvalue = {}
        for t in self.union.targets_analysed:
            sources = self.union._single_target[t].sources
            if not sources.size:
                continue
            self.significance[t] = np.zeros(len(sources), dtype=bool)
            self.pvalue[t] = np.zeros(len(sources))
            for i in range(len(sources)):
                [sig, p] = stats._find_pvalue(
                    statistic=self.cmi_diff[t][i],
                    distribution=self.cmi_surr[t][i, :],
                    alpha=self.settings['alpha_comp'],
                    tail=self.settings['tail_comp'])
                self.significance[t][i] = sig
                self.pvalue[t][i] = p

    def _union_indices_to_lags(self):
        """Clean up bevor returning."""
        # convert time indices to lags for selected variables
        for t in self.union.targets_analysed:
            self.union._single_target[t]['selected_vars_sources'] = (
                self._idx_to_lag(
                    self.union._single_target[t]['selected_vars_sources'],
                    self.union['max_lag']))
            self.union._single_target[t]['selected_vars_target'] = (
                self._idx_to_lag(
                    self.union._single_target[t]['selected_vars_target'],
                    self.union['max_lag']))

    def _get_permuted_replications(self, data_a, data_b, target):
        """Return realisations with replications permuted betw. two data sets.

        Return surrogate data for a given target for the conditioning and the

        where original data

        Create surrogate data by permuting realisations of the conditioning set
        over replications. All realisations in one replication get swapped
        between the two conditions to generate surrogate data.

        Args:
            data_a : Data instance
                raw data, condition A
            data_b : Data instance
                raw data, condition B
            target : int
                index of the target in the union network

        Returns:
            cond_a_perm, cur_val_a_perm, cond_b_perm, cur_val_b_perm

        """
        # Get indices of current value and full conditioning set in the
        # union network.
        current_val = (target, self.union['max_lag'])
        idx_cond_full = (
            self.union._single_target[target]['selected_vars_target'] +
            self.union._single_target[target]['selected_vars_sources'])

        # Get realisations of the current value and the full conditioning set.
        assert data_a.n_replications == data_b.n_replications, (
                            'Unequal no. replications in the two data sets.')
        [cur_val_a_real, repl_idx_a] = data_a.get_realisations(current_val,
                                                               [current_val])
        [cur_val_b_real, repl_idx_b] = data_b.get_realisations(current_val,
                                                               [current_val])
        cond_a_real = data_a.get_realisations(current_val, idx_cond_full)[0]
        cond_b_real = data_b.get_realisations(current_val, idx_cond_full)[0]

        # Get no. replications and no. samples per replication.
        n_repl = max(repl_idx_a) + 1
        n_per_repl = sum(repl_idx_a == 0)

        # Make copies such that arrays in the caller scope are not overwritten.
        cond_a_perm = cp.copy(cond_a_real)
        cond_b_perm = cp.copy(cond_b_real)
        cur_val_a_perm = cp.copy(cur_val_a_real)
        cur_val_b_perm = cp.copy(cur_val_b_real)

        # Swap or permute realisations of the conditioning set depending on the
        # stats type.
        if self.settings['stats_type'] == 'dependent':
            swap = np.repeat(np.random.randint(2, size=n_repl).astype(bool),
                             n_per_repl)
            cond_a_perm[swap, :] = cond_b_real[swap, :]
            cond_b_perm[swap, :] = cond_a_real[swap, :]
            cur_val_a_perm[swap, :] = cur_val_b_real[swap, :]
            cur_val_b_perm[swap, :] = cur_val_a_real[swap, :]

        elif self.settings['stats_type'] == 'independent':
            # Pool replications from both data sets and draw two samples of
            # size n_repl.
            resample_a = np.random.choice(2 * n_repl, n_repl, replace=False)
            resample_b = np.setdiff1d(np.arange(2 * n_repl), resample_a)

            # Get resampled realisations for group A.
            i_0 = 0
            i_1 = n_per_repl
            for r in resample_a:
                if r >= n_repl:     # take realisation from cond B
                    r_perm = r - n_repl
                    cond_a_perm[i_0:i_1, ] = cond_b_real[repl_idx_b ==
                                                         r_perm, :]
                    cur_val_a_perm[i_0:i_1, ] = cur_val_b_real[repl_idx_b ==
                                                               r_perm, :]
                else:               # take realisation from cond A otherwise
                    cond_a_perm[i_0:i_1, ] = cond_a_real[repl_idx_a == r, :]
                    cur_val_a_perm[i_0:i_1, ] = cur_val_a_real[repl_idx_a == r,
                                                               :]
                i_0 = i_1
                i_1 = i_0 + n_per_repl

            # Get resampled realisations for group B.
            i_0 = 0
            i_1 = n_per_repl
            for r in resample_b:
                if r >= n_repl:     # take realisation from cond B
                    r_perm = r - n_repl
                    cond_b_perm[i_0:i_1, ] = cond_b_real[repl_idx_b ==
                                                         r_perm, :]
                    cur_val_b_perm[i_0:i_1, ] = cur_val_b_real[repl_idx_b ==
                                                               r_perm, :]
                else:               # take realisation from cond A otherwise
                    cond_b_perm[i_0:i_1, ] = cond_a_real[repl_idx_a == r, :]
                    cur_val_b_perm[i_0:i_1, ] = cur_val_a_real[repl_idx_a == r,
                                                               :]
                i_0 = i_1
                i_1 = i_0 + n_per_repl
        else:
            raise ValueError('Unkown "stats_type": {0}, should be "dependent" '
                             'or "independent".'.format(
                                                self.settings['stats_type']))
        return cond_a_perm, cur_val_a_perm, cond_b_perm, cur_val_b_perm

    def _initialise(self, settings):
        """Check input and set analysis settings to initial values."""
        # Check if the type of comparison was specified.
        try:
            settings['stats_type']
        except KeyError:
            raise KeyError('You have to provide a "stats_type": "dependent" '
                           'or "independent".')

        # Add CMI estimator to class.
        try:
            EstimatorClass = find_estimator(settings['cmi_estimator'])
        except KeyError:
            raise KeyError('Please provide an estimator class or name!')
        self._cmi_estimator = EstimatorClass(settings)

        if 'local_values' in settings and settings['local_values']:
            raise RuntimeError('Can''t run network comparison on local values.')

        # Set defaults for statistical tests.
        self.settings = settings.copy()
        self.settings['local_values'] = False
        self.settings.setdefault('verbose', True)
        self.settings.setdefault('n_perm_comp', 500)
        self.settings.setdefault('alpha_comp', 0.05)
        self.settings.setdefault('tail_comp', 'two')
        self.settings.setdefault('permute_in_time', False)
        stats.check_n_perm(self.settings['n_perm_comp'],
                           self.settings['alpha_comp'])

    def _reset(self):
        """Reset instance after analysis."""
        self.__init__()
        del self.settings
        del self.union
        del self.cmi_diff
        del self.cmi_surr
        del self._cmi_estimator

    def _link_exists(self, link):
        """Check if link is part of the union network."""
        source = link[0]
        target = link[1]
        if target not in self.union.targets_analysed:
            return False
        if source not in self.union._single_target[target].sources:
            return False
        return True
