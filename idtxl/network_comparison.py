"""Perform inference statistics on groups of data."""
import copy as cp
import numpy as np
from scipy.special import binom
from .estimator import find_estimator
from . import stats
from . import idtxl_utils as utils
from .network_analysis import NetworkAnalysis


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

    Attributes:
        union : dict
            union of all networks entering the comparison, used as the basis
            for statistical comparison
        settings : dict
            parameters for CMI estimation
        cmi_diff : dict
            original difference in CMI estimates for each source variable >
            target combination in the union network
        cmi_surr : dict
            differences in CMI estimates from surrogate data, used as test
            distribution
        alpha : float
            critical alpha level for network comparison
        n_permutations : int
            number of permutations
        stats_type : str
            type of statistics ('dependent' or 'independent')
        tail : str
            test tail ('one' or 'two')
    """

    def __init__(self):
        super().__init__()

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
                  test, 'two' for two-sided test (default='two')
                - n_perm_comp : int [optional] - number of permutations
                  (default=500)
                - alpha_comp : float - critical alpha level for statistical
                  significance (default=0.05)
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
            dict
                results of network comparison, contains the union network
                ('union_network'), parameters used for statistical comparison
                and for CMI estimation ('settings', includes critical alpha
                level, 'alpha_comp'; number of permutations, 'n_perm_comp';
                statistics type, 'stats_type', test tail, 'tail_comp'),
                the original absolute CMI differences per source variable-
                target combination in the union network ('cmi_diff_abs'), the
                direction of the effect, i.e., if TE was stronger in condition
                A than B ('a>b'), the surrogate CMI difference values
                ('cmi_surr'), the p-value for each source variable > target
                combination ('pval') and their statistical significance
                ('sign').
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
        [pvalue, sign] = self._p_value_union()

        self._union_indices_to_lags()
        results = {
            # Return both the absolute difference and the direction of the
            # effect. Returning just the difference and evalutating the sign
            # does not give the direction of the effect if one or both values
            # are negative (which may happen due to estimator bias).
            'cmi_diff_abs': self._get_abs_diff(self.cmi_diff),
            'a>b': self.cmi_comp,
            'cmi_surr': self.cmi_surr,
            'union_network': self.union,
            'pval': pvalue,
            'sign': sign,
            'settings': self.settings
            }

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
            dict
                results of network comparison, contains the union network
                ('union_network'), parameters used for statistical comparison
                and for CMI estimation ('settings', includes critical alpha
                level, 'alpha_comp'; number of permutations, 'n_perm_comp';
                statistics type, 'stats_type', test tail, 'tail_comp', number
                of subjects per group), the original absolute CMI differences
                per source variable- target combination in the union network
                ('cmi_diff_abs'), the direction of the effect, i.e., if TE was
                stronger in condition A than B ('a>b'), the surrogate CMI
                difference values ('cmi_surr'), the p-value for each source
                variable > target combination ('pval') and their statistical
                significance ('sign').
        """
        # Check input and analysis parameters.
        self._initialise(settings)
        self._check_n_subjects(data_set_a, data_set_b)
        data_all = np.hstack((data_set_a, data_set_b))
        self._check_equal_realisations(*data_all)

        # Main comparison.
        print('\n-------------------------- (1) create union of networks')
        network_all = np.hstack((network_set_a, network_set_b))
        self._create_union(*network_all)
        self._calculate_union_cmi(data_set_a, data_set_b)
        print('\n-------------------------- (2) calculate differences in TE '
              'values')
        self._calculate_cmi_diff_between()
        print('\n-------------------------- (3) create surrogate distribution')
        self._create_surrogate_distribution_between()
        print('\n-------------------------- (4) determine p-value')
        [pvalue, sign] = self._p_value_union()

        self._union_indices_to_lags()
        results = {
            # Return both the absolute difference and the direction of the
            # effect. Returning just the difference and evalutating the sign
            # does not give the direction of the effect if one or both values
            # are negative (which may happen due to estimator bias).
            'cmi_diff_abs': self._get_abs_diff(self.cmi_diff),
            'a>b': self.cmi_comp,
            'cmi_surr': self.cmi_surr,
            'union_network': self.union,
            'pval': pvalue,
            'sign': sign,
            'settings': self.settings
            }

        self._reset()  # remove attributes
        return results

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
        targets = []
        for nw in networks:
            k = [i for i in nw.keys()]
            try:
                k.remove('fdr_corrected')
            except ValueError:
                pass
            targets = targets + k
        targets = np.unique(np.array(targets))

        # Get the maximum lags from the networks, we need this to get
        # realisations of variables later on.
        self.union = {}
        self.union['targets'] = targets
        self.union['max_lag'] = networks[0][targets[0]]['current_value'][1]

        # Get the union of sources for each target in the union network.
        for t in targets:
            self.union[t] = {}
            self.union[t]['selected_vars_sources'] = []
            self.union[t]['selected_vars_target'] = []
            for nw in networks:
                # Check if the max_lag is the same for each network going into
                # the comparison.
                try:
                    lag = nw[t]['current_value'][1]
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
                    cond_src = self._lag_to_idx(nw[t]['selected_vars_sources'],
                                                self.union['max_lag'])
                except KeyError:
                    cond_src = []
                try:
                    cond_tgt = self._lag_to_idx(nw[t]['selected_vars_target'],
                                                self.union['max_lag'])
                except KeyError:
                    cond_tgt = []

                # Add conditional if it isn't already in the union network.
                for c in cond_src:
                    if c not in self.union[t]['selected_vars_sources']:
                        self.union[t]['selected_vars_sources'].append(c)
                for c in cond_tgt:
                    if c not in self.union[t]['selected_vars_target']:
                        self.union[t]['selected_vars_target'].append(c)

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
        # re-calculate CMI for each data object using the union network mask
        cmi_a = self._calculate_cmi_all_links(data_a)
        cmi_b = self._calculate_cmi_all_links(data_b)
        self.cmi_diff = self._calculate_diff(cmi_a, cmi_b)
        # compare raw TE values betw. cond.
        self.cmi_comp = self._compare_union_cmi_within(cmi_a, cmi_b)

    def _calculate_cmi_diff_between(self):
        """Calculate the difference in CMI between two groups of subjects.

        Calculate the difference in the conditional mutual information (CMI)
        for each source > target combination in the union network between data
        sets recorded from subjects measured under one of two experimental
        conditions. Compare the absolute mean TE values between the two groups.

        Returns:
            numpy array
                CMI differences
        """
        self.cmi_diff = self._calculate_diff_of_mean(permute=False)
        self._compare_union_cmi_between()  # compare raw TE values betw. cond.
        # TODO Idea: loop over pairs of data in data_set_a and *_b and feed it
        # to the within function? BUT: such an implementation doesn't allow for
        # unbalanced designs, which is a problem and needs to be changed in the
        # within function as well

    def _calculate_diff_of_mean(self, permute=False):
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
            cmi_all = self.cmi_a + self.cmi_b
            new_partition_a = np.random.choice(range(len(cmi_all)),
                                               size=len(self.cmi_a),
                                               replace=False)
            new_partition_b = list(set(range(0, len(cmi_all))) -
                                   set(new_partition_a))
            cmi_a_perm = [cmi_all[i] for i in new_partition_a]
            cmi_b_perm = [cmi_all[i] for i in new_partition_b]

            return self._calculate_diff(self._calculate_mean(cmi_a_perm),
                                        self._calculate_mean(cmi_b_perm))

        else:
            return self._calculate_diff(self._calculate_mean(self.cmi_a),
                                        self._calculate_mean(self.cmi_b))

    def _calculate_union_cmi(self, data_set_a, data_set_b):
        """Calculate CMI for each data object using the union network mask.

        Args:
            data_set_a : list/array of Data instances
                first set of raw data
            data_set_b : list/array of Data instances
                second set of raw data
        """
        self.cmi_a = []
        for d in data_set_a:
            self.cmi_a.append(self._calculate_cmi_all_links(d))
        self.cmi_b = []
        for d in data_set_b:
            self.cmi_b.append(self._calculate_cmi_all_links(d))

    def _calculate_cmi_all_links(self, data):
        """Calculate CMI for each source>target combi in the union network."""
        cmi = {}
        for t in self.union['targets']:
            cmi_temp = []

            # if there are no sources for the current target, continue to next
            if not self.union[t]['selected_vars_sources']:
                cmi[t] = np.array(cmi_temp)
                continue

            # get realisations of the current value and the full cond. set.
            # TODO why is there a global max_lag and not one for each target?
            current_val = (t, self.union['max_lag'])
            idx_cond_full = (self.union[t]['selected_vars_target'] +
                             self.union[t]['selected_vars_sources'])
            [cur_val_real, repl_idx] = data.get_realisations(current_val,
                                                             [current_val])
            cond_full_real = data.get_realisations(current_val,
                                                   idx_cond_full)[0]

            # Calculate TE from each source variable to current target t
            n_sources = len(self.union[t]['selected_vars_sources'])
            cmi_temp = np.zeros(n_sources)
            i = 0
            for idx_source in self.union[t]['selected_vars_sources']:
                # get realisations of current TE-source from the set of all
                # conditionals and calculate the CMI
                [cond_real, source_real] = utils.separate_arrays(
                                                             idx_cond_full,
                                                             idx_source,
                                                             cond_full_real)
                cmi_temp[i] = self._cmi_estimator.estimate(cur_val_real,
                                                           source_real,
                                                           cond_real)
                i += 1

            cmi[t] = cmi_temp

        return cmi

    def _compare_union_cmi_between(self):
        """Compare mean TE between conditions to get direction of effect."""
        self.cmi_comp = {}
        cmi_a_mean = self._calculate_mean(self.cmi_a)
        cmi_b_mean = self._calculate_mean(self.cmi_b)
        for t in self.union['targets']:
            self.cmi_comp[t] = cmi_a_mean[t] > cmi_b_mean[t]

    def _compare_union_cmi_within(self, cmi_a, cmi_b):
        """Compare TE between conditions to get direction of effect."""
        self.cmi_comp = {}
        for t in self.union['targets']:
            self.cmi_comp[t] = cmi_a[t] > cmi_b[t]

    def _get_abs_diff(self, cmi_diff):
        """Get the absolute value for each difference in the union network."""
        cmi_diff_abs = {}
        for t in self.union['targets']:
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
        for t in self.union['targets']:
            cmi_temp_a = []
            cmi_temp_b = []
            # If there are no sources for current target, continue  to the next
            if not self.union[t]['selected_vars_sources']:
                cmi_a[t] = np.array(cmi_temp_a)
                cmi_b[t] = np.array(cmi_temp_b)
                continue

            # Get full conditioning set for current target.
            idx_cond_full = (self.union[t]['selected_vars_target'] +
                             self.union[t]['selected_vars_sources'])
            # Get realisations, where realisations are permuted/swapped
            # replication-wise between two data sets (e.g., from different
            # conditions)
            [cond_full_perm_a,
             cur_val_perm_a,
             cond_full_perm_b,
             cur_val_perm_b] = self._get_permuted_replications(data_a,
                                                               data_b,
                                                               t)
            # Calculate CMI from each source to current target t from permuted
            # data
            n_sources = len(self.union[t]['selected_vars_sources'])
            cmi_temp_a = np.zeros(n_sources)
            cmi_temp_b = np.zeros(n_sources)
            i = 0
            for idx_source in self.union[t]['selected_vars_sources']:
                # Get realisations of current source from the set of all
                # surrogate conditionals and calculate the CMI. Do this for
                # both conditions.
                [temp_cond_real_a, source_real_a] = utils.separate_arrays(
                                                             idx_cond_full,
                                                             idx_source,
                                                             cond_full_perm_a)
                cmi_temp_a[i] = self._cmi_estimator.estimate(cur_val_perm_a,
                                                             source_real_a,
                                                             temp_cond_real_a)
                [temp_cond_real_b, source_real_b] = utils.separate_arrays(
                                                             idx_cond_full,
                                                             idx_source,
                                                             cond_full_perm_b)
                cmi_temp_b[i] = self._cmi_estimator.estimate(cur_val_perm_b,
                                                             source_real_b,
                                                             temp_cond_real_b)
                i += 1

            cmi_a[t] = cmi_temp_a
            cmi_b[t] = cmi_temp_b

        return cmi_a, cmi_b

    def _calculate_mean(self, cmi_set):
        """Calculate the mean CMI over multiple data sets for all targets."""
        if type(cmi_set) is dict:
            raise TypeError('Input needs to be 1-D array-like of dicts.')
        cmi_mean = {}
        for t in self.union['targets']:
            n_sources = cmi_set[0][t].shape[0]
            n_datasets = len(cmi_set)
            temp = np.empty((n_datasets, n_sources))
            i = 0
            for c in cmi_set:
                temp[i, :] = c[t]
                i += 1
            cmi_mean[t] = np.mean(temp, axis=0)
        return cmi_mean

    def _calculate_diff(self, cmi_a, cmi_b):
        """Calculate the difference between two CMI estimates for all targets.

        Calculate the differene in CMI for each source > target combination in
        cmi_a/_b. The inputs are assumed to be dictionaries with one entry for
        each target and for each target, cmi values are given as a numpy
        array.
        """
        cmi_diff = {}
        for t in self.union['targets']:
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
        self.cmi_surr = []
        for p in range(self.settings['n_perm_comp']):
            if self.settings['verbose']:
                print('Creating surrogate data set {0} of {1}.'.format(
                        p + 1, self.settings['n_perm_comp']))
            [cmi_a, cmi_b] = self._calculate_cmi_all_links_permuted(data_a,
                                                                    data_b)
            self.cmi_surr.append(self._calculate_diff(cmi_a, cmi_b))

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
        self.cmi_surr = []
        for p in range(self.settings['n_perm_comp']):
            self.cmi_surr.append(self._calculate_diff_of_mean(permute=True))

    def _p_value_union(self):
        """Calculate the p-value for the CMI between each source and target."""
        # Test each original difference against its surrogate distribution.
        significance = {}
        pvalue = {}
        for t in self.union['targets']:
            n_sources = self.cmi_surr[0][t].shape[0]
            if n_sources == 0:
                continue
            surr_temp = np.zeros((self.settings['n_perm_comp'], n_sources))
            significance[t] = np.zeros(n_sources, dtype=bool)
            pvalue[t] = np.empty(n_sources)
            for p in range(self.settings['n_perm_comp']):
                surr_temp[p, :] = self.cmi_surr[p][t]
            for s in range(n_sources):
                [sig, p] = stats._find_pvalue(
                                            statistic=self.cmi_diff[t][s],
                                            distribution=surr_temp[:, s],
                                            alpha=self.settings['alpha_comp'],
                                            tail=self.settings['tail_comp'])
                significance[t][s] = sig
                pvalue[t][s] = p
        return pvalue, significance

    def _union_indices_to_lags(self):
        """Clean up bevor returning."""
        # convert time indices to lags for selected variables
        for t in self.union['targets']:
            self.union[t]['selected_vars_sources'] = self._idx_to_lag(
                                        self.union[t]['selected_vars_sources'],
                                        self.union['max_lag'])
            self.union[t]['selected_vars_target'] = self._idx_to_lag(
                                        self.union[t]['selected_vars_target'],
                                        self.union['max_lag'])

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
        idx_cond_full = (self.union[target]['selected_vars_target'] +
                         self.union[target]['selected_vars_sources'])

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

        # Set defaults for statistical tests.
        settings.setdefault('verbose', True)
        settings.setdefault('n_perm_comp', 500)
        settings.setdefault('alpha_comp', 0.05)
        settings.setdefault('tail_comp', 'two')
        stats.check_n_perm(settings['n_perm_comp'], settings['alpha_comp'])
        self.settings = settings

    def _reset(self):
        """Reset instance after analysis."""
        self.__init__()
        del self.settings
        del self.union
        del self.cmi_diff
        del self.cmi_surr
        del self._cmi_estimator
