"""Perform inference statistics on groups of data."""
import copy as cp
import numpy as np
from scipy.special import binom
from .set_estimator import Estimator_cmi
from . import stats
from . import idtxl_utils as utils
from .network_analysis import Network_analysis

VERBOSE = True


class Network_comparison(Network_analysis):
    """Set up network comparison between two experimental conditions.

    The class provides methods for the comparison of networks inferred from
    data recorded under two experimental conditions. Four statistical tests
    are implemented:

    |                      | stats_type | example                             |
    |----------------------|------------|-------------------------------------|
    | __within__ subjects  |dependent   |base line (A) vs. task (B)           |
    |                      |independent |detect house (A) vs. face (B)        |
    | __between__ subjects |dependent   |patients (A) vs. matched controls (B)|
    |                      |independent |male (A) vs. female (B) participants |

    The stats_type is determined through an option passed to the class
    constructor, whether the analysis is performed withing or between subjects
    is determined through the use of the class method for comparison:
    'compare_within' or 'compare_between'.

    Args:
        options : dict
            options for statistical comparison of networks

            - 'stats_type' - 'dependent' or 'independent' for dependent or
              independent units of observation
            - 'cmi_calc_name' - estimator to be used for CMI calculation
              (For estimator options see the respective documentation.)
            - 'tail' - test tail, 'one' for one-sided test, 'two' for two-sided
              test (default='two')
            - 'n_perm_comp' - number of permutations (default=500)
            - 'alpha_comp' - critical alpha level for statistical significance
            (default=0.05)

    Attributes:
        union : dict
            union of all networks entering the comparison, used as the basis
            for statistical comparison
        cmi_opts : dict
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

    def __init__(self, options):
        try:
            self.stats_type = options['stats_type']
        except KeyError:
            raise KeyError('You have to provide a "stats_type": "dependent" '
                           'or "independent".')
        try:
            self._cmi_calculator = Estimator_cmi(options['cmi_calc_name'])
        except KeyError:
            raise KeyError('No CMI calculator was specified!')
        self.n_permutations = options.get('n_perm_comp', 10)
        self.alpha = options.get('alpha_comp', 0.05)
        self.tail = options.get('tail', 'two')
        self.cmi_opts = options
        stats.check_n_perm(self.n_permutations, self.alpha)

    def compare_within(self, network_a, network_b, data_a, data_b):
        """Compare networks inferred under two conditions within one subject.

        Compare two networks inferred from data recorded under two different
        experimental conditions within one subject (units of observations are
        replications of one experimental condition within one subject).

        Arguments:
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
                (critical alpha level, 'alpha'; number of permutations,
                'n_permutations'; statistics type, 'stats_type', test tail,
                'tail'), parameters used for CMI estimation ('cmi_opts'),
                the original CMI difference ('cmi_diff'), the surrogate CMI
                values ('cmi_surr'), the p-value for each source variable >
                target combination in the union network ('pval') and their
                statistical significance ('sign')
        """
        # Check input and analysis parameters.
        self._check_n_replications(data_a, data_b)
        self._check_equal_realisations(data_a, data_b)

        # Main comparison.
        print('\n-------------------------- (1) create union of networks')
        self._create_union(network_a, network_b)
        print('\n-------------------------- (2) calculate original TE values')
        self._calculate_cmi_diff_within(data_a, data_b)
        print('\n-------------------------- (3) create surrogate distribution')
        self._create_surrogate_distribution_within(data_a, data_b)
        print('\n-------------------------- (4) determine p-value')
        [pvalue, sign] = self._p_value_union()

        self._clean_up()  # remove temp. attributes, Convert lags to indices

        results = {
            'alpha': self.alpha,
            'n_permutations': self.n_permutations,
            'stats_type': self.stats_type,
            'tail': self.tail,
            'cmi_diff': self.cmi_diff,
            'cmi_surr': self.cmi_surr,
            'cmi_opts': self.cmi_opts,
            'union_network': self.union,
            'pval': pvalue,
            'sign': sign
            }
        return results

    def compare_between(self, network_set_a, network_set_b, data_set_a,
                        data_set_b):
        """Compare networks inferred under two conditions between subjects.

        Compare two sets of networks inferred from two sets of data recorded
        under different experimental conditions within multiple subjects, i.e.,
        data have been recorded from subjects assigned to one of two
        experimental conditions (units of observations are subjects).

        Arguments:
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
                (critical alpha level, 'alpha'; number of permutations,
                'n_permutations'; statistics type, 'stats_type', test tail,
                'tail'), parameters used for CMI estimation ('cmi_opts'),
                the original CMI difference ('cmi_diff'), the surrogate CMI
                values ('cmi_surr'), the p-value for each source variable >
                target combination in the union network ('pval') and their
                statistical significance ('sign')
        """
        # Check input and analysis parameters.
        network_all = np.hstack((network_set_a, network_set_b))
        data_all = np.hstack((data_set_a, data_set_b))
        self._check_n_subjects(data_set_a, data_set_b)
        self._check_equal_realisations(*data_all)

        # Main comparison.
        print('\n-------------------------- (1) create union of networks')
        self._create_union(*network_all)
        print('\n-------------------------- (2) calculate original TE values')
        self._calculate_cmi_diff_between(data_set_a, data_set_b)
        print('\n-------------------------- (3) create surrogate distribution')
        self._create_surrogate_distribution_between(data_set_a, data_set_b)
        print('\n-------------------------- (4) determine p-value')
        [pvalue, sign] = self._p_value_union()

        self._clean_up()  # remove temp. attributes, Convert lags to indices

        results = {
            'alpha': self.alpha,
            'n_permutations': self.n_permutations,
            'stats_type': self.stats_type,
            'tail': self.tail,
            'cmi_diff': self.cmi_diff,
            'cmi_surr': self.cmi_surr,
            'cmi_opts': self._idx_to_lag,
            'union_network': self.union,
            'pval': pvalue,
            'sign': sign
            }
        return results

    def _check_n_replications(self, data_a, data_b):
        """Check if no. replications is sufficient request no. permutations."""
        assert data_a.n_replications == data_b.n_replications, (
                            'Unequal no. replications in the two data sets.')
        n_replications = data_a.n_replications
        if self.stats_type == 'dependent':
            if 2**n_replications < self.n_permutations :
                raise RuntimeError('The number of replications {0} in the data'
                                   ' are not sufficient to enable the '
                                   'requested no. permutations {1}'.format(
                                                       n_replications,
                                                       self.n_permutations))
        elif self.stats_type == 'independent':
            if binom(2*n_replications, n_replications) < self.n_permutations:
                raise RuntimeError('The number of replications {0} in the data'
                                   ' are not sufficient to enable the '
                                   'requested no. permutations {1}'.format(
                                                       n_replications,
                                                       self.n_permutations))
        else:
            raise RuntimeError('Unknown ''stats_type''!')

    def _check_n_subjects(self, data_set_a, data_set_b):
        """Check if no. subjects is sufficient request no. permutations."""
        if self.stats_type == 'dependent':
            assert len(data_set_a) == len(data_set_b), (
                    'The number of data sets is not equal between conditions.')
            n_data_sets = len(data_set_a)
            if 2**n_data_sets < self.n_permutations:
                raise RuntimeError('The number of data sets per condition {0} '
                                   'is not sufficient to enable the '
                                   'requested no. permutations {1}'.format(
                                                       n_data_sets,
                                                       self.n_permutations))
        elif self.stats_type == 'independent':
            max_len = max(len(data_set_a), len(data_set_b))
            total_len = len(data_set_a) + len(data_set_b)
            if binom(total_len, max_len) < self.n_permutations:
                raise RuntimeError('The total number of data sets {0} is not '
                                   'sufficient to enable the requested no. '
                                   'permutations {1}'.format(
                                                       total_len,
                                                       self.n_permutations))
        else:
            raise RuntimeError('Unknown ''stats_type''!')

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
                k.remove('fdr')
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

    # def _lag_to_idx(self, lag_list, max_lag):
    #     """Convert sample lags to time indices.
    #
    #     Use method from Network_analysis class to convert lags.
    #     """
    #     if not hasattr(self, 'netw_analysis'):
    #         self.netw_analysis = Network_analysis()
    #         self.netw_analysis.current_value = (0, max_lag)
    #     return self.netw_analysis._lag_to_idx(lag_list)
    #
    # def _idx_to_lag(self, idx_list, max_lag):
    #     """Convert sample time indices to lags.
    #
    #     Use method from Network_analysis class to convert time indices.
    #     """
    #     if not hasattr(self, 'netw_analysis'):
    #         self.netw_analysis = Network_analysis()
    #         self.netw_analysis.current_value = (0, max_lag)
    #     return self.netw_analysis._idx_to_lag(idx_list)

    def _calculate_cmi_diff_within(self, data_a, data_b, permute=False):
        """Calculate the difference in CMI within a subject.

        Calculate the difference in the conditional mutual information (CMI)
        for each source > target combination in the union network between data
        recorded under two experimental conditions within one subject.

        Args:
            data_a : Data instance
                first set of raw data
            data_a : Data instance
                second set of raw data
            permute : bool [optional]
                if True, permute data from same replications between sets a and
                b, depending on the stats type set for the instance

        Returns:
            numpy array
                CMI differences
        """
        # re-calculate CMI for each data object using the union network mask
        self.cmi_diff = self._calculate_diff(self._calculate_cmi(data_a),
                                             self._calculate_cmi(data_b))

    def _calculate_cmi_diff_between(self, data_set_a, data_set_b):
        """Calculate the difference in CMI between two groups of subjects.

        Calculate the difference in the conditional mutual information (CMI)
        for each source > target combination in the union network between data
        sets recorded from subjects measured under one of two experimental
        conditions.

        Args:
            data_set_a : list/array of Data instances
                first set of raw data
            data_set_b : list/array of Data instances
                second set of raw data

        Returns:
            numpy array
                CMI differences
        """
        self.cmi_diff = self._calculate_diff_of_mean(data_set_a, data_set_b,
                                                     permute=False)
        # TODO idea: loop over pairs of data in data_set_a and *_b and feed it
        # to the within function? is the mean of differences the same as the
        # difference of the mean? > yes
        # BUT: such an implementation doesn't allow for unbalanced designs
        # > this sucks and needs to be changed in the within function as well

    def _calculate_diff_of_mean(self, data_set_a, data_set_b, permute=False):
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
        # re-calculate CMI for each data object using the union network mask
        cmi_union_a = []
        for d in data_set_a:
            cmi_union_a.append(self._calculate_cmi(d))
        cmi_union_b = []
        for d in data_set_b:
            cmi_union_b.append(self._calculate_cmi(d))

        # permute data obejcts between conditions a and b before calculating
        # the CMI
        if permute:
            cmi_all = cmi_union_a + cmi_union_b
            new_partition_a = np.random.choice(range(len(cmi_all)),
                                               size=len(cmi_union_a),
                                               replace=False)
            new_partition_b = list(set(range(0, len(cmi_all))) -
                                   set(new_partition_a))
            cmi_a_perm = [cmi_all[i] for i in new_partition_a]
            cmi_b_perm = [cmi_all[i] for i in new_partition_b]

            return self._calculate_diff(self._calculate_mean(cmi_a_perm),
                                        self._calculate_mean(cmi_b_perm))

        else:
            return self._calculate_diff(self._calculate_mean(cmi_union_a),
                                        self._calculate_mean(cmi_union_b))

        # get the mean difference between the two sets of CMI estimates

    def _calculate_cmi(self, data):
        """Calculate CMI for each source>target combi in the union network."""
        cmi = {}
        for t in self.union['targets']:
            cmi_temp = []

            # if there are no sources for the current target, continue to next
            if not self.union[t]['selected_vars_sources']:
                cmi[t] = np.array(cmi_temp)
                continue

            # get realisations of the current value and the full cond. set.
            current_val = (t, self.union['max_lag'])
            cond_full = (self.union[t]['selected_vars_target'] +
                         self.union[t]['selected_vars_sources'])
            [cur_val_real, repl_idx] = data.get_realisations(current_val,
                                                             [current_val])
            cond_full_real = data.get_realisations(current_val, cond_full)[0]

            # Calculate TE from each source variable to current target t
            for c in self.union[t]['selected_vars_sources']:
                current_cond = cp.copy(self.union[t]['selected_vars_sources'])
                current_cond.pop(current_cond.index(c))

                # get realisations of current TE-source from the set of all
                # conditionals and calculate the CMI
                [cond_real, source_real] = utils.separate_arrays(
                                                             cond_full,
                                                             c,
                                                             cond_full_real)
                cmi_temp.append(self._cmi_calculator.estimate(
                                                        cur_val_real,
                                                        source_real,
                                                        cond_real,
                                                        self.cmi_opts))

            cmi[t] = np.array(cmi_temp)

        return cmi

    def _calculate_cmi_permuted(self, data_a, data_b):
        """Calculate surrogate CMI for union network.

        Calculate conditional mutual information (CMI) for each source > target
        combination in the union network after permuting realisations of
        sources between the two data sets (comming from two conditions).
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
            cond_full = (self.union[t]['selected_vars_target'] +
                         self.union[t]['selected_vars_sources'])
            # Get realisations, where realisations are permuted replication-
            # wise between two data sets (e.g., from different conditions)
            [cond_full_perm_a,
             cur_val_perm_a,
             cond_full_perm_b,
             cur_val_perm_b] = self._get_permuted_replications(data_a,
                                                               data_b,
                                                               t)
            # Calculate TE from each source to current target t
            for c in self.union[t]['selected_vars_sources']:
                current_cond = cp.copy(self.union[t]['selected_vars_sources'])
                current_cond.pop(current_cond.index(c))

                # Get realisations of current (permuted) TE-source from the set
                # of all conditionals and calculate the CMI.
                [cond_real_a, source_real_a] = utils.separate_arrays(
                                                             cond_full,
                                                             c,
                                                             cond_full_perm_a)
                cmi_temp_a.append(self._cmi_calculator.estimate(
                                                        cur_val_perm_a,  # TODO do we need the permuted data here?
                                                        source_real_a,
                                                        cond_real_a,
                                                        self.cmi_opts))
                [cond_real_b, source_real_b] = utils.separate_arrays(
                                                             cond_full,
                                                             c,
                                                             cond_full_perm_b)
                cmi_temp_b.append(self._cmi_calculator.estimate(
                                                        cur_val_perm_b,
                                                        source_real_b,
                                                        cond_real_b,
                                                        self.cmi_opts))
            cmi_a[t] = np.array(cmi_temp_a)
            cmi_b[t] = np.array(cmi_temp_b)

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
        for p in range(self.n_permutations):
            [cmi_a, cmi_b] = self._calculate_cmi_permuted(data_a, data_b)
            self.cmi_surr.append(self._calculate_diff(cmi_a, cmi_b))

    def _create_surrogate_distribution_between(self, data_set_a, data_set_b):
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

        Args:
            data_set_a : list/array of Data instances
                first set of raw data
            data_set_b : list/array of Data instances
                second set of raw data
        """
        self.cmi_surr = []
        for p in range(self.n_permutations):
            self.cmi_surr.append(self._calculate_diff_of_mean(data_set_a,
                                                              data_set_b,
                                                              permute=True))

    def _p_value_union(self):
        """Calculate the p-value for the CMI between each source and target."""
        # Test each original difference against its surrogate distribution.
        significance = {}
        pvalue = {}
        for t in self.union['targets']:
            n_sources = self.cmi_surr[0][t].shape[0]
            if n_sources == 0:
                continue
            surr_temp = np.zeros((self.n_permutations, n_sources))
            significance[t] = np.zeros(n_sources, dtype=bool)
            pvalue[t] = np.empty(n_sources)
            for p in range(self.n_permutations):
                surr_temp[p, :] = self.cmi_surr[p][t]
            for s in range(n_sources):
                [sig, p] = stats._find_pvalue(statistic=self.cmi_diff[t][s],
                                              distribution=surr_temp[:, s],
                                              alpha=self.alpha,
                                              tail=self.tail)
                significance[t][s] = sig
                pvalue[t][s] = p
        return pvalue, significance

    def _clean_up(self):
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
        cond_full = (self.union[target]['selected_vars_target'] +
                     self.union[target]['selected_vars_sources'])

        # Get realisations of the current value and the full conditioning set.
        assert data_a.n_replications == data_b.n_replications, (
                            'Unequal no. replications in the two data sets.')
        [cur_val_a, repl_idx_a] = data_a.get_realisations(current_val,
                                                          [current_val])
        [cur_val_b, repl_idx_b] = data_b.get_realisations(current_val,
                                                          [current_val])
        cond_a = data_a.get_realisations(current_val, cond_full)[0]
        cond_b = data_b.get_realisations(current_val, cond_full)[0]

        # Get no. replications and no. samples per replication.
        n_repl = max(repl_idx_a) + 1
        n_per_repl = sum(repl_idx_a == 0)

        # Make copies such that arrays in the caller scope are not overwritten.
        cond_a_perm = np.empty(cond_a.shape)
        cond_b_perm = np.empty(cond_b.shape)
        cur_val_a_perm = np.empty(cur_val_a.shape)
        cur_val_b_perm = np.empty(cur_val_b.shape)

        # Swap or permute realisations of the conditioning set depending on the
        # stats type.
        if self.stats_type == 'dependent':
            swap = np.repeat(np.random.randint(2, size=n_repl).astype(bool),
                             n_per_repl)
            cond_a_perm[swap, :] = cond_b[swap, :]
            cond_b_perm[swap, :] = cond_a[swap, :]
            cond_a_perm[np.invert(swap), :] = cond_a[np.invert(swap), :]
            cond_b_perm[np.invert(swap), :] = cond_b[np.invert(swap), :]

        elif self.stats_type == 'independent':
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
                    cond_a_perm[i_0:i_1, ] = cond_b[repl_idx_b == r_perm, :]
                    cur_val_a_perm[i_0:i_1, ] = cur_val_b[repl_idx_b== r_perm, :]
                else:     # take original realisation from cond A
                    cond_a_perm[i_0:i_1, ] = cond_a[repl_idx_a == r, :]
                    cur_val_a_perm[i_0:i_1, ] = cur_val_a[repl_idx_a == r, :]
                i_0 = i_1
                i_1 = i_0 + n_per_repl

            # Get resampled realisations for group B.
            i_0 = 0
            i_1 = n_per_repl
            for r in resample_b:
                if r >= n_repl:     # take realisation from cond A
                    r_perm = r - n_repl
                    cond_b_perm[i_0:i_1, ] = cond_b[repl_idx_b == r_perm, :]
                    cur_val_b_perm[i_0:i_1, ] = cur_val_b[repl_idx_b == r_perm, :]
                else:
                    cond_b_perm[i_0:i_1, ] = cond_a[repl_idx_a == r, :]
                    cur_val_b_perm[i_0:i_1, ] = cur_val_a[repl_idx_a == r, :]
                i_0 = i_1
                i_1 = i_0 + n_per_repl
        else:
            raise ValueError('Unkown "stats_type": {0}, should be "dependent" '
                             'or "independent".'.format(self.stats_type))
        return cond_a_perm, cur_val_a_perm, cond_b_perm, cur_val_b_perm
