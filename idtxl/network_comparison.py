"""Perform inference statistics on groups of data."""
import sys
import copy as cp
import numpy as np
from scipy.special import binom
from .set_estimator import Estimator_cmi
from . import stats
from . import idtxl_utils

VERBOSE = True


class Network_comparison():
    """Set up network comparison for inference on networks.

    Args:
        network_1 : dict
            results from network inference
        network_2 : dict
            results from network inference
        options : dict
            options for statistical comparison of networks

            - 'stats_type' - 'dependent' or 'independent' for dependent or
              independent units of observation
            - 'cmi_calc_name' - estimator to be used for CMI calculation
              (For estimator options see the respective documentation.)
            - 'n_perm_comp' - number of permutations (default=500)
            - 'alpha_comp' - critical alpha level for statistical significance
            (default=0.05)

    Returns:
        numpy array, bool
            statistical significance of difference of each source
        numpy array, float
            the test's p-values for each difference
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
        """Compare two networks within an unit of observation under two conditions.

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
                results of network comparison
        """
        # Check input and analysis parameters.
        self._check_n_perm_within(data_a, data_b)
        self._check_n_realisations(data_a, data_b)

        # Main comparison.
        print('\n-------------------------- (1) create union of networks')
        self._create_union(network_a, network_b)
        print('\n-------------------------- (2) calculate original TE values')
        self._calculate_cmi_diff_within(data_a, data_b)
        print('\n-------------------------- (3) create surrogate distribution')
        self._create_surrogate_distribution_within(data_a, data_b)
        print('\n-------------------------- (4) determine p-value')
        [pvalue, sign] = self._p_value_union()
        return self.union, pvalue, sign

    def compare_between(self, network_set_a, network_set_b, data_set_a,
                        data_set_b):
        """Compare networks between units of observation under two conditions.

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
                results of network comparison
        """
        # Check input and analysis parameters.
        network_all = np.hstack((network_set_a, network_set_b))
        data_all = np.hstack((data_set_a, data_set_b))
        self._check_n_perm_between(data_set_a, data_set_b)
        self._check_n_realisations(*data_all)

        # Main comparison.
        print('\n-------------------------- (1) create union of networks')
        self._create_union(*network_all)
        print('\n-------------------------- (2) calculate original TE values')
        self._calculate_cmi_diff_between(data_set_a, data_set_b)
        print('\n-------------------------- (3) create surrogate distribution')
        self._create_surrogate_distribution_between(data_set_a, data_set_b)
        print('\n-------------------------- (4) determine p-value')
        [pvalue, sign] = self._p_value_union()
        return self.union, pvalue, sign

    def _check_n_perm_within(self, data_a, data_b):
        """Check if requested no. permutations is sufficient for comparison."""
        assert data_a.n_replications == data_b.n_replications, ('Unequal no. '
                                'replications in the two data sets.')
        n_replications = data_a.n_replications
        if self.stats_type == 'dependent':
            if self.n_permutations > 2**n_replications:
                raise RuntimeError('The number of replications {0} in the data'
                                   ' are not sufficient to enable the '
                                   'requested no. permutations {1}'.format(
                                                       n_replications,
                                                       self.n_permutations))
        elif self.stats_type == 'independent':
            if self.n_permutations > binom(2*n_replications, n_replications):
                raise RuntimeError('The number of replications {0} in the data'
                                   ' are not sufficient to enable the '
                                   'requested no. permutations {1}'.format(
                                                       n_replications,
                                                       self.n_permutations))
        else:
            raise RuntimeError('Unknown ''stats_type''!')

    def _check_n_perm_between(self, data_set_a, data_set_b):
        """Check if requested no. permutations is sufficient for comparison."""

        if self.stats_type == 'dependent':
            assert len(data_set_a) == len(data_set_b), ('The number of data '
                                    'sets is not equal between conditions.')
            n_data_sets = len(data_set_a)
            if self.n_permutations > 2**n_data_sets:
                raise RuntimeError('The number of data sets per condition {0} '
                                   'is not sufficient to enable the '
                                   'requested no. permutations {1}'.format(
                                                       n_data_sets,
                                                       self.n_permutations))
        elif self.stats_type == 'independent':
            max_len = max(len(data_set_a), len(data_set_b))
            total_len = len(data_set_a) + len(data_set_b)
            if self.n_permutations > binom(total_len, max_len):
                raise RuntimeError('The total number of data sets {0} is not '
                                   'sufficient to enable the requested no. '
                                   'permutations {1}'.format(
                                                       total_len,
                                                       self.n_permutations))
        else:
            raise RuntimeError('Unknown ''stats_type''!')

    def _check_n_realisations(self, *data_sets):
        """Check if all data sets have an equal no. realisations."""
        n_data_sets = len(data_sets)
        n_realisations = data_sets[0].n_realisations()
        for d in data_sets:
            if d.n_realisations() != n_realisations:
                raise RuntimeError('Unequal no. realisations between data '
                                   'sets.')

    def _create_union(self, *networks):
        """Create the union from a set of individual networks."""

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
        for nw in networks:
            for t in targets:
                self.union[t] = {}
                self.union[t]['conditional_sources'] = []
                self.union[t]['conditional_target'] = []

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
                # network and target. Use an empty array if no sources were
                # selected for that target.
                try:
                    cond_src = nw[t]['selected_vars_sources']
                except KeyError:
                    cond_src = []
                try:
                    cond_tgt = nw[t]['selected_vars_target']
                except KeyError:
                    cond_tgt = []
                # TODO convert lags to indices before adding them to the union

                # Add conditional if it isn't already in the union network.
                for c in cond_src:
                    if c not in self.union[t]['conditional_sources']:
                        self.union[t]['conditional_sources'].append(c)
                for c in cond_tgt:
                    if c not in self.union[t]['conditional_target']:
                        self.union[t]['conditional_target'].append(c)


    def _calculate_cmi_diff_within(self, data_a, data_b, permute=False):
        """Calculate the difference in TE for each source and target.

        Args:
            data_a : Data instance
                first set of raw data
            data_a : Data instance
                second set of raw data
            cmi_options : dict [optional]
                options for CMI estimation
            permute : bool [optional]
                if True, permute data from same replications between sets a and
                b, depending on the stats type set for the instance

        Returns:
            numpy array
                TE differences
        """
        # re-calculate TE for each data object using the union network mask
        self.cmi_diff = self._calculate_diff(self._calculate_cmi(data_a),
                                             self._calculate_cmi(data_b))

    def _calculate_cmi_diff_between(self, data_set_a, data_set_b):
        """Calculate the difference in CMI between two groups of data.
        """
        self.cmi_diff = self._get_diff_of_mean(data_set_a, data_set_b,
                                               permute=False)
        # TODO idea: loop over pairs of data in data_set_a and *_b and feed it to the within function?
        # is the mean of differences the same as the difference of the mean? > yes
        # BUT: such an implementation doesn't allow for unbalanced designs > this sucks and needs to
        # be changed in the within function as well

    def _get_diff_of_mean(self, data_set_a, data_set_b, permute=False):
        # re-calculate TE for each data object using the union network mask
        cmi_union_a = []
        for d in data_set_a:
            cmi_union_a.append(self._calculate_cmi(d))
        cmi_union_b = []
        for d in data_set_b:
            cmi_union_b.append(self._calculate_cmi(d))

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

        # get the mean difference between the two sets of TE estimates


    def _calculate_cmi(self, data):
        """Calculate CMI for each source/target combi in the union network."""

        cmi = {}
        for t in self.union['targets']:
            current_val = (t, self.union['max_lag'])
            cond_full = (self.union[t]['conditional_target'] +
                         self.union[t]['conditional_sources'])
            # get realisations of the current value and the full cond. set.
            [cur_val_real, repl_idx] = data.get_realisations(current_val,
                                                             [current_val])
            cond_full_real = data.get_realisations(current_val, cond_full)[0]

#            if permute:
#                [cond_full_real_a,
#                 cond_full_real_b,
#                 cur_val_real_a,
#                 cur_val_real_b] = self._permute_replications(cond_full_real_a,
#                                                              cond_full_real_b,
#                                                              cur_val_real_a,
#                                                              cur_val_real_b,
#                                                              repl_idx)

            # Calculate TE from each source to current target t
            cmi_temp = []
            for c in self.union[t]['conditional_sources']:
                current_cond = cp.copy(self.union[t]['conditional_sources'])
                current_cond.pop(current_cond.index(c))

                # get realisations of current TE-source from the set of all
                # conditionals and calculate the CMI
                [cond_real, source_real] = self._separate_realisations(
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

    def _calculate_cmi_permuted(self, data, data_perm):
        """Calculate surrogate CMI for union network

        Calculate CMI for each source/target combi after permuting realisations
        of sources between the two data sets. Results can be used in a
        surrogate permutation test of the CMI.
        """

        cmi_a = {}
        cmi_b = {}
        for t in self.union['targets']:

            # Get full conditioning set for current target.
            cond_full = (self.union[t]['conditional_target'] +
                         self.union[t]['conditional_sources'])
            # Get realisations, where realisations are permuted replication-
            # wise between two data sets (e.g., from different conditions)
            [cond_full_perm_a,
             cur_val_perm_a,
             cond_full_perm_b,
             cur_val_perm_b] = self._get_permuted_replications(data,
                                                               data_perm,
                                                               t)
            # Calculate TE from each source to current target t
            cmi_temp_a = []
            cmi_temp_b = []
            for c in self.union[t]['conditional_sources']:
                current_cond = cp.copy(self.union[t]['conditional_sources'])
                current_cond.pop(current_cond.index(c))

                # Get realisations of current (permuted) TE-source from the set
                # of all conditionals and calculate the CMI.
                [cond_real_a, source_real_a] = self._separate_realisations(
                                                             cond_full,
                                                             c,
                                                             cond_full_perm_a)
                cmi_temp_a.append(self._cmi_calculator.estimate(
                                                        cur_val_perm_a,
                                                        source_real_a,
                                                        cond_real_a,
                                                        self.cmi_opts))
                [cond_real_b, source_real_b] = self._separate_realisations(
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
        """Calculate the mean CMI over multiple networks for all targets."""
        cmi_mean = {}
        for t in self.union['targets']:
            n_sources = cmi_set[0][t].shape[0]
            n_datasets = len(cmi_set)
            temp = np.empty((n_datasets, n_sources))
            i = 0
            for c in cmi_set:
                temp[i,:] = c[t]
                i += 1
            cmi_mean[t] = np.mean(temp, axis=0)
        return cmi_mean

    def _calculate_diff(self, cmi_a, cmi_b):
        """Calculate the difference between two CMI estimates over all targets.
        """
        cmi_diff = {}
        for t in self.union['targets']:
            cmi_diff[t] = cmi_a[t] - cmi_b[t]
        return cmi_diff

    def _create_surrogate_distribution_within(self, data_a, data_b):
        """Create the surrogate distribution for network inference.

        Create distribution by permuting realisations between conditions and
        re-calculating the conditional mutual information (CMI). Realisations
        are shuffled as whole trials, the permutation strategy depends on the
        stats type set in the instance (dependent or independent).

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

        Create distribution by permuting CMI estimates between conditions and
        re-calculating the mean of differences. The permutation strategy
        depends on the stats type set in the instance (dependent or
        independent).

        Args:
            data_a : Data instance
                first set of raw data
            data_a : Data instance
                second set of raw data
        """
        self.cmi_surr = []
        for p in range(self.n_permutations):
            self.cmi_surr.append(self._get_diff_of_mean(data_set_a,
                                                        data_set_b,
                                                        permute=True))

    def _p_value_union(self):
        """Calculate the p-value for the TE between each source and target."""
        # Test each original difference against its surrogate distribution.
        significance = {}
        pvalue = {}
        for t in self.union['targets']:
            n_sources = self.cmi_surr[0][t].shape[0]
            if n_sources == 0:
                continue
            surr_temp = np.zeros((self.n_permutations, n_sources))
            significance[t] = np.empty(n_sources, dtype=bool)
            pvalue[t] = np.empty(n_sources)
            for p in range(self.n_permutations):
                surr_temp[p, :] = self.cmi_surr[p][t]
            for s in range(n_sources):
                [sign, pval] = stats._find_pvalue(self.cmi_diff[t][s],
                                                  surr_temp[:, s],
                                                  self.alpha,
                                                  self.tail)
                significance[t][s] = sign
                pvalue[t][s] = pval
        return pvalue, significance

    def _get_permuted_replications(self, data_a, data_b, target):
        """Return realisations with replications permuted between conditions."""

        # Get indices of current value and full conditioning set in the
        # union network.
        current_val = (target, self.union['max_lag'])
        cond_full = (self.union[target]['conditional_target'] +
                     self.union[target]['conditional_sources'])

        # Get realisations of the current value and the full cond. set.
        [cur_val_a, repl_idx] = data_a.get_realisations(current_val,
                                                        [current_val])
        cur_val_b = data_b.get_realisations(current_val, [current_val])[0]
        cond_a = data_a.get_realisations(current_val, cond_full)[0]
        cond_b = data_b.get_realisations(current_val, cond_full)[0]

        # Get no. replications and no. samples per replication.
        n_repl = max(repl_idx) + 1
        n_per_repl = sum(repl_idx == 0)

        # Make copies such as to not overwrite the arrays in the caller scope.
        cond_a_perm = np.empty(cond_a.shape)
        cond_b_perm = np.empty(cond_b.shape)
        cur_val_a_perm = np.empty(cur_val_a.shape)
        cur_val_b_perm = np.empty(cur_val_b.shape)

        # Swap or permute arrays depending on the stats type.
        if self.stats_type == 'dependent':
            swap = np.repeat(np.random.randint(2, size=n_repl).astype(bool),
                             n_per_repl)
            cond_a_perm[swap, :] = cond_b[swap, :]
            cond_b_perm[swap, :] = cond_a[swap, :]
            cond_a_perm[np.invert(swap), :] = cond_b[np.invert(swap), :]
            cond_b_perm[np.invert(swap), :] = cond_a[np.invert(swap), :]

        elif self.stats_type == 'independent':
            resample_a = np.random.choice(n_repl * 2, n_repl, replace=False)
            resample_b = np.setdiff1d(np.arange(n_repl * 2), resample_a)

            # Resample group A.
            ind_0 = 0
            ind_1 = n_per_repl
            for r in resample_a:
                if r >= n_repl:
                    cond_a_perm[ind_0:ind_1, ] = cond_b[
                                                repl_idx == (r - n_repl), :]
                    cur_val_a_perm[ind_0:ind_1, ] = cur_val_b[
                                                repl_idx == (r - n_repl), :]
                else:
                    cond_a_perm[ind_0:ind_1, ] = cond_a[repl_idx == r, :]
                    cur_val_a_perm[ind_0:ind_1, ] = cur_val_a[repl_idx == r, :]
                ind_0 = ind_1
                ind_1 = ind_0 + n_per_repl
            # Resample group B.
            ind_0 = 0
            ind_1 = n_per_repl
            for r in resample_b:
                if r >= n_repl:
                    cond_b_perm[ind_0:ind_1, ] = cond_b[
                                                repl_idx == (r - n_repl), :]
                    cur_val_b_perm[ind_0:ind_1, ] = cur_val_b[
                                                repl_idx == (r - n_repl), :]
                else:
                    cond_b_perm[ind_0:ind_1, ] = cond_a[repl_idx == r, :]
                    cur_val_b_perm[ind_0:ind_1, ] = cur_val_a[repl_idx == r, :]
                ind_0 = ind_1
                ind_1 = ind_0 + n_per_repl
        else:
            raise ValueError('Unkown "stats_type": {0}, should be "dependent" '
                             'or "independent".'.format(self.stats_type))
        return cond_a_perm, cur_val_a_perm, cond_b_perm, cur_val_b_perm

    def _separate_realisations(self, idx_full, idx_single, real_full):
        """Remove single indexes' realisations from a set of realisations.

        Remove the realisations of a single index from a set of realisations.
        Return both the single realisation and realisations for the remaining
        set. This allows us to reuse the collected realisations when pruning
        the conditional set after candidates have been included.

        Args:
            idx_full: list of indices indicating the full set
            idx_single: index to be removed

        Returns:
            realisation_single: numpy array with realisations for single index
            realisations_remaining: numpy array with remaining realisations
        """
        assert(len(idx_full) == real_full.shape[1]), ('Index list does not '
                                                      'correspond with full '
                                                      'realisations.')
        array_idx_single = idx_full.index(idx_single)
        real_single = np.expand_dims(real_full[:, array_idx_single], axis=1)
        real_remaining = idtxl_utils.remove_column(real_full, array_idx_single)
        return real_remaining, real_single
