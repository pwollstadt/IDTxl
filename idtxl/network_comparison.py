# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:32:46 2016

@author: patricia
"""
import sys
import copy as cp
import numpy as np
from . import set_estimator.Estimator_cmi
from . import data.Data
from . import stats
from . import utils

VERBOSE = True


class Network_comparison():

    def __init__(self, options):
        try:
            self.cmi_calculator = Estimator_cmi(options['cmi_calc_name'])
        except KeyError:
            raise KeyError('You have to provide an estimator name.')
        try:
            self.n_permutations = options['n_perm_comp']
        except KeyError:
            self.n_permutations = 10
        try:
            self.alpha = options['alpha_comp']
        except KeyError:
            self.alpha = 0.05
        try:
            self.tail = options['tail']
        except KeyError:
            self.tail = 'two'
        try:
            self.stats_type = options['stats_type']
        except KeyError:
            raise KeyError('You have to provide a "stats_type", "dependent" '
                           'or "independent".')
        self._check_n_perm()


class Compare_single_recording(Network_comparison):
    """Statisticall compare two networks from a single recording.

    Arguments:
        network_1 : dict
            results from network inference
        network_2 : dict
            results from network inference
        options : dict
            options for statistical comparison of networks
            'cmi_calc_name' - estimator to be used for CMI calculation
            (For estimator options see the respective documentation.)
            'n_perm_comp' - number of permutations (default=500)
            'alpha_comp' - critical alpha level for statistical significance
            (default=0.05)

    Returns:
        numpy array, bool
            statistical significance of difference of each source
        numpy array, float
            the test's p-values for each difference

    """

    def __init__(self, options):
        super().__init__(options)

    def compare(self, network_a, network_b, data_a, data_b):
        # Generate union of links from both networks.
        if VERBOSE:
            print('\n------------------- (1) create union of networks')
        self._create_union(network_a, network_b)

        # Calculate test statistic from original data.
        if VERBOSE:
            print('\n------------------- (2) calculate original TE values')
        cmi_diff = self._calculate_cmi_diff(data_a, data_b, options)

        # Generate surrogate distribution.
        surr_distribution = np.zeros((self.n_permutations,
                                      self.union['n_sources_tot']))
        if VERBOSE:
            print('\n------------------- (3) generate surrogate distributions')
            print('n_perm: {0}. Done:    '.format(self.n_permutations), end='')
        for p in range(self.n_permutations):
            surr_distribution[p, ] = self._calculate_cmi_diff(data_a, data_b,
                                                              options,
                                                              permute=True)
            if VERBOSE:
                print('\b\b\b{num:03d}'.format(num=p + 1), end='')
                sys.stdout.flush()
        if VERBOSE:
            print(' ')

        # Test each original difference against its surrogate distribution.
        significance = np.empty(self.union['n_sources_tot'])
        pvalue = np.empty(self.union['n_sources_tot'])
        for c in range(self.union['n_sources_tot']):
            [s, p] = stats._find_pvalue(cmi_diff[c], surr_distribution[:, c],
                                        self.alpha, tail=self.tail)
            significance[c] = s
            pvalue[c] = p

        return significance, pvalue

    def _check_n_perm(self):
        sufficient = True
        # TODO add this
        if not sufficient:
            raise ValueError('The number of realisations is not high enough to'
                             ' allow for a sufficient number of permuations.')

    def _create_union(self, network_a, network_b):
        """Create the union of sources for two networks."""
        targets = np.array([i for i in network_a.keys()] +
                           [i for i in network_b.keys()])
        targets = np.unique(targets)
        self.union = {}

        # Get the maximum lags from the networks, we need this to get
        # realisations of variables later on.
        self.union['targets'] = targets
        self.union['n_sources_tot'] = 0
        self.union['max_lag'] = network_a[targets[0]]['current_value'][1]

        for t in targets:
            self.union[t] = {}
            try:
                lag = network_a[t]['current_value'][1]
                if self.union['max_lag'] != lag:
                    raise ValueError('Networks seem to have been analyzed '
                                     'using different lags.')
            except KeyError:
                pass
            try:
                lag = network_b[t]['current_value'][1]
                if self.union['max_lag'] != lag:
                    raise ValueError('Networks seem to have been analyzed '
                                     'using different lags.')
            except KeyError:
                pass

            try:
                cond_src_a = network_a[t]['conditional_sources']
                cond_tgt_a = network_a[t]['conditional_target']
            except KeyError:
                cond_src_a = []
                cond_tgt_a = []
            try:
                cond_src_b = network_b[t]['conditional_sources']
                cond_tgt_b = network_b[t]['conditional_target']
            except KeyError:
                cond_src_b = []
                cond_tgt_b = []

            # TODO convert lags to indices before adding them to the union

            try:
                self.union[t]['conditional_sources'] += cond_src_a
                self.union[t]['conditional_target'] += cond_tgt_a
            except KeyError:
                self.union[t]['conditional_sources'] = cond_src_a
                self.union[t]['conditional_target'] = cond_tgt_a
            for c in cond_src_b:
                try:
                    self.union[t]['conditional_sources'].index(c)
                except ValueError:
                    self.union[t]['conditional_sources'].append(c)
            for c in cond_tgt_b:
                try:
                    self.union[t]['conditional_target'].index(c)
                except ValueError:
                    self.union[t]['conditional_target'].append(c)

            self.union['n_sources_tot'] += len(
                                        self.union[t]['conditional_sources'])

    def _calculate_cmi_diff(self, data_a, data_b, cmi_options=None,
                            permute=False):
        """Calculate the difference in TE for each source and target.

        Arguments:
            data_a : Data instance
                first set of raw data
            data_a : Data instance
                second set of raw data
            cmi_options : dict [optional]
                options for CMI estimation
            permute : bool [optional]
                if True, permute data from same replications between sets a and
                b

        Returns:
            numpy array
                TE differences
        """
        te_diff = np.arange(0)
        # Calculate TE for each candidate for each target.
        for t in self.union['targets']:
            # Get realisations for the current target only once and reuse in
            # the CMI calculation for each source.
            current_val = (t, self.union['max_lag'])
            cond_full = (self.union[t]['conditional_target'] +
                         self.union[t]['conditional_sources'])
            [cur_val_real_a, repl_idx] = data_a.get_realisations(current_val,
                                                                 [current_val])
            cur_val_real_b = data_b.get_realisations(current_val,
                                                     [current_val])[0]
            cond_full_real_a = data_a.get_realisations(current_val,
                                                       cond_full)[0]
            cond_full_real_b = data_b.get_realisations(current_val,
                                                       cond_full)[0]

            if permute:
                [cond_full_real_a,
                 cond_full_real_b,
                 cur_val_real_a,
                 cur_val_real_b, ] = (self._permute_replications(
                                                             cond_full_real_a,
                                                             cond_full_real_b,
                                                             cur_val_real_a,
                                                             cur_val_real_b,
                                                             repl_idx))

            te_diff_temp = np.zeros(len(self.union[t]['conditional_sources']))
            i = 0
            for c in self.union[t]['conditional_sources']:
                current_cond = cp.copy(self.union[t]['conditional_sources'])
                current_cond.pop(current_cond.index(c))

                # Calculate first TE.
                [cond_real, source_real] = self._separate_realisations(
                                                             cond_full,
                                                             c,
                                                             cond_full_real_a)

                cmi_a = self.cmi_calculator.estimate(cur_val_real_a,
                                                     source_real,
                                                     cond_real,
                                                     options)

                # Calculate second TE.
                [cond_real, source_real] = self._separate_realisations(
                                                             cond_full,
                                                             c,
                                                             cond_full_real_b)
                cmi_b = self.cmi_calculator.estimate(cur_val_real_b,
                                                     source_real,
                                                     cond_real,
                                                     options)
                te_diff_temp[i] = cmi_a - cmi_b
                i += 1
            te_diff = np.hstack((te_diff, te_diff_temp))
        return te_diff

    def _permute_replications(self, cond_a, cond_b, cur_val_a, cur_val_b,
                              repl_idx):
        """Permute realisations in replications between two conditions."""
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
        return cond_a_perm, cond_b_perm, cur_val_a_perm, cur_val_b_perm

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
        real_remaining = utils.remove_column(real_full, array_idx_single)
        return real_remaining, real_single

if __name__ == '__main__':
    r0 = {
        'current_value': (0, 5),
        'conditional_sources': [(1, 1), (1, 2), (1, 3), (2, 1), (2, 0)],
        'conditional_target': [(0, 1), (0, 2), (0, 3)],
        'conditional_full': [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
                             (2, 1), (2, 0)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.001, 0.0014, 0.01, 0.045, 0.047])
        }
    r1 = {
        'current_value': (1, 5),
        'conditional_sources': [(2, 0), (2, 1), (2, 2), (3, 1), (3, 2)],
        'conditional_target': [(1, 0), (1, 1)],
        'conditional_full': [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1),
                             (3, 2)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.00001, 0.00014, 0.01, 0.035, 0.02])
        }
    r2 = {
        'current_value': (2, 5),
        'conditional_sources': [],
        'conditional_target': [(3, 0), (3, 1)],
        'conditional_full': [(3, 0), (3, 1)],
        'omnibus_sign': False,
        'cond_sources_pval': None
        }
    res_0 = {
        0: r0,
        1: r1,
        2: r2
    }

    r0 = {
        'current_value': (0, 5),
        'conditional_sources': [(2, 1), (2, 0)],
        'conditional_target': [(0, 1), (0, 2), (0, 4)],
        'conditional_full': [(0, 1), (0, 2), (0, 3), (2, 1), (2, 0)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.001, 0.0014, 0.01, 0.045, 0.047])
        }
    r1 = {
        'current_value': (1, 5),
        'conditional_sources': [(2, 0), (2, 1), (3, 2), (3, 0)],
        'conditional_target': [(1, 1), (1, 2), (1, 3)],
        'conditional_full': [(1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (3, 2),
                             (3, 0)],
        'omnibus_sign': True,
        'cond_sources_pval': np.array([0.00001, 0.00014, 0.01, 0.035, 0.02])
        }
    r3 = {
        'current_value': (3, 5),
        'conditional_sources': [(2, 0), (2, 1), (1, 0), (1, 1), (1, 2)],
        'conditional_target': [(3, 1), (3, 3)],
        'conditional_full': [(3, 1), (3, 3), (2, 0), (2, 1), (1, 0), (1, 1),
                             (1, 2)],
        'omnibus_sign': False,
        'cond_sources_pval': None
        }
    res_1 = {
        0: r0,
        1: r1,
        3: r3
    }

    data_1 = Data()
    data_1.generate_mute_data(100, 5)
    data_2 = Data()
    data_2.generate_mute_data(100, 5)
    options = {
        'cmi_calc_name': 'jidt_kraskov',
        'stats_type': 'independent',
        'n_perm_comp': 10,
        'alpha_comp': 0.5
        }

    comp = Compare_single_recording(options)
    comp.compare(res_0, res_1, data_1, data_2)
