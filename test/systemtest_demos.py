"""Provide demos of all user interfaces.

This module tests if demos of all user interfaces run. Individual functions can
later be distributed as demos of the toolboxes' functions.

"""
import time
import random as rn
import numpy as np
from idtxl.multivariate_te import Multivariate_te
from idtxl.bivariate_te import Bivariate_te
from idtxl.single_process_storage import Single_process_storage
from idtxl.set_estimator import (Estimator_te, Estimator_ais, Estimator_cmi,
                                 Estimator_mi, Estimator_pid)
from idtxl.data import Data


# NETWORK INFERENCE
def test_multivariate_te():
    """Test network inference using multivariate TE."""
    dat = Data()
    dat.generate_mute_data(n_samples=100, n_replications=5)
    analysis_opts = {
            'cmi_calc_name': 'jidt_kraskov',
            'n_perm_max_stat': 200,
            'n_perm_min_stat': 200,
            'n_perm_omnibus': 500,
            'n_perm_max_seq': 500,
            }
    start_time = time.time()
    mte = Multivariate_te(max_lag_sources=5, min_lag_sources=1,
                          max_lag_target=5, tau_sources=1,
                          tau_target=1, options=analysis_opts)
    # Analyse target 0 with all possible sources
    res_1 = mte.analyse_single_target(data=dat, target=0, sources='all')
    # Analyse target 1 with two specific sources (0, 2)
    res_2 = mte.analyse_single_target(data=dat, target=1, sources=[0, 2])
    # Analyse the whole network - all-to-all
    res_3 = mte.analyse_network(data=dat, targets='all', sources='all')
    runtime = time.time() - start_time
    print("---- {0} minutes".format(runtime / 60))


def test_bivariate_te():
    """Test network inference using bivariate TE."""
    dat = Data()
    dat.generate_mute_data(n_samples=100, n_replications=5)
    analysis_opts = {
            'cmi_calc_name': 'jidt_kraskov',
            'n_perm_max_stat': 200,
            'n_perm_min_stat': 200,
            'n_perm_omnibus': 500,
            'n_perm_max_seq': 500,
            }
    start_time = time.time()
    bte = Bivariate_te(max_lag_sources=50, min_lag_sources=40,
                       max_lag_target=30, tau_sources=1,
                       tau_target=3, options=analysis_opts)
    # Analyse target 0 with all possible sources
    res_1 = bte.analyse_single_target(data=dat, target=0, sources='all')
    # Analyse target 1 with two specific sources (0, 2)
    res_2 = bte.analyse_single_target(data=dat, target=1, sources=[0, 2])
    # Analyse the whole network - all-to-all
    res_3 = bte.analyse_network(data=dat, targets='all', sources='all')
    runtime = time.time() - start_time
    print("---- {0} minutes".format(runtime / 60))


def test_single_process_storage():
    """Test AIS estimation on network nodes."""
    dat = Data()
    dat.generate_mute_data(n_samples=100, n_replications=5)
    analysis_opts = {
            'cmi_calc_name': 'jidt_kraskov',
            'n_perm_max_stat': 200,
            'n_perm_min_stat': 200
            }
    start_time = time.time()
    sps = Single_process_storage(max_lag=5, options=analysis_opts, tau=1)
    # Estimate AIS in the first process only
    res_1 = sps.analyse_single_process(data=dat, process=0)
    # Estimate AIS in processes 0, 1, 2
    res_2 = sps.analyse_network(dat, processes=[0, 1, 2])
    # Estimate AIS in all processes
    res_3 = sps.analyse_network(dat, processes='all')
    runtime = time.time() - start_time
    print("---- {0} minutes".format(runtime / 60))


# HIGH-LEVEL INFORMATION-THEORY
def test_te_estimation():
    """Test TE estimation on correlated Gaussian time series."""
    n = 1000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    target = [0] + [sum(pair) for pair in zip(
        [cov * y for y in source[0:n - 1]],
        [(1 - cov) * y for y in
            [rn.normalvariate(0, 1) for r in range(n - 1)]])]
    analysis_opts = {
        'kraskov_k': 4,
        'normalise': 'false',
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False,
        'tau_target': 1,
        'tau_source': 1,
        'source_target_delay': 1,
        'history_target': 1,
        'history_source': 1,
        }
    te_est = Estimator_te('jidt_kraskov')
    te_res = te_est.estimate(np.array(source), np.array(target),
                             analysis_opts)
    print('correlated Gaussians: TE result {0:.4f} bits; expected to be'
          '{1:0.4f} bit for the copy'
          .format(te_res, np.log(1 / (1 - cov ** 2))))


def test_ais_estimation():
    """Test AIS estimation on random time series."""
    n = 1000
    process = [rn.normalvariate(0, 1) for r in range(n)]
    analysis_opts = {
        'kraskov_k': 4,
        'normalise': 'false',
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False,
        'tau': 1,
        'history': 1
        }
    ais_est = Estimator_ais('jidt_kraskov')
    ais_res = ais_est.estimate(np.array(process), analysis_opts)
    print('random data: AIS result {0:.4f} bits; expected to be close to zero'
          .format(ais_res))


def test_pid_tartu():
    """Test network inference using PID estimation with Tartu estimator."""
    n = 1000
    source1 = np.random.randint(2, size=n)
    source2 = np.random.randint(2, size=n)
    target = np.logical_xor(source1, source2).astype(int)

    analysis_opts = {}

    pid_est = Estimator_pid('pid_tartu')
    pid_est = pid_est.estimate(source1, source2, target, analysis_opts)


def test_pid_sydney():
    """Test network inference using PID estimation with Sydney estimator."""
    n = 1000
    source1 = np.random.randint(2, size=n)
    source2 = np.random.randint(2, size=n)
    target = np.logical_xor(source1, source2).astype(int)

    analysis_opts = {
        'alph_s1': 2,
        'alph_s2': 2,
        'alph_t': 2,
        'max_unsuc_swaps_row_parm': 1000,
        'num_reps': 60,
        'max_iters': 100
        }
    pid_est = Estimator_pid('pid_sydney')
    pid_est = pid_est.estimate(source1, source2, target, analysis_opts)
    print('binary XOR PID result: {0} bits shared information; {1} '
          'synergistic information.' .format(pid_est['shd_s1_s2'],
                                             pid_est['syn_s1_s2']))


# LOW-LEVEL INFORMATION-THEORY
def test_mi_estimation():
    """Test MI estimation on continuous and discrete data."""
    # continuous data: correlated Gaussians
    n = 1000
    cov = 0.4
    var1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    var2 = [sum(pair) for pair in zip(
                [cov * y for y in var1],
                [(1 - cov) * y for y in [rn.normalvariate(0, 1) for
                                         r in range(n)]])]
    analysis_opts = {
        'kraskov_k': 4,
        'normalise': 'false',
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False
        }
    mi_est = Estimator_mi('jidt_kraskov')
    mi_res = mi_est.estimate(np.array(var1), np.array(var2), analysis_opts)
    print('correlated Gaussians: MI result {0:.4f} bits; expected to be'
          '{1:0.4f} bit for the copy ??'  # TODO expected result?
          .format(mi_res, np.log(1 / (1 - cov ** 2))))

    # discrete data: copy of integers
    var1_discrete = np.zeros(n, np.int_)
    var2_discrete = var1_discrete
    analysis_opts = {
        'num_discrete_bins': 2,
        'time_diff': 0,
        'discretise_method': 'none'
        }
    mi_est = Estimator_mi('jidt_discrete')
    res_1 = mi_est.estimate(var1_discrete, var2_discrete, opts=analysis_opts)
    print('Example 1: MI result {0:.4f} bits; expected to be 1 bit for the '
          'copy'.format(res_1))


def test_cmi_estimation():
    """Test CMI estimation on continuous and discrete data."""
    # continuous data: correlated Gaussians
    n = 1000
    cov = 0.4
    source = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    target = [sum(pair) for pair in zip(
        [cov * y for y in source],
        [(1 - cov) * y for y in [rn.normalvariate(0, 1) for r in range(n)]])]
    # Cast everything to numpy so the idtxl estimator understands it.
    source = np.expand_dims(np.array(source), axis=1)
    target = np.expand_dims(np.array(target), axis=1)
    analysis_opts = {
        'kraskov_k': 4,
        'normalise': True,
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False
        }
    cmi_est = Estimator_cmi('jidt_kraskov')
    res = cmi_est.estimate(var1=source[1:], var2=target[1:],
                           conditional=target[:-1], opts=analysis_opts)
    print('Example 1: TE result {0:.4f} nats; expected to be close to {1:.4f} '
          'nats for these correlated Gaussians.'.format(
                        res, np.log(1 / (1 - np.pow(cov, 2)))))

    # discrete data: copy of integers
    analysis_opts = {
        'num_discrete_bins': 2,
        'time_diff': 0,
        'discretise_method': 'none'
        }

    n = 1000  # Need this to be an even number for the test to work
    source = np.zeros(n, np.int_)
    target = source
    cmi_est = Estimator_cmi('jidt_discrete')
    res = cmi_est.estimate(var1=source, var2=target, conditional=None,
                           opts=analysis_opts)
    print('Example 1a: CMI result {0:.4f} bits; expected to be 1 bit for the '
          'copy (no conditional)'.format(res))


if __name__ == "__main__":
    test_multivariate_te()
    test_bivariate_te()
    test_single_process_storage()
    test_pid_tartu()
    test_pid_sydney()
    test_te_estimation()
    test_ais_estimation()
    test_mi_estimation()
    test_cmi_estimation()
