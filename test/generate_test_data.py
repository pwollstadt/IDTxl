"""Generate test data for IDTxl network comparison unit and system tests.

Generate test data for IDTxl network comparison unit and system tests. Simulate
discrete and continous data from three correlated Gaussian data sets. Perform
network inference using bivariate/multivariate mutual information (MI)/transfer
entropy (TE) analysis. Results are saved used for unit and system testing of
network comparison (systemtest_network_comparison.py).

A coupling is simulated as a lagged, linear correlation between three Gaussian
variables and looks like this:

    1 -> 2 -> 3  with a delay of 1 sample for each coupling
"""
import pickle
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.multivariate_mi import MultivariateMI
from idtxl.bivariate_mi import BivariateMI
from idtxl.estimators_jidt import JidtDiscreteCMI
from idtxl.data import Data

# path = os.path.join(os.path.dirname(__file__) + '/data/')
path = 'data/'


def analyse_mute_te_data():
    # Generate example data: the following was ran once to generate example
    # data, which is now in the data sub-folder of the test-folder.
    data = Data()
    data.generate_mute_data(100, 5)
    # analysis settings
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 50,
        'n_perm_min_stat': 50,
        'n_perm_omnibus': 200,
        'n_perm_max_seq': 50,
        'max_lag_target': 5,
        'max_lag_sources': 5,
        'min_lag_sources': 1,
        'permute_in_time': True
        }
    # network inference for individual data sets
    nw_0 = MultivariateTE()
    res_0 = nw_0.analyse_network(
        settings, data, targets=[0, 1], sources='all')
    pickle.dump(res_0, open(path + 'mute_results_0.p', 'wb'))
    res_1 = nw_0.analyse_network(
        settings, data,  targets=[1, 2], sources='all')
    pickle.dump(res_1, open(path + 'mute_results_1.p', 'wb'))
    res_2 = nw_0.analyse_network(
        settings, data,  targets=[0, 2], sources='all')
    pickle.dump(res_2, open(path + 'mute_results_2.p', 'wb'))
    res_3 = nw_0.analyse_network(
        settings, data,  targets=[0, 1, 2], sources='all')
    pickle.dump(res_3, open(path + 'mute_results_3.p', 'wb'))
    res_4 = nw_0.analyse_network(
        settings, data,  targets=[1, 2], sources='all')
    pickle.dump(res_4, open(path + 'mute_results_4.p', 'wb'))
    res_5 = nw_0.analyse_network(settings, data)
    pickle.dump(res_5, open(path + 'mute_results_full.p', 'wb'))


def generate_discrete_data(n_replications=1):
    """Generate Gaussian test data: 1 -> 2 -> 3, delay 1."""
    d = generate_gauss_data(n_replications=n_replications, discrete=True)
    data = Data(d, dim_order='psr', normalise=False)
    return data


def generate_continuous_data(n_replications=1):
    """Generate Gaussian test data: 1 -> 2 -> 3, delay 1."""
    d = generate_gauss_data(n_replications=n_replications, discrete=False)
    data = Data(d, dim_order='psr', normalise=True)
    return data


def generate_gauss_data(n_replications=1, discrete=False):
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 5}
    est = JidtDiscreteCMI(settings)
    covariance_1 = 0.4
    covariance_2 = 0.3
    n = 10000
    delay = 1
    if discrete:
        d = np.zeros((3, n - 2*delay, n_replications), dtype=int)
    else:
        d = np.zeros((3, n - 2*delay, n_replications))
    for r in range(n_replications):
        proc_1 = np.random.normal(0, 1, size=n)
        proc_2 = (covariance_1 * proc_1 + (1 - covariance_1) *
                  np.random.normal(0, 1, size=n))
        proc_3 = (covariance_2 * proc_2 + (1 - covariance_2) *
                  np.random.normal(0, 1, size=n))
        proc_1 = proc_1[(2*delay):]
        proc_2 = proc_2[delay:-delay]
        proc_3 = proc_3[:-(2*delay)]

        if discrete:  # discretise data
            proc_1_dis, proc_2_dis = est._discretise_vars(
                var1=proc_1, var2=proc_2)
            proc_1_dis, proc_3_dis = est._discretise_vars(
                var1=proc_1, var2=proc_3)
            d[0, :, r] = proc_1_dis
            d[1, :, r] = proc_2_dis
            d[2, :, r] = proc_3_dis
        else:
            d[0, :, r] = proc_1
            d[1, :, r] = proc_2
            d[2, :, r] = proc_3
    return d


def analyse_discrete_data():
    """Run network inference on discrete data."""
    data = generate_discrete_data()
    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,  # alphabet size of the variables analysed
        'min_lag_sources': 1,
        'max_lag_sources': 3,
        'max_lag_target': 1}

    nw = MultivariateTE()
    res = nw.analyse_network(settings=settings, data=data)
    pickle.dump(res, open('{0}discrete_results_mte_{1}.p'.format(
        path, settings['cmi_estimator']), 'wb'))

    nw = BivariateTE()
    res = nw.analyse_network(settings=settings, data=data)
    pickle.dump(res, open('{0}discrete_results_bte_{1}.p'.format(
        path, settings['cmi_estimator']), 'wb'))

    nw = MultivariateMI()
    res = nw.analyse_network(settings=settings, data=data)
    pickle.dump(res, open('{0}discrete_results_mmi_{1}.p'.format(
        path, settings['cmi_estimator']), 'wb'))

    nw = BivariateMI()
    res = nw.analyse_network(settings=settings, data=data)
    pickle.dump(res, open('{0}discrete_results_bmi_{1}.p'.format(
        path, settings['cmi_estimator']), 'wb'))


def analyse_continuous_data():
    """Run network inference on continuous data."""
    data = generate_continuous_data()
    settings = {
        'min_lag_sources': 1,
        'max_lag_sources': 3,
        'max_lag_target': 1}

    nw = MultivariateTE()
    for estimator in ['JidtGaussianCMI', 'JidtKraskovCMI']:
        settings['cmi_estimator'] = estimator
        res = nw.analyse_network(settings=settings, data=data)
        pickle.dump(res, open('{0}continuous_results_mte_{1}.p'.format(
            path, estimator), 'wb'))

    nw = BivariateTE()
    for estimator in ['JidtGaussianCMI', 'JidtKraskovCMI']:
        settings['cmi_estimator'] = estimator
        res = nw.analyse_network(settings=settings, data=data)
        pickle.dump(res, open('{0}continuous_results_bte_{1}.p'.format(
            path, estimator), 'wb'))

    nw = MultivariateMI()
    for estimator in ['JidtGaussianCMI', 'JidtKraskovCMI']:
        settings['cmi_estimator'] = estimator
        res = nw.analyse_network(settings=settings, data=data)
        pickle.dump(res, open('{0}continuous_results_mmi_{1}.p'.format(
            path, estimator), 'wb'))

    nw = BivariateMI()
    for estimator in ['JidtGaussianCMI', 'JidtKraskovCMI']:
        settings['cmi_estimator'] = estimator
        res = nw.analyse_network(settings=settings, data=data)
        pickle.dump(res, open('{0}continuous_results_bmi_{1}.p'.format(
            path, estimator), 'wb'))


def assert_results():
    for algo in ['mmi', 'mte', 'bmi', 'bte']:
        # Test continuous data:
        for estimator in ['JidtGaussianCMI', 'JidtKraskovCMI']:
            res = pickle.load(open(
                'data/continuous_results_{0}_{1}.p'.format(
                    algo, estimator), 'rb'))
            print('\nInference algorithm: {0} (estimator: {1})'.format(
                algo, estimator))
            _print_result(res)

        # Test discrete data:
        estimator = 'JidtDiscreteCMI'
        res = pickle.load(open(
            'data/discrete_results_{0}_{1}.p'.format(
                algo, estimator), 'rb'))
        print('\nInference algorithm: {0} (estimator: {1})'.format(
            algo, estimator))
        _print_result(res)


def _print_result(res):
    res.adjacency_matrix.print_matrix()
    tp = 0
    fp = 0
    if res.adjacency_matrix._edge_matrix[0, 1] == True: tp += 1
    if res.adjacency_matrix._edge_matrix[1, 2] == True: tp += 1
    if res.adjacency_matrix._edge_matrix[0, 2] == True: fp += 1
    fn = 2 - tp
    print('TP: {0}, FP: {1}, FN: {2}'.format(tp, fp, fn))


if __name__ == '__main__':
    analyse_discrete_data()
    analyse_mute_te_data()
    analyse_continuous_data()
    assert_results()
