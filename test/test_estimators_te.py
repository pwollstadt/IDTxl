import random as rn
import numpy as np
from idtxl.set_estimator import Estimator_te
from idtxl.data import Data
import idtxl.idtxl_utils


def test_multivariate_te_corr_gaussian():
    """Test multivariate TE estimation on correlated Gaussians.

    Run the multivariate TE algorithm on two sets of random Gaussian data with
    a given covariance. The second data set is shifted by one sample creating
    a source-target delay of one sample. This example is modeled after the
    JIDT demo 4 for transfer entropy. The resulting TE can be compared to the
    analytical result (but expect some error in the estimate).

    Note:
        This test runs considerably faster than other system tests.
        This produces strange small values for non-coupled sources.  TODO
    """
    n = 1000
    cov = 0.4
    source_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    # source_2 = [rn.normalvariate(0, 1) for r in range(n)]  # uncorrelated src
    target = [sum(pair) for pair in zip(
        [cov * y for y in source_1],
        [(1 - cov) * y for y in [rn.normalvariate(0, 1) for r in range(n)]])]
    # Cast everything to numpy so the idtxl estimator understands it.
    source_1 = np.expand_dims(np.array(source_1), axis=1).astype(float)
    # source_2 = np.expand_dims(np.array(source_2), axis=1)
    target = np.expand_dims(np.array(target), axis=1).astype(float)

    dat = Data(normalise=True)
    dat.set_data(np.vstack((source_1[1:].T, target[:-1].T)), 'ps')
    analysis_opts = {
        'kraskov_k': 4,
        'normalise': 'false',
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False,
        'tau_target': 1,
        'tau_source': 1,
        'source_target_delay': 1,
        'history_target':1,
        'history_source': 1,
        }
    te_est = Estimator_te('jidt_kraskov')
    te_est.estimate(source_1, target, analysis_opts)

if __name__ == '__main__':
    test_multivariate_te_corr_gaussian()    