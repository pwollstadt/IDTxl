"""Test Active Information Storag(AIS) estimators.

This module provides unit tests for AIS estimators.
"""
import pytest
import random as rn
import numpy as np
from idtxl.set_estimator import Estimator_ais
from test_estimators_cmi import jpype_missing


@jpype_missing
def test_jidt_kraskov_input():
    """Test handling of wrong inputs to the JIDT Kraskov TE-estimator."""
    ais_est = Estimator_ais('jidt_kraskov')
    process = np.empty((100))

    # Wrong type for options dictinoary
    with pytest.raises(TypeError):
        ais_est.estimate(process=process, opts=None)
    # Missing history for the target
    analysis_opts = {}
    with pytest.raises(RuntimeError):
        ais_est.estimate(process=process, opts=analysis_opts)
    # Run analysis with all default vales
    analysis_opts = {'history': 3}
    ais_est.estimate(process=process, opts=analysis_opts)


@jpype_missing
def test_ais_estimator_kraskov():
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
    source = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    # Cast everything to numpy so the idtxl estimator understands it.
    source = np.array(source)

    analysis_opts = {
        'kraskov_k': 4,
        'normalise': False,
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False,
        'tau': 1,
        'history': 1,
        }
    ais_est = Estimator_ais('jidt_kraskov')
    ais = ais_est.estimate(source, analysis_opts)
    assert np.abs(ais) < 0.1, ('AIS should be close to 0, but is '
                               '{0}.'.format(ais))


@jpype_missing
def test_ais_local_values():
    """Test local AIS estimation."""
    n = 1000
    source = [rn.normalvariate(0, 1) for r in range(n)]
    analysis_opts = {
        'kraskov_k': 4,
        'normalise': False,
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': True,
        'history': 3,
        }
    ais_est = Estimator_ais('jidt_kraskov')
    ais_res = ais_est.estimate(np.array(source),
                               analysis_opts)
    assert ais_res.shape[0] == n, 'Local AIS estimator did not return an array'


@jpype_missing
def test_ais_estimator_jidt_gaussian():
    """Test Gaussian AIS estimation on normally-distributed time series.
    """
    # Set length of time series
    n = 1000
    # Set tolerance for assert
    assert_tolerance = 0.01
    # Generate random normally-distributed time series
    source = [rn.normalvariate(0, 1) for r in range(n)]
    # Cast everything to numpy so the idtxl estimator understands it.
    source = np.array(source)
    # Call JIDT to perform estimation
    opts = {
        'normalise': False,
        'theiler_t': 0,
        'noise_level': 1e-8,
        'local_values': False,
        'tau': 1,
        'history': 1,
        }
    est = Estimator_ais('jidt_gaussian')
    res = est.estimate(process=source, opts=opts)
    # Compute theoretical value for comparison
    theoretical_res = 0
    print('AIS result: {0:.4f} nats; expected to be '
          '{1:.4f} nats.'.format(res, theoretical_res))
    assert (np.abs(res - theoretical_res) < assert_tolerance),\
        ('Test for Gaussians AIS estimator on Gaussian data failed'
         '(error larger than {1:.4f}).'.format(assert_tolerance))


if __name__ == '__main__':
    test_jidt_kraskov_input()
    test_ais_local_values()
    test_ais_estimator_kraskov
    test_ais_estimator_jidt_gaussian()
