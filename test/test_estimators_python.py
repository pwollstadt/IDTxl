import numpy as np
import time

import pytest

from idtxl.estimators_jidt import JidtKraskovCMI
from idtxl.estimators_python import PythonKraskovCMI

SEED = 42

def _compute_gaussian_mi(Sigma, s_dim, t_dim):
    SigmaS = Sigma[:s_dim, :s_dim]
    SigmaT = Sigma[s_dim:s_dim+t_dim, s_dim:s_dim+t_dim]

    I = 0.5 * np.log(np.linalg.det(SigmaS) * np.linalg.det(SigmaT) / np.linalg.det(Sigma))

    return I

def _compute_gaussian_cmi(Sigma, s_dim, t_dim, c_dim):

    I_S_TC = _compute_gaussian_mi(Sigma, s_dim, t_dim + c_dim)
    
    Sigma_T_C = Sigma[s_dim:, s_dim:]
    I_T_C = _compute_gaussian_mi(Sigma_T_C, t_dim, c_dim)

    return I_S_TC - I_T_C

_Sigmas = np.array([

    # Test one: Strong corr. between S and T, 
    # no corr. between S and C, 
    # no corr. between T and C
    [[1, 0.99, 0],
     [0.99, 1, 0],
     [0, 0, 1]],

    # Test two: Strong corr. between S and T,
    # some corr. between S and C,
    # some corr. between T and C
    [[1, 0.99, 0.5],
     [0.99, 1, 0.5],
     [0.5, 0.5, 1]],

    # Test three: Strong corr. between S and T,
    # strong corr. between S and C,
    # strong corr. between T and C
    [[1, 0.99, 0.99],
     [0.99, 1, 0.99],
     [0.99, 0.99, 1]],
])

@pytest.mark.parametrize('Sigma', _Sigmas)
def test_cmi_gaussian(Sigma):
    
    rng = np.random.default_rng(SEED)
    S, T, C = rng.multivariate_normal(np.zeros(3), Sigma, 10_000).T
    S, T, C = S[:, np.newaxis], T[:, np.newaxis], C[:, np.newaxis]

    cmi_gaussian = _compute_gaussian_cmi(Sigma, 1, 1, 1)

    print(f'Analytical CMI: {cmi_gaussian}')

    mi_gaussian = _compute_gaussian_mi(Sigma[:2, :2], 1, 1)

    print(f'Analytical MI: {mi_gaussian}')

    # Run JIDT estimator as a reference

    jidt_estimator = JidtKraskovCMI({'kraskov_k': 4, 'noise_level':0, 'num_threads':1})

    itic = time.perf_counter()
    cmi_jidt = jidt_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f'JIDT CMI: {cmi_jidt} (took {itoc - itic} seconds)')
    assert np.isclose(cmi_gaussian, cmi_jidt, rtol=0.08)

    # Run Python estimators with different knn_finders

    python_estimator = PythonKraskovCMI({'kraskov_k':4, 'noise_level':0, 'knn_finder':'scipy_kdtree'})

    itic = time.perf_counter()
    cmi_python = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f'Python CMI (scipy_kdtree): {cmi_python} (took {itoc - itic} seconds)')
    assert np.isclose(cmi_jidt, cmi_python, rtol=1e-4)

    python_estimator = PythonKraskovCMI({'kraskov_k':4, 'noise_level':0, 'knn_finder':'sklearn_kdtree'})

    itic = time.perf_counter()
    cmi_python = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f'Python CMI (sklearn_kdtree): {cmi_python} (took {itoc - itic} seconds)')
    assert np.isclose(cmi_jidt, cmi_python, rtol=1e-4)

    python_estimator = PythonKraskovCMI({'kraskov_k':4, 'noise_level':0, 'knn_finder':'sklearn_balltree'})

    itic = time.perf_counter()
    cmi_python = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()

    print(f'Python CMI (sklearn_balltree): {cmi_python} (took {itoc - itic} seconds)')
    assert np.isclose(cmi_jidt, cmi_python, rtol=1e-4)

if __name__ == '__main__':
    for sigma in _Sigmas:
        test_cmi_gaussian(sigma)
    print('All tests passed.')