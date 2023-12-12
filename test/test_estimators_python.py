import numpy as np
import time

import pytest

from idtxl.estimators_jidt import JidtKraskovCMI, JidtDiscreteCMI
from idtxl.estimators_python import PythonKraskovCMI, PythonDiscretePluginCMI

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

_Sigmas_3var = np.array([

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

@pytest.mark.parametrize('Sigma', _Sigmas_3var)
def test_cmi_gaussian(Sigma):
    
    rng = np.random.default_rng(SEED)
    S, T, C = rng.multivariate_normal(np.zeros(3), Sigma, 10_000).T
    S, T, C = S[:, np.newaxis], T[:, np.newaxis], C[:, np.newaxis]

    cmi_gaussian = _compute_gaussian_cmi(Sigma, 1, 1, 1)

    print(f'Analytical CMI: {cmi_gaussian}')

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

_Sigmas_2var = np.array([
    # Test one: No corr. between S and T
    [[1, 0],
     [0, 1]],

    # Test two: Some corr. between S and T
    [[1, 0.5],
     [0.5, 1]],

    # Test three: Strong corr. between S and T
    [[1, 0.99],
     [0.99, 1]],
])

@pytest.mark.parametrize('Sigma', _Sigmas_2var)
def test_mi_gaussian(Sigma):

    rng = np.random.default_rng(SEED)
    S, T = rng.multivariate_normal(np.zeros(2), Sigma, 10_000).T
    S, T = S[:, np.newaxis], T[:, np.newaxis]

    mi_gaussian = _compute_gaussian_mi(Sigma, 1, 1)
    print(f'Analytical MI: {mi_gaussian}')

    # Run JIDT estimator as a reference

    jidt_estimator = JidtKraskovCMI({'kraskov_k': 4, 'noise_level':0, 'num_threads':1})

    itic = time.perf_counter()
    mi_jidt = jidt_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f'JIDT MI: {mi_jidt} (took {itoc - itic} seconds)')
    assert np.isclose(mi_gaussian, mi_jidt, rtol=0.08, atol=0.01)

    # Run Python estimators with different knn_finders

    python_estimator = PythonKraskovCMI({'kraskov_k':4, 'noise_level':0, 'knn_finder':'scipy_kdtree'})

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f'Python MI (scipy_kdtree): {mi_python} (took {itoc - itic} seconds)')
    assert np.isclose(mi_jidt, mi_python, rtol=1e-4, atol=1e-4)

    python_estimator = PythonKraskovCMI({'kraskov_k':4, 'noise_level':0, 'knn_finder':'sklearn_kdtree'})

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f'Python MI (sklearn_kdtree): {mi_python} (took {itoc - itic} seconds)')
    assert np.isclose(mi_jidt, mi_python, rtol=1e-4, atol=1e-4)

    python_estimator = PythonKraskovCMI({'kraskov_k':4, 'noise_level':0, 'knn_finder':'sklearn_balltree'})

    itic = time.perf_counter()
    mi_python = python_estimator.estimate(var1=S, var2=T)
    itoc = time.perf_counter()

    print(f'Python MI (sklearn_balltree): {mi_python} (took {itoc - itic} seconds)')
    assert np.isclose(mi_jidt, mi_python, rtol=1e-4, atol=1e-4)

def _create_binary_data(n_samples, st_corr, tc_corr):
    rng = np.random.default_rng(SEED)
    T = rng.binomial(1, 0.5, size=(n_samples, 1))
    S = T ^ rng.binomial(1, 1 - st_corr, size=(n_samples, 1))
    C = T ^ rng.binomial(1, 1 - tc_corr, size=(n_samples, 1))

    return S, T, C

def test_binary_cmi():

    S, T, C = _create_binary_data(10, 1, .05)

    jidt_estimator = JidtDiscreteCMI()
    itic = time.perf_counter()
    cmi_jidt = jidt_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()
    print(f'JIDT CMI: {cmi_jidt} (took {itoc - itic} seconds)')

    python_estimator = PythonDiscretePluginCMI({'sparsity':'dense'})
    itic = time.perf_counter()
    cmi_python1 = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()
    print(f'Python CMI (dense): {cmi_python1} (took {itoc - itic} seconds)')

    python_estimator = PythonDiscretePluginCMI({'sparsity':'sparse'})
    itic = time.perf_counter()
    cmi_python2 = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()
    print(f'Python CMI (sparse): {cmi_python2} (took {itoc - itic} seconds)')

    assert np.isclose(cmi_python1, cmi_python2, rtol=1e-100)
    assert np.isclose(cmi_python1, cmi_jidt, atol=0.0001)

def _create_discrete_data(n_samples, alph1, alph2, alphc, dim1=1, dim2=1, dimc=1):
    rng = np.random.default_rng(SEED)
    T = rng.integers(0, alph1, size=(n_samples, dim1))
    S = rng.integers(0, alph2, size=(n_samples, dim2))
    C = rng.integers(0, alphc, size=(n_samples, dimc))

    return S, T, C

def test_discrete_cmi():
    alph1 = 1000
    alph2 = 1000
    alphc = 1000
    S, T, C = _create_discrete_data(1_000_000, alph1, alph2, alphc)

    python_estimator = PythonDiscretePluginCMI({'sparsity':'sparse', 'alph1':alph1, 'alph2':alph2, 'alphc':alphc})
    itic = time.perf_counter()
    cmi_python2 = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()
    print(f'Python CMI (sparse): {cmi_python2} (took {itoc - itic} seconds)')

    python_estimator = PythonDiscretePluginCMI({'sparsity':'dense', 'alph1':alph1, 'alph2':alph2, 'alphc':alphc})
    itic = time.perf_counter()
    cmi_python1 = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()
    print(f'Python CMI (dense): {cmi_python1} (took {itoc - itic} seconds)')

    jidt_estimator = JidtDiscreteCMI({'alph1':alph1, 'alph2':alph2, 'alphc':alphc})
    itic = time.perf_counter()
    cmi_jidt = jidt_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()
    print(f'JIDT CMI: {cmi_jidt} (took {itoc - itic} seconds)')

    assert np.isclose(cmi_python1, cmi_python2, rtol=1e-100)
    assert np.isclose(cmi_python1, cmi_jidt, atol=0.0001)

def test_discrete_mi():
    alph1 = 1000
    alph2 = 1000
    S, T, C = _create_discrete_data(10_000, alph1, alph2, 1)

    python_estimator = PythonDiscretePluginCMI({'sparsity':'sparse', 'alph1':alph1, 'alph2':alph2})
    itic = time.perf_counter()
    cmi_python2 = python_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'Python CMI (sparse): {cmi_python2} (took {itoc - itic} seconds)')

    python_estimator = PythonDiscretePluginCMI({'sparsity':'dense', 'alph1':alph1, 'alph2':alph2})
    itic = time.perf_counter()
    cmi_python1 = python_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'Python CMI (dense): {cmi_python1} (took {itoc - itic} seconds)')

    jidt_estimator = JidtDiscreteCMI({'alph1':alph1, 'alph2':alph2})
    itic = time.perf_counter()
    cmi_jidt = jidt_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'JIDT CMI: {cmi_jidt} (took {itoc - itic} seconds)')

    assert np.isclose(cmi_python1, cmi_python2, rtol=1e-100)
    assert np.isclose(cmi_python1, cmi_jidt, atol=0.0001)

def test_discrete_cmi2d():
    alph1 = 20
    alph2 = 20
    alphc= 20
    S, T, C = _create_discrete_data(10_000, alph1, alph2, alphc, 2, 2, 2)

    python_estimator = PythonDiscretePluginCMI({'sparsity':'sparse', 'alph1':alph1, 'alph2':alph2, 'alphc':alphc})
    itic = time.perf_counter()
    cmi_python2 = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()
    print(f'Python CMI (sparse): {cmi_python2} (took {itoc - itic} seconds)')

    python_estimator = PythonDiscretePluginCMI({'sparsity':'dense', 'alph1':alph1, 'alph2':alph2, 'alphc':alphc})
    itic = time.perf_counter()
    cmi_python1 = python_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()
    print(f'Python CMI (dense): {cmi_python1} (took {itoc - itic} seconds)')

    jidt_estimator = JidtDiscreteCMI({'alph1':alph1, 'alph2':alph2, 'alphc':alphc})
    itic = time.perf_counter()
    cmi_jidt = jidt_estimator.estimate(var1=S, var2=T, conditional=C)
    itoc = time.perf_counter()
    print(f'JIDT CMI: {cmi_jidt} (took {itoc - itic} seconds)')

    assert np.isclose(cmi_python1, cmi_python2, rtol=1e-100)
    assert np.isclose(cmi_python1, cmi_jidt, atol=0.0001)

def test_discrete_mi2d():
    alph1 = 100
    alph2 = 100
    S, T, C = _create_discrete_data(10_000, alph1, alph2, 1, 2, 2, 2)

    python_estimator = PythonDiscretePluginCMI({'sparsity':'sparse', 'alph1':alph1, 'alph2':alph2})
    itic = time.perf_counter()
    cmi_python2 = python_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'Python CMI (sparse): {cmi_python2} (took {itoc - itic} seconds)')

    python_estimator = PythonDiscretePluginCMI({'sparsity':'dense', 'alph1':alph1, 'alph2':alph2})
    itic = time.perf_counter()
    cmi_python1 = python_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'Python CMI (dense): {cmi_python1} (took {itoc - itic} seconds)')

    jidt_estimator = JidtDiscreteCMI({'alph1':alph1, 'alph2':alph2})
    itic = time.perf_counter()
    cmi_jidt = jidt_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'JIDT CMI: {cmi_jidt} (took {itoc - itic} seconds)')

    assert np.isclose(cmi_python1, cmi_python2, rtol=1e-100)
    assert np.isclose(cmi_python1, cmi_jidt, atol=0.0001)

@pytest.mark.parametrize('Sigma', _Sigmas_3var)
def test_discrete_gaussian_cmi(Sigma: np.ndarray):
    rng = np.random.default_rng(SEED)
    S, T, C = rng.multivariate_normal(np.zeros(3), Sigma, 10_000).T
    S, T, C = S[:, np.newaxis], T[:, np.newaxis], C[:, np.newaxis]

    alph1 = 100
    alph2 = 100
    alphc = 100
    kwargs = {'alph1':alph1, 'alph2':alph2, 'alphc':alphc, 'discretise_method':'max_ent'}

    python_estimator = PythonDiscretePluginCMI({'sparsity':'sparse', **kwargs})
    itic = time.perf_counter()
    cmi_python2 = python_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'Python CMI (sparse): {cmi_python2} (took {itoc - itic} seconds)')

    python_estimator = PythonDiscretePluginCMI({'sparsity':'dense', **kwargs})
    itic = time.perf_counter()
    cmi_python1 = python_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'Python CMI (dense): {cmi_python1} (took {itoc - itic} seconds)')

    jidt_estimator = JidtDiscreteCMI(kwargs)
    itic = time.perf_counter()
    cmi_jidt = jidt_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'JIDT CMI: {cmi_jidt} (took {itoc - itic} seconds)')

    assert np.isclose(cmi_python1, cmi_python2, rtol=1e-100)
    assert np.isclose(cmi_python1, cmi_jidt, atol=0.0001)

@pytest.mark.parametrize('Sigma', _Sigmas_2var)
def test_discrete_gaussian_mi(Sigma: np.ndarray):
    rng = np.random.default_rng(SEED)
    S, T = rng.multivariate_normal(np.zeros(2), Sigma, 10_000).T
    S, T = S[:, np.newaxis], T[:, np.newaxis]

    alph1 = 100
    alph2 = 100
    kwargs = {'alph1':alph1, 'alph2':alph2, 'discretise_method':'max_ent'}

    python_estimator = PythonDiscretePluginCMI({'sparsity':'sparse', **kwargs})
    itic = time.perf_counter()
    cmi_python2 = python_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'Python CMI (sparse): {cmi_python2} (took {itoc - itic} seconds)')

    python_estimator = PythonDiscretePluginCMI({'sparsity':'dense', **kwargs})
    itic = time.perf_counter()
    cmi_python1 = python_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'Python CMI (dense): {cmi_python1} (took {itoc - itic} seconds)')

    jidt_estimator = JidtDiscreteCMI(kwargs)
    itic = time.perf_counter()
    cmi_jidt = jidt_estimator.estimate(var1=S, var2=T, conditional=None)
    itoc = time.perf_counter()
    print(f'JIDT CMI: {cmi_jidt} (took {itoc - itic} seconds)')

    assert np.isclose(cmi_python1, cmi_python2, rtol=1e-100)
    assert np.isclose(cmi_python1, cmi_jidt, atol=0.0001)

if __name__ == '__main__':
    print('Testing Kraskov CMI estimator...')
    for sigma in _Sigmas_3var:
        test_cmi_gaussian(sigma)
    for sigma in _Sigmas_2var:
        test_mi_gaussian(sigma)
    print('All tests passed.')

    print('Testing discrete CMI estimator...')
    test_binary_cmi()
    test_discrete_cmi()
    test_discrete_mi()
    test_discrete_cmi2d()
    test_discrete_mi2d()
    for sigma in _Sigmas_3var:
        test_discrete_gaussian_cmi(sigma)
    for sigma in _Sigmas_2var:
        test_discrete_gaussian_mi(sigma)
    print('All tests passed.')
