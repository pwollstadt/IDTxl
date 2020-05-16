# -*- coding: utf-8 -*-
"""Demo for spectral transfer entropy analysis.

Computes spectrally resolved transfer entropy (TE) on example data. To save
running time, load pre-computed TE from disk (then the script sruns in approx.
2 h for a small number of surrogates, ~50). For details, refer to the paper.

This script generates Figures 5 and 6 from the paper.

References:

- Pinzuti, E., Wollstadt, P., Gutknecht, A., Tüscher, O., Wibral, M. (2020).
  Measuring spectrally-resolved information transfer for sender- and receiver-
  specific frequencies.
  https://www.biorxiv.org/content/10.1101/2020.02.08.939744v1

@author: edoardo
"""
# Import classes
import numpy as np
import os
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.multivariate_spectral_te import MultivariateSpectralTE
from idtxl.visualise_graph import plot_spectral_result, plot_SOSO_result
import pickle
import time

start = time.time()

# a) Load data set and optionally load pre-calculated multivariate TE to save
# running time.
load_te_results = True
dataf = np.load(os.path.join(os.path.dirname(__file__),
                'data/Cross_frequency_randomPhase125Hzbivariate3.npy'))
data = Data(dataf[:, :, :], dim_order='psr')

# b) Multivariate TE network analysis
if load_te_results:
    path = os.path.join(os.path.dirname(__file__), 'data/resultTE_0.pkl')
    with open(path, 'rb') as fp:
        result = pickle.load(fp)
else:
    network_analysis = MultivariateTE()
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'max_lag_sources': 3,
        'min_lag_sources': 1,
        'max_lag_target': 6,
        'tau_sources': 1,
        'tau_target': 1,
        'n_perm_max_stat': 200,
        'n_perm_min_stat':  200,
        'n_perm_omnibus':  200,
        'n_perm_max_seq': 200,
        'alpha': 0.05,
        'fdr_correction': False
    }
    result = network_analysis.analyse_network(settings=settings, data=data)

# c) Multivariate spectral TE analysis. Options for spectral surrogate
# creation:
#   - random permutation of wavelet coeff (spectr)
#   - iterative Amplitude Adjustment Fuorier Transform (iaaft)
source = 0
target = 1
spectral_settings = {'cmi_estimator': 'JidtKraskovCMI',
                     'n_perm_spec': 41,
                     'n_scale': 5,
                     'wavelet': 'la8',  # or la16, mother wavelets
                     'alpha_spec': 0.05,
                     'permute_in_time_spec': True,
                     'perm_type_spec': 'block',
                     'block_size_spec': 1,
                     'perm_range_spec': int(data.n_samples/1),
                     'spectral_analysis_type': 'both',
                     'fdr_corrected': False,
                     'parallel_surr': True,
                     'surr_type': 'spectr',  # or 'iaaft'
                     'n_jobs': 6,
                     'verb_parallel': 50}
# Run spectral TE analysis on significant source from Multivariate TE.
spectral_analysis = MultivariateSpectralTE()
result_spectral = spectral_analysis.analyse_network(
    spectral_settings, data, result, sources=[source], targets=[target])

# d) Run SOSO algortihm and plot results.
print('Computing SOSO')
spectral_settings['scale_source'] = [3]
spectral_settings['scale_target'] = 0
spectral_settings['spectral_analysis_type'] = 'SOSO'
spectral_settings['n_perm_spec'] = 41
spectral_settings['delta'] = (
    result_spectral._single_target[1]['source'][3]['deltaO'])
result_spectral.settings['n_perm_spec'] = 41
result_spectral.settings['spectral_analysis_type'] = 'SOSO'

result_soso = spectral_analysis.analyse_network(
    spectral_settings, data, result, sources=[source], targets=[target])

print('Elapsed time: {:.2f} min'.format((time.time() - start) / 60))
plot_spectral_result(result_spectral, freq_rate=125)
plot_SOSO_result(result_soso,
                 target=target,
                 source_scale=spectral_settings['scale_source'][0],
                 target_scale=spectral_settings['scale_target'])
