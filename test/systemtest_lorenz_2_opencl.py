import os
import time
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

start_time = time.time()
# load simulated data from 2 coupled Lorenz systems 1->2, u = 45 ms
d = np.load(os.path.join(os.path.dirname(__file__),
            'data/lorenz_2_exampledata.npy'))
dat = Data()
dat.set_data(d[:, :, 0:100], 'psr')
analysis_opts = {
        'cmi_estimator': 'OpenCLKraskovCMI',
        'n_perm_max_stat': 200,
        'n_perm_min_stat': 200,
        'n_perm_omnibus': 500,
        'n_perm_max_seq': 500,
        }
lorenz_analysis = MultivariateTE(max_lag_sources=50, min_lag_sources=40,
                                 max_lag_target=30, tau_sources=1,
                                 tau_target=3, options=analysis_opts)
res_1 = lorenz_analysis.analyse_single_target(dat, 0)
res_2 = lorenz_analysis.analyse_single_target(dat, 1)
runtime = time.time() - start_time
print("---- {0} minutes".format(runtime / 60))

path = os.path.dirname(__file__) + 'output/'
np.savez(path + 'test_lorenz', res_1, res_2)
np.save(path + 'test_lorenz_time', runtime)
