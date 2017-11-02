import os
import time
import pickle
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

start_time = time.time()
# load simulated data from 2 coupled Lorenz systems 1->2, u = 45 ms
d = np.load(os.path.join(os.path.dirname(__file__),
            'data/lorenz_2_exampledata.npy'))
data = Data()
data.set_data(d[:, :, 0:100], 'psr')
settings = {
        'cmi_estimator':  'JidtKraskovCMI',
        'max_lag_sources': 50,
        'min_lag_sources': 40,
        'max_lag_target': 30,
        'tau_sources': 1,
        'tau_target': 3,
        'n_perm_max_stat': 200,
        'n_perm_min_stat': 200,
        'n_perm_omnibus': 500,
        'n_perm_max_seq': 500,
        }
lorenz_analysis = MultivariateTE()
res_0 = lorenz_analysis.analyse_single_target(settings, data, 0)
res_1 = lorenz_analysis.analyse_single_target(settings, data, 1)
runtime = time.time() - start_time
print("---- {0} minutes".format(runtime / 60))

path = '{0}output/'.format(os.path.dirname(__file__))
pickle.dump(res_0, open('{0}test_lorenz_res_{1}'.format(path, 0), 'wb'))
pickle.dump(res_1, open('{0}test_lorenz_res_{1}'.format(path, 1), 'wb'))
