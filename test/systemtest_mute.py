import time
import numpy as np
from idtxl.multivariate_te import Multivariate_te 
from idtxl.data import Data

start_time = time.time()
dat = Data()  # initialise an empty data object
dat.generate_mute_data(n_samples=1000, n_replications=10)
max_lag = 5
analysis_opts = {
        'cmi_calc_name': 'jidt_kraskov',
        'n_perm_max_stat': 200,
        'n_perm_min_stat': 200,
        'n_perm_omnibus': 500,
        'n_perm_max_seq': 500,
        }

network_analysis = Multivariate_te(max_lag, analysis_opts)
res = network_analysis.analyse_network(dat)
runtime = time.time() - start_time
print("---- {0} minutes".format(runtime / 60))

np.save('/home/patriciaw/Dropbox/BIC/#idtxl/test/test', res)
np.save('/home/patriciaw/Dropbox/BIC/#idtxl/test/test_time', runtime)
