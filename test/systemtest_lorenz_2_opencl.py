"""Run test on two bi-directionally coupled Lorenz systems using GPU estimators

Simulated delays between the systems are
\delta_{0->1} = 45
\delta_{1->0} = 75.
"""
import os
import time
import pickle
from pathlib import Path
import numpy as np

from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

start_time = time.time()
# load simulated data from 2 coupled Lorenz systems 1->2, u = 45 ms
d = np.load(Path(os.path.dirname(__file__)).joinpath("data/lorenz_2_exampledata.npy"))
data = Data(d[:, :, 0:100], "psr")
settings = {
    "cmi_estimator": "OpenCLKraskovCMI",
    "max_lag_target": 30,
    "tau_sources": 1,
    "tau_target": 3,
    "n_perm_max_stat": 200,
    "n_perm_min_stat": 200,
    "n_perm_omnibus": 500,
    "n_perm_max_seq": 500,
}
lorenz_analysis = MultivariateTE()
settings["min_lag_sources"] = 40
settings["max_lag_sources"] = 50
res_0 = lorenz_analysis.analyse_single_target(settings, data, target=0)
settings["min_lag_sources"] = 70
settings["max_lag_sources"] = 80
res_1 = lorenz_analysis.analyse_single_target(settings, data, target=1)
runtime = time.time() - start_time
print("---- {0:.2f} minutes".format(runtime / 60))

path = Path(os.path.dirname(__file__)).joinpath("data")
with open(path.joinpath("test_lorenz_opencl_res_0"), "wb") as output_file:
    pickle.dump(res_0, output_file)
with open(path.joinpath("test_lorenz_opencl_res_1"), "wb") as output_file:
    pickle.dump(res_1, output_file)
