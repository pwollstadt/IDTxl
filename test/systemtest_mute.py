import pickle
import time
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

start_time = time.time()
data = Data()  # initialise an empty data object
data.generate_mute_data(n_samples=1000, n_replications=10)
settings = {
        'cmi_estimator':  'JidtKraskovCMI',
        'n_perm_max_stat': 500,
        'n_perm_min_stat': 200,
        'n_perm_omnibus': 500,
        'n_perm_max_seq': 500,
        'max_lag_sources': 5,
        'min_lag_sources': 1
        }

network_analysis = MultivariateTE()
results = network_analysis.analyse_network(settings, data)
runtime = time.time() - start_time
print("---- {0} minutes".format(runtime / 60))

# Save results
# pickle.dump(results, open('test_mute_results.p', 'wb'))
