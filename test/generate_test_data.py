"""Generate test data for IDTxl unit tests."""
import pickle
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data


# Generate example data: the following was ran once to generate example data,
# which is now in the data sub-folder of the test-folder.
data = Data()
data.generate_mute_data(100, 5)
# analysis settings
settings = {
    'cmi_estimator': 'JidtKraskovCMI',
    'n_perm_max_stat': 50,
    'n_perm_min_stat': 50,
    'n_perm_omnibus': 200,
    'n_perm_max_seq': 50,
    'max_lag_target': 5,
    'max_lag_sources': 5,
    'min_lag_sources': 1,
    'permute_in_time': True
    }
# network inference for individual data sets
# path = os.path.join(os.path.dirname(__file__) + '/data/')
path = 'data/'
nw_0 = MultivariateTE()
res_0 = nw_0.analyse_network(settings, data, targets=[0, 1], sources='all')
pickle.dump(res_0, open(path + 'mute_results_0.p', 'wb'))
res_1 = nw_0.analyse_network(settings, data,  targets=[1, 2], sources='all')
pickle.dump(res_1, open(path + 'mute_results_1.p', 'wb'))
res_2 = nw_0.analyse_network(settings, data,  targets=[0, 2], sources='all')
pickle.dump(res_2, open(path + 'mute_results_2.p', 'wb'))
res_3 = nw_0.analyse_network(settings, data,  targets=[0, 1, 2], sources='all')
pickle.dump(res_3, open(path + 'mute_results_3.p', 'wb'))
res_4 = nw_0.analyse_network(settings, data,  targets=[1, 2], sources='all')
pickle.dump(res_4, open(path + 'mute_results_4.p', 'wb'))
