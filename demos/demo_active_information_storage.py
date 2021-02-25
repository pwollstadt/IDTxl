# Import classes
from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.data import Data

# a) Generate test data
data = Data()
data.generate_mute_data(n_samples=1000, n_replications=5)

# b) Initialise analysis object and define settings
network_analysis = ActiveInformationStorage()
settings = {'cmi_estimator':  'JidtGaussianCMI',
            'max_lag': 5}

# c) Run analysis
results = network_analysis.analyse_network(settings=settings, data=data)

# d) Plot list of processes with significant AIS to console
print(results.get_significant_processes(fdr=False))
