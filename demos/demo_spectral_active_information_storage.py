# Import classes

from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.data import Data
from idtxl.ais_spectral import ActiveInformationStorageSpectral
import h5py
import sys
from random import gauss

dataf=[]
import numpy as np
import scipy.io as spio


f = h5py.File('/...IDTxl/data/Ar.mat')
dataf = f.get('time_series')
data=dataf[:,:,:].T

lag=4
nscale=5

d1 = Data(data[:,:,0:50], dim_order='psr')

# b) Initialise analysis object and define settings
network_analysis = ActiveInformationStorage()
settings = {'cmi_estimator':  'JidtKraskovCMI',
            'max_lag': lag,
            'tau':1,
            'n_perm_omnibus': 200,
            'n_perm_max_seq':  200,
            'n_perm_min_stat': 200,
            'n_perm_max_stat':  200}

# c) Run analysis time domain
results = network_analysis.analyse_network(settings=settings, data=d1)#,processes=[pp]




# c) Run analysis spectral domain domain
testing=['source'] # change to 'both' to destroy also future of the process 
for test in testing:

    spectral_settings = {'cmi_estimator':'JidtKraskovCMI',
                                                 'n_perm_spec': 200,
            
                                                 'n_scale':nscale,
                                                 'wavelet':'la8',
                                                 'block_size_spec':1 ,
                                                 'perm_range_spec':int(d1.n_samples/1),
    
                                                 'alpha_spec': 0.05,
                                                 'permute_in_time_spec':True,
                                                 'perm_type_spec': 'block',
                                                 'spectral_analysis_type': test,                                             
                                                 'parallel_surr':True,
                                                 'surr_type': 'spectr',
                                                 'n_jobs':6,
                                                 'verb_parallel':50
                                                 }
        
    network_analysis_spectral = ActiveInformationStorageSpectral()    
    resultsSpectral = network_analysis_spectral.analyse_network_spectral(settings=spectral_settings,data=d1,results= results)
      