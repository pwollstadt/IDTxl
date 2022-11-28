

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:47:45 2022

@author: edoardo
"""

# Import classes

from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.data import Data
from idtxl.ais_spectral import ActiveInformationStorageSpectral
import h5py
import sys
from random import gauss
#from idtxl.visualise_graph import plot_spectral_result_ais
dataf=[]
import numpy as np
import scipy.io as spio
# a) Generate test data
# a) Generate test data
#data = Data()
#data.generate_mute_data(n_samples=1000, n_replications=5)

#



idx = sys.argv[1]
idxx = int(idx)

dda=np.load('/scratch1/users/epinzut/IDTxl/demos/data/arfi05.npy')
data=dda

data_str='arfi'+str(idxx)

d1 = Data(data[:,:], dim_order='sr')

lag=40
# b) Initialise analysis object and define settings
network_analysis = ActiveInformationStorage()
settings = {'cmi_estimator':  'JidtKraskovCMI',
            'max_lag': lag,
            'tau':3,
            'n_perm_omnibus':200,
            'n_perm_max_seq':   200,
            'n_perm_min_stat': 200,
            'n_perm_max_stat':  200}

# c) Run analysis
results = network_analysis.analyse_network(settings=settings, data=d1)#,processes=[pp]

# d) Plot list of processes with significant AIS to console
print(results.get_single_process(process=0,fdr=False))
import pickle

nscale=8

testing=['both','source','target']
for test in testing:

    spectral_settings = {'cmi_estimator':'JidtKraskovCMI',
                                                 'n_perm_spec': 41,
            
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
                                                 'n_jobs':12,
                                                 'verb_parallel':50
                                                 }
        
    network_analysis_spectral = ActiveInformationStorageSpectral()    
    resultsSpectral = network_analysis_spectral.analyse_network_spectral(settings=spectral_settings,data=d1,results= results)
    
   
    
    

    import numpy, scipy.io
    import os
    n_scale=nscale
    directory1='/scratch1/users/epinzut/IDTxl/demos/data/arfi/' +data_str+'/'
    if  not os.path.exists(directory1):
                                
        
        os.makedirs(directory1)
    for level in range(0,n_scale):
    
        scipy.io.savemat(directory1+str(level)+'_'+ data_str+'spec_domain'+test +'.mat',{"foo":resultsSpectral._single_process[0]['source'][level]['ais_surrogate']})
        scipy.io.savemat(directory1+ data_str+'spec_domain0'+test +'.mat',{"ais_orig":resultsSpectral._single_process[0]['source'][level]['ais_full_orig']})
        scipy.io.savemat(directory1+ data_str+'spec_domain'+test +'.mat',{"emb":results._single_process[0]['selected_vars']})


