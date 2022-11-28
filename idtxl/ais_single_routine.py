# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 17:53:08 2021

@author: edoardo
"""


import numpy as np
import sys
import random
import sys
import os
import numpy as np
from .single_process_analysis import SingleProcessAnalysis
import time
#sys.path.append('/scratch/user/epinzut/idtx_env/IDTxl/')
# Import classes
from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.data import Data
from idtxl.ais_spectral import ActiveInformationStorageSpectral
import h5py
#from idtxl.visualise_graph import plot_spectral_result_ais
import scipy.io as spio
import time
import sys
from joblib import Parallel, delayed
import idtxl.idtxl_utils as utils
import pickle
import scipy.io as spio
from idtxl.estimator import find_estimator
from idtxl.stats  import _get_surrogates,_find_pvalue
from idtxl.network_analysis  import NetworkAnalysis
import copy as cp
class compute_single_ais_ferret(SingleProcessAnalysis):
    def __init__(self):
            super().__init__()

    def _initialise(self, settings, results, data,process):
        
        self.settings = settings.copy()
         # Set CMI estimator.
       # self._set_cmi_estimator()
        try:
            EstimatorClass = find_estimator(settings['cmi_estimator'])
        except KeyError:
            raise KeyError('Estimator was not specified!')      
                     
                    
        self._cmi_estimator = EstimatorClass(settings)        
        self.current_value =results._single_process[process]['current_value']       
                
        
      
    
      
        
    
    def compute_mi_orig_single_trials(self,results_union_source,settings,results,data,process):
     
        n_perm=500
      
        self._initialise(settings,results,data,process)
        
        
        
     
        selected_vars_full =self._lag_to_idx(
                lag_list=(list(results_union_source) ), current_value_sample=self.current_value[1])
        
        
        cur_value_source_realisations = data.get_realisations(
                                         self.current_value,
                                         [self.current_value])[0]
        cur_set_realisations = data.get_realisations(
                   self.current_value, selected_vars_full)[0]
        
        surr_realisations =_get_surrogates(data,
                        self.current_value,
                        [self.current_value],
                        n_perm,
                        settings)
       
        
        
        surr_dist =self._cmi_estimator.estimate_parallel(
                            n_chunks=n_perm,
                            re_use=['var2', 'conditional'],
                            var1=surr_realisations,
                            var2=cur_set_realisations,
                            conditional=None)
        orig_mi =self._cmi_estimator.estimate(
                            var1= cur_value_source_realisations,
                            var2=cur_set_realisations,
                            conditional=None
                            )
        [significance, p_value] =_find_pvalue(statistic=orig_mi,
                                           distribution=surr_dist,
                                           alpha=0.05,
                                           tail='one_bigger')  
      
    
        
        results_ais={
                    'current_value':self.current_value,
                    'selected_vars': self._idx_to_lag(selected_vars_full),
                    'ais':orig_mi,
                    'ais_pval': p_value,
                    'ais_sign': significance
                }
        return results_ais   