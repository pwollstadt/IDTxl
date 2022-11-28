# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:03:57 2021

@author: edoardo
"""

import numpy as np
from . import stats

from idtxl.data import Data
import copy as cp
from .single_process_analysis import SingleProcessAnalysis
from .estimator import find_estimator
from .results import ResultsSingleProcessAnalysis
from . import idtxl_utils as utils
from .network_analysis import NetworkAnalysis
from .results import DotDict
from . import idtxl_exceptions as ex
#from multiprocessing import Pool
from .results import ResultsSpectralTE
try:
    from . import modwt
except ImportError as err:
    ex.package_missing(err, 'modwt is not available')




class ActiveInformationStorageSpectral(SingleProcessAnalysis):
    
    
    def __init__(self):
        super().__init__()

    def analyse_network_spectral(self, settings, data, results,processes='all'):
        
        # Set defaults for AIS estimation.
        settings.setdefault('verbose', True)
        settings.setdefault('fdr_correction', True)




        # Check provided processes for analysis.
        if processes == 'all':
            processes = [t for t in range(data.n_processes)]
        if (type(processes) is list) and (type(processes[0]) is int):
            pass
        else:
            raise ValueError('Processes were not specified correctly: '
                             '{0}.'.format(processes))

        # Perform AIS estimation for each target individually.
        results_spectral = ResultsSingleProcessAnalysis(
            n_nodes=data.n_processes,
            n_realisations=data.n_realisations(),
            normalised=data.normalise)
        
        # results_spectral = ResultsSpectralTE(n_nodes=data.n_processes,
        #                                  n_realisations=data.n_realisations(),
        #                                  normalised=data.normalise)
        
        
        for t in range(len(processes)):
            if settings['verbose']:
                print('\n####### analysing process {0} of {1}'.format(
                                                processes[t], processes))
                
            #res_single={}    
            res_single = self.analyse_single_process_spectral(
                    settings, data, processes[t],results)
            results_spectral.combine_results(res_single)    
                
          
        
        results_spectral.data_properties.n_realisations = (
            res_single.data_properties.n_realisations)
        
        return results_spectral
    
    
    
    def spectral_surrogates(self,data_slice,scale):

        """Return spectral surrogate data for statistical testing.

            Args:
            data : Data instance
                raw data for analysis
            scale : current_scale under analysis

        Returns:
            numpy array
                surrogate data with dimensions
                (realisations * replication)
        """

        # MODWT  (all replication in one step)

        [w_transform, approx_coeff] = modwt.modwt_C(data_slice, self.settings['wavelet'],self.max_scale)

        w_transform1=np.transpose(w_transform,(1,0,2))


        ww=w_transform1[:,:,:]
        wav_stored = Data(ww, dim_order='psr', normalise=False)
        wav_stored1 = Data(w_transform1, dim_order='psr', normalise=False)
        #wav_stored = Data(w_transform1, dim_order='psr', normalise=False)


       

        # Create surrogates by shuffling coefficients in given scale.

        if self.settings['perm_type_spec'] =='block':
            param={'perm_in_time': self.settings['permute_in_time_spec'],
                        'perm_type': self.settings['perm_type_spec'],

                        'block_size': self.settings['block_size_spec'],
                        'perm_range':self.settings['perm_range_spec']

               }
        elif self.settings['perm_type_spec'] =='circular':

                         param={'wavelet': self.settings['wavelet'],
                               'perm_in_time': self.settings['permute_in_time_spec'],
                               'perm_type': self.settings['perm_type_spec'],
                               'max_scale': self.max_scale,
                               'max_shift': self.settings['max_shift_spec']

                                }

        else:

            param={'perm_in_time': self.settings['permute_in_time_spec'],
                        'perm_type': self.settings['perm_type_spec']

                                      }







        spectral_surr = stats._generate_spectral_surrogates(
                                                      wav_stored,
                                                      scale,
                                                      1,
                                                      perm_settings=param)

       


        # stack together the coeff
        #f_s=w_transform1[scale,:,:]
        #s_s=w_transform1[scale,:,:]
        #aa=spectral_surr[:, :, 0]
        #aka=np.vstack((f_s,aa,s_s))

        

        wav_stored._data[scale, :, :] = spectral_surr[:, :, 0]
 
      

        #wav_stored1._data[scale, :, :] = aka
        merged_coeff=np.transpose(wav_stored._data,(1,0,2))

 

        # IMODWT  (all replication in one step)
        rec_surrogate=modwt.imodwt_c(merged_coeff,approx_coeff,self.settings['wavelet'],self.max_scale)
     

        # samples x replications

        return  rec_surrogate    
    
    
    
    
    
    
    
    def analyse_single_process_spectral(self, settings, data, process,results):
        """Estimate active information storage for a single process.
        
        # Check input and clean up object if it was used before."""
        self._initialise(settings,results, data, process)
    
        if self.settings['spectral_analysis_type']== 'both':
            results_spec = {}
            results_spec['source'] = {}
            for current_scale in range(0,settings['n_scale']):
           
               print('Testing Scale n: {0}'.format(current_scale))    
               ais_scale=self.ais_single_scales( settings, data,current_scale)
               results_spec['source'][current_scale] = ais_scale
        elif self.settings['spectral_analysis_type']== 'source':    
            results_spec = {}
            results_spec['source'] = {}
            for current_scale in range(0,settings['n_scale']):
           
               print('Testing Scale n: {0}'.format(current_scale))    
               ais_scale=self.ais_single_scales_source( settings, data,current_scale)
               results_spec['source'][current_scale] = ais_scale

        elif self.settings['spectral_analysis_type']== 'target':    
            results_spec = {}
            results_spec['source'] = {}
            for current_scale in range(0,settings['n_scale']):
           
               print('Testing Scale n: {0}'.format(current_scale))    
               ais_scale=self.ais_single_scales_target( settings, data,current_scale)
               results_spec['source'][current_scale] = ais_scale



        # Add analyis info.
        results = ResultsSpectralTE(
            n_nodes=data.n_processes,
            n_realisations=data.n_realisations(self.current_value),
            normalised=data.normalise)
        results._add_single_result(
            settings=self.settings,
            target=self.process,
            results=results_spec)    
            
            
        return results
    
    
    
    
    
    def  ais_single_scales(self, settings, data,current_scale):
          
    
          
               
        i_1 = 0
        i_2 = data.n_realisations(self.current_value)
        temp_source_realisation_perm = np.empty(
              (data.n_realisations(self.current_value) * self.settings['n_perm_spec'],
               1)).astype(data.data_type)
        
        i_3 = 0
        i_4 = data.n_realisations(self.current_value)
       
        temp_source_realisationFull_perm = np.empty(
              (data.n_realisations( self.current_value) 
                     * self.settings['n_perm_spec'],
                len(self.source_vars))).astype(data.data_type)
        
     

       
        data_slice = data._get_data_slice(self.process)[0]
        _surr_source = np.zeros(self.settings['n_perm_spec'])
        for perm in range(0, self.settings['n_perm_spec']):   
    
            rec_surrogate=self.spectral_surrogates(data_slice,current_scale)
        
            d_temp = cp.copy(data.data)
            d_temp[self.process, :, :] =  rec_surrogate
            data_surr = Data(d_temp, 'psr', normalise=False)
            cur_value_source_realisations = data_surr.get_realisations(
                                                     self.current_value,
                                                     [self.current_value])[0]


            cur_set_realisations =data_surr.get_realisations(
                               self.current_value, self.selected_vars_full)[0]
        
            # print(np.shape( temp_source_realisationFull_perm))
            # temp_source_realisation_perm[i_1:i_2,:] = cur_value_source_realisations
            # temp_source_realisationFull_perm[i_3:i_4,:] = cur_set_realisations

            # i_1 = i_2
            # i_2 += data.n_realisations(self.current_value)

            # i_3 = i_4
            # i_4 += data.n_realisations(self.current_value)
            _surr_source[perm]= self._cmi_estimator.estimate(
                                            var1=cur_value_source_realisations,
                                            var2=cur_set_realisations,
                                            conditional=None) 
    
        # surr_dist = self._cmi_estimator.estimate_parallel(
        #                     n_chunks=self.settings['n_perm_spec'],
        #                     re_use=['var2', 'conditional'],
        #                     var1= temp_source_realisation_perm,
        #                     var2=temp_source_realisationFull_perm,
        #                     conditional=None)
        
        
        surr_dist = _surr_source
    
       
    
        [significance, pvalue] =stats._find_pvalue(statistic= self.orig_mi ,
                                               distribution=surr_dist,
                                               alpha=0.05,
                                               tail='one')
    
    
        mean_spectral_ais=np.median(surr_dist)
        result_scale = {
            'ais_surrogate': surr_dist,
            'ais_spectral': mean_spectral_ais,
            'spec_pval': pvalue,
            'spec_sign': significance,
            'ais_full_orig':self.orig_mi
          }
        return  result_scale
    
    def  ais_single_scales_source(self, settings, data,current_scale):
          
    
          
               
        i_1 = 0
        i_2 = data.n_realisations(self.current_value)
        temp_source_realisation_perm = np.empty(
              (data.n_realisations(self.current_value) * self.settings['n_perm_spec'],
               1)).astype(data.data_type)
        
        i_3 = 0
        i_4 = data.n_realisations(self.current_value)
       
        temp_source_realisationFull_perm = np.empty(
              (data.n_realisations( self.current_value) 
                     * self.settings['n_perm_spec'],
                len(self.source_vars))).astype(data.data_type)
        
     

       
        data_slice = data._get_data_slice(self.process)[0]
        _surr_source = np.zeros(self.settings['n_perm_spec'])
        for perm in range(0, self.settings['n_perm_spec']):   
    
            rec_surrogate=self.spectral_surrogates(data_slice,current_scale)
        
            d_temp = cp.copy(data.data)
            d_temp[self.process, :, :] =  rec_surrogate
            data_surr = Data(d_temp, 'psr', normalise=False)
            cur_value_source_realisations = data.get_realisations(
                                                     self.current_value,
                                                     [self.current_value])[0]


            cur_set_realisations =data_surr.get_realisations(
                               self.current_value, self.selected_vars_full)[0]
        
            # print(np.shape( temp_source_realisationFull_perm))
            # temp_source_realisation_perm[i_1:i_2,:] = cur_value_source_realisations
            # temp_source_realisationFull_perm[i_3:i_4,:] = cur_set_realisations

            # i_1 = i_2
            # i_2 += data.n_realisations(self.current_value)

            # i_3 = i_4
            # i_4 += data.n_realisations(self.current_value)
            _surr_source[perm]= self._cmi_estimator.estimate(
                                            var1=cur_value_source_realisations,
                                            var2=cur_set_realisations,
                                            conditional=None) 
    
        # surr_dist = self._cmi_estimator.estimate_parallel(
        #                     n_chunks=self.settings['n_perm_spec'],
        #                     re_use=['var2', 'conditional'],
        #                     var1= temp_source_realisation_perm,
        #                     var2=temp_source_realisationFull_perm,
        #                     conditional=None)
        
        
        surr_dist = _surr_source
    
       
    
        [significance, pvalue] =stats._find_pvalue(statistic= self.orig_mi ,
                                               distribution=surr_dist,
                                               alpha=0.05,
                                               tail='one')
    
    
        mean_spectral_ais=np.median(surr_dist)
        result_scale = {
            'ais_surrogate': surr_dist,
            'ais_spectral': mean_spectral_ais,
            'spec_pval': pvalue,
            'spec_sign': significance,
            'ais_full_orig':self.orig_mi
          }
        return  result_scale        
    

    def  ais_single_scales_target(self, settings, data,current_scale):
        print('shuffling X+')  
    
          
               
        i_1 = 0
        i_2 = data.n_realisations(self.current_value)
        temp_source_realisation_perm = np.empty(
              (data.n_realisations(self.current_value) * self.settings['n_perm_spec'],
               1)).astype(data.data_type)
        
        i_3 = 0
        i_4 = data.n_realisations(self.current_value)
       
        temp_source_realisationFull_perm = np.empty(
              (data.n_realisations( self.current_value) 
                     * self.settings['n_perm_spec'],
                len(self.source_vars))).astype(data.data_type)
        
     

       
        data_slice = data._get_data_slice(self.process)[0]
        _surr_source = np.zeros(self.settings['n_perm_spec'])
        for perm in range(0, self.settings['n_perm_spec']):   
    
            rec_surrogate=self.spectral_surrogates(data_slice,current_scale)
        
            d_temp = cp.copy(data.data)
            d_temp[self.process, :, :] =  rec_surrogate
            data_surr = Data(d_temp, 'psr', normalise=False)
            cur_value_source_realisations =data_surr.get_realisations(
                                                     self.current_value,
                                                     [self.current_value])[0]


            cur_set_realisations =data.get_realisations(
                               self.current_value, self.selected_vars_full)[0]
        
            # print(np.shape( temp_source_realisationFull_perm))
            # temp_source_realisation_perm[i_1:i_2,:] = cur_value_source_realisations
            # temp_source_realisationFull_perm[i_3:i_4,:] = cur_set_realisations

            # i_1 = i_2
            # i_2 += data.n_realisations(self.current_value)

            # i_3 = i_4
            # i_4 += data.n_realisations(self.current_value)
            _surr_source[perm]= self._cmi_estimator.estimate(
                                            var1=cur_value_source_realisations,
                                            var2=cur_set_realisations,
                                            conditional=None) 
    
        # surr_dist = self._cmi_estimator.estimate_parallel(
        #                     n_chunks=self.settings['n_perm_spec'],
        #                     re_use=['var2', 'conditional'],
        #                     var1= temp_source_realisation_perm,
        #                     var2=temp_source_realisationFull_perm,
        #                     conditional=None)
        
        
        surr_dist = _surr_source
    
       
    
        [significance, pvalue] =stats._find_pvalue(statistic= self.orig_mi ,
                                               distribution=surr_dist,
                                               alpha=0.05,
                                               tail='one')
    
    
        mean_spectral_ais=np.median(surr_dist)
        result_scale = {
            'ais_surrogate': surr_dist,
            'ais_spectral': mean_spectral_ais,
            'spec_pval': pvalue,
            'spec_sign': significance,
            'ais_full_orig':self.orig_mi
          }
        return  result_scale        
        

    
    def _initialise(self, settings, results, data,process):
        
        self.settings = settings.copy()
         # Set CMI estimator.
       # self._set_cmi_estimator()
        try:
            EstimatorClass = find_estimator(settings['cmi_estimator'])
        except KeyError:
            raise KeyError('Estimator was not specified!')
        # Don't add results with conflicting settings
        #if utils.conflicting_entries(self.settings, settings):
        if utils.conflicting_entries(settings, settings):
            raise RuntimeError(
                'Conflicting entries in spectral TE and network inference settings.')
       
        self.process = process
        self.current_value =results._single_process[self.process]['current_value']
        self.source_vars=results._single_process[self.process]['selected_vars']
        
        self.selected_vars_full =self._lag_to_idx(
         lag_list=(self.source_vars ), current_value_sample=self.current_value[1])
        self._cmi_estimator = EstimatorClass(settings)
        
        self.orig_mi = results._single_process[self.process]['ais']
       
        self.max_scale = int(np.log2(data.n_samples))
     #   self.settings['wavelet']='la8'
        self.settings.setdefault('wavelet', 'la8')
        #self.settings.setdefault('block_size', 1)
        #self.settings.setdefault('perm_range', 10 )
        self.settings.setdefault('spectral_analysis_type', 'source')
