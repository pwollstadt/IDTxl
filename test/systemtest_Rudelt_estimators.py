import numpy as np
import os
from idtxl.data_spiketime import Data_spiketime
from sys import stderr
import idtxl.hde_utils as utl
from idtxl.estimators_Rudelt import RudeltNSBEstimatorSymbolsMI, RudeltPluginEstimatorSymbolsMI, RudeltBBCEstimator, RudeltShufflingEstimator

FAST_EMBEDDING_AVAILABLE = True
try:
    import idtxl.hde_fast_embedding as fast_emb
except:
    FAST_EMBEDDING_AVAILABLE = False
    print("""
    Error importing Cython fast embedding module. Continuing with slow Python implementation.\n
    This may take a long time.\n
    """, file=stderr, flush=True)

settings = {'debug': False,
            'embedding_past_range_set': [0.005, 0.00998, 0.01991, 0.03972, 0.07924, 0.15811, 0.31548, 0.62946, 1.25594,
                                       2.50594, 5.0],
            'number_of_bootstraps_R_max': 10,
            'number_of_bootstraps_R_tot': 10,
            'auto_MI_bin_size_set': [0.01, 0.025, 0.05, 0.25],
            'embedding_step_size': 0.005}
            #'parallel': True,
           #'numberCPUs': 8}


#spiketimes = np.loadtxt(os.path.join(os.path.dirname(__file__),
#            'data/spike_times.dat'), dtype=float)


#nr_processes = 10
#nr_replications = 3

#spiketimedata = np.empty(shape=(nr_processes, nr_replications), dtype=np.ndarray)

#for i in range(nr_processes):
#    for r in range(nr_replications):
#        if i == 0 and r == 0:
#            spiketimedata[i, r] = spiketimes
#        else:
#            ran = np.random.rand(len(spiketimes)) * 100
#            new = spiketimes + ran
#            sampl = int(np.random.uniform(low=0.6*len(spiketimes), high=0.9*len(spiketimes), size=(1,)))
#            spiketimedata[i, r] = new[0:sampl]

#data = Data_spiketime(spiketimedata, dim_order='pr')

#
#spiketimedata2 = np.empty(shape=(1, 1), dtype=np.ndarray)
#spiketimedata2[0, 0] = spiketimes
#data = Data_spiketime(spiketimedata2, dim_order='pr')

data = Data_spiketime()              # initialise empty data object
data.load_Rudelt_data()


process_list = [0]
symbol_array, past_symbol_array, current_symbol_array, symbol_array_length, spiketimes = \
    data.get_realisations_symbols(process_list, 0.005, 1, 0.0, 0.005, output_spike_times=True)
number_of_bins_d = np.array(list(np.binary_repr(max(symbol_array[0, 0])))).astype(np.int8)

estnsb = RudeltNSBEstimatorSymbolsMI()
I_nsb, R_nsb = estnsb.estimate(symbol_array[0, 0], past_symbol_array[0, 0], current_symbol_array[0, 0])

estplugin = RudeltPluginEstimatorSymbolsMI()
I_plugin, R_plugin = estplugin.estimate(symbol_array[0, 0], past_symbol_array[0, 0], current_symbol_array[0, 0])

estbbc = RudeltBBCEstimator()
I_bbc, history_dependence, bbc_term = \
    estbbc.estimate(symbol_array[0, 0], past_symbol_array[0, 0], current_symbol_array[0, 0])

estsh = RudeltShufflingEstimator()
I_sh, R_sh = estsh.estimate(symbol_array[0, 0])
