# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 12:31:15 2016

@author: wibral
"""
import argparse
import pickle
import json
import scipy.io as sio
import numpy as np
import time as tm
from idtxl.estimators_fast_pid_ext_rep import pid


def evaluate_PID(datapath, day, channel1, channel2):
    print('-' * 80)
    filename1 = datapath + 'statesW2-1_d' + day + '_ch' + channel1 + '.mat'
    filename2 = datapath + 'statesW2-1_d' + day + '_ch' + channel2 +'.mat'
    print("Using target and history from: {0}".format(filename1))
    print("Using inout history from: {0}".format(filename2))
    contents_file1 = sio.loadmat(filename1)
    contents_file2 = sio.loadmat(filename2)
    target = np.squeeze(contents_file1['NextValue'].astype('int32'))
    inputs = [np.squeeze(contents_file1['PastVector1'].astype('int32')),
              np.squeeze(contents_file2['PastVector1'].astype('int32'))]

    uniques_1 = []
    uniques_2 = []
    shareds = []
    synergies = []

    tic = tm.clock()
    cfg = {'alph_s1': len(np.unique(inputs[0])),
           'alph_s2': len(np.unique(inputs[1])),
           'alph_t': len(np.unique(target)),
           'max_unsuc_swaps_row_parm': 100,
           'num_reps': 62,
           'max_iters': 300}
    print('unsuccessful_swap_param: {0}'.format(cfg['max_unsuc_swaps_row_parm']))
    estimate = pid(inputs[0], inputs[1], target, cfg)
    estimate['cfg'] = cfg
    # storing deep copies of the results as the object holding
    # them is mutable and reused in this loop
    print("unq_s1: {0}".format(estimate['unq_s1']))
    uniques_1.append(estimate['unq_s1'].copy())
    print("unq_s2: {0}".format(estimate['unq_s2']))
    uniques_2.append(estimate['unq_s2'].copy())
    print("shd_s1_s2: {0}".format(estimate['shd_s1_s2']))
    shareds.append(estimate['shd_s1_s2'].copy())
    print("syn_s1_s2:{0}".format(estimate['syn_s1_s2']))
    synergies.append(estimate['syn_s1_s2'].copy())
##
##    print("maximum synergy: {0} on channels: {1}".format(np.max(synergies),
##              np.where(synergies == np.max(synergies))))
##    print("maximum shared info: {0} on channels: {1}".format(np.max(shareds),
##              np.where(shareds == np.max(shareds))))
##    print("maximum unq_s1 (color x motion): {0} on channels: {1}".format(np.max(uniques_1),
##              np.where(uniques_1 == np.max(uniques_1))))
##    print("maximum unq_s2 (conjunction on surface): {0} on channels: {1}".format(np.max(uniques_2),
##              np.where(uniques_2 == np.max(uniques_2))))
#
    toc = tm.clock()
    print('\nTotal elapsed time: {0} seconds'.format(toc - tic))
#    filename = 'results_' + analysis_band + '_.p'
#    with open(filename, 'wb') as fp:
#        pickle.dump(estimate, fp)
#
#    summary = {}
#    summary['unique_info_colorxmotion'] = uniques_1.astype('float32').tolist()
#    summary['unique_info_conjunction'] = uniques_2.astype('float32').tolist()
#    summary['shared information'] = shareds.astype('float32').tolist()
#    summary['synergistic information'] = synergies.astype('float32').tolist()
#
#    filename = 'summary_' + analysis_band + '_.json'
#    with open(filename, 'w') as fp:
#        json.dump(summary, fp, sort_keys=True, indent=4)
    return estimate



if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('day')
    args = parser.parse_args()
    day = args.day

    datapath = ('/home/wibral/unison/projects/'
                'TransferEntropy/Application_Projects/'
                'Culture_PID/')
    first_channels = ['53']
    second_channels = ['17']

#    for day in days:
    for channel1 in first_channels:
        for channel2 in second_channels:
            if channel1 != channel2:
                estimate = evaluate_PID(datapath, day, channel1, channel2)
                savename = (datapath + 'SINGLECOMB_results_day_' + day + '_uswaps_' +
                            str(estimate['cfg']['max_unsuc_swaps_row_parm']) +
                            '_ch1_' +
                            channel1 + '_ch2_' + channel2 + '_target_' +
                            channel1 + '.p')
                print(savename)
                with open(savename, 'wb') as fp:
                    pickle.dump(estimate, fp)

                results = pickle.load(open(savename, "rb"))
                print(results)
