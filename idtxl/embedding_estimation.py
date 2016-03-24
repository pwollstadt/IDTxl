# -*- coding: utf-8 -*-
"""
module estimtaes embedding parameters for use in transfer entropy
ans active information storage calculations.

@author: Michael Wibral
"""

import numpy as np
import neighbour_search_opencl as ns

# TODO check consistency with our naming conventions
def estimate_auto_embedding_parameters(ft2idtxl_data, channel_index, algorithm, est_parameters):
    """
    wrapper function for all the various ways to estimate (self-)embedding
    ft2txl_data= input data as a dictionary created by ft2xl
    algorithm = the type of estimation procedure
    est_parameters = dictionary of parameters (history dimension and lags,
    Theiler correction etc.) to be investigated in the embedding

    output is in the form of an embedding delay 'comb' vector that has the
    relative indices of the past embedding points
    """

    # get the data out of the ft2txl_data format
    data_matrix = ft2idtxl_data['np_timeseries']
    timeseries = data_matrix[:, channel_index, :]
    # TODO zscore timeseries

    if algorithm == 'ragwitz_ocl':
        delay_comb = _estimate_auto_embedding_ragwitz_ocl(timeseries, est_parameters)
    elif algorithm == 'nonuniform':
        delay_comb = _estimate_auto_embedding_nonuniform(timeseries, est_parameters)
    else:
        print("error requested embedding estimation algorithm" + algorithm + "not found")

    return delay_comb

# @profile
def _estimate_auto_embedding_ragwitz_ocl(timeseries, est_parameters):
    """takes a single (!) timeseries as a one-dimensional numpy array
    end returns a uniform embedding comb vector using the Ragwitz method
    """
    # set up the parameters to iterate over
    errors = np.zeros((len(est_parameters['tau']), len(est_parameters['dim'])))
    tau_count = -1
    for tau in est_parameters['tau']:
        tau_count += 1
        print('testing embedding for tau: ', tau)
        dim_count = -1
        for dim in est_parameters['dim']:
            dim_count += 1
            print('testing embedding for dim: ', dim)
            num_trials = timeseries.shape[1]
            tmp_errors = np.zeros((num_trials,))
            # note below: trial-based computations of neighbors and their future fates are suboptimal for GPU
            for trial in range(0, num_trials):
                # print('trial:', trial)
                # print('working on trial : ', trial)
                trial_timeseries = timeseries[:, trial]
                # calculate how many timepoints will remain, remember last
                # points needs to be predicted -> don't include it
                # '-1' for the future of the last embedded point  (n-dims also lead to n-1 taus)
                num_remain_points = trial_timeseries.shape[0] - (dim - 1) * tau - 1  # TODO switch to dimmax and taumax so that all tau and dim are estiamted on the same points
#                 print('total points: ', trial_timeseries.shape[0])
#                 print('index of last point is: ', trial_timeseries.shape[0] - 1)
#                 print('num_remain_points:', num_remain_points)

                # preallocate with zeros - there might be a better way ?
                embedded_data = np.zeros((dim, num_remain_points))
                # create an array of embedded points
                # Note: first point is zero, first useful one is (dim-1)*tau,
                # -- this point is the the ((dim-1)*tau+1)th point (!).
                # last useful point is num_remain_points-1 + dim * tau
                # last point would be at remain_points + dim * tau + 1, but we need it for prediction
                for sample_index in range((dim - 1) * tau , num_remain_points + (dim - 1) * tau):
                    # print('sample index', sample_index)
                    # TODO import slicing based embedding from pwollstadt
                    for d in range(0, dim):
                        embedded_data[d, sample_index - (dim - 1) * tau] = trial_timeseries[sample_index - d * tau]

                # TODO pool all embedded data over trials
                # and find neighbors and indices in one go on the GPU
                # print('shape of embedded data: ', embedded_data.shape)
                # find the -nearest neighbors (opencl)
                # TODO fix the neighbour search
                pointset = embedded_data.astype('float32')
                queryset = embedded_data.astype('float32')
                # next line calls opencl, after its execution indexes and distances are filled with the values computed in opencl
                (indices, distances) = ns.knn_search(pointset, queryset, est_parameters['num_knn'], est_parameters['theiler'], 1, dim, num_remain_points , est_parameters['gpuid'])

                # find future fate of neighbors (augment the indexes by 1 each)
                # indexes is a num_knn by num_remain_points size array, we can
                # transport everything into the future by just adding +1 to
                # every single entry :-)
                future_neighbor_indices = indices + 1
                # print('min neighbor index', np.amin(indexes))
                # print('max neighbor index:', np.amax(indexes))
                # print('max future index:', np.amax(future_neighbor_indices))

                # average the values (num_knn different values) of the future
                # neighbors (remember to add back the offset that was removed
                # at the beginning of the embedding)
                # get all the values
                future_neighbor_values = np.zeros((future_neighbor_indices.shape))
                for k in range(0, future_neighbor_indices.shape[0]):
                    for t in range(0, future_neighbor_indices.shape[1]):
                        # print('t:', t)
                        # print('num_remain_points:', num_remain_points)
                        # print('current index: ', future_neighbor_indices[n, t])
                        future_neighbor_values[k, t] = trial_timeseries[future_neighbor_indices[k, t] + (dim - 1) * tau]

                mean_future_neighbor_values = np.mean(future_neighbor_values, axis=0)
                # print(mean_future_neighbor_values)

                future_orig_points = trial_timeseries[(dim - 1) * tau + 1 : num_remain_points + (dim - 1) * tau + 1]

                diffs = future_orig_points - mean_future_neighbor_values
                MSE = np.mean(diffs * diffs)  # /std(timeSeries);
                # print('MSE for trial: ', MSE)
                tmp_errors[trial] = MSE

            errors[tau_count, dim_count] = np.mean(tmp_errors)



    print(errors)

def _estimate_auto_embedding_nonuniform(timeseries, est_parameters):
    pass
