import numpy as np
from errors import *
import neighbour_search_opencl as ns

VERBOSE = False

def uniform_embedding(data, candidate_dimension, candidate_tau, cfg):
    """Find uniform embedding using the Ragwitz criterion.
    
    Find embedding parameters dim and tau to uniformly embed the time series 
    given in data. The time series is assumed to be an [1xN] numpy array.
    Returns an numpy array of size [N-embedding_length x dim] that holds 
    embedded data points.
    """
    
    try:
        source_target_delay = cfg["source_target_delay"]
    except IdtxlParamError as e:
        print("Parameter missing: " + e.missing_parameter)
    try:
        pred_points = cfg["pred_points"]
    except IdtxlParamError as e:
        print("Parameter missing: " + e.missing_parameter)
    try:
        knn = cfg["knn"]
    except IdtxlParamError as e:
        print("Parameter missing: " + e.missing_parameter)
    try:
        theiler_t = cfg["theiler_t"]
    except IdtxlParamError as e:
        print("Parameter missing: " + e.missing_parameter)
    
    mse = np.empty([candidate_dimension.shape[0], candidate_tau.shape[0]])
    count_dim = 0
    count_tau = 0
    for dim in candidate_dimension:
        for tau in candidate_tau:
            pointset = embed_timeseries(data, dim, tau, pred_points)
            mse[count_dim, count_tau]= prediction_error(data, pointset, knn, theiler_t)
            count_dim += 1
            count_tau += 1


def prediction_error(data, pointset, knn, theiler_t):
    """ Calculate the prediction error given the current embedding of pointset.
    
    Return the mean squared error between the point predicted by the current 
    embedding and the actual next point in the time series.
    """
    
    queryset = pointset
    [knn_neighbour, distance] = ns.knn_search(pointset, queryset, knn, theiler_t)
    predicted_point = np.empty(pointset.shape[0])
    actual_point = np.empty(pointset.shape[0])
    for point in pointset:
        predicted_point = data(knn_neighbour[point,:]) + 1 + embedding_length / knn
        actual_point = data[point] + 1 + embedding_length
    
    raw_error = predicted_point - actual_point
    return (sum(np.power(raw_error,2))/pointset.size[0])/std(timeSeries)


def embed_timeseries(timeseries, dim, tau, pred_points=None):
    """Do a uniform embedding of the time series using a fixed dim and tau.
    """
    
    if pred_points is None:
        pred_points = timeseries.shape[0]+1
        
    embedding_length = (dim-1) * tau
    timeseries_length = timeseries.shape[0]
    n_embedded_points = timeseries_length - embedding_length
    pointset = np.empty((n_embedded_points, dim))
    count = 0
    for point in range(embedding_length+1, pred_points):
        if VERBOSE: print("point no {0}".format(point))
        pointset[count,:] = timeseries[point-(embedding_length+1):point:tau]
        count += 1
    
    #embedding = np.fliplr(embedding)
    return pointset

if __name__ == "__main__":
    """ Do a quick check if eveything is called correctly."""
    
    timeseries = np.arange(21)
    dim = 3
    tau = 1
    print("testing embedding for dim={0}, tau={1}".format(dim,tau))
    embedding_1 = embed_timeseries(timeseries, dim, tau)
    print(embedding_1)

    timeseries = np.arange(21)
    dim = 3
    tau = 2
    print("testing embedding for dim={0}, tau={1}".format(dim,tau))
    embedding_2 = embed_timeseries(timeseries, dim, tau)
    print(embedding_2)

    embedding_1_correct = np.array([
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9, 10],
        [9, 10, 11],
        [10, 11, 12],
        [11, 12, 13],
        [12, 13, 14],
        [13, 14, 15],
        [14, 15, 16],
        [15, 16, 17],
        [16, 17, 18],
        [17, 18, 19],
        [18, 19, 20]])

    embedding_2_correct = np.array([
        [0, 2, 4],
        [1, 3, 5],
        [2, 4, 6],
        [3, 5, 7],
        [4, 6, 8],
        [5, 7, 9],
        [6, 8, 10],
        [7, 9, 11],
        [8, 10, 12],
        [9, 11, 13],
        [10, 12, 14],
        [11, 13, 15],
        [12, 14, 16],
        [13, 15, 17],
        [14, 16, 18],
        [15, 17, 19],
        [16, 18, 20]])

    print("Embedding 1 correct: {0}".format((embedding_1_correct==embedding_1).all()))
    print("Embedding 2 correct: {0}".format((embedding_2_correct==embedding_2).all()))
    
    dim_cand = np.arange(1,11)
    tau_cand = np.arange(1,5)
    cfg = {
        "source_target_delay": 3, 
        "pred_points": 5,
        "theiler_t": 1,
        "knn": 4
        }
    uniform_embedding(timeseries, dim_cand, tau_cand, cfg)
