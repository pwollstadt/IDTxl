from numpy import empty, fliplr

VERBOSE = False

def embed_timeseries(timeseries, dim, tau):
    """ Embeds a time series using dimension 'dim' and
    step size 'tau'. The time series is assumed to be an
    [1xN] numpy array 'timeseries'.
    Returns an numpy array of size [N-embedding_length x dim]
    that holds embedded data points.
    """

    embedding_length = (dim-1) * tau
    
    timeseries_length = timeseries.shape[0]
    n_embedded_points = timeseries_length - embedding_length
    
    embedding = empty((n_embedded_points, dim))

    count = 0
    
    for point in range(embedding_length+1, timeseries_length+1):
        if VERBOSE: print("point no {0}".format(point))
        embedding[count,:] = timeseries[point-(embedding_length+1):point:tau]
        count += 1
    
    #embedding = fliplr(embedding)
    return embedding
