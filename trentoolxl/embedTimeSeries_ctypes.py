import ctypes 
import numpy as np

c_embedding = ctypes.CDLL("libembedding.so.1.0.1")

timeSeries = np.arange(0,99999)
print timeSeries.shape[0]
dim = 5
tau = 2
delay = 0
nEmbeddedPoints = (timeSeries.shape[0] - delay - dim*tau) + 2

embeddedPoints = np.empty(nEmbeddedPoints*dim)

# http://wiki.scipy.org/Cookbook/Ctypes#head-4ee0c35d45f89ef959a7d77b94c1c973101a562f
ts_p = timeSeries.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
embeddedPoints_p = embeddedPoints.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

embedding_c.embed(embeddedPoints_p, ts_p, ts_len, dim, tau, delay) 


print "Nachher: ", [i for i in a]
