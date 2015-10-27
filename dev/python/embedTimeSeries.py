#def embedTimeSeries(timeSeries, dim, tau, delay):

import numpy as np
 
#timeSeries = np.random.rand(1000)
timeSeries = np.arange(0,99999)
print timeSeries.shape[0]
dim = 5
tau = 2
delay = 0
   	
#timeSeriesZstand = (timeSeries - np.mean(timeSeries)) / np.std(timeSeries)
timeSeriesZstand = timeSeries
nEmbeddedPoints = (timeSeries.shape[0] - delay - dim*tau) + 2
nEmbeddedPoints = np.int(nEmbeddedPoints)
embeddedPoints = np.empty([nEmbeddedPoints,dim])
firstPredictionPoint = delay
lastPredictionPoint = timeSeries.shape[0]-dim*tau+1
count = 0

for point in range(firstPredictionPoint, lastPredictionPoint+1):
    
    firstPoint = point
    lastPoint  = firstPoint + dim*tau
    embeddedPoints[count,:] = timeSeriesZstand[firstPoint:lastPoint:tau]
    count += 1

#return embeddedPoints
