from jpype import *
import random
import math

jarLocation = "/home/patriciaw/jidt_1_3/infodynamics-dist-1.3/infodynamics.jar"
if not isJVMStarted():
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# Generate some random normalised data.
numObservations = 1000
covariance=0.4
source_correlated = [random.normalvariate(0,1) for r in range(numObservations)]
destArray = [0] + [sum(pair) for pair in zip([covariance*y for y in source_correlated[0:numObservations-1]], \
                  [(1-covariance)*y for y in [random.normalvariate(0,1) for r in range(numObservations-1)]] ) ]
source_uncorrelated = [random.normalvariate(0,1) for r in range(numObservations)]

# initialise the whole Java stuff
teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
teCalc = teCalcClass()
teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
teCalc.initialise(1)                    # Use history length 1 (Schreiber k=1)
teCalc.setProperty("k", "4")            # Use Kraskov parameter K=4 for 4 nearest points (knn search)

# run two anlyses
teCalc.setObservations(JArray(JDouble, 1)(source_correlated), JArray(JDouble, 1)(destArray))
result_correlated = teCalc.computeAverageLocalOfObservations()
teCalc.initialise() # Initialise leaving the parameters the same
teCalc.setObservations(JArray(JDouble, 1)(source_uncorrelated), JArray(JDouble, 1)(destArray))
result_uncorrelated = teCalc.computeAverageLocalOfObservations()

print("TE result %.4f nats; expected to be close to %.4f nats for these correlated Gaussians" % \
    (result_correlated, math.log(1/(1-math.pow(covariance,2)))))
print("TE result %.4f nats; expected to be close to 0 nats for these uncorrelated Gaussians" % result_uncorrelated)
