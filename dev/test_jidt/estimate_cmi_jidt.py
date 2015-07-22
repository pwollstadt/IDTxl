from jpype import *
import random
import math

jarLocation = "/home/patriciaw/jidt_1_3/infodynamics-dist-1.3/infodynamics.jar"
if not isJVMStarted():
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# generate data, taken from the TE example
numObservations = 1000
covariance=0.4
source = [random.normalvariate(0,1) for r in range(numObservations)]
target = [0] + [sum(pair) for pair in zip([covariance*y for y in source[0:numObservations-1]], \
                  [(1-covariance)*y for y in [random.normalvariate(0,1) for r in range(numObservations-1)]] ) ]
conditional = [random.normalvariate(0,1) for r in range(numObservations)]

cmiCalcClass = JPackage("infodynamics.measures.continuous.kraskov").ConditionalMutualInfoCalculatorMultiVariateKraskov2
cmiCalc = cmiCalcClass()
#cmiCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
cmiCalc.initialise(1, 1, 1)               # needs the dimensions of all three vars
cmiCalc.setProperty("k", "4")             # Use Kraskov parameter K=4 for 4 nearest points (knn search)

cmiCalc.setObservations(JArray(JDouble, 1)(source), 
                        JArray(JDouble, 1)(target),
                        JArray(JDouble, 1)(conditional))
cmi = cmiCalc.computeAverageLocalOfObservations()
print("result: %.4f nats." % cmi)
