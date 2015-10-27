from jpype import *
import random
import math

jarLocation = "/home/patriciaw/jidt_1_3/infodynamics-dist-1.3/infodynamics.jar"
if not isJVMStarted():
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# generate data, taken from the TE example
numObservations = 1000
covariance=0.4
#source = [random.normalvariate(0,1) for r in range(numObservations)]
#target = [0] + [sum(pair) for pair in zip([covariance*y for y in source[0:numObservations-1]], \
#                  [(1-covariance)*y for y in [random.normalvariate(0,1) for r in range(numObservations-1)]] ) ]
#conditional = [random.normalvariate(0,1) for r in range(numObservations)]

source = [[random.normalvariate(0,1) for x in range(5)] for x in range(numObservations)]
target = [[random.normalvariate(0,1) for x in range(5)] for x in range(numObservations)]


miCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov2
miCalc = miCalcClass()
miCalc.initialise(10)               # needs the summed dimensions of the two vars
miCalc.setObservations(JArray(JDouble, 2)(source), 
                        JArray(JDouble, 2)(target))
mi = miCalc.computeAverageLocalOfObservations()
print("result: %.4f nats." % mi)
