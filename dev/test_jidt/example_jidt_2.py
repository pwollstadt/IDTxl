# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 19:13:05 2016

@author: patricia
"""

import jpype as jp
import numpy as np

# Change location of jar to match yours:
jarLocation = "infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
if not jp.isJVMStarted():
    jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" +
                jarLocation)

# generate random data
base = 2
x = np.random.random_integers(0, base - 1, 10000)
y = np.random.random_integers(0, base - 1, 10000)

hCalcClass = jp.JPackage("infodynamics.measures.discrete").EntropyCalculatorDiscrete
hCalc = hCalcClass(base)
hCalc.initialise()
hCalc.addObservations(x)
hCalc.addObservations(jp.JArray(jp.JInt, 1)(x.tolist()))
hCalc.addObservations(jp.JArray(jp.JInt, 1)(x))
H = hCalc.computeAverageLocalOfObservations()
print("H(X): " + str(H))


miCalcClass = jp.JPackage("infodynamics.measures.discrete").MutualInformationCalculatorDiscrete
miCalc = miCalcClass(base)
miCalc.initialise()
miCalc.addObservations(x, y)
MI = miCalc.computeAverageLocalOfObservations()
print("I(X;Y): " + str(MI))
