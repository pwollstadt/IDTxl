# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 19:03:40 2016

@author: patricia
"""
import jpype as jp
import numpy as np

n = 1000
x = np.random.randint(2, size=(n, 1))
y = np.random.randint(2, size=(n, 1))

jarLocation = 'infodynamics.jar'
if not jp.isJVMStarted():
    jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                jarLocation))
mi_calc = jp.JPackage('infodynamics.measures.discrete').MutualInformationCalculatorDiscrete
mi_calc.initialise(2)
mi_calc.addObservations(x, y)
mi = mi_calc.computeAverageLocalOfObservations()
