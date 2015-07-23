from jpype import *
import pyinfo

def jidt_kraskov(self, var1, var2, knn):
    """Return the mutual information calculated by JIDT using the Kraskov
    estimator."""
    
    jarLocation = "/home/patriciaw/jidt_1_3/infodynamics-dist-1.3/infodynamics.jar"
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov2
    calc = calcClass()
    calc.setProperty("NORMALISE", "true")
    calc.setProperty("k", str(knn))
    #calc.initialise(var1.size[1]+var2.size[2])
    calc.initialise(len(var1[1])+len(var2[1]))
    calc.setObservations(JArray(JDouble, 2)(var1), 
                         JArray(JDouble, 2)(var2))
    return calc.computeAverageLocalOfObservations()
