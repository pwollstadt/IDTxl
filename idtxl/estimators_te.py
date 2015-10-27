from jpype import *
import pyinfo

def jidt_kraskov(self, source, target, knn, history_length):
    """Return the transfer entropy calculated by JIDT using the Kraskov
    estimator."""
    
    jarLocation = "/home/patriciaw/jidt_1_3/infodynamics-dist-1.3/infodynamics.jar"
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calc = calcClass()
    calc.setProperty("NORMALISE", "true")
    calc.initialise(history_length)
    calc.setProperty("k", str(knn))
    calc.setObservations(JArray(JDouble, 1)(source), JArray(JDouble, 1)(target))
    return calc.computeAverageLocalOfObservations()

def pyinfo_kraskov(self, source, target, knn, history_length):
    """Return the transfer entropy calculated by the pyinfo module using the
    Kraskov estimator."""
    
    return pyinfo.te_kraskov(source, target)
