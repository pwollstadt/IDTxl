from jpype import *
import pyinfo

def jidt_kraskov(self, source, target, conditional, knn):
    """Return the conditional mutual information calculated by JIDT using the
    Kraskov estimator."""
    
    jarLocation = "/home/patriciaw/jidt_1_3/infodynamics-dist-1.3/infodynamics.jar"
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
    calcClass = JPackage("infodynamics.measures.continuous.kraskov").ConditionalMutualInfoCalculatorMultiVariateKraskov2
    calc = calcClass()
    calc.setProperty("NORMALISE", "true")
    calc.setProperty("k", str(knn))
    calc.initialise(source.size[1], target.size[1], conditional.size[1])
    calc.setObservations(JArray(JDouble, 2)(source), 
                         JArray(JDouble, 2)(target),
                         JArray(JDouble, 2)(conditional))
    return calc.computeAverageLocalOfObservations()

def pyinfo_kraskov(self, source, target, conditional, knn):
    """Return the conditional mutual information calculated by the pyinfo module
    using the Kraskov estimator."""
    
    return pyinfo.cmi_kraskov(source, target, conditional)
