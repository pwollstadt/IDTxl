import jpype as jp
import pyinfo


# TODO this should take numpy arrays
def jidt_kraskov(self, var1, var2, knn):
    """Calculate mutual information with JIDT's Kraskov implementation."""

    jarLocation = 'infodynamics.jar'
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                    jarLocation))
    calcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                 MultiInfoCalculatorKraskov1)
    calc = calcClass()
    calc.setProperty('NORMALISE', 'true')
    calc.setProperty('k', str(knn))
    calc.initialise(len(var1[0])+len(var2[0]))
    calc.setObservations(jp.JArray(jp.JDouble, 2)(var1),
                         jp.JArray(jp.JDouble, 2)(var2))
    return calc.computeAverageLocalOfObservations()
