import jpype as jp
import pyinfo

# TODO this should take numpy arrays
def jidt_kraskov(self, source, target, knn, history_length):
    """Calculate transfer entropy with JIDT's Kraskov implementation."""

    jarLocation = 'infodynamics.jar'
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                                                    jarLocation))
    calcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                 TransferEntropyCalculatorKraskov)
                 # TODO does this use Kraskov 1 or 2 as default? -> use 1
    calc = calcClass()
    calc.setProperty('NORMALISE', 'true')
    calc.initialise(history_length)
    calc.setProperty('k', str(knn))
    calc.setObservations(jp.JArray(jp.JDouble, 1)(source),
                         jp.JArray(jp.JDouble, 1)(target))
    return calc.computeAverageLocalOfObservations()


def pyinfo_kraskov(self, source, target, knn, history_length):
    """Calculate transfer entropy with pyinfo's Kraskov implementation."""

    return pyinfo.te_kraskov(source, target)
