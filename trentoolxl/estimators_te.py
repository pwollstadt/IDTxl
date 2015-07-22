from jpype import *
import pyinfo
#def jidt_kraskov_intialise(self):

def jidt_kraskov(self, source, target, knn, history_length):
    jarLocation = "/home/patriciaw/jidt_1_3/infodynamics-dist-1.3/infodynamics.jar"
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true")
    teCalc.initialise(history_length)
    teCalc.setProperty("k", str(knn))
    teCalc.setObservations(JArray(JDouble, 1)(source), JArray(JDouble, 1)(target))
    return teCalc.computeAverageLocalOfObservations()

def pyinfo_kraskov(self, source, target, knn, history_length):
    return pyinfo.te_kraskov(source, target)
