import jpype as jp
import pyinfo


def jidt_kraskov(self, var1, var2, conditional, knn=4):
    """Calculate conditional mutual infor with JIDT's Kraskov implementation.

    Calculate the conditional mutual information between three variables. Call
    JIDT via jpype and use the Kraskov 2 estimator. If no conditional is given
    (is None), the function returns the mutual information between var1 and
    var2.

    This function is ment to be imported into the set_estimator module and used
    as a method in the Estimator_cmi class.

    Args:
        self: instance of Estimator_cmi
        var1: numpy array with realisations of the first random variable, where
            dimensions are realisations x variable dimension
        var2: numpy array with realisations of the second random variable
        conditional: numpy array with realisations of the random variable for
            conditioning
        knn: int, number of nearest neighbours for the Kraskov estimator

    Returns:
        conditional mutual information
    """

    jarLocation = 'infodynamics.jar'
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                    jarLocation))

    # print('Size var1: {0}, var2: {1}'.format(var1.size, var2.size))
    if conditional is not None:
        assert(conditional.size != 0), 'Conditional Array is empty.'
        calcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                     ConditionalMutualInfoCalculatorMultiVariateKraskov2)
    else:
        calcClass = (jp.JPackage("infodynamics.measures.continuous.kraskov").
                     MultiInfoCalculatorKraskov2)

    calc = calcClass()
    calc.setProperty('NORMALISE', 'true')
    calc.setProperty('k', str(knn))

    if conditional is not None:
        calc.initialise(var1.shape[1], var2.shape[1],  # needs dims of vars
                        conditional.shape[1])
        # calc.setObservations(JArray(JDouble, 2)(var1),
        #                     JArray(JDouble, 2)(var2),
        #                     JArray(JDouble, 2)(conditional))
        calc.setObservations(var1, var2, conditional)
        return calc.computeAverageLocalOfObservations()
    else:
        calc.initialise(var1.shape[1] + var2.shape[1])
        # calc.setObservations(jp.JArray(jp.JDouble, 2)(var1),
        #                      jp.JArray(jp.JDouble, 2)(var2))
        calc.setObservations(var1, var2)
        return calc.computeAverageLocalOfObservations()


def pyinfo_kraskov(self, var1, var2, conditional, knn):
    """Return the conditional mutual information calculated by the pyinfo module
    using the Kraskov estimator."""

    return pyinfo.cmi_kraskov(var1, var2, conditional)
