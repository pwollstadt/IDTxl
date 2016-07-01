from pkg_resources import resource_filename
import jpype as jp
import numpy as np
import random as rn


def jidt_kraskov(self, process, opts):
    """Calculate active information storage with JIDT's Kraskov implementation.

    Calculate active information storage (AIS) for some process using JIDT's 
    implementation of the Kraskov type 1 estimator. AIS is defined as the 
    mutual information between the processes' past state and current value.

    The past state needs to be defined in the opts dictionary, where a past 
    state is defined as a uniform embedding with parameters history and tau. 
    The history describes the number of samples taken from a processes' past, 
    tau describes the embedding delay, i.e., the spacing between every two 
    samples from the processes' past.

    References:

    Lizier, Joseph T., Mikhail Prokopenko, and Albert Y. Zomaya. (2012). Local 
    measures of information storage in complex distributed computation.
    Information Sciences, 208, 39-54.

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    This function is meant to be imported into the set_estimator module and 
    used as a method in the Estimator_cmi class.

    Args:
        self : instance of Estimator_cmi
            function is supposed to be used as part of the Estimator_cmi class
        process : numpy array
            realisations of the process        
        opts : dict [optional]
            sets estimation parameters:

            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='ACT', the autocorr. time of the target)
            - 'noise_level' - random noise added to the data (default=1e-8)
            - 'local_values' - return local AIS instead of average AIS
              (default=False)
            - 'history' - number of samples in the processes' past to consider
            - 'tau' - the processes' embedding delay (default=1)
            
    Returns:
        float
            active information storage in the process
        OR
        numpy array of floats
            local active information storage if local_values is set

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the AIS calculator to save
        computation time. The Theiler window ignores trial boundaries. The
        AIS estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """
    if opts is None:
        opts = {}
    elif type(opts) is not dict:
        raise TypeError('Opts should be a dictionary.')

    # Get defaults for estimator options
    kraskov_k = str(opts.get('kraskov_k', 4))
    normalise = str(opts.get('normalise', 'false'))
    theiler_t = str(opts.get('theiler_t', 0)) # TODO necessary?
    noise_level = str(opts.get('noise_level', 1e-8))
    local_values = opts.get('local_values', False)
    tau = opts.get('tau', 1)
    try:
        history = opts['history']
    except KeyError:
        raise RuntimeError('No history was provided for AIS estimation.')

    jarLocation = resource_filename(__name__, 'infodynamics.jar')
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                                                    jarLocation))
    calcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                 ActiveInfoStorageCalculatorKraskov)
    calc = calcClass()
    calc.setProperty('NORMALISE', normalise)
    calc.setProperty('k', kraskov_k)
    calc.setProperty('PROP_KRASKOV_ALG_NUM', str(1))
    calc.setProperty('NOISE_LEVEL_TO_ADD', noise_level)
    calc.setProperty('DYN_CORR_EXCL', theiler_t)
    calc.initialise(history, tau)
    calc.setObservations(process)
    if local_values:
        return calc.computeLocalOfPreviousObservations()
    else:
        return calc.computeAverageLocalOfObservations()
