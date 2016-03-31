import jpype as jp
import pyinfo


def jidt_kraskov(self, source, target, opts):
    """Calculate transfer entropy with JIDT's Kraskov implementation.

    Calculate transfer entropy between a source and a target variable using
    JIDT's implementation of the Kraskov type 1 estimator. Transfer entropy is
    defined as the conditional mutual information between the source's past
    state and the target's current value, conditional on the target's past.

    Past states need to be defined in the opts dictionary, where a past state
    is defined as a uniform embedding with parameters history and tau. The
    history describes the number of samples taken from a variable's past, tau
    descrices the embedding delay, i.e., the spacing between every two samples
    from the processes' past.

    References:

        Schreiber, T. (2000). Measuring information transfer. Physical Review
        Letters, 85(2), 461.

        Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual
        information. Physical Review E, 69(6), 066138.

        Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
        studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    This function is ment to be imported into the set_estimator module and used
    as a method in the Estimator_cmi class.

    Args:
        self (Estimator_cmi): instance of Estimator_cmi
        source : numpy array
            realisations of the source variable
        target : numpy array
            realisations of the target variable
        opts : dict [optional]
            sets estimation parameters:
            'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            'normalise' - z-standardise data (default=False)
            'theiler_t' - no. next temporal neighbours ignored in KNN and
            range searches (default='ACT', the autocorr. time of the target)
            'noise_level' - random noise added to the data (default=1e-8)
            'history_source' - number of samples in the source's past to
            consider
            'history_target' - number of samples in the target's past to
            consider (default=same as the source history)
            'tau_source' - source's embedding delay (default=1)
            'tau_target' - target's embedding delay (default=1)
            'source_target_delay' - information transfer delay between source
            and target (default=1)

    Returns:
        float
            transfer entropy from source to target

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the CMI calculator to save
        computation time. The Theiler window ignores trial boundaries. The
        CMI estimator does add noise to the data as a default. To make analysis
        runs replicable set noise_level to 0.
    """
    if opts is None:
        opts = {}
    try:
        kraskov_k = str(opts['kraskov_k'])
    except KeyError:
        kraskov_k = str(4)
    try:
        if opts['normalise']:
            normalise = 'true'
        else:
            normalise = 'false'
    except KeyError:
        normalise = 'false'
    try:
        theiler_t = str(opts['theiler_t'])
    except KeyError:
        theiler_t = str(0)
    try:
        noise_level = str(opts['noise_level'])
    except KeyError:
        noise_level = str(1e-8)
    try:
        history_target = opts['history_target']
    except KeyError:
        raise RuntimeError('No history was provided for TE estimation.')
    try:
        history_source = opts['history_source']
    except KeyError:
        history_source = history_target
    try:
        tau_target = opts['tau_target']
    except KeyError:
        tau_target = 1
    try:
        tau_source = opts['tau_source']
    except KeyError:
        tau_source = 1
    try:
        delay = opts['source_target_delay']
    except KeyError:
        delay = 1

    jarLocation = 'infodynamics.jar'
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                                                    jarLocation))
    calcClass = (jp.JPackage('infodynamics.measures.continuous.kraskov').
                 TransferEntropyCalculatorKraskov)
    calc = calcClass()
    calc.setProperty('NORMALISE', normalise)
    calc.setProperty('k', kraskov_k)
    calc.setProperty('PROP_KRASKOV_ALG_NUM', str(1))
    calc.setProperty('NOISE_LEVEL_TO_ADD', noise_level)
    calc.setProperty('DYN_CORR_EXCL', theiler_t)
    calc.initialise(history_target, tau_target,
                    history_source, tau_source,
                    delay)
    calc.setObservations(source, target)
    return calc.computeAverageLocalOfObservations()


def pyinfo_kraskov(self, source, target, knn, history_length):
    """Calculate transfer entropy with pyinfo's Kraskov implementation."""

    return pyinfo.te_kraskov(source, target)
