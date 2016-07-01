from pkg_resources import resource_filename
import jpype as jp


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

    Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004). Estimating mutual
    information. Physical review E, 69(6), 066138.

    Lizier, Joseph T. (2014). JIDT: an information-theoretic toolkit for
    studying the dynamics of complex systems. Front. Robot. AI, 1(11).

    This function is ment to be imported into the set_estimator module and used
    as a method in the Estimator_cmi class.

    Args:
        self : instance of Estimator_cmi
            function is supposed to be used as part of the Estimator_cmi class
        source : numpy array
            realisations of the source variable
        target : numpy array
            realisations of the target variable
        opts : dict [optional]
            sets estimation parameters:

            - 'kraskov_k' - no. nearest neighbours for KNN search (default=4)
            - 'normalise' - z-standardise data (default=False)
            - 'theiler_t' - no. next temporal neighbours ignored in KNN and
              range searches (default='ACT', the autocorr. time of the target)
            - 'noise_level' - random noise added to the data (default=1e-8)
            - 'local_values' - return local TE instead of average TE 
              (default=False)
            - 'history_source' - number of samples in the source's past to
              consider
            - 'history_target' - number of samples in the target's past to
              consider (default=same as the source history)
            - 'tau_source' - source's embedding delay (default=1)
            - 'tau_target' - target's embedding delay (default=1)
            - 'source_target_delay' - information transfer delay between source
              and target (default=1)

    Returns:
        float
            transfer entropy from source to target
        OR
        numpy array of floats
            local transfer entropy if local_values is set

    Note:
        Some technical details: JIDT normalises over realisations, IDTxl
        normalises over raw data once, outside the CMI calculator to save
        computation time. The Theiler window ignores trial boundaries. The
        CMI estimator does add noise to the data as a default. To make analysis
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
    tau_target = opts.get('tau_target', 1)
    tau_source = opts.get('tau_source', 1)
    delay = opts.get('source_target_delay', 1)
    try:
        history_target = opts['history_target']
    except KeyError:
        raise RuntimeError('No history was provided for TE estimation.')
    history_source = opts.get('history_source', history_target)    

    jarLocation = resource_filename(__name__, 'infodynamics.jar')
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
    if local_values:
        return calc.computeLocalOfPreviousObservations()
    else:
        return calc.computeAverageLocalOfObservations()    
