"""Partical information decomposition for discrete random variables.

This module provides an estimator for partial information decomposition 
as proposed in

Bertschinger, Rauh, Olbrich, Jost, Ay; Quantifying Unique Information,
Entropy 2014, 16, 2161-2183; doi:10.3390/e16042161

The pid estimator returns estimates of shared information, unique 
information and synergistic information that two random variables X and 
Y have about a third variable Z. The estimator finds these estimates by
permuting the initial joint probability distribution of X, Y, and Z to 
find a permuted distribution Q that minimizes the unique information in  
X about Z (as proposed by Bertschinger and colleagues). The unique in-
formation is defined as the conditional mutual information I(X;Z|Y).

The estimator iteratively permutes the joint probability distribution of 
X, Y, and Z under the constraint that the marginal distributions (X, Z) 
and (Y, Z) stay constant. This is done by swapping two realizations of X 
which have the same corresponding value in Z, e.g.:

    X [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    Y [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    ---------------------------------
    Z [1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
    
    Possible swaps: X[0] and X[1]; X[0] and X[4]; X[2] and X[8]; ...
    
After each swap, I(X;Z|Y) is re-calculated under the new distribution; 
if the CMI is lower than the current permutation is kept and the next 
swap is tested. The iteration stops after the provided number of 
iterations.

Example:
    import numpy as np
    import pid
    
    n = 5000
    alph = 2
    x = np.random.randint(0, alph, n)
    y = np.random.randint(0, alph, n)
    z = np.logical_xor(x, y).astype(int)
    cfg = {
        'alphabetsize': 2, 
        'jarpath': '/home/user/infodynamics-dist-1.3/infodynamics.jar',
        'iterations': 10000
    }
    [est, opt] = pid(x, y, z, cfg)
    
This program is free software; you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the 
Free Software Foundation;

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY.

Version 1.0 by Patricia Wollstadt, Raul Vicente, Michael Wibral
Frankfurt, Germany, 2016

"""
import sys
import jpype as jp
import numpy as np
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time as tm


def pid(x, y, z, cfg):
    """Estimate partial information decomposition of discrete variables.
    
    The estimator finds shared information, unique information and 
    synergistic information between three discrete input variables.
    
    Args:
        x (numpy array): 1D array containing realizations of a discrete 
            random variable
        y (numpy array): 1D array containing realizations of a discrete 
            random variable
        z (numpy array): 1D array containing realizations of a discrete 
            random variable 
        cfg (dict): dictionary with estimation parameters, must contain 
            values for 'alphabetsize' (no. values in each variable x, y,
            z), 'jarpath' (string with path to JIDT jar file), 
            'iterations' (no. iterations of the estimator)
    
    Returns:
        dict: estimated decomposition, contains: MI/CMI values computed 
            from non-permuted distributions; PID estimates (shared, 
            synergistic, unique information); I(X;Y,Z) under permuted 
            distribution Q
        dict: additional information about iterative optimization, 
            contains: final permutation Q; cfg dictionary; array with
            I(X;Z|Y) for each iteration; array with delta I(X;Z|Y) for 
            each iteration; I(X;Y,Z) for each iteration
    """
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
        raise ValueError('Inputs x, y, z have to be vectors' 
                         '(1D-arrays).')
    
    try:
        jarpath = cfg['jarpath']
    except TypeError:
        print "The cfg argument should be a dictionary."
        raise
    except KeyError:
        print "'jarpath' is missing from the cfg dictionary."
        raise
    try:
        alphabet = cfg['alphabetsize']
    except KeyError:
        print "'alphabetsize' is missing from the cfg dictionary."
        raise
    try:
        iterations = cfg['iterations']
    except KeyError:
        print "'iterations' is missing from the cfg dictionary."
        raise
    
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), 
                    "-ea", "-Djava.class.path=" + jarpath)
    
    Cmi_calc_class = (jp.JPackage("infodynamics.measures.discrete")
                      .ConditionalMutualInformationCalculatorDiscrete)
    Mi_calc_class = (jp.JPackage("infodynamics.measures.discrete")
                     .MutualInformationCalculatorDiscrete)
    
    cmi_calc = Cmi_calc_class(alphabet,alphabet,alphabet)
    mi_calc  = Mi_calc_class(alphabet)
    multimi_calc  = Mi_calc_class(alphabet ** 2)
    
    cmi_xz_y = _calculate_cmi(cmi_calc, z, x, y)
    mi_xyz = _calculate_multimi(multimi_calc, x, y, z)
    mi_xz = _calculate_mi(mi_calc, x, z)
    mi_yz = _calculate_mi(mi_calc, y, z)
    
    n = z.shape[0]
    reps = iterations + 1
    ind = np.arange(n)
    cmi_q_xz_y_all = _nan(reps)  # collect estimates in each iteration
    cmi_q_xz_y_delta = _nan(reps)  # collect delta of estimates
    mi_q_xyz_all = _nan(reps)  # collect joint MI of all three vars
    cmi_q_xz_y_all[0] = cmi_xz_y  # initial I(X;Z|Y)
    unsuccessful = 0
    
    print 'Starting [                   ]',
    print '\b' * 21,
    sys.stdout.flush()
    for i in range(1, reps):
        steps = reps/20
        if i%steps == 0:
            print '\b.',
            sys.stdout.flush()
        
        #print "iteration " + str(i + 1) + " of " + str(reps - 1)
        
        x_new  = x
        ind_new = ind
        
        # swapping: pick sample at random, find all other samples that 
        # are potential matches (have the same value in Z), pick one of
        # the matches for the actual swap
        swap_1 = np.random.randint(n)
        swap_candidates = np.where(z == z[swap_1])[0]
        swap_2 = np.random.choice(swap_candidates)
        
        # swap value in X and index to keep track
        x_new[swap_1], x_new[swap_2] = x_new[swap_2], x_new[swap_1]
        ind_new[swap_1], ind_new[swap_2] = (ind_new[swap_2], 
                                            ind_new[swap_1])
        
        # calculate CMI under new swapped distribution
        cmi_new = _calculate_cmi(cmi_calc, x_new, z, y)
        
        if cmi_new < cmi_q_xz_y_all[i - 1]:
            x = x_new
            ind = ind_new
            cmi_q_xz_y_all[i] = cmi_new
            cmi_q_xz_y_delta[i] = cmi_q_xz_y_all[i - 1] - cmi_new 
            mi_q_xyz_all[i] = _calculate_multimi(multimi_calc, x, y, z)
        else:
            cmi_q_xz_y_all[i] = cmi_q_xz_y_all[i - 1]
            unsuccessful += 1
    
    print '\b]  Done!\n',
    print 'Unsuccessful swaps: ' + str(unsuccessful)
    
    # estimate unq/syn/shd information
    mi_q_xyz = _get_last_value(mi_q_xyz_all)
    unq_x = _get_last_value(cmi_q_xz_y_all)  # Bertschinger, 2014, p. 2163
    unq_y = _calculate_cmi(cmi_calc, z, y, x)  # Bertschinger, 2014, p. 2166
    syn_xy = mi_xyz - mi_q_xyz  # Bertschinger, 2014, p. 2163
    shd_xy = mi_xz + mi_yz - mi_q_xyz  # Bertschinger, 2014, p. 2167
    
    estimate = {
        'unq_x': unq_x,
        'unq_y': unq_y,
        'shd_xy': shd_xy,
        'syn_xy': syn_xy,
        'mi_q_xyz': mi_q_xyz,
        'orig_cmi_zx_y': cmi_xz_y,  # orignial values (empirical P)
        'orig_mi_xyz': mi_xyz,
        'orig_mi_xz': mi_xz,
        'orig_mi_yz': mi_yz
    }
    # useful outputs for plotting/debugging
    optimization = {
        'q': ind_new,
        'unsuc_swaps': unsuccessful,
        'cmi_q_xz_y_all': cmi_q_xz_y_all,
        'cmi_q_xz_y_delta': cmi_q_xz_y_delta,
        'mi_q_xyz_all': mi_q_xyz_all,
        'cfg': cfg
    }
    return estimate, optimization


def _nan(shape):
    """Return 1D numpy array of nans.
    
    Args:
        shape (int): length of array
    
    Returns:
        numpy array: array filled with nans
    """
    a = np.empty(shape)
    a.fill(np.nan)
    return a


def _calculate_cmi(cmi_calc, var_1, var_2, cond):
    """Calculate conditional MI from three variables usind JIDT.
    
    Args:
        cmi_calc (JIDT calculator object): JIDT calculator for conditio-
            nal mutual information
        var_1, var_2 (1D numpy array): realizations of two discrete 
            random variables
        cond (1D numpy array): realizations of a discrete random 
            variable for conditioning
    
    Returns:
        double: conditional mutual information between var_1 and var_2
            conditional on cond
    """
    var_1_java = jp.JArray(jp.JInt, var_1.ndim)(var_1.tolist())
    var_2_java = jp.JArray(jp.JInt, var_2.ndim)(var_2.tolist())
    cond_java = jp.JArray(jp.JInt, cond.ndim)(cond.tolist())
    cmi_calc.initialise()
    cmi_calc.addObservations(var_1_java, var_2_java, cond_java)
    cmi = cmi_calc.computeAverageLocalOfObservations()
    return cmi


def _calculate_mi(mi_calc, var_1, var_2):
    """Calculate MI from two variables usind JIDT.
    
    Args:
        mi_calc (JIDT calculator object): JIDT calculator for mutual 
            information
        var_1, var_2, (1D numpy array): realizations of some discrete
            random variables
    
    Returns:
        double: mutual information between input variables
    """
    mi_calc.initialise()
    mi_calc.addObservations(var_1, var_2)
    mi = mi_calc.computeAverageLocalOfObservations()
    return mi


def _calculate_multimi(multimi_calc, var_1, var_2, var_3):
    """Calculate MI from three variables usind JIDT.
    
    Args:
        multimi_calc (JIDT calculator object): JIDT calculator for 
            mutual information
        var_1, var_2, var_3 (1D numpy array): realizations of some 
            discrete random variables
    
    Returns:
        double: mutual information between all three input variables
    """
    mUtils = jp.JPackage('infodynamics.utils').MatrixUtils
    xy = mUtils.computeCombinedValues(np.column_stack((x, y)), 2)
    multimi_calc.initialise()
    multimi_calc.addObservations(xy, z)
    multimi = multimi_calc.computeAverageLocalOfObservations()
    return multimi


def _get_last_value(x):
    """Return the highest-index value that is not a NaN from an array.
    
    Args:
        x (1D numpy array): array where some entries are nan
    
    Returns:
        int/double: entry in x with highest index, which is not nan (if
            no such value exists, nan is returned)
    """
    ind = np.where(~np.isnan(x))[0]
    try:
        return x[ind[-1]]
    except IndexError:
        print "Couldn't find a value that is not NaN."
        return np.NaN


if __name__ == '__main__':
    
    n = 10000
    alph = 2
    x = np.random.randint(0, alph, n)
    y = np.random.randint(0, alph, n)
    z = np.logical_xor(x, y).astype(int)
    cfg = {
        'alphabetsize': 2, 
        'jarpath': '/home/patriciaw/jidt_1_3/infodynamics-dist-1.3/infodynamics.jar',
        'iterations': 10000
    }
    print "Testing PID estimator on binary XOR, iterations: " + \
        str(n) + ", N: " + str(n)
    tic = tm.clock()
    [est, opt] = pid(x, y, z, cfg)
    toc = tm.clock()
    print "Elapsed time: " + str(toc - tic) + " seconds"
    
    # plot results
    text_x_pos = opt['cfg']['iterations'] * 0.05
    plt.figure
    plt.subplot(2, 2, 1)
    plt.plot(est['orig_mi_xz'] + est['orig_mi_yz'] - opt['mi_q_xyz_all'])
    plt.ylim([-1, 0.1])
    plt.title('shared info')
    plt.ylabel('SI_Q(Z:X;Y)')
    plt.subplot(2, 2, 2)
    plt.plot(est['orig_mi_xyz'] - opt['mi_q_xyz_all'])
    plt.plot([0, opt['cfg']['iterations']],[1, 1], 'r')
    plt.text(text_x_pos, 0.9, 'XOR', color='red')
    plt.ylim([0, 1.1])
    plt.title('synergistic info')
    plt.ylabel('CI_Q(Z:X;Y)')
    plt.subplot(2, 2, 3)
    plt.plot(opt['cmi_q_xz_y_all'])
    plt.title('unique info X')
    plt.ylabel('UI_Q(X:Z|Y)')
    plt.xlabel('iteration')
    plt.subplot(2, 2, 4)
    plt.plot(opt['cmi_q_xz_y_delta'], 'r')
    plt.title('delta unique info X')
    plt.ylabel('delta UI_Q(X:Z|Y)')
    plt.xlabel('iteration')


