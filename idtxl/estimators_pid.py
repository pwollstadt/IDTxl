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


def pid(s1, s2, target, cfg):
    """Estimate partial information decomposition of discrete variables.

    The estimator finds shared information, unique information and
    synergistic information between three discrete input variables.

    Args:
        s1 (numpy array): 1D array containing realizations of a discrete
            random variable (this is the source variable the algorithm calculates
            the actual UI for)
        s2 (numpy array): 1D array containing realizations of a discrete
            random variable (the other source variable)
        target (numpy array): 1D array containing realizations of a discrete
            random variable
        cfg (dict): dictionary with estimation parameters, must contain
            values for 'alphabetsize' (no. values in each variable s1, s2,
            target), 'jarpath' (string with path to JIDT jar file),
            'iterations' (no. iterations of the estimator)

    Returns:
        dict: estimated decomposition, contains: MI/CMI values computed
            from non-permuted distributions; PID estimates (shared,
            synergistic, unique information); I(target;s1,s2) under permuted
            distribution Q
        dict: additional information about iterative optimization,
            contains: final permutation Q; cfg dictionary; array with
            I(target:s1|s2) for each iteration; array with delta I(target:s1|s2) for
            each iteration; I(target:s1,s2) for each iteration

    Note:   variables names joined by "_" enter a mutual information computation
            together i.e. mi_va1_var2 --> I(var1 : var2).
            variables names joined directly form a new joint variable
            mi_var1var2_var3 --> I(var3:(var1,var2))
    """
    if s1.ndim != 1 or s2.ndim != 1 or target.ndim != 1:
        raise ValueError('Inputs s1, s2, target have to be vectors'
                         '(1D-arrays).')

    try:
        jarpath = cfg['jarpath']
    except TypeError:
        print('The cfg argument should be a dictionary.')
        raise
    except KeyError:
        print('"jarpath" is missing from the cfg dictionary.')
        raise
    try:
        alphabet = cfg['alphabetsize']
    except KeyError:
        print('"alphabetsize" is missing from the cfg dictionary.')
        raise
    try:
        iterations = cfg['iterations']
    except KeyError:
        print('"iterations" is missing from the cfg dictionary.')
        raise

    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(),
                    '-ea', '-Djava.class.path=' + jarpath)

    Cmi_calc_class = (jp.JPackage('infodynamics.measures.discrete')
                      .ConditionalMutualInformationCalculatorDiscrete)
    Mi_calc_class = (jp.JPackage('infodynamics.measures.discrete')
                     .MutualInformationCalculatorDiscrete)

    cmi_calc = Cmi_calc_class(alphabet,alphabet,alphabet)
    mi_calc  = Mi_calc_class(alphabet)
    jointmi_calc  = Mi_calc_class(alphabet ** 2)

    cmi_target_s1_cond_s2 = _calculate_cmi(cmi_calc, target, s1, s2)
    mi_s1s2_target = _calculate_jointmi(jointmi_calc, s1, s2, target)
    mi_target_s1 = _calculate_mi(mi_calc, s1, target)
    mi_target_s2 = _calculate_mi(mi_calc, s2, target)

    n = target.shape[0]
    reps = iterations + 1
    ind = np.arange(n)
    cmi_q_target_s1_cond_s2_all = _nan(reps)  # collect estimates in each iteration
    cmi_q_target_s1_cond_s2_delta = _nan(reps)  # collect delta of estimates
    mi_q_s1s2_target_all = _nan(reps)  # collect joint MI of the two sources with the target
    cmi_q_target_s1_cond_s2_all[0] = cmi_target_s1_cond_s2  # initial I(s1;target|Y)
    unsuccessful = 0

#    print('Starting [                   ]', end='')
#    print('\b' * 21, end='')
    sys.stdout.flush()
    for i in range(1, reps):
#        steps = reps/20
#        if i%steps == 0:
#            print('\b.', end='')
#            sys.stdout.flush()

        #print('iteration ' + str(i + 1) + ' of ' + str(reps - 1)

        s1_new  = s1
        ind_new = ind

        # swapping: pick sample at random, find all other samples that
        # are potential matches (have the same value in target), pick one of
        # the matches for the actual swap
        swap_1 = np.random.randint(n)
        swap_candidates = np.where(target == target[swap_1])[0]
        swap_2 = np.random.choice(swap_candidates)

        # swap value in s1 and index to keep track
        s1_new[swap_1], s1_new[swap_2] = s1_new[swap_2], s1_new[swap_1]
        ind_new[swap_1], ind_new[swap_2] = (ind_new[swap_2],
                                            ind_new[swap_1])

        # calculate CMI under new swapped distribution
        cmi_new = _calculate_cmi(cmi_calc, target, s1_new, s2)

        if cmi_new < cmi_q_target_s1_cond_s2_all[i - 1]:
            s1 = s1_new
            ind = ind_new
            cmi_q_target_s1_cond_s2_all[i] = cmi_new
            cmi_q_target_s1_cond_s2_delta[i] = cmi_q_target_s1_cond_s2_all[i - 1] - cmi_new
            mi_q_s1s2_target_all[i] = _calculate_jointmi(jointmi_calc, s1, s2, target)
        else:
            cmi_q_target_s1_cond_s2_all[i] = cmi_q_target_s1_cond_s2_all[i - 1]
            unsuccessful += 1

    print('\b]  Done!\n', end='')
    print('Unsuccessful swaps: {0}'.format(unsuccessful))

    # estimate unq/syn/shd information
    mi_q_s1s2_target = _get_last_value(mi_q_s1s2_target_all)
    unq_s1 = _get_last_value(cmi_q_target_s1_cond_s2_all)  # Bertschinger, 2014, p. 2163
    unq_s2 = _calculate_cmi(cmi_calc, target, s2, s1)  # Bertschinger, 2014, p. 2166
    syn_s1s2 = mi_s1s2_target - mi_q_s1s2_target  # Bertschinger, 2014, p. 2163
    shd_s1s2 = mi_target_s1 + mi_target_s2 - mi_q_s1s2_target  # Bertschinger, 2014, p. 2167

    estimate = {
        'unq_s1': unq_s1,
        'unq_s2': unq_s2,
        'shd_s1s2': shd_s1s2,
        'syn_s1s2': syn_s1s2,
        'mi_q_s1s2_target': mi_q_s1s2_target,
        'orig_cmi_target_s1_cond_s2': cmi_target_s1_cond_s2,  # orignial values (empirical P)
        'orig_mi_s1s2_target': mi_s1s2_target,
        'orig_mi_target_s1': mi_target_s1,
        'orig_mi_target_s2': mi_target_s2
    }
    # useful outputs for plotting/debugging
    optimization = {
        'q': ind_new,
        'unsuc_swaps': unsuccessful,
        'cmi_q_target_s1_cond_s2_all': cmi_q_target_s1_cond_s2_all,
        'cmi_q_target_s1_cond_s2_delta': cmi_q_target_s1_cond_s2_delta,
        'mi_q_s1s2_target_all': mi_q_s1s2_target_all,
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
    mi_calc.addObservations(jp.JArray(jp.JInt, target.ndim)(var_1.tolist()),
                            jp.JArray(jp.JInt, target.ndim)(var_2.tolist()))
    mi = mi_calc.computeAverageLocalOfObservations()
    return mi


def _calculate_jointmi(jointmi_calc, s1, s2, target):
    """Calculate MI from three variables usind JIDT.

    Args:
        jointmi_calc (JIDT calculator object): JIDT calculator for
            mutual information
        var_1, var_2, var_3 (1D numpy array): realizations of some
            discrete random variables

    Returns:
        double: mutual information between all three input variables
    """
    # mUtils = jp.JPackage('infodynamics.utils').MatrixUtils
    # xy = mUtils.computeCombinedValues(np.column_stack((x, y)), 2)
    [s12, alph_joined] = _join_variables(s1, s2, 2, 2)
    jointmi_calc.initialise()
    jointmi_calc.addObservations(jp.JArray(jp.JInt, s12.T.ndim)(s12.T.tolist()),
                                 jp.JArray(jp.JInt, target.ndim)(target.tolist()))
    jointmi = jointmi_calc.computeAverageLocalOfObservations()
    return jointmi


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
        print('Couldn not find a value that is not NaN.')
        return np.NaN


def _join_variables(a, b, alph_a, alph_b):
    """Join two sequences of random variables (RV) into a new RV.

    Works like the method 'computeCombinedValues' implemented in JIDT
    (https://github.com/jlizier/jidt/blob/master/java/source/
    infodynamics/utils/MatrixUtils.java).

    Args:
        a, b (np array): sequence of integer numbers of arbitrary base
            (representing observations from two RVs)
        alph_a, alph_b (int): alphabet size of a and b

    Returns:
        np array, int: joined RV
        int: alphabet size of new RV
    """
    if a.shape[0] != b.shape[0]:
        raise Error

    if alph_b < alph_a:
        a, b = b, a
        alph_a, alph_b = alph_b, alph_a

    joined = np.zeros(a.shape[0])

    for i in range(joined.shape[0]):
        mult = 1
        joined[i] += mult * b[i]
        mult *= alph_a
        joined[i] += mult * a[i]

    alph_new = max(a) * alph_a + alph_b
    '''
    for (int r = 0; r < rows; r++) {
        // For each row in vec1
        int combinedRowValue = 0;
        int multiplier = 1;
        for (int c = columns - 1; c >= 0; c--) {
            // Add in the contribution from each column
            combinedRowValue += separateValues[r][c] * multiplier;
            multiplier *= base;
        }
        combinedValues[r] = combinedRowValue;
    } '''

    return joined.astype(int), alph_new

if __name__ == '__main__':

    n = 10000
    alph = 2
    s1 = np.random.randint(0, alph, n)
    s2 = np.random.randint(0, alph, n)
    target = np.logical_xor(s1, s2).astype(int)
    cfg = {
        'alphabetsize': 2,
        'jarpath': 'infodynamics.jar',
        'iterations': 0000
    }
    print('Testing PID estimator on binary XOR, iterations: {0}, {1}'.format(
                                                        n, cfg['iterations']))
    tic = tm.clock()
    [est, opt] = pid(s1, s2, target, cfg)
    toc = tm.clock()
    print('Elapsed time: {0} seconds'.format(toc - tic))

    # plot results
    text_x_pos = opt['cfg']['iterations'] * 0.05
    plt.figure
    plt.subplot(2, 2, 1)
    plt.plot(est['orig_mi_target_s1'] + est['orig_mi_target_s2'] - opt['mi_q_s1s2_target_all'])
    plt.ylim([-1, 0.1])
    plt.title('shared info')
    plt.ylabel('SI_Q(target:s1;s2)')
    plt.subplot(2, 2, 2)
    plt.plot(est['orig_mi_s1s2_target'] - opt['mi_q_s1s2_target_all'])
    plt.plot([0, opt['cfg']['iterations']],[1, 1], 'r')
    plt.text(text_x_pos, 0.9, 'XOR', color='red')
    plt.ylim([0, 1.1])
    plt.title('synergistic info')
    plt.ylabel('CI_Q(target:s1;s2)')
    plt.subplot(2, 2, 3)
    plt.plot(opt['cmi_q_target_s1_cond_s2_all'])
    plt.title('unique info s1')
    plt.ylabel('UI_Q(s1:target|s2)')
    plt.xlabel('iteration')
    plt.subplot(2, 2, 4)
    plt.plot(opt['cmi_q_target_s1_cond_s2_delta'], 'r')
    plt.title('delta unique info s1')
    plt.ylabel('delta UI_Q(s1:target|s2)')
    plt.xlabel('iteration')


