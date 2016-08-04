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
from . import idtxl_exceptions as ex
try:
    import jpype as jp
except ImportError:
    ex.jpype_missing('Jpype is not available on this system. To use '
                     'JAVA/JIDT-powered PID estimation install it from '
                     'https://pypi.python.org/pypi/JPype1')

def pid(s1_o, s2_o, target_o, cfg):
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
        est (dict): estimated decomposition, contains: MI/CMI values computed
            from non-permuted distributions; PID estimates (shared,
            synergistic, unique information); I(target;s1,s2) under permuted
            distribution Q
        opt (dict): additional information about iterative optimization,
            contains: final permutation Q; cfg dictionary; array with
            I(target:s1|s2) for each iteration; array with delta I(target:s1|s2) for
            each iteration; I(target:s1,s2) for each iteration

    Note:   variables names joined by "_" enter a mutual information computation
            together i.e. mi_va1_var2 --> I(var1 : var2).
            variables names joined directly form a new joint variable
            mi_var1var2_var3 --> I(var3:(var1,var2))
    """
    # make deep copies of input arrays to avoid side effects
    s1 = s1_o.copy()
    s2 = s2_o.copy()
    target = target_o.copy()



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
        alph_s1 = cfg['alph_s1']
    except KeyError:
        print('"alphabetsize" is missing from the cfg dictionary.')
        raise
    try:
        alph_s2 = cfg['alph_s2']
    except KeyError:
        print('"alphabetsize" is missing from the cfg dictionary.')
        raise
    try:
        alph_t = cfg['alph_t']
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
                    '-ea', '-Djava.class.path=' + jarpath, "-Xmx3000M")
    # what if it's there already - do we have to attach to it?

    # transform variables as far as possible outside the loops below
    # (note: only these variables should change when executing the loop)
    target_jA = jp.JArray(jp.JInt, target.ndim)(target.tolist())
    s2_jA = jp.JArray(jp.JInt, s2.ndim)(s2.tolist())
    s1_list = s1.tolist()
    s1_dim = s1.ndim


    Cmi_calc_class = (jp.JPackage('infodynamics.measures.discrete')
                      .ConditionalMutualInformationCalculatorDiscrete)
    Mi_calc_class = (jp.JPackage('infodynamics.measures.discrete')
                     .MutualInformationCalculatorDiscrete)
#
#   cmi_calc = Cmi_calc_class(alphabet,alphabet,alphabet)
    cmi_calc_target_s1_cond_s2 = Cmi_calc_class(alph_t,alph_s1,alph_s2)

    # MAX THE CORRECT WAY TO GO?
    alph_max_s1_t = max(alph_s1, alph_t)
    alph_max_s2_t = max(alph_s2, alph_t)
    alph_max = max(alph_s1*alph_s2, alph_t)

#   mi_calc  = Mi_calc_class(alphabet)
    mi_calc_s1 = Mi_calc_class(alph_max_s1_t)
    mi_calc_s2 = Mi_calc_class(alph_max_s2_t)

#   jointmi_calc  = Mi_calc_class(alphabet ** 2)
    jointmi_calc  = Mi_calc_class(alph_max)

    print("initialized all estimators")
    cmi_target_s1_cond_s2 = _calculate_cmi(cmi_calc_target_s1_cond_s2, target, s1, s2)
    jointmi_s1s2_target = _calculate_jointmi(jointmi_calc, s1, s2, target)
#   print("Original joint mutual information: {0}".format(jointmi_s1s2_target))
    mi_target_s1 = _calculate_mi(mi_calc_s1, s1, target)
#   print("Original mutual information I(target:s1): {0}".format(mi_target_s1))
    mi_target_s2 = _calculate_mi(mi_calc_s2, s2, target)
#   print("Original mutual information I(target:s2): {0}".format(mi_target_s2))
    print("Original redundancy - synergy: {0}".format(mi_target_s1 +
                                                mi_target_s2 - jointmi_s1s2_target))

    n = target.shape[0]
    reps = iterations + 1
    ind = np.arange(n)
    # collect estimates in each iteration
    cmi_q_target_s1_cond_s2_all = -np.inf * np.ones(reps).astype('float128')
    cmi_q_target_s1_cond_s2_delta = -np.inf * np.ones(reps).astype('float128')  # collect delta of estimates
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

        s1_new_list  = s1_list

        ind_new = ind

        # swapping: pick sample at random, find all other samples that
        # are potential matches (have the same value in target), pick one of
        # the matches for the actual swap
        swap_1 = np.random.randint(n)
        swap_candidates = np.where(target == target[swap_1])[0]
        swap_2 = np.random.choice(swap_candidates)

        # swap value in s1 and index to keep track
        s1_new_list[swap_1], s1_new_list[swap_2] = s1_new_list[swap_2], s1_new_list[swap_1]
        ind_new[swap_1], ind_new[swap_2] = (ind_new[swap_2],
                                            ind_new[swap_1])

        # calculate CMI under new swapped distribution
        cmi_new = _calculate_cmi_from_jA_list(cmi_calc_target_s1_cond_s2, target_jA, s1_new_list, s1_dim, s2_jA)


        if (np.less_equal(cmi_new, cmi_q_target_s1_cond_s2_all[i - 1])):
            s1_list = s1_new_list
            ind = ind_new
            cmi_q_target_s1_cond_s2_all[i] = cmi_new
            cmi_q_target_s1_cond_s2_delta[i] = cmi_q_target_s1_cond_s2_all[i - 1] - cmi_new
        else:
            cmi_q_target_s1_cond_s2_all[i] = cmi_q_target_s1_cond_s2_all[i - 1]
            unsuccessful += 1

    print('Unsuccessful swaps: {0}'.format(unsuccessful))
    # convert the final s1 back to an array
    s1_final =np.asarray(s1_new_list, dtype=np.int)
    # estimate unq/syn/shd information
    jointmi_q_s1s2_target = _calculate_jointmi(jointmi_calc, s1_final, s2, target)
    unq_s1 = _get_last_value(cmi_q_target_s1_cond_s2_all)  # Bertschinger, 2014, p. 2163

    # NEED TO REINITIALISE the calculator
    cmi_calc_target_s2_cond_s1 = Cmi_calc_class(alph_t,alph_s2,alph_s1)
    unq_s2 = _calculate_cmi(cmi_calc_target_s2_cond_s1, target, s2, s1_final)  # Bertschinger, 2014, p. 2166
    syn_s1s2 = jointmi_s1s2_target - jointmi_q_s1s2_target  # Bertschinger, 2014, p. 2163
    shd_s1s2 = mi_target_s1 + mi_target_s2 - jointmi_q_s1s2_target  # Bertschinger, 2014, p. 2167

    estimate = {
        'unq_s1': unq_s1,
        'unq_s2': unq_s2,
        'shd_s1s2': shd_s1s2,
        'syn_s1s2': syn_s1s2,
        'jointmi_q_s1s2_target': jointmi_q_s1s2_target,
        'orig_cmi_target_s1_cond_s2': cmi_target_s1_cond_s2,  # orignial values (empirical P)
        'orig_jointmi_s1s2_target': jointmi_s1s2_target,
        'orig_mi_target_s1': mi_target_s1,
        'orig_mi_target_s2': mi_target_s2
    }
    # useful outputs for plotting/debugging
    optimization = {
        'q': ind_new,
        'unsuc_swaps': unsuccessful,
        'cmi_q_target_s1_cond_s2_all': cmi_q_target_s1_cond_s2_all,
        'cmi_q_target_s1_cond_s2_delta': cmi_q_target_s1_cond_s2_delta,
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

def _calculate_cmi_from_jA_list(cmi_calc, var_1_java, var_2_list, var_2_ndim, cond_java):
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
#    var_1_java = jp.JArray(jp.JInt, var_1.ndim)(var_1.tolist())
    var_2_java = jp.JArray(jp.JInt, var_2_ndim)(var_2_list)
#    cond_java = jp.JArray(jp.JInt, cond.ndim)(cond.tolist())
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
    mi_calc.addObservations(jp.JArray(jp.JInt, var_1.ndim)(var_1.tolist()),
                            jp.JArray(jp.JInt, var_2.ndim)(var_2.tolist()))
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
    mUtils = jp.JPackage('infodynamics.utils').MatrixUtils
    # speed critical line ?
    s12 = mUtils.computeCombinedValues(jp.JArray(jp.JInt, 2)(np.column_stack((s1, s2)).tolist()), 2)
#    [s12, alph_joined] = _join_variables(s1, s2, 2, 2)
    jointmi_calc.initialise()
#    jointmi_calc.addObservations(jp.JArray(jp.JInt, s12.T.ndim)(s12.T.tolist()),
#                                 jp.JArray(jp.JInt, target.ndim)(target.tolist()))
    jointmi_calc.addObservations(s12,jp.JArray(jp.JInt, target.ndim)(target.tolist()))

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
    ind = np.where(x > -np.inf)[0]
    try:
        return x[ind[-1]]
    except IndexError:
        print('Couldn not find a value that is not -inf.')
        return np.NaN

#def test_logical_xor():
#
#    # logical XOR
#    n = 1000
#    alph = 2
#    s1 = np.random.randint(0, alph, n)
#    s2 = np.random.randint(0, alph, n)
#    target = np.logical_xor(s1, s2).astype(int)
#    cfg = {
#        'alph_s1': 2,
#        'alph_s2': 2,
#        'alph_t': 2,
#        'jarpath': 'infodynamics.jar',
#        'iterations': 1000
#    }
#    print('Testing PID estimator on binary AND, pointsset size{0}, iterations: {1}'.format(
#                                                        n, cfg['iterations']))
#    [est, opt] = pid(s1, s2, target, cfg)
#    print("----Results: ----")
#    print("unq_s1: {0}".format(est['unq_s1']))
#    print("unq_s2: {0}".format(est['unq_s2']))
#    print("shd_s1s2: {0}".format(est['shd_s1s2']))
#    print("syn_s1s2: {0}".format(est['syn_s1s2']))
#    assert 0.9 < est['syn_s1s2'] <=1.1, 'incorrect synergy: {0}, expected was {1}'.format(est['syn_s1s2'], 0.98)
#
#if __name__ == '__main__':
#    test_logical_xor()