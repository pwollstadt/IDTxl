"""Partical information decomposition for discrete random variables.

This module provides an estimator for partial information decomposition
as proposed in

Bertschinger, Rauh, Olbrich, Jost, Ay; Quantifying Unique Information,
Entropy 2014, 16, 2161-2183; doi:10.3390/e16042161

"""
import sys
import numpy as np
from . import synergy_tartu
from . import idtxl_exceptions as ex
try:
    import jpype as jp
except ImportError:
    ex.jpype_missing('Jpype is not available on this system. To use '
                     'JAVA/JIDT-powered PID estimation install it from '
                     'https://pypi.python.org/pypi/JPype1')


def pid_frankfurt(s1_o, s2_o, target_o, opts):
    """Estimate partial information decomposition of discrete variables.

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

    Args:
        s1 (numpy array): 1D array containing realizations of a discrete
            random variable (this is the source variable the algorithm
            calculates the actual UI for)
        s2 (numpy array): 1D array containing realizations of a discrete
            random variable (the other source variable)
        target (numpy array): 1D array containing realizations of a discrete
            random variable
        opts (dict): dictionary with estimation parameters, must contain
            values for 'alphabetsize' (no. values in each variable s1, s2,
            target), 'jarpath' (string with path to JIDT jar file),
            'iterations' (no. iterations of the estimator)

    Returns:
        est (dict): estimated decomposition, contains: MI/CMI values computed
            from non-permuted distributions; PID estimates (shared,
            synergistic, unique information); I(target;s1,s2) under permuted
            distribution Q
        opt (dict): additional information about iterative optimization,
            contains: final permutation Q; opts dictionary; array with
            I(target:s1|s2) for each iteration; array with delta
            I(target:s1|s2) for each iteration; I(target:s1,s2) for each
            iteration

    Note:   variables names joined by "_" enter a mutual information
            computation together i.e. mi_va1_var2 --> I(var1 : var2).
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
        jarpath = opts['jarpath']
    except TypeError:
        print('The opts argument should be a dictionary.')
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
    cmi_calc_target_s1_cond_s2 = Cmi_calc_class(alph_t, alph_s1, alph_s2)

    # MAX THE CORRECT WAY TO GO?
    alph_max_s1_t = max(alph_s1, alph_t)
    alph_max_s2_t = max(alph_s2, alph_t)
    alph_max = max(alph_s1*alph_s2, alph_t)

#   mi_calc  = Mi_calc_class(alphabet)
    mi_calc_s1 = Mi_calc_class(alph_max_s1_t)
    mi_calc_s2 = Mi_calc_class(alph_max_s2_t)

#   jointmi_calc = Mi_calc_class(alphabet ** 2)
    jointmi_calc = Mi_calc_class(alph_max)

    print("initialized all estimators")
    cmi_target_s1_cond_s2 = _calculate_cmi(cmi_calc_target_s1_cond_s2,
                                           target, s1, s2)
    jointmi_s1s2_target = _calculate_jointmi(jointmi_calc, s1, s2, target)
#   print("Original joint mutual information: {0}".format(jointmi_s1s2_target))
    mi_target_s1 = _calculate_mi(mi_calc_s1, s1, target)
#   print("Original mutual information I(target:s1): {0}".format(mi_target_s1))
    mi_target_s2 = _calculate_mi(mi_calc_s2, s2, target)
#   print("Original mutual information I(target:s2): {0}".format(mi_target_s2))
    print("Original redundancy - synergy: {0}".format(
                            mi_target_s1 + mi_target_s2 - jointmi_s1s2_target))

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

        # print('iteration ' + str(i + 1) + ' of ' + str(reps - 1)

        s1_new_list = s1_list

        ind_new = ind

        # swapping: pick sample at random, find all other samples that
        # are potential matches (have the same value in target), pick one of
        # the matches for the actual swap
        swap_1 = np.random.randint(n)
        swap_candidates = np.where(target == target[swap_1])[0]
        swap_2 = np.random.choice(swap_candidates)

        # swap value in s1 and index to keep track
        s1_new_list[swap_1], s1_new_list[swap_2] = (s1_new_list[swap_2],
                                                    s1_new_list[swap_1])
        ind_new[swap_1], ind_new[swap_2] = (ind_new[swap_2],
                                            ind_new[swap_1])

        # calculate CMI under new swapped distribution
        cmi_new = _calculate_cmi_from_jA_list(cmi_calc_target_s1_cond_s2,
                                              target_jA,
                                              s1_new_list,
                                              s1_dim,
                                              s2_jA)

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
    s1_final = np.asarray(s1_new_list, dtype=np.int)
    # estimate unq/syn/shd information
    jointmi_q_s1s2_target = _calculate_jointmi(
                                            jointmi_calc, s1_final, s2, target)
    unq_s1 = _get_last_value(cmi_q_target_s1_cond_s2_all)  # Bertschinger, 2014, p. 2163

    # NEED TO REINITIALISE the calculator
    cmi_calc_target_s2_cond_s1 = Cmi_calc_class(alph_t, alph_s2, alph_s1)
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

def _calculate_cmi_from_jA_list(cmi_calc, var_1_java, var_2_list, var_2_ndim,
                                cond_java):
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



def pid_sydney(s1, s2, t, cfg):
    """Provide a fast implementation of the PDI estimator for discrete data.

    This module exports a fast implementation of the partial information
    decomposition (PID) estimator for discrete data. The estimator does not
    require JAVA or GPU modules to run.

    Improved version with larger initial swaps and checking for convergence of
    both the unique information from sources 1 and 2.
    """

    if s1.ndim != 1 or s2.ndim != 1 or t.ndim != 1:
        raise ValueError('Inputs s1, s2, target have to be vectors'
                         '(1D-arrays).')
    if (len(t) != len(s1) or len(t) != len(s2)):
        raise ValueError('Number of samples s1, s2 and t must be equal')

    try:
        alph_s1 = cfg['alph_s1']
    except TypeError:
        print('The cfg argument should be a dictionary.')
        raise
    except KeyError:
        print('"alph_s1" is missing from the cfg dictionary.')
        raise
    try:
        alph_s2 = cfg['alph_s2']
    except KeyError:
        print('"alph_s2" is missing from the cfg dictionary.')
        raise
    try:
        alph_t = cfg['alph_t']
    except KeyError:
        print('"alph_t" is missing from the cfg dictionary.')
        raise
    try:
        max_unsuc_swaps_row_parm = cfg['max_unsuc_swaps_row_parm']
    except KeyError:
        print('"max_unsuc_swaps_row_parm" is missing from the cfg dictionary.')
        raise
    try:
        num_reps = cfg['num_reps']
    except KeyError:
        print('"num_reps" is missing from the cfg dictionary.')
        raise
    if (num_reps > 63):
        raise ValueError('Number of reps must be 63 or less to prevent integer overflow')
    try:
        max_iters = cfg['max_iters']
    except KeyError:
        print('"max_iters" is missing from the cfg dictionary.')
        raise

    # -- DEFINE PARAMETERS -- #

    num_samples = len(t)

    # Max swaps = number of possible swaps * control parameter
    num_pos_swaps = alph_t * alph_s1 * (alph_s1-1) * alph_s2 * (alph_s2-1)
    max_unsuc_swaps_row = np.floor(num_pos_swaps * max_unsuc_swaps_row_parm)

    # -- CALCULATE PROBABLITIES -- #

    # Declare arrays for counts
    t_count = np.zeros(alph_t, dtype=np.int)
    s1_count = np.zeros(alph_s1, dtype=np.int)
    s2_count = np.zeros(alph_s2, dtype=np.int)
    joint_t_s1_count = np.zeros((alph_t, alph_s1), dtype=np.int)
    joint_t_s2_count = np.zeros((alph_t, alph_s2), dtype=np.int)
    joint_s1_s2_count = np.zeros((alph_s1, alph_s2), dtype=np.int)
    joint_t_s1_s2_count = np.zeros((alph_t, alph_s1, alph_s2),
                                   dtype=np.int)

    # Count observations
    for obs in range(0, num_samples):
        t_count[t[obs]] += 1
        s1_count[s1[obs]] += 1
        s2_count[s2[obs]] += 1
        joint_t_s1_count[t[obs], s1[obs]] += 1
        joint_t_s2_count[t[obs], s2[obs]] += 1
        joint_s1_s2_count[s1[obs], s2[obs]] +=1
        joint_t_s1_s2_count[t[obs], s1[obs], s2[obs]] += 1
#    min_joint_nonzero_count = np.min(
#				np.min(
#				np.min(
#				joint_t_s1_s2_count[np.nonzero(joint_t_s1_s2_count)])))

    max_joint_nonzero_count = np.max(joint_t_s1_s2_count[np.nonzero(joint_t_s1_s2_count)])


    # Fixed probabilities
    t_prob = np.divide(t_count, num_samples).astype('float128')
    s1_prob = np.divide(s1_count, num_samples).astype('float128')
    s2_prob = np.divide(s2_count, num_samples).astype('float128')
    joint_t_s1_prob = np.divide(joint_t_s1_count, num_samples).astype('float128')
    joint_t_s2_prob = np.divide(joint_t_s2_count, num_samples).astype('float128')

    # Variable probabilities
    joint_s1_s2_prob = np.divide(joint_s1_s2_count, num_samples).astype('float128')
    joint_t_s1_s2_prob = np.divide(joint_t_s1_s2_count, num_samples).astype('float128')
    max_prob = np.max(joint_t_s1_s2_prob[np.nonzero(joint_t_s1_s2_prob)])

#    # make copies of the variable probabilities for independent second
#    # optimization and comparison of KLDs for convergence check:
#    # KLDs should initially rise and then fall when close to the minimum
#    joint_s1_s2_prob_alt = joint_s1_s2_prob.copy()
#    joint_t_s1_s2_prob_alt = joint_t_s1_s2_prob.copy()

    # -- VIRTUALISED SWAPS -- #

    # Calculate the initial cmi's and store them
    cond_mut_info1 = _cmi_prob(
        s2_prob, joint_t_s2_prob, joint_s1_s2_prob, joint_t_s1_s2_prob)
    cur_cond_mut_info1 = cond_mut_info1

    joint_s2_s1_prob = np.transpose(joint_s1_s2_prob)
    joint_t_s2_s1_prob = np.ndarray.transpose(joint_t_s1_s2_prob,[0,2,1])

    cond_mut_info2 = _cmi_prob(
        s1_prob, joint_t_s1_prob, joint_s2_s1_prob,joint_t_s2_s1_prob)
    cur_cond_mut_info2 = cond_mut_info2

    # sanity check: the curr cmi must be smaller than the joint, else something
    # is fishy
    #
    jointmi_s1s2_t = _joint_mi(s1, s2, t, alph_s1, alph_s2, alph_t)

    if cond_mut_info1 > jointmi_s1s2_t:
        raise ValueError('joint MI {0} smaller than cMI {1}'
                         ''.format(jointmi_s1s2_t, cond_mut_info1))
    else:
        print('Passed sanity check on jMI and cMI')


    # Declare reps array of repeated doubling to half the prob_inc
    # WARNING: num_reps greater than 63 results in integer overflow
    # TODO: in principle we could divide the increment until we run out of fp
    # precision, e.g.
    # we can get some extra reps by not starting with a swap of size 1/n
    # but soemthing larger, by adding as many steps here as we are powers of 2
    # larger in the max probability than 1/n
    # and by starting with swaps in the size of the max probability
    # this will keep almost all of the bins of the joint pdf from being swapped
    # but they will joint swapping later, or after being swapped into
    # unfortunatley this does not run with the current code as it uses large
    # powers of integers
    # another idea would be to decrement by something slightly smaller than 2
#    num_reps = num_reps + np.int32(np.floor(np.log(max_joint_nonzero_count)/np.log(2)))
    print("num_reps:")
    print(num_reps)
    reps = np.array(np.power(2,range(0,num_reps)))

    # Replication loop
    for rep in reps:
        prob_inc = np.multiply(
            np.float128(max_prob),
            np.divide(np.float128(1),np.float128(rep)))
        # Want to store number of succesive unsuccessful swaps
        unsuccessful_swaps_row = 0
        # SWAP LOOP
        for attempt_swap in range(0, max_iters):
            # Pick a random candidate from the targets
            t_cand = np.random.randint(0, alph_t)
            s1_cand = np.random.randint(0, alph_s1)
            s2_cand = np.random.randint(0, alph_s2)

            # Pick a swap candidate
            s1_prim = np.random.randint(0, alph_s1-1)
            if (s1_prim >= s1_cand):
                s1_prim += 1
            s2_prim = np.random.randint(0, alph_s2-1)
            if (s2_prim >= s2_cand):
                s2_prim += 1

#            unsuccessful_swaps_row = _try_swap(cur_cond_mut_info,
#                                               joint_t_s1_s2_prob,
#                                               joint_s1_s2_prob,
#                                               joint_t_s2_prob, s2_prob,
#                                               t_cand, s1_prim, s2_prim,
#                                               s1_cand, s2_cand,
#                                               prob_inc,
#                                               unsuccessful_swaps_row)
#            print("unsuccessful_swaps_row: {0}".format(unsuccessful_swaps_row))

            # START of a possible try_swap function
            # based on a fixed set of candidates
            # that can then be used recursively until the swap direction
            # becomes unsuccessful

            # Ensure we can decrement without introducing neg probs
            # this is very important as we start swaps in the size of the
            # maximum probability
            if (joint_t_s1_s2_prob[t_cand, s1_cand, s2_cand] >= prob_inc
                and joint_t_s1_s2_prob[t_cand, s1_prim, s2_prim] >= prob_inc
                and joint_s1_s2_prob[s1_cand, s2_cand] >= prob_inc
                and joint_s1_s2_prob[s1_prim, s2_prim] >= prob_inc):

                joint_t_s1_s2_prob[t_cand, s1_cand, s2_cand] -= prob_inc
                joint_t_s1_s2_prob[t_cand, s1_prim, s2_prim] -= prob_inc
                joint_t_s1_s2_prob[t_cand, s1_cand, s2_prim] += prob_inc
                joint_t_s1_s2_prob[t_cand, s1_prim, s2_cand] += prob_inc

                joint_s1_s2_prob[s1_cand, s2_cand] -= prob_inc
                joint_s1_s2_prob[s1_prim, s2_prim] -= prob_inc
                joint_s1_s2_prob[s1_cand, s2_prim] += prob_inc
                joint_s1_s2_prob[s1_prim, s2_cand] += prob_inc

                # Calculate the cmi after this virtual swap
                cond_mut_info1 = _cmi_prob(
                    s2_prob, joint_t_s2_prob, joint_s1_s2_prob, joint_t_s1_s2_prob)
                cond_mut_info2 = _cmi_prob(
                    s2_prob, joint_t_s2_prob, joint_s1_s2_prob, joint_t_s1_s2_prob)

                # If at least one of the cmis is improved keep it,
                # reset the unsuccessful swap counter
                if ( cond_mut_info1 < cur_cond_mut_info1 or
                     cond_mut_info2 < cur_cond_mut_info2):
                    cur_cond_mut_info1 = cond_mut_info1
                    cur_cond_mut_info2 = cond_mut_info2
                    unsuccessful_swaps_row = 0
                    # TODO: if this swap direction was successful - repeat it !
                # Else undo the changes, record unsuccessful swap
                else:
                    joint_t_s1_s2_prob[t_cand, s1_cand, s2_cand] += prob_inc
                    joint_t_s1_s2_prob[t_cand, s1_prim, s2_prim] += prob_inc
                    joint_t_s1_s2_prob[t_cand, s1_cand, s2_prim] -= prob_inc
                    joint_t_s1_s2_prob[t_cand, s1_prim, s2_cand] -= prob_inc

                    joint_s1_s2_prob[s1_cand, s2_cand] += prob_inc
                    joint_s1_s2_prob[s1_prim, s2_prim] += prob_inc
                    joint_s1_s2_prob[s1_cand, s2_prim] -= prob_inc
                    joint_s1_s2_prob[s1_prim, s2_cand] -= prob_inc

                    unsuccessful_swaps_row += 1
            else:
                unsuccessful_swaps_row += 1
            # END of a possible try_swap function

            if (unsuccessful_swaps_row >= max_unsuc_swaps_row):
                break

    # print(cond_mut_info, '\t', prob_inc, '\t', unsuccessful_swaps_row)

    # -- PID Evaluation -- #

    # Classical mutual information terms
    mi_target_s1 = _mi_prob(t_prob, s1_prob, joint_t_s1_prob)
    mi_target_s2 = _mi_prob(t_prob, s2_prob, joint_t_s2_prob)
    jointmi_s1s2_target = _joint_mi(s1, s2, t, alph_s1, alph_s2, alph_t)
    print('jointmi_s1s2_target: {0}'.format(jointmi_s1s2_target))

    # PID terms
    unq_s1 = cond_mut_info1
    shd_s1_s2 = mi_target_s1 - unq_s1
    unq_s2 = mi_target_s2 - shd_s1_s2
    syn_s1_s2 = jointmi_s1s2_target - unq_s1 - unq_s2 - shd_s1_s2

    estimate = {
        'joint_mi_s1s2_t': jointmi_s1s2_target,
        'unq_s1': unq_s1,
        'unq_s2': unq_s2,
        'shd_s1_s2': shd_s1_s2,
        'syn_s1_s2': syn_s1_s2,
    }

    return estimate


def _cmi_prob(s2cond_prob, joint_t_s2cond_prob,
             joint_s1_s2cond_prob, joint_t_s1_s2cond_prob):

    total = np.zeros(1).astype('float128')

    [alph_t, alph_s1, alph_s2cond] = np.shape(joint_t_s1_s2cond_prob)

    for sym_s1 in range(0, alph_s1):
        for sym_s2cond in range(0, alph_s2cond):
            for sym_t in range(0, alph_t):

                # print(sym_s1, '\t', sym_s2cond, '\t', sym_t, '\t', joint_t_s2cond_prob[sym_t, sym_s2cond], '\t', joint_s1_s2cond_prob[sym_s1, sym_s2cond], '\t', joint_t_s1_s2cond_prob[sym_t,sym_s1, sym_s2cond], '\t', s2cond_prob[sym_s2cond])

                if ( s2cond_prob[sym_s2cond]
                     * joint_t_s2cond_prob[sym_t, sym_s2cond]
                     * joint_s1_s2cond_prob[sym_s1, sym_s2cond]
                     * joint_t_s1_s2cond_prob[sym_t, sym_s1, sym_s2cond] > 0 ):

                    local_contrib = (
                        np.log(joint_t_s1_s2cond_prob[sym_t, sym_s1, sym_s2cond])
                        + np.log(s2cond_prob[sym_s2cond])
                        - np.log(joint_t_s2cond_prob[sym_t,sym_s2cond])
                        - np.log(joint_s1_s2cond_prob[sym_s1, sym_s2cond])
                        ) / np.log(2)

                    weighted_contrib = (
                        joint_t_s1_s2cond_prob[sym_t, sym_s1, sym_s2cond]
                        * local_contrib)
                else:
                    weighted_contrib = 0
                total += weighted_contrib

    return total


def _mi_prob(s1_prob, s2_prob, joint_s1_s2_prob):
    """
    MI calculator in the prob domain
    """
    total = np.zeros(1).astype('float128')

    [alph_s1, alph_s2] = np.shape(joint_s1_s2_prob)

    for sym_s1 in range(0, alph_s1):
        for sym_s2 in range(0, alph_s2):

#            print(sym_s1, '\t', sym_s2, '\t', s1_prob[sym_s1], '\t', s2_prob[sym_s2], '\t', joint_s1_s2_prob[sym_s1, sym_s2])

            if ( s1_prob[sym_s1] * s2_prob[sym_s2]
                 * joint_s1_s2_prob[sym_s1, sym_s2] > 0 ):

                local_contrib = (
                    np.log(joint_s1_s2_prob[sym_s1, sym_s2])
                    - np.log(s1_prob[sym_s1])
                    - np.log(s2_prob[sym_s2])
                    ) / np.log(2)

                weighted_contrib = (
                    joint_s1_s2_prob[sym_s1, sym_s2]
                    * local_contrib)
            else:
                weighted_contrib = 0
            total += weighted_contrib

    return total


def _joint_mi(s1, s2, t, alph_s1, alph_s2, alph_t):
    """
    Joint MI calculator in the samples domain
    """

    [s12, alph_s12] = _join_variables(s1, s2, alph_s1, alph_s2)

    t_count = np.zeros(alph_t, dtype=np.int)
    s12_count = np.zeros(alph_s12, dtype=np.int)
    joint_t_s12_count = np.zeros((alph_t, alph_s12), dtype=np.int)

    num_samples = len(t)

    for obs in range(0, num_samples):
        t_count[t[obs]] += 1
        s12_count[s12[obs]] += 1
        joint_t_s12_count[t[obs], s12[obs]] += 1

    t_prob = np.divide(t_count, num_samples).astype('float128')
    s12_prob = np.divide(s12_count, num_samples).astype('float128')
    joint_t_s12_prob = np.divide(joint_t_s12_count, num_samples).astype('float128')

    jmi = _mi_prob(t_prob, s12_prob, joint_t_s12_prob)

    return jmi


def _join_variables(a, b, alph_a, alph_b):

    alph_new = alph_a * alph_b

    if alph_b < alph_a:
        a, b = b, a
        alph_a, alph_b = alph_b, alph_a

    ab = alph_b * a + b

    return ab, alph_new


# TODO fix this - no idea why it does not yield the correct results
#def _try_swap(cur_cond_mut_info, joint_t_s1_s2_prob, joint_s1_s2_prob,
#              joint_t_s2_prob, s2_prob,
#              t_cand, s1_prim, s2_prim, s1_cand, s2_cand,
#              prob_inc, unsuccessful_swaps_row):
##            unsuccessful_swaps_row_local = unsuccessful_swaps_row
##            print("unsuccessful_swaps_row_local: {0}".format(unsuccessful_swaps_row_local))
#            if (joint_t_s1_s2_prob[t_cand, s1_cand, s2_cand] >= prob_inc
#            and joint_t_s1_s2_prob[t_cand, s1_prim, s2_prim] >= prob_inc
#            and joint_s1_s2_prob[s1_cand, s2_cand] >= prob_inc
#            and joint_s1_s2_prob[s1_prim, s2_prim] >= prob_inc):
#
#                joint_t_s1_s2_prob[t_cand, s1_cand, s2_cand] -= prob_inc
#                joint_t_s1_s2_prob[t_cand, s1_prim, s2_prim] -= prob_inc
#                joint_t_s1_s2_prob[t_cand, s1_cand, s2_prim] += prob_inc
#                joint_t_s1_s2_prob[t_cand, s1_prim, s2_cand] += prob_inc
#
#                joint_s1_s2_prob[s1_cand, s2_cand] -= prob_inc
#                joint_s1_s2_prob[s1_prim, s2_prim] -= prob_inc
#                joint_s1_s2_prob[s1_cand, s2_prim] += prob_inc
#                joint_s1_s2_prob[s1_prim, s2_cand] += prob_inc
#
#                # Calculate the cmi after this virtual swap
#                cond_mut_info = _cmi_prob(
#                    s2_prob, joint_t_s2_prob, joint_s1_s2_prob, joint_t_s1_s2_prob)
#
#                # If improved keep it, reset the unsuccessful swap counter
#                if ( cond_mut_info < cur_cond_mut_info ):
#                    cur_cond_mut_info = cond_mut_info
#                    unsuccessful_swaps_row = 0
#                    # TODO: if this swap direction was successful - repeat it !
#                # Else undo the changes, record unsuccessful swap
#                else:
#                    joint_t_s1_s2_prob[t_cand, s1_cand, s2_cand] += prob_inc
#                    joint_t_s1_s2_prob[t_cand, s1_prim, s2_prim] += prob_inc
#                    joint_t_s1_s2_prob[t_cand, s1_cand, s2_prim] -= prob_inc
#                    joint_t_s1_s2_prob[t_cand, s1_prim, s2_cand] -= prob_inc
#
#                    joint_s1_s2_prob[s1_cand, s2_cand] += prob_inc
#                    joint_s1_s2_prob[s1_prim, s2_prim] += prob_inc
#                    joint_s1_s2_prob[s1_cand, s2_prim] -= prob_inc
#                    joint_s1_s2_prob[s1_prim, s2_cand] -= prob_inc
#
#                    unsuccessful_swaps_row += 1
#            else:
#                unsuccessful_swaps_row += 1
#            return unsuccessful_swaps_row # need to return this to make it visible outside
#        # END of a possible try_swap function

def pid_tartu(t, s1, s2, ):

    ''' TODO this doen't do anything in the original code
    # add noise
    noise_level = .05
    for i in range(old_L, L):
        noise = 0
        p = random.random()
        if p < noise_level:
            t[i] = fun_obj.noise(t[i])
    '''
    # update counts
    for i in range(old_L, L):
        if (t[i], s1[i], s2[i]) in counts.keys():
            counts[(t[i], s1[i], s2[i])] += 1
        else:
            counts[(t[i], s1[i], s2[i])] = 1

    # make pdf from counts
    pdf = dict()
    for xyz, c in counts.items():
        pdf[xyz] = c / float(L)

    old_L = L
    # END OF creation of pdf

    print("pdf=", TartuSynergy.sorted_pdf(pdf))
    print("test__solve_time_series(): L = ", L)
    retval = TartuSynergy.solve_PDF(pdf, fun_obj.true_input_distrib(),
                                    fun_obj.true_result_distrib(),
                                    fun_obj.true_CI(),
                                    fun_obj.true_SI(), verbose=verbose)
    optpdf, feas, kkt, CI, SI = retval
    print("CI:", CI, "  SI:", SI, "  sum of marginal eqns violations:",
          feas, "  maximal KKT-system constraint violation:", kkt)

    synergy_tartu.solve_PDF(pdf)
