"""Partical information decomposition for discrete random variables.

This module provides an estimator for partial information decomposition
as proposed in

Bertschinger, Rauh, Olbrich, Jost, Ay; Quantifying Unique Information,
Entropy 2014, 16, 2161-2183; doi:10.3390/e16042161

"""
import numpy as np
from . import synergy_tartu
from . import idtxl_exceptions as ex
from .estimator import Estimator
try:
    import jpype as jp
except ImportError as err:
    ex.package_missing(err, 'Jpype is not available on this system. Install it'
                            ' from https://pypi.python.org/pypi/JPype1 to use '
                            'JAVA/JIDT-powered CMI estimation.')

VERBOSE = False

# TODO add support for multivariate estimation for Tartu and Sydney estimator


def pid_frankfurt(self, s1, s2, t, opts):
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
        s1 : numpy array
            1D array containing realizations of a discrete random variable
            (this is the source variable the algorithm calculates the actual
            UI for)
        s2 : numpy array
            1D array containing realizations of a discrete random variable (the
            other source variable)
        t : numpy array
            1D array containing realizations of a discrete random variable
        opts : dict
            estimation parameters

            - 'alphabetsize' - no. values in each variable s1, s2, t
            - 'jarpath' - string with path to JIDT jar file
            - 'iterations' - no. iterations of the estimator

    Returns:
        dict
            estimated decomposition, contains: MI/CMI values computed
            from non-permuted distributions; PID estimates (shared,
            synergistic, unique information); I(target;s1,s2) under permuted
            distribution Q
        dict
            additional information about iterative optimization,
            contains: final permutation Q; opts dictionary; array with
            I(target:s1|s2) for each iteration; array with delta
            I(target:s1|s2) for each iteration; I(target:s1,s2) for each
            iteration

    Note:
        variables names joined by "_" enter a mutual information computation
        together i.e. mi_va1_var2 --> I(var1 : var2). Variables names joined
        directly form a new joint variable
        mi_var1var2_var3 --> I(var3:(var1,var2))
    """
    _check_input(s1, s2, t, opts)
    # make deep copies of input arrays to avoid side effects
    s1_cp = s1.copy()
    s2_cp = s2.copy()
    t_cp = t.copy()

    # get estimation parameters
    try:
        jarpath = opts['jarpath']
    except TypeError:
        print('The opts argument should be a dictionary.')
        raise
    except KeyError:
        print('"jarpath" is missing from the opts dictionary.')
        raise
    try:
        alph_s1 = opts['alph_s1']
    except KeyError:
        print('"alphabetsize" is missing from the opts dictionary.')
        raise
    try:
        alph_s2 = opts['alph_s2']
    except KeyError:
        print('"alphabetsize" is missing from the opts dictionary.')
        raise
    try:
        alph_t = opts['alph_t']
    except KeyError:
        print('"alphabetsize" is missing from the opts dictionary.')
        raise
    try:
        iterations = opts['iterations']
    except KeyError:
        print('"iterations" is missing from the opts dictionary.')
        raise

    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(),
                    '-ea', '-Djava.class.path=' + jarpath, "-Xmx3000M")
    # what if it's there already - do we have to attach to it?

    # transform variables as far as possible outside the loops below
    # (note: only these variables should change when executing the loop)
    target_jA = jp.JArray(jp.JInt, t.ndim)(t.tolist())
    s2_jA = jp.JArray(jp.JInt, s2.ndim)(s2.tolist())
    s1_list = s1_cp.tolist()
    s1_dim = s1_cp.ndim

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
    alph_max = max(alph_s1 * alph_s2, alph_t)

#   mi_calc  = Mi_calc_class(alphabet)
    mi_calc_s1 = Mi_calc_class(alph_max_s1_t)
    mi_calc_s2 = Mi_calc_class(alph_max_s2_t)

#   jointmi_calc = Mi_calc_class(alphabet ** 2)
    jointmi_calc = Mi_calc_class(alph_max)

    print("initialized all estimators")
    cmi_target_s1_cond_s2 = _calculate_cmi(cmi_calc_target_s1_cond_s2,
                                           t_cp, s1_cp, s2_cp)
    jointmi_s1s2_target = _calculate_jointmi(jointmi_calc, s1_cp, s2_cp, t_cp)
#   print("Original joint mutual information: {0}".format(jointmi_s1s2_target))
    mi_target_s1 = _calculate_mi(mi_calc_s1, s1_cp, t_cp)
#   print("Original mutual information I(target:s1): {0}".format(mi_target_s1))
    mi_target_s2 = _calculate_mi(mi_calc_s2, s2_cp, t_cp)
#   print("Original mutual information I(target:s2): {0}".format(mi_target_s2))
    print("Original redundancy - synergy: {0}".format(
                            mi_target_s1 + mi_target_s2 - jointmi_s1s2_target))

    n = t_cp.shape[0]
    reps = iterations + 1
    ind = np.arange(n)
    # collect estimates in each iteration
    cmi_q_target_s1_cond_s2_all = -np.inf * np.ones(reps).astype('float128')
    cmi_q_target_s1_cond_s2_delta = -np.inf * np.ones(reps).astype('float128')
    cmi_q_target_s1_cond_s2_all[0] = cmi_target_s1_cond_s2
    unsuccessful = 0

    for i in range(1, reps):
        s1_new_list = s1_list
        ind_new = ind

        # swapping: pick sample at random, find all other samples that
        # are potential matches (have the same value in target), pick one of
        # the matches for the actual swap
        swap_1 = np.random.randint(n)
        swap_candidates = np.where(t_cp == t_cp[swap_1])[0]
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
    jointmi_q_s1s2_target = _calculate_jointmi(jointmi_calc, s1_final,
                                               s2_cp, t_cp)
    unq_s1 = _get_last_value(cmi_q_target_s1_cond_s2_all)  # Bertschinger, 2014, p. 2163

    # NEED TO REINITIALISE the estimator
    cmi_calc_target_s2_cond_s1 = Cmi_calc_class(alph_t, alph_s2, alph_s1)
    unq_s2 = _calculate_cmi(cmi_calc_target_s2_cond_s1, t_cp, s2_cp, s1_final)  # Bertschinger, 2014, p. 2166
    syn_s1s2 = jointmi_s1s2_target - jointmi_q_s1s2_target  # Bertschinger, 2014, p. 2163
    shd_s1s2 = mi_target_s1 + mi_target_s2 - jointmi_q_s1s2_target  # Bertschinger, 2014, p. 2167

    estimate = {
        'unq_s1': unq_s1,
        'unq_s2': unq_s2,
        'shd_s1s2': shd_s1s2,
        'syn_s1s2': syn_s1s2,
        'jointmi_q_s1s2_target': jointmi_q_s1s2_target,
        'orig_cmi_target_s1_cond_s2': cmi_target_s1_cond_s2,
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
        'opts': opts
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
        cmi_calc (JIDT estimator object): JIDT estimator for conditio-
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
        cmi_calc (JIDT estimator object): JIDT estimator for conditio-
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
        mi_calc (JIDT estimator object): JIDT estimator for mutual
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
        jointmi_calc (JIDT estimator object): JIDT estimator for
            mutual information
        var_1, var_2, var_3 (1D numpy array): realizations of some
            discrete random variables

    Returns:
        double: mutual information between all three input variables
    """
    mUtils = jp.JPackage('infodynamics.utils').MatrixUtils
    # speed critical line ?
    s12 = mUtils.computeCombinedValues(
                jp.JArray(jp.JInt, 2)(np.column_stack((s1, s2)).tolist()), 2)
#    [s12, alph_joined] = _join_variables(s1, s2, 2, 2)
    jointmi_calc.initialise()
#    jointmi_calc.addObservations(
#                           jp.JArray(jp.JInt, s12.T.ndim)(s12.T.tolist()),
#                           jp.JArray(jp.JInt, target.ndim)(target.tolist()))
    jointmi_calc.addObservations(
                            s12,
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
    ind = np.where(x > -np.inf)[0]
    try:
        return x[ind[-1]]
    except IndexError:
        print('Couldn not find a value that is not -inf.')
        return np.NaN


class SydneyPID(Estimator):
    """Estimate partial information decomposition of discrete variables.

    Fast implementation of the partial information decomposition (PID)
    estimator for discrete data. The estimator does not require JAVA or GPU
    modules to run.

    The estimator finds shared information, unique information and
    synergistic information between the two inputs s1 and s2 with respect to
    the output t.

    Improved version with larger initial swaps and checking for convergence of
    both the unique information from sources 1 and 2. The function counts the
    empirical observations, calculates probabilities and the initial CMI, then
    does the vitrualised swaps until it has converged, and finally calculates
    the PID. The virtualised swaps stage contains two loops. An inner loop
    which actually does the virtualised swapping, keeping the changes if the
    CMI decreases; and an outer loop which decreases the size of the
    probability mass increment the virtualised swapping utilises.

    Args:
        opts : dict
            estimation parameters

            - 'alph_s1' - alphabet size of s1
            - 'alph_s2' -  alphabet size of s2
            - 'alph_t' - alphabet size of t
            - 'max_unsuc_swaps_row_parm' - soft limit for virtualised swaps
              based on the number of unsuccessful swaps attempted in a row.
              If there are too many unsuccessful swaps in a row, then it
              will break the inner swap loop; the outer loop decrements the
              size of the probability mass increment and then attemps
              virtualised swaps again with the smaller probability increment.
              The exact number of unsuccessful swaps allowed before breaking
              is the total number of possible swaps (given our alphabet
              sizes) times the control parameter 'max_unsuc_swaps_row_parm',
              e.g., if the parameter is set to 3, this gives a high degree of
              confidence that nearly (if not) all of the possible swaps have
              been attempted before this soft limit breaks the swap loop.
            - 'num_reps' -  number of times the outer loop will halve the
              size of the probability increment used for the virtualised
              swaps. This is in direct correspondence with the number of times
              the empirical data was replicated in your original
              implementation.
            - 'max_iters' - provides a hard upper bound on the number of times
              it will attempt to perform virtualised swaps in the inner loop.
              However, this hard limit is (practically) never used as it should
              always hit the soft limit defined above (parameter may be removed
              in the future).
    """
    def __init__(self, opts):
        try:
            opts['alph_s1']
        except KeyError:
            print('"alph_s1" is missing from the opts dictionary.')
            raise
        try:
            opts['alph_s2']
        except KeyError:
            print('"alph_s2" is missing from the opts dictionary.')
            raise
        try:
            opts['alph_t']
        except KeyError:
            print('"alph_t" is missing from the opts dictionary.')
            raise
        try:
            opts['max_unsuc_swaps_row_parm']
        except KeyError:
            print('"max_unsuc_swaps_row_parm" is missing from the opts '
                  'dictionary.')
            raise
        try:
            opts['num_reps']
        except KeyError:
            print('"num_reps" is missing from the opts dictionary.')
            raise
        if opts['num_reps'] > 63:
            raise ValueError('Number of reps must be 63 or less to prevent '
                             'integer overflow.')
        try:
            opts['max_iters']
        except KeyError:
            print('"max_iters" is missing from the opts dictionary.')
            raise
        self.opts = opts

    def is_parallel():
        return False

    def estimate(self, s1, s2, t):
        """
        Args:
            s1 : numpy array
                1D array containing realizations of a discrete random variable
            s2 : numpy array
                1D array containing realizations of a discrete random variable
            t : numpy array
                1D array containing realizations of a discrete random variable

        Returns:
            dict
                estimated decomposition, contains the joint distribution,
                unique, shared, and synergistic information
        """
        s1, s2, t, self.opts = _check_input(s1, s2, t, self.opts)

        # Check if float128 is supported by the architecture
        try:
            np.float128()
        except AttributeError as err:
            if "'module' object has no attribute 'float128'" == err.args[0]:
                raise RuntimeError(
                        'This system doesn''t seem to support float128 '
                        '(requirement for using the Sydney PID-estimator.')
            else:
                raise

        # -- DEFINE PARAMETERS -- #

        num_samples = len(t)
        alph_t = self.opts['alph_t']
        alph_s1 = self.opts['alph_s1']
        alph_s2 = self.opts['alph_s2']
        max_unsuc_swaps_row_parm = self.opts['max_unsuc_swaps_row_parm']
        # Max swaps = number of possible swaps * control parameter
        num_pos_swaps = alph_t * alph_s1 * (alph_s1-1) * alph_s2 * (alph_s2-1)
        max_unsuc_swaps_row = np.floor(num_pos_swaps *
                                       max_unsuc_swaps_row_parm)

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
            joint_s1_s2_count[s1[obs], s2[obs]] += 1
            joint_t_s1_s2_count[t[obs], s1[obs], s2[obs]] += 1
        #    min_joint_nonzero_count = np.min(
        #   			np.min(
        #   			np.min(
        #   			joint_t_s1_s2_count[np.nonzero(joint_t_s1_s2_count)])))

        max_joint_nonzero_count = np.max(
                        joint_t_s1_s2_count[np.nonzero(joint_t_s1_s2_count)])

        # Fixed probabilities
        t_prob = np.divide(t_count, num_samples).astype('float128')
        s1_prob = np.divide(s1_count, num_samples).astype('float128')
        s2_prob = np.divide(s2_count, num_samples).astype('float128')
        joint_t_s1_prob = np.divide(joint_t_s1_count,
                                    num_samples).astype('float128')
        joint_t_s2_prob = np.divide(joint_t_s2_count,
                                    num_samples).astype('float128')

        # Variable probabilities
        joint_s1_s2_prob = np.divide(joint_s1_s2_count,
                                     num_samples).astype('float128')
        joint_t_s1_s2_prob = np.divide(joint_t_s1_s2_count,
                                     num_samples).astype('float128')
        max_prob = np.max(joint_t_s1_s2_prob[np.nonzero(joint_t_s1_s2_prob)])

    #    # make copies of the variable probabilities for independent second
    #    # optimization and comparison of KLDs for convergence check:
    #    # KLDs should initially rise and then fall when close to the minimum
    #    joint_s1_s2_prob_alt = joint_s1_s2_prob.copy()
    #    joint_t_s1_s2_prob_alt = joint_t_s1_s2_prob.copy()

        # -- VIRTUALISED SWAPS -- #

        # Calculate the initial cmi's and store them
        cond_mut_info1 = self._cmi_prob(
            s2_prob, joint_t_s2_prob, joint_s1_s2_prob, joint_t_s1_s2_prob)
        cur_cond_mut_info1 = cond_mut_info1

        joint_s2_s1_prob = np.transpose(joint_s1_s2_prob)
        joint_t_s2_s1_prob = np.ndarray.transpose(joint_t_s1_s2_prob,
                                                  [0, 2, 1])

        cond_mut_info2 = self._cmi_prob(
            s1_prob, joint_t_s1_prob, joint_s2_s1_prob, joint_t_s2_s1_prob)
        cur_cond_mut_info2 = cond_mut_info2

        # sanity check: the curr cmi must be smaller than the joint, else
        # something is fishy
        jointmi_s1s2_t = self._joint_mi(s1, s2, t, alph_s1, alph_s2, alph_t)

        if cond_mut_info1 > jointmi_s1s2_t:
            raise ValueError('joint MI {0} smaller than cMI {1}'
                             ''.format(jointmi_s1s2_t, cond_mut_info1))
        else:
            if VERBOSE:
                print('Passed sanity check on jMI and cMI')

        # Declare reps array of repeated doubling to half the prob_inc
        # WARNING: num_reps greater than 63 results in integer overflow
        # TODO: in principle we could divide the increment until we run out of
        # fp precision, e.g. we can get some extra reps by not starting with a
        # swap of size 1/n but soemthing larger, by adding as many steps here
        # as we are powers of 2 larger in the max probability than 1/n and by
        # starting with swaps in the size of the max probability this will keep
        # almost all of the bins of the joint pdf from being swapped but they
        # will joint swapping later, or after being swapped into unfortunatley
        # this does not run with the current code as it uses large powers of
        # integers another idea would be to decrement by something slightly
        # smaller than 2
    #    num_reps = num_reps + np.int32(np.floor(np.log(max_joint_nonzero_count)/np.log(2)))
        if VERBOSE:
            print('num_reps: {0}'.format(self.opts['num_reps']))
        reps = np.array(np.power(2, range(0, self.opts['num_reps'])))

        # Replication loop
        for rep in reps:
            prob_inc = np.multiply(
                np.float128(max_prob),
                np.divide(np.float128(1), np.float128(rep)))
            # Want to store number of succesive unsuccessful swaps
            unsuccessful_swaps_row = 0
            # SWAP LOOP
            for attempt_swap in range(0, self.opts['max_iters']):
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
                if (joint_t_s1_s2_prob[t_cand, s1_cand, s2_cand] >= prob_inc and
                        joint_t_s1_s2_prob[t_cand, s1_prim, s2_prim] >= prob_inc and
                        joint_s1_s2_prob[s1_cand, s2_cand] >= prob_inc and
                        joint_s1_s2_prob[s1_prim, s2_prim] >= prob_inc):

                    joint_t_s1_s2_prob[t_cand, s1_cand, s2_cand] -= prob_inc
                    joint_t_s1_s2_prob[t_cand, s1_prim, s2_prim] -= prob_inc
                    joint_t_s1_s2_prob[t_cand, s1_cand, s2_prim] += prob_inc
                    joint_t_s1_s2_prob[t_cand, s1_prim, s2_cand] += prob_inc

                    joint_s1_s2_prob[s1_cand, s2_cand] -= prob_inc
                    joint_s1_s2_prob[s1_prim, s2_prim] -= prob_inc
                    joint_s1_s2_prob[s1_cand, s2_prim] += prob_inc
                    joint_s1_s2_prob[s1_prim, s2_cand] += prob_inc

                    # Calculate the cmi after this virtual swap
                    cond_mut_info1 = self._cmi_prob(s2_prob,
                                                    joint_t_s2_prob,
                                                    joint_s1_s2_prob,
                                                    joint_t_s1_s2_prob)
                    cond_mut_info2 = self._cmi_prob(s2_prob,
                                                    joint_t_s2_prob,
                                                    joint_s1_s2_prob,
                                                    joint_t_s1_s2_prob)

                    # If at least one of the cmis is improved keep it,
                    # reset the unsuccessful swap counter
                    if (cond_mut_info1 < cur_cond_mut_info1 or
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
        mi_target_s1 = self._mi_prob(t_prob, s1_prob, joint_t_s1_prob)
        mi_target_s2 = self._mi_prob(t_prob, s2_prob, joint_t_s2_prob)
        jointmi_s1s2_target = self._joint_mi(s1, s2, t, alph_s1, alph_s2,
                                             alph_t)
        if VERBOSE:
            print('jointmi_s1s2_target: {0}'.format(jointmi_s1s2_target))

        # PID terms
        unq_s1 = cond_mut_info1
        shd_s1_s2 = mi_target_s1 - unq_s1
        unq_s2 = mi_target_s2 - shd_s1_s2
        syn_s1_s2 = jointmi_s1s2_target - unq_s1 - unq_s2 - shd_s1_s2

        # Return scalars instead of 1-element numpy arrays
        return {'joint_mi_s1s2_t': jointmi_s1s2_target[0],
                'unq_s1': unq_s1[0],
                'unq_s2': unq_s2[0],
                'shd_s1_s2': shd_s1_s2[0],
                'syn_s1_s2': syn_s1_s2[0]}

    def _cmi_prob(self, s2cond_prob, joint_t_s2cond_prob,
                  joint_s1_s2cond_prob, joint_t_s1_s2cond_prob):
        total = np.zeros(1).astype('float128')

        [alph_t, alph_s1, alph_s2cond] = np.shape(joint_t_s1_s2cond_prob)

        for sym_s1 in range(0, alph_s1):
            for sym_s2cond in range(0, alph_s2cond):
                for sym_t in range(0, alph_t):

                    if (s2cond_prob[sym_s2cond] *
                            joint_t_s2cond_prob[sym_t, sym_s2cond] *
                            joint_s1_s2cond_prob[sym_s1, sym_s2cond] *
                            joint_t_s1_s2cond_prob[sym_t, sym_s1, sym_s2cond] >
                            0):

                        local_contrib = (
                               np.log(joint_t_s1_s2cond_prob[sym_t, sym_s1,
                                                             sym_s2cond]) +
                               np.log(s2cond_prob[sym_s2cond]) -
                               np.log(joint_t_s2cond_prob[sym_t, sym_s2cond]) -
                               np.log(joint_s1_s2cond_prob[sym_s1, sym_s2cond])
                                       ) / np.log(2)

                        weighted_contrib = (
                            joint_t_s1_s2cond_prob[sym_t, sym_s1, sym_s2cond] *
                            local_contrib)
                    else:
                        weighted_contrib = 0
                    total += weighted_contrib

        return total

    def _mi_prob(self, s1_prob, s2_prob, joint_s1_s2_prob):
        """MI estimator in the prob domain."""
        total = np.zeros(1).astype('float128')
        [alph_s1, alph_s2] = np.shape(joint_s1_s2_prob)

        for sym_s1 in range(0, alph_s1):
            for sym_s2 in range(0, alph_s2):

                if (s1_prob[sym_s1] * s2_prob[sym_s2] *
                        joint_s1_s2_prob[sym_s1, sym_s2] > 0):

                    local_contrib = (
                        np.log(joint_s1_s2_prob[sym_s1, sym_s2]) -
                        np.log(s1_prob[sym_s1]) -
                        np.log(s2_prob[sym_s2])
                                    ) / np.log(2)

                    weighted_contrib = (joint_s1_s2_prob[sym_s1, sym_s2] *
                                        local_contrib)
                else:
                    weighted_contrib = 0
                total += weighted_contrib

        return total

    def _joint_mi(self, s1, s2, t, alph_s1, alph_s2, alph_t):
        """Joint MI estimator in the samples domain."""

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
        joint_t_s12_prob = np.divide(joint_t_s12_count,
                                     num_samples).astype('float128')

        return self._mi_prob(t_prob, s12_prob, joint_t_s12_prob)


def _join_variables(a, b, alph_a, alph_b):
    """Join two variables into a new one."""
    alph_new = alph_a * alph_b

    if alph_b < alph_a:
        a, b = b, a
        alph_a, alph_b = alph_b, alph_a

    ab = alph_b * a + b

    return ab, alph_new


# TODO fix this - no idea why it does not yield the correct results
# def _try_swap(cur_cond_mut_info, joint_t_s1_s2_prob, joint_s1_s2_prob,
#              joint_t_s2_prob, s2_prob,
#              t_cand, s1_prim, s2_prim, s1_cand, s2_cand,
#              prob_inc, unsuccessful_swaps_row):
# #            unsuccessful_swaps_row_local = unsuccessful_swaps_row
# #            print("unsuccessful_swaps_row_local: {0}".format(unsuccessful_swaps_row_local))
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
#                cond_mut_info = self._cmi_prob(
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


class TartuPID(Estimator):
    """Estimate partial information decomposition for two inputs and one output

    Fast implementation of the partial information decomposition (PID)
    estimator for discrete data. The estimator does require a gurobi
    installation.

    The estimator finds shared information, unique information and
    synergistic information between the two inputs s1 and s2 with respect to
    the output t.

    Improved version with larger initial swaps and checking for convergence of
    both the unique information from sources 1 and 2.

    Args:
        opts : dict
            estimation parameters (with default parameters)

            - 'get_sorted_pdf' -False
            - 'true_pdf' - None
            - 'true_result' - None
            - 'true_CI' - None
            - 'true_SI' - None
            - 'feas_eps' - 1.e-10
            - 'kkt_eps' - 1.e-5
            - 'feas_eps_2' - 1.e-6
            - 'kkt_eps_2' - 0.01
            - 'kkt_search_eps' - 0.5
            - 'max_zero_probability' - 1.e-5
            - 'verbose' - False
    """

    def __init__(self, opts):
        # get estimation parameters
        # get_sorted_pdf = opts.get('sorted_pdf', False)
        # true_pdf = opts.get('true_pdf', None)
        # true_result = opts.get('true_result', None)
        # true_CI = opts.get('true_CI', None)
        # true_SI = opts.get('true_SI', None)
        # feas_eps = opts.get('feas_eps', 1.e-10)
        # kkt_eps = opts.get('kkt_eps', 1.e-5)
        # feas_eps_2 = opts.get('feas_eps_2', 1.e-6)
        # kkt_eps_2 = opts.get('kkt_eps_2', .01)
        # kkt_search_eps = opts.get('kkt_search_eps', .5)
        # max_zero_probability = opts.get('max_zero_probability', 1.e-5)
        # verbose = opts.get('verbose', False)
        opts.setdefault('sorted_pdf', False)
        opts.setdefault('true_pdf', None)
        opts.setdefault('true_result', None)
        opts.setdefault('true_CI', None)
        opts.setdefault('true_SI', None)
        opts.setdefault('feas_eps', 1.e-10)
        opts.setdefault('kkt_eps', 1.e-5)
        opts.setdefault('feas_eps_2', 1.e-6)
        opts.setdefault('kkt_eps_2', .01)
        opts.setdefault('kkt_search_eps', .5)
        opts.setdefault('max_zero_probability', 1.e-5)
        opts.setdefault('verbose', False)
        self.opts = opts

    def is_parallel():
        return False

    def estimate(self, s1, s2, t):
        """
        Args:
            s1 : numpy array
                1D array containing realizations of a discrete random variable
            s2 : numpy array
                1D array containing realizations of a discrete random variable
            t : numpy array
                1D array containing realizations of a discrete random variable


        Returns:
            dict
                estimated decomposition, contains the optimised PDF, shared,
                and synergistic information
        """
        s1, s2, t, self.opts = _check_input(s1, s2, t, self.opts)
        counts = dict()
        n_samples = s1.shape[0]

        # count occurences
        for i in range(n_samples):
            if (t[i], s1[i], s2[i]) in counts.keys():
                counts[(t[i], s1[i], s2[i])] += 1
            else:
                counts[(t[i], s1[i], s2[i])] = 1

        # make pdf from counts
        pdf = dict()
        for xyz, c in counts.items():
            pdf[xyz] = c / float(n_samples)

        retval = synergy_tartu.solve_PDF(pdf,
                                         self.opts['true_pdf'],
                                         self.opts['true_result'],
                                         self.opts['true_CI'],
                                         self.opts['true_SI'],
                                         self.opts['feas_eps'],
                                         self.opts['kkt_eps'],
                                         self.opts['feas_eps_2'],
                                         self.opts['kkt_eps_2'],
                                         self.opts['kkt_search_eps'],
                                         self.opts['max_zero_probability'],
                                         self.opts['verbose'])
        optpdf, feas, kkt, CI, SI, UI_s1, UI_s2 = retval
        res = {
            'kkt': kkt,
            'feas': feas,
            'optpdf': optpdf,
            'shd_s1_s2': SI,
            'syn_s1_s2': CI,
            'unq_s1': UI_s1,
            'unq_s2': UI_s2,
        }
        if self.opts['sorted_pdf']:
            res['sorted_pdf'] = synergy_tartu.sorted_pdf(pdf)
        return res


def _check_input(s1, s2, t, opts):
    """Check input to PID estimators."""
#    if s1.ndim != 1 or s2.ndim != 1 or t.ndim != 1:
#        raise ValueError('Inputs s1, s2, target have to be vectors'
#                         '(1D-arrays).')
#    if (len(t) != len(s1) or len(t) != len(s2)):
#        raise ValueError('Number of samples s1, s2 and t must be equal')

    # In general, IDTxl expects 2D inputs because JIDT/JPYPE only accepts those
    # and we have a multivariate approach, i.e., a vector is a special case of
    # 2D-data. Squeeze 2D arrays if the dimension of the second axis is 1.
    # Otherwise combine multivariate sources into a single variable for
    # estimation.

    if (type(s1) != np.ndarray or type(s2) != np.ndarray or
            type(t) != np.ndarray):
        raise TypeError('All inputs, s1, s2, t, must be numpy arrays.')

    # Convert IDTxl 2D vectors to 1D arrays. Remove unneeded axis or combine
    # multivariate inputs into a single variable.
    if s1.ndim != 1:
        if s1.shape[1] == 1:
            s1 = np.squeeze(s1)
        elif s1.ndim == 2 and s1.shape[1] > 1:
            s1_joint = s1[:, 0]
            alph_new = len(np.unique(s1[:, 0]))
            for col in range(1, s1.shape[1]):
                alph_col = len(np.unique(s1[:, col]))
                s1_joint, alph_new = _join_variables(s1_joint, s1[:, col],
                                                     alph_new, alph_col)
            opts['alph_s1'] = alph_new
        else:
            raise ValueError('Input source 1 s1 has to be a 1D or 2D numpy '
                             'array.')

    if s2.ndim != 1:
        if s2.shape[1] == 1:
            s2 = np.squeeze(s2)
        elif s2.ndim == 2 and s2.shape[1] > 1:
            s2_joint = s2[:, 0]
            alph_new = len(np.unique(s2[:, 0]))
            for col in range(1, s2.shape[1]):
                alph_col = len(np.unique(s2[:, col]))
                s2_joint, alph_new = _join_variables(s2_joint, s2[:, col],
                                                     alph_new, alph_col)
            opts['alph_s2'] = alph_new
        else:
            raise ValueError('Input source 2 s2 has to be a 1D or 2D numpy '
                             'array.')
    if t.ndim != 1:
        if t.shape[1] == 1:
            t = np.squeeze(t)
        else:  # For now we only allow 1D-targets
            raise ValueError('Input target t has to be a vector '
                             '(t.shape[1]=1).')

    # Check types of remaining inputs.
    if type(opts) != dict:
        raise TypeError('The opts argument should be a dictionary.')

    if not issubclass(s1.dtype.type, np.integer):
        raise TypeError('Input s1 (source 1) must be an integer numpy array.')
    if not issubclass(s2.dtype.type, np.integer):
        raise TypeError('Input s2 (source 2) must be an integer numpy array.')
    if not issubclass(t.dtype.type, np.integer):
        raise TypeError('Input t (target) must be an integer numpy array.')

    return s1, s2, t, opts
