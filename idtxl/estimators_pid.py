"""Partical information decomposition for discrete random variables.

This module provides an estimator for partial information decomposition
as proposed in

Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014). Quantifying
Unique Information. Entropy, 16(4), 2161–2183. http://doi.org/10.3390/e16042161
"""
import numpy as np
from . import synergy_tartu
from .estimator import Estimator

# TODO add support for multivariate estimation for Tartu and Sydney estimator


class SydneyPID(Estimator):
    """Estimate partial information decomposition of discrete variables.

    Fast implementation of the BROJA partial information decomposition (PID)
    estimator for discrete data (Bertschinger, 2014). The estimator does not
    require JAVA or GPU modules to run.

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

    References

    - Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014).
      Quantifying unique information. Entropy, 16(4), 2161–2183.
      http://doi.org/10.3390/e16042161

    Args:
        settings : dict
            estimation parameters

            - alph_s1 : int - alphabet size of s1
            - alph_s2 : int - alphabet size of s2
            - alph_t : int - alphabet size of t
            - max_unsuc_swaps_row_parm : int - soft limit for virtualised swaps
              based on the number of unsuccessful swaps attempted in a row.
              If there are too many unsuccessful swaps in a row, then it
              will break the inner swap loop; the outer loop decrements the
              size of the probability mass increment and then attemps
              virtualised swaps again with the smaller probability increment.
              The exact number of unsuccessful swaps allowed before breaking
              is the total number of possible swaps (given our alphabet
              sizes) times the control parameter max_unsuc_swaps_row_parm,
              e.g., if the parameter is set to 3, this gives a high degree of
              confidence that nearly (if not) all of the possible swaps have
              been attempted before this soft limit breaks the swap loop.
            - num_reps : int -  number of times the outer loop will halve the
              size of the probability increment used for the virtualised
              swaps. This is in direct correspondence with the number of times
              the empirical data was replicated in your original
              implementation.
            - max_iters : int - provides a hard upper bound on the number of
              times it will attempt to perform virtualised swaps in the inner
              loop. However, this hard limit is (practically) never used as it
              should always hit the soft limit defined above (parameter may be
              removed in the future).
            - verbose : bool [optional] - print output to console
              (default=False)
    """
    def __init__(self, settings):
        try:
            settings['alph_s1']
        except KeyError:
            print('"alph_s1" is missing from the settings dictionary.')
            raise
        try:
            settings['alph_s2']
        except KeyError:
            print('"alph_s2" is missing from the settings dictionary.')
            raise
        try:
            settings['alph_t']
        except KeyError:
            print('"alph_t" is missing from the settings dictionary.')
            raise
        try:
            settings['max_unsuc_swaps_row_parm']
        except KeyError:
            print('"max_unsuc_swaps_row_parm" is missing from the settings '
                  'dictionary.')
            raise
        try:
            settings['num_reps']
        except KeyError:
            print('"num_reps" is missing from the settings dictionary.')
            raise
        if settings['num_reps'] > 63:
            raise ValueError('Number of reps must be 63 or less to prevent '
                             'integer overflow.')
        try:
            settings['max_iters']
        except KeyError:
            print('"max_iters" is missing from the settings dictionary.')
            raise
        self.settings = settings.copy()
        self.settings.setdefault('verbose', False)

    def is_parallel():
        return False

    def is_analytic_null_estimator(self):
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
        s1, s2, t, self.settings = _check_input(s1, s2, t, self.settings)

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
        alph_t = self.settings['alph_t']
        alph_s1 = self.settings['alph_s1']
        alph_s2 = self.settings['alph_s2']
        max_unsuc_swaps_row_parm = self.settings['max_unsuc_swaps_row_parm']
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
            if self.settings['verbose']:
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
        if self.settings['verbose']:
            print('num_reps: {0}'.format(self.settings['num_reps']))
        reps = np.array(np.power(2, range(0, self.settings['num_reps'])))

        # Replication loop
        for rep in reps:
            prob_inc = np.multiply(
                np.float128(max_prob),
                np.divide(np.float128(1), np.float128(rep)))
            # Want to store number of succesive unsuccessful swaps
            unsuccessful_swaps_row = 0
            # SWAP LOOP
            for attempt_swap in range(0, self.settings['max_iters']):
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
        if self.settings['verbose']:
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


class TartuPID(Estimator):
    """Estimate partial information decomposition for two inputs and one output

    Implementation of the partial information decomposition (PID) estimator for
    discrete data. The estimator finds shared information, unique information
    and synergistic information between the two inputs s1 and s2 with respect
    to the output t.

    The algorithm uses exponential cone programming and requires the Python
    package for ECOS: Embedded Cone Solver (https://pypi.python.org/pypi/ecos).

    References:

    - Makkeh, A., Theis, D.O., & Vicente, R. (2017). Bivariate Partial
      Information Decomposition: The Optimization Perspective. Entropy, 19(10),
      530.
    - Makkeh, A., Theis, D.O., & Vicente, R. (2018). BROJA-2PID: A cone
      programming based Partial Information Decomposition estimator. Entropy,
      20(271), https://github.com/Abzinger/BROJA_2PID.

    Args:
        settings : dict
            estimation parameters (with default parameters)

            - verbose : bool [optional] - print output to console
              (default=False)
            - cone_solver : str [optional] - which cone solver to use
              (default='ECOS')
            - solver_args : dict [optional] - solver arguments (default={})
    """

    def __init__(self, settings):
        # get estimation parameters
        self.settings = settings.copy()
        self.settings.setdefault('verbose', False)
        self.settings.setdefault('cone_solver', 'ECOS')
        self.settings.setdefault('solver_args', {'keep_solver_object': False})

    def is_parallel():
        return False

    def is_analytic_null_estimator(self):
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
                estimated decomposition, solver used, numerical error
        """
        s1, s2, t, self.settings = _check_input(s1, s2, t, self.settings)
        pdf = _get_pdf_dict(s1, s2, t)

        retval = synergy_tartu.pid(pdf_dirty=pdf,
                                   cone_solver=self.settings['cone_solver'],
                                   output=int(self.settings['verbose']),
                                   **self.settings['solver_args'])

        results = {
            'num_err': retval['Num_err'],
            'solver': retval['Solver'],
            'shd_s1_s2': retval['SI'],
            'syn_s1_s2': retval['CI'],
            'unq_s1': retval['UIY'],
            'unq_s2': retval['UIZ'],
        }
        return results


def _get_pdf_dict(s1, s2, t):
    # Create dictionary with probability mass function
    counts = dict()
    n_samples = s1.shape[0]

    # Count occurences.
    for i in range(n_samples):
        if (t[i], s1[i], s2[i]) in counts.keys():
            counts[(t[i], s1[i], s2[i])] += 1
        else:
            counts[(t[i], s1[i], s2[i])] = 1

    # Create PMF from counts.
    pmf = dict()
    for xyz, c in counts.items():
        pmf[xyz] = c / float(n_samples)
    return pmf


def _check_input(s1, s2, t, settings):
    """Check input to PID estimators."""
    # Check if inputs are numpy arrays.
    if (type(s1) != np.ndarray or type(s2) != np.ndarray or
            type(t) != np.ndarray):
        raise TypeError('All inputs, s1, s2, t, must be numpy arrays.')

    # In general, IDTxl expects 2D inputs because JIDT/JPYPE only accepts those
    # and we have a multivariate approach, i.e., a vector is a special case of
    # 2D-data. The PID estimators on the other hand, expect 1D data. Squeeze 2D
    # arrays if the dimension of the second axis is 1. Otherwise combine
    # multivariate sources into a single variable for estimation.
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
            settings['alph_s1'] = alph_new
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
            settings['alph_s2'] = alph_new
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
    if type(settings) != dict:
        raise TypeError('The settings argument should be a dictionary.')

    if not issubclass(s1.dtype.type, np.integer):
        raise TypeError('Input s1 (source 1) must be an integer numpy array.')
    if not issubclass(s2.dtype.type, np.integer):
        raise TypeError('Input s2 (source 2) must be an integer numpy array.')
    if not issubclass(t.dtype.type, np.integer):
        raise TypeError('Input t (target) must be an integer numpy array.')

    # Check if variables have equal length.
    if (len(t) != len(s1) or len(t) != len(s2)):
        raise ValueError('Number of samples s1, s2 and t must be equal')

    return s1, s2, t, settings
