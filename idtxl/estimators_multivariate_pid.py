"""Multivariate Partical information decomposition for discrete random variables.

This module provides an estimator for multivariate partial information
decomposition as proposed in

- Makkeh, A. & Gutknecht, A. & Wibral, M. (2020). A Differentiable measure
  for shared information. 1- 27 Retrieved from
  http://arxiv.org/abs/2002.03356
"""
import numpy as np
from . import lattices as lt
from . import pid_goettingen
from .estimator import Estimator
from .estimators_pid import _join_variables

# TODO add support for multivariate estimation for Tartu and Sydney estimator


class SxPID(Estimator):
    """Estimate partial information decomposition for multiple inputs.

    Implementation of the multivariate partial information decomposition (PID)
    estimator for discrete data with (up to 4 inputs) and one output. The
    estimator finds shared information, unique information and synergistic
    information between the multiple inputs s1, s2, ..., sn with respect to the
    output t for each realization (t, s1, ..., sn) and then average them
    according to their distribution weights p(t, s1, ..., sn). Both the
    pointwise (on the realization level) PID and the averaged PID are returned
    (see the 'return' of 'estimate()').

    The algorithm uses recursion to compute the partial information
    decomposition.

    References:

    - Makkeh, A. & Wibral, M. (2020). A differentiable pointwise partial
      Information Decomposition estimator. https://github.com/Abzinger/SxPID.

    Args:
        settings : dict
            estimation parameters (with default parameters)

            - verbose : bool [optional] - print output to console
              (default=False)
    """

    def __init__(self, settings):
        # get estimation parameters
        self.settings = settings.copy()
        self.settings.setdefault('verbose', False)

    def is_parallel():
        return False

    def is_analytic_null_estimator(self):
        return False

    def estimate(self, s, t):
        """
        Args:
            s : list of numpy arrays
                1D arrays containing realizations of a discrete random variable
            t : numpy array
                1D array containing realizations of a discrete random variable

        Returns:
            dict of dict
                {
                 'ptw' -> { realization -> {alpha -> [float, float, float]} }

                 'avg' -> {alpha -> [float, float, float]}
                }
            where the list of floats is ordered
            [informative, misinformative, informative - misinformative]
            ptw stands for pointwise decomposition
            avg stands for average decomposition
        """
        s, t, self.settings = _check_input(s, t, self.settings)
        pdf = _get_pdf_dict(s, t)

        # Read lattices from a file
        # Stored as {
        #             n -> [{alpha -> children}, (alpha_1,...) ]
        #           }
        # children is a list of tuples
        lattices = lt.lattices
        num_source_vars = len(s)
        retval_ptw, retval_avg = pid_goettingen.pid(
            num_source_vars,
            pdf_orig=pdf,
            chld=lattices[num_source_vars][0],
            achain=lattices[num_source_vars][1],
            printing=self.settings['verbose'])

        # TODO AskM: Trivariate: does it make sense to name the alphas
        #    for example shared_syn_s1_s2__syn_s1_s3 ?
        results = {
            'ptw': retval_ptw,
            'avg': retval_avg,
        }
        return results


def _get_pdf_dict(s, t):
    """"Write probability mass function estimated via counting to a dict."""
    # Create dictionary with probability mass function
    counts = dict()
    n_samples = s[0].shape[0]

    # Count occurences.
    for i in range(n_samples):
        key = tuple([s[j][i] for j in range(len(s))]) + (t[i],)
        if key in counts.keys():
            counts[key] += 1
        else:
            counts[key] = 1

    # Create PMF from counts.
    pmf = dict()
    for xyz, c in counts.items():
        pmf[xyz] = c / float(n_samples)
    return pmf


def _check_input(s, t, settings):
    """Check input to PID estimators."""
    # Check if inputs are numpy arrays.
    if type(t) != np.ndarray:
        raise TypeError('Input t must be a numpy array.')
    for i in range(len(s)):
        if type(s[i]) != np.ndarray:
            raise TypeError('All inputs s{0} must be numpy arrays.'.format(i+1))

    # In general, IDTxl expects 2D inputs because JIDT/JPYPE only accepts those
    # and we have a multivariate approach, i.e., a vector is a special case of
    # 2D-data. The PID estimators on the other hand, expect 1D data. Squeeze 2D
    # arrays if the dimension of the second axis is 1. Otherwise combine
    # multivariate sources into a single variable for estimation.
    for i in range(len(s)):
        if s[i].ndim != 1:
            if s[i].shape[1] == 1:
                s[i] = np.squeeze(s[i])
            elif s[i].ndim == 2 and s[i].shape[1] > 1:
                si_joint = s[i][:, 0]
                alph_new = len(np.unique(s[i][:, 0]))
                for col in range(1, s[i].shape[1]):
                    alph_col = len(np.unique(s[i][:, col]))
                    si_joint, alph_new = _join_variables(
                        si_joint, s[i][:, col], alph_new, alph_col)
                settings['alph_s'+str(i+1)] = alph_new
            else:
                raise ValueError('Input source {0} s{0} has to be a 1D or 2D '
                                 'numpy array.'.format(i+1))

    if t.ndim != 1:
        if t.shape[1] == 1:
            t = np.squeeze(t)
        else:  # For now we only allow 1D-targets
            raise ValueError('Input target t has to be a vector '
                             '(t.shape[1]=1).')

    # Check types of remaining inputs.
    if type(settings) != dict:
        raise TypeError('The settings argument should be a dictionary.')
    for i in range(len(s)):
        if not issubclass(s[i].dtype.type, np.integer):
            raise TypeError('Input s{0} (source {0}) must be an integer numpy '
                            'array.'.format(i+1))
    # ^ for
    if not issubclass(t.dtype.type, np.integer):
        raise TypeError('Input t (target) must be an integer numpy array.')

    # Check if variables have equal length.
    for i in range(len(s)):
        if len(t) != len(s[i]):
            raise ValueError('Number of samples s and t must be equal')

    return s, t, settings
