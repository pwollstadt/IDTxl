"""Shared exclusion partial information decomposition (SxPID)."""
import numpy as np
import math
from itertools import chain, combinations
from . import idtxl_exceptions as ex

try:
    from prettytable import PrettyTable
except ImportError as err:
    ex.package_missing(
        err,
        "PrettyTable is not available on this system. Install it from "
        "https://pypi.org/project/PrettyTable/ to use the Goettinge PID "
        "estimator.",
    )


class Lattice:
    """Generates the redundancy lattice for 'n' sources
    The algebraic structure on which partial information decomposition is
    build on.
    """

    def __init__(self, n):
        self.n = n
        self.lis = [i for i in range(1, self.n + 1)]

    # ^ _init_()

    def powerset(self):
        return chain.from_iterable(
            combinations(self.lis, r) for r in range(1, len(self.lis) + 1)
        )

    # ^ powerset()

    def less_than(self, beta, alpha):
        """compare whether an antichain beta is smaller than antichain
        alpha"""
        return all(any(frozenset(b) <= frozenset(a) for b in beta) for a in alpha)

    # ^compare()s

    def comparable(self, a, b):
        return a < b or a > b

    # ^comparable()

    def antichain(self):
        """Generates the nodes (antichains) of the lattice"""
        # dummy expensive function might use dit or networkx functions
        assert self.n < 5, "antichain(n): number of sources should be less than 5"
        achain = []
        for r in range(1, math.floor((2**self.n - 1) / 2) + 2):
            # enumerate the power set of the powerset
            for alpha in combinations(self.powerset(), r):
                flag = 1
                # check if alpha is an antichain
                for a in list(alpha):
                    for b in list(alpha):
                        if a < b and self.comparable(frozenset(a), frozenset(b)):
                            flag = 0
                        # ^if
                    # ^for b
                # ^for a
                if flag:
                    achain.append(alpha)
            # ^for alpha
        # ^for r
        return achain

    # ^antichain()

    def children(self, alpha, achain):
        """Enumerates the direct nodes (antichains) ordered by the node
        (antichain) 'alpha'"""
        chl = []
        downset = [
            beta for beta in achain if self.less_than(beta, alpha) and beta != alpha
        ]
        for beta in downset:
            if all(
                not self.less_than(beta, gamma) for gamma in downset if gamma != beta
            ):
                chl.append(beta)
            # ^if
        # ^for beta
        return chl

    # ^children()


# ^Lattice()

# ---------------
# pi^+(t:alpha)
#    and
# pi^-(t:alpha)
# ---------------


def powerset(m):
    lis = [i for i in range(1, m + 1)]
    return chain.from_iterable(combinations(lis, r) for r in range(1, len(lis) + 1))


# ^powerset()


def marg(pdf, rlz, uset):
    """compute the marginal probability mass
    e.g. p(t,s1,s2)"""
    idxs = [idx - 1 for idx in list(uset)]
    summ = 0.0
    for k in pdf.keys():
        if all(k[idx] == rlz[idx] for idx in idxs):
            summ += pdf[k]
    # ^for
    return summ


# ^marg()


def prob(n, pdf, rlz, gamma, target=False):
    """Compute the Probability mass on a  lattice node
    e.g. node = {1}{2} p(s1 \cup s2) using inclusion-exclusion"""
    m = len(gamma)
    pset = powerset(m)
    summ = 0
    for idxs in pset:
        if target:
            uset = frozenset((n + 1,))
        else:
            uset = frozenset(())
        # ^if
        for i in list(idxs):
            uset |= frozenset(gamma[i - 1])
        # ^for i
        summ += (-1) ** (len(idxs) + 1) * marg(pdf, rlz, uset)
    # ^for idxs
    return summ


# ^prob()


def differs(n, pdf, rlz, alpha, chl, target=False):
    """Compute the probability mass difference
    For a node 'alpha' and any child gamma of alpha it computes p(gamma) -
    p(alpha) for all gamma"""
    if chl == [] and target:
        base = prob(n, pdf, rlz, [()], target) / prob(n, pdf, rlz, alpha, target)
    else:
        base = prob(n, pdf, rlz, alpha, target)
    # ^if bottom
    temp_diffs = [prob(n, pdf, rlz, gamma, target) - base for gamma in chl]
    temp_diffs.sort()
    return [base] + temp_diffs


# ^differs()


def sgn(num_chld):
    """Recurrsive function that generates the signs (+ or -) for the
    inclusion-exculison principle"""
    if num_chld == 0:
        return np.array([+1])
    else:
        return np.concatenate((sgn(num_chld - 1), -sgn(num_chld - 1)), axis=None)
    # ^if bottom


# ^ sgn()


def vec(num_chld, diffs):
    """Recurrsive function that returns a numpy vector used in evaluating
       the moebuis inversion (compute the PPID atoms)
    Args:
        num_chld: int - the number of the children of alpha: (gamma_1,...,
                  gamma_{num_chld})
        diffs: list of floats - vector of probability differences (d_i)_i
               where d_i = p(gamma_i) - p(alpha) and d_0 = p(alpha)
    """
    # print(diffs)
    if num_chld == 0:
        return np.array([diffs[0]])
    else:
        temp = vec(num_chld - 1, diffs) + diffs[num_chld] * np.ones(2 ** (num_chld - 1))
        return np.concatenate((vec(num_chld - 1, diffs), temp), axis=None)
    # ^if bottom


# ^vec()


def pi_plus(n, pdf, rlz, alpha, chld, achain):
    """Compute the informative PPID"""
    diffs = differs(n, pdf, rlz, alpha, chld[tuple(alpha)], False)
    return np.dot(sgn(len(chld[alpha])), -np.log2(vec(len(chld[alpha]), diffs)))


# ^pi_plus()


def pi_minus(n, pdf, rlz, alpha, chld, achain):
    """Compute the misinformative PPID"""
    diffs = differs(n, pdf, rlz, alpha, chld[alpha], True)
    if chld[alpha] == []:
        return np.dot(sgn(len(chld[alpha])), np.log2(vec(len(chld[alpha]), diffs)))
    else:
        return np.dot(sgn(len(chld[alpha])), -np.log2(vec(len(chld[alpha]), diffs)))
    # ^if bottom


# ^pi_minus()


def pid(n, pdf_orig, chld, achain, printing=False):
    """Estimate partial information decomposition for 'n' inputs and one output

    Implementation of the partial information decomposition (PID) estimator for
    discrete data. The estimator finds shared information, unique information
    and synergistic information between the two, three, or four inputs with
    respect to the output t.

    P.S. The implementation can be extended to any number 'n' of variables if
    their corresponding redundancy lattice is provided ( check Lattice() )

    Args:
            n : int - number of pid sources
            pdf_orig : dict - the original joint distribution of the inputs and
                       the output (realizations are the keys). It doesn't have
                       to be a full support distribution, i.e., it can contain
                       realizations with 'zero' mass probability
            chld : dict - list of children for each node in the redundancy
                   lattice (nodes are the keys)
            achain : tuple - tuple of all the nodes (antichains) in the
                     redundacy lattice
            printing: Bool - If true prints the results using PrettyTables

    Returns:
            tuple
                pointwise decomposition, averaged decomposition
    """
    assert (
        type(pdf_orig) is dict
    ), "pid_goettingen.pid(pdf, chld, achain): pdf must be a dictionary"
    assert (
        type(chld) is dict
    ), "pid_goettingen.pid(pdf, chld, achain): chld must be a dictionary"
    assert (
        type(achain) is list
    ), "pid_goettingen.pid(pdf, chld, achain): pdf must be a list"

    if __debug__:
        sum_p = 0.0
        for k, v in pdf_orig.items():
            assert (
                type(k) is tuple
            ), "pid_goettingen.pid(pdf, chld, achain): pdf keys must be tuples"
            assert len(k) < 6, (
                "pid_goettingen.pid(pdf, chld, achain): pdf keys must be tuples"
                "of length at most 5"
            )
            assert type(v) is float or (
                type(v) == int and v == 0
            ), "pid_goettingen.pid(pdf, chld, achain): pdf values must be floats"
            assert (
                v > -0.1
            ), "pid_goettingen.pid(pdf, chld, achain): pdf values must be nonnegative"
            sum_p += v
        # ^for

        assert abs(sum_p - 1) < 1.0e-7, (
            "pid_goettingen.pid(pdf, chld, achain): pdf keys must sum up to 1"
            "(tolerance of precision is 1.e-7)"
        )
    # ^if debug

    assert (
        type(printing) is bool
    ), "pid_goettingen.pid(pdf, chld, achain, printing): printing must be a bool"

    # Remove the impossible realization
    pdf = {k: v for k, v in pdf_orig.items() if v > 1.0e-300}

    # Initialize the output where
    # ptw = { rlz -> { alpha -> pi_alpha } }
    # avg = { alpha -> PI_alpha }
    ptw = dict()
    # avg = defaultdict(lambda : [0.,0.,0.])
    avg = dict()
    # Compute and store the (+, -, +-) atoms
    for rlz in pdf.keys():
        ptw[rlz] = dict()
        for alpha in achain:
            piplus = pi_plus(n, pdf, rlz, alpha, chld, achain)
            piminus = pi_minus(n, pdf, rlz, alpha, chld, achain)
            ptw[rlz][alpha] = (piplus, piminus, piplus - piminus)
            # avg[alpha][0] += pdf[rlz]*ptw[rlz][alpha][0]
            # avg[alpha][1] += pdf[rlz]*ptw[rlz][alpha][1]
            # avg[alpha][2] += pdf[rlz]*ptw[rlz][alpha][2]

        # ^for
    # ^for
    # compute and store the average of the (+, -, +-) atoms
    for alpha in achain:
        avgplus = 0.0
        avgminus = 0.0
        avgdiff = 0.0
        for rlz in pdf.keys():
            avgplus += pdf[rlz] * ptw[rlz][alpha][0]
            avgminus += pdf[rlz] * ptw[rlz][alpha][1]
            avgdiff += pdf[rlz] * ptw[rlz][alpha][2]
            avg[alpha] = (avgplus, avgminus, avgdiff)
        # ^for
    # ^for

    # Print the result if asked
    if printing:
        table = PrettyTable()
        table.field_names = ["RLZ", "Atom", "pi+", "pi-", "pi"]
        for rlz in pdf.keys():
            count = 0
            for alpha in achain:
                stalpha = ""
                for a in alpha:
                    stalpha += "{"
                    for i in a:
                        stalpha += str(i)
                    # ^for i
                    stalpha += "}"
                # ^for a
                if count == 0:
                    table.add_row(
                        [
                            str(rlz),
                            stalpha,
                            str(ptw[rlz][alpha][0]),
                            str(ptw[rlz][alpha][1]),
                            str(ptw[rlz][alpha][2]),
                        ]
                    )
                else:
                    table.add_row(
                        [
                            " ",
                            stalpha,
                            str(ptw[rlz][alpha][0]),
                            str(ptw[rlz][alpha][1]),
                            str(ptw[rlz][alpha][2]),
                        ]
                    )
                count += 1
            # ^for alpha
            table.add_row(["*", "*", "*", "*", "*"])
        # ^for realization

        table.add_row(["-", "-", "-", "-", "-"])
        count = 0
        for alpha in achain:
            stalpha = ""
            for a in alpha:
                stalpha += "{"
                for i in a:
                    stalpha += str(i)
                # ^for i
                stalpha += "}"
            # ^for a
            if count == 0:
                table.add_row(
                    [
                        "avg",
                        stalpha,
                        str(avg[alpha][0]),
                        str(avg[alpha][1]),
                        str(avg[alpha][2]),
                    ]
                )
            else:
                table.add_row(
                    [
                        " ",
                        stalpha,
                        str(avg[alpha][0]),
                        str(avg[alpha][1]),
                        str(avg[alpha][2]),
                    ]
                )
            count += 1
        # ^for alpha
        print(table)
    # ^if printing

    return ptw, avg


# ^pid()
