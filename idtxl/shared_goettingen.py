"""A
JxPID
"""
import numpy as np
import math
import time
from itertools import chain, combinations
from collections import defaultdict
from prettytable import PrettyTable

#---------
# Lattice 
#---------
class Lattice:
    def __init__(self, n):
        self.n = n
        self.lis = [i for i in range(1,self.n+1)]
    #^ _init_()
    
    def powerset(self):
        return chain.from_iterable(combinations(self.lis, r) for r in range(1,len(self.lis) + 1) )
    #^ powerset()

    def less_than(self, beta, alpha):
        # compare whether an antichain beta is smaller than antichain alpha
        return all(any(frozenset(b) <= frozenset(a) for b in beta) for a in alpha)
    #^ compare()

    def comparable(self, a,b):
        return a < b or a > b
    #^ comparable()

    def antichain(self):
        # dummy expensive function might use dit or networkx functions
        # assert self.n < 5, "antichain(n): number of sources should be less than 5"
        achain = []
        for r in range(1, math.floor((2**self.n - 1)/2) + 2):
            # enumerate the power set of the powerset
            for alpha in combinations(self.powerset(), r):
                flag = 1
                # check if alpha is an antichain
                for a in list(alpha):
                    for b in list(alpha):
                        if a < b and self.comparable(frozenset(a),frozenset(b)): flag = 0 
                    #^ for b
                #^ for a
                if flag: achain.append(alpha)
            #^ for alpha
        #^ for r 
        return achain
    #^ antichain()

    def children(self, alpha, achain):
        chl = []
        downset = [beta for beta in achain if self.less_than(beta,alpha) and beta != alpha]
        for beta in downset:
            if all(not self.less_than(beta,gamma) for gamma in downset if gamma != beta):
                chl.append(beta)
            #^ if
        #^ for beta
        return chl
    #^ children()

#^ Lattice()

#---------------
# pi^+(t:alpha)
#    and
# pi^-(t:alpha) 
#---------------

def powerset(m):
    lis = [i for i in range(1, m+1)]
    return chain.from_iterable(combinations(lis, r) for r in range(1,len(lis) + 1) )
#^ powerset()

def marg(pdf, rlz, uset):
    idxs = [ idx - 1 for idx in list(uset)]
    summ = 0.
    for k in pdf.keys():
        if all(k[idx] == rlz[idx] for idx in idxs): summ += pdf[k]
    #^ for
    return summ
#^ marg()
    
def prob(n, pdf, rlz, gamma, target=False):
    m = len(gamma)
    pset = powerset(m)
    summ = 0
    for idxs in pset:
        if target:
            uset = frozenset((n+1,))
        else:
            uset = frozenset(())
        #^ if 
        for i in list(idxs):
            uset |= frozenset(gamma[i-1])
        #^ for i
        summ += (-1)**(len(idxs) + 1) * marg(pdf, rlz, uset)
    #^ for idxs
    return summ
#^ prob()

def differs(n, pdf, rlz, alpha, chl, target=False):
    if chl == [] and target:
        base = prob(n, pdf, rlz, [()], target)/prob(n, pdf, rlz, alpha, target)
    else:
        base = prob(n, pdf, rlz, alpha, target)
    #^ if bottom
    temp_diffs = [prob(n, pdf, rlz, gamma, target) - base for gamma in chl]
    temp_diffs.sort()
    return [base] + temp_diffs
#^ differs()

def sgn(num_chld):
    if num_chld == 0:
        return np.array([+1])
    else:
        return np.concatenate((sgn(num_chld - 1), -sgn(num_chld - 1)), axis=None)
    #^ if bottom 
#^sgn()
    
def vec(num_chld, diffs):
    """
    Args: 
    num_chld : the number of the children of alpha: (gamma_1,...,gamma_{num_chld}) 
    diffs : vector of probability differences (d_i)_i where d_i = p(gamma_i) - p(alpha) and d_0 = p(alpha)  
    """
    # print(diffs)
    if num_chld == 0:
        return np.array([diffs[0]])
    else:
        temp = vec(num_chld - 1, diffs) + diffs[num_chld]*np.ones(2**(num_chld - 1))
        return np.concatenate((vec(num_chld - 1, diffs), temp), axis=None)
    #^ if bottom
#^ vec()

def pi_plus(n, pdf, rlz, alpha, chld, achain):
    diffs = differs(n, pdf, rlz, alpha, chld[tuple(alpha)], False)
    return np.dot(sgn(len(chld[alpha])), -np.log2(vec(len(chld[alpha]),diffs)))
#^ pi_plus()

def pi_minus(n, pdf, rlz, alpha, chld, achain):
    diffs = differs(n, pdf, rlz, alpha, chld[alpha], True)
    if chld[alpha] == []:
        return np.dot(sgn(len(chld[alpha])), np.log2(vec(len(chld[alpha]),diffs)))
    else:
        return np.dot(sgn(len(chld[alpha])), -np.log2(vec(len(chld[alpha]),diffs)))
    #^ if bottom
#^ pi_minus()


def pid(n, pdf_dirty, chld, achain, printing=False):
    """Estimate partial information decomposition for two inputs and one output
    
    Implementation of the partial information decomposition (PID) estimator for
    discrete data. The estimator finds shared information, unique information
    and synergistic information between the two, three, or four inputs with respect
    to the output t.
    
    P.S. The implementation can be extended to any number n of variables if their 
    corresponding redundancy lattice is provided (check the class Lattice())

    Args:
            n : int
               number of sources
            pdf_dirty : dict
                       the joint distribution of the inputs and the output 
                       (realizations are the keys)
            chld : dict
                  list of children for each node in the redundancy lattice 
                   (nodes are the keys)
            achain : tuple
                    tuple of all the nodes (tuple) in the redundacy lattice
            printing: Bool
                     If true prints the results using PrettyTables
        Returns:
            tuple
                pointwise decomposition, averaged decomposition
    """
    assert type(pdf_dirty) is dict, "jx_pid.pid(pdf, chld, achain): pdf must be a dictionary"
    assert type(chld) is dict, "jx_pid.pid(pdf, chld, achain): chld must be a dictionary"
    assert type(achain) is list, "jx_pid.pid(pdf, chld, achain): pdf must be a list"

    if __debug__:
        sum_p = 0.
        for k,v in pdf_dirty.items():
            assert type(k) is tuple,                              "jx_pid.pid(pdf, chld, achain): pdf's keys must be tuples"
            assert len(k) < 6,                                    "jx_pid.pid(pdf, chld, achain): pdf's keys must be tuples of length at most 5"
            assert type(v) is float or ( type(v)==int and v==0 ), "jx_pid.pid(pdf, chld, achain): pdf's values must be floats"
            assert v >-.1,                                        "jx_pid.pid(pdf, chld, achain): pdf's values must be nonnegative"
            sum_p += v
        #^ for

        assert abs(sum_p - 1) < 1.e-7,                           "jx_pid.pid(pdf, chld, achain): pdf's keys must sum up to 1 (tolerance of precision is 1.e-7)"
    #^ if debug

    assert type(printing) is bool,                                "jx_pid.pid(pdf, chld, achain, printing): printing must be a bool"

    # Remove the impossible realization
    pdf = {k:v for k,v in pdf_dirty.items() if v > 1.e-300 }

    # Initialize the output where
    # ptw = { rlz -> { alpha -> pi_alpha } }
    # avg = { alpha -> PI_alpha }
    ptw = dict()
    #avg = defaultdict(lambda : [0.,0.,0.])
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
             
        #^ for
    #^ for
    # compute and store the average of the (+, -, +-) atoms 
    for alpha in achain:
        avgplus = 0.
        avgminus = 0.
        avgdiff = 0.
        for rlz in pdf.keys():
            avgplus  += pdf[rlz]*ptw[rlz][alpha][0]
            avgminus += pdf[rlz]*ptw[rlz][alpha][1]
            avgdiff  += pdf[rlz]*ptw[rlz][alpha][2]
            avg[alpha] = (avgplus, avgminus, avgdiff)
        #^ for
    #^ for

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
                    #^ for i
                    stalpha += "}" 
                #^ for a
                if count == 0: table.add_row( [str(rlz), stalpha, str(ptw[rlz][alpha][0]), str(ptw[rlz][alpha][1]), str(ptw[rlz][alpha][2])] )
                else:          table.add_row( [" ", stalpha, str(ptw[rlz][alpha][0]), str(ptw[rlz][alpha][1]), str(ptw[rlz][alpha][2])] )
                count += 1 
            #^ for alpha
            table.add_row(["*", "*", "*", "*", "*"])
        #^ for realization

        table.add_row(["-", "-", "-", "-", "-"])
        count = 0
        for alpha in achain:
            stalpha = ""
            for a in alpha:
                stalpha += "{"
                for i in a:
                    stalpha += str(i)
                #^ for i
                stalpha += "}" 
            #^ for a
            if count == 0: table.add_row( ["avg", stalpha, str(avg[alpha][0]), str(avg[alpha][1]), str(avg[alpha][2])] )
            else:          table.add_row( [" ", stalpha, str(avg[alpha][0]), str(avg[alpha][1]), str(avg[alpha][2])] )
            count += 1
        #^ for alpha
        print(table)
    #^ if printing
    
    return ptw, avg
#^ jxpid()

# EOF
