"""Test fast PID estimator on logical AND.
"""
import random
import numpy as np
import time as tm
from bitstring import BitArray, Bits
# import estimators_fast_pid as epid
from idtxl.set_estimator import Estimator_pid

# LOGICAL AND
alph_x = 2
alph_y = 2
alph_z = 2

n = 5000

x = np.random.randint(0, alph_x, n)
y = np.random.randint(0, alph_y, n)
z = np.logical_and(x, y).astype(int)

cfg = {
    'alph_s1': alph_x,
    'alph_s2': alph_y,
    'alph_t': alph_z,
    'max_unsuc_swaps_row_parm': 3,
    'num_reps': 63,
    'max_iters': 10000
}

pid_sydney = Estimator_pid('pid_sydney')

tic = tm.time()
est = pid_sydney.estimate(x, y, z, cfg)
toc = tm.time()

print('\n\nLOGICAL AND')
print('\nPID evaluation       {:.3f} seconds\n'.format(toc - tic))
print('Uni s1              ', est['unq_s1'])
print('Uni s2              ', est['unq_s2'])
print('Shared s1_s2        ', est['shd_s1_s2'])
print('Synergy s1_s2       ', est['syn_s1_s2'])

# LOGICAL XOR
z = np.logical_xor(x, y).astype(int)

cfg = {
    'alph_s1': alph_x,
    'alph_s2': alph_y,
    'alph_t': alph_z,
    'max_unsuc_swaps_row_parm': 3,
    'num_reps': 63,
    'max_iters': 10000
}

tic = tm.time()
est = pid_sydney.estimate(x, y, z, cfg)
toc = tm.time()

print('\nPID evaluation       {:.3f} seconds\n'.format(toc - tic))
print('Uni s1              ', est['unq_s1'])
print('Uni s2              ', est['unq_s2'])
print('Shared s1_s2        ', est['shd_s1_s2'])
print('Synergy s1_s2       ', est['syn_s1_s2'])

# SINGLE INPUT COPY
z = x

cfg = {
    'alph_s1': alph_x,
    'alph_s2': alph_y,
    'alph_t': alph_z,
    'max_unsuc_swaps_row_parm': 3,
    'num_reps': 63,
    'max_iters': 10000
}

tic = tm.time()
est = pid_sydney.estimate(x, y, z, cfg)
toc = tm.time()

print('\nPID evaluation       {:.3f} seconds\n'.format(toc - tic))
print('Uni s1              ', est['unq_s1'])
print('Uni s2              ', est['unq_s2'])
print('Shared s1_s2        ', est['shd_s1_s2'])
print('Synergy s1_s2       ', est['syn_s1_s2'])

# PARITY
a = random.randint(0, 2**(2*n) - 1)
b = random.randint(0, 2**n - 1)

A = BitArray(uint=a, length=2*n)
B = BitArray(uint=b, length=n)


def parity(bytestring):
    """Return parity function for a bitstring."""
    par = 0
    string = Bits(bytes=bytestring)

    for bit in string:
        par ^= int(bit)

    return par

x = np.zeros((n,), dtype=np.int)
y = np.zeros((n,), dtype=np.int)
z = np.zeros((n,), dtype=np.int)

for i in range(n):
    x[i] = (B[i:i+1]).uint
    y[i] = (A[i * 2: (i + 1) * 2]).uint
    z[i] = parity(A[i * 2: (i + 1) * 2] + B[i: i + 1])

alph_x = 2
alph_y = 4
alph_z = 2

cfg = {
    'alph_s1': alph_x,
    'alph_s2': alph_y,
    'alph_t': alph_z,
    'max_unsuc_swaps_row_parm': 3,
    'num_reps': 63,
    'max_iters': 10000
}

tic = tm.time()
est = pid_sydney.estimate(x, y, z, cfg)
toc = tm.time()

print('\n\nPARITY')
print('\nPID evaluation       {:.3f} seconds\n'.format(toc - tic))
print('Uni s1              ', est['unq_s1'])
print('Uni s2              ', est['unq_s2'])
print('Shared s1_s2        ', est['shd_s1_s2'])
print('Synergy s1_s2       ', est['syn_s1_s2'])
