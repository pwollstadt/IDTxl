# encoding: utf-8
# cython: profile=True
# filename: calc_pi_cython_inline.pyx

def recip_square(i):
    return 1./(i*i)

def approx_pi(n=10000000):
    val = 0.
    for k in range(1,n+1):
        val += recip_square(k)
    return (6 * val)**.5
