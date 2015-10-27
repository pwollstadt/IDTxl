#!/usr/bin/env python
# encoding: utf-8
# filename: profile_cython.py

import pstats, cProfile

import pyximport
pyximport.install()

import calc_pi_cython

cProfile.runctx("calc_pi_cython.approx_pi()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()