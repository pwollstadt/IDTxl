# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:11:52 2016

@author: patricia
"""

from joblib import Parallel, delayed

def f(x):
    return x

l = range(5)
results = Parallel(n_jobs=-1)(delayed(f)(i) for i in l)