# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:39:10 2016

https://docs.python.org/2/tutorial/controlflow.html#keyword-arguments

@author: patricia
"""
def l(a=None,b=None):
    print(a)
    print(b)





def f1(a=5, **options):
    opts = options
    keys = sorted(options.keys())
    for kw in keys:
        print(kw + ": " + options[kw])


def f2(**myargs):
    try:
        my_b = myargs['b']
    except KeyError as e:
        raise KeyError('No opt b!')
        #raise
    for k in sorted(myargs.keys()):
        print(myargs[k])


def f3(*myargs):
    for k in sorted(myargs.keys()):
        print(myargs[k])

o = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
f1(5, **o)
f1(1)

# this is a bit nasty, because a doesn't get printed, it is automatically
# removed from the options dict
o2= {"a": 4, "state": "bleedin' demised", "action": "VOOM"}
f1(**o2)

d = {'a': 1, 'b': 2, 'c': 3}
f2(**d)
f2()