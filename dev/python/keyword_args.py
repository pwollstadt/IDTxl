# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:39:10 2016

https://docs.python.org/2/tutorial/controlflow.html#keyword-arguments

@author: patricia
"""


def f(a=5, **options):

    keys = sorted(options.keys())
    for kw in keys:
        print(kw + ":" + options[kw])


o = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
f(5, **o)

# this is a bit nasty, because a doesn't get printed, it is automatically
# removed from the options dict
o2= {"a": 4, "state": "bleedin' demised", "action": "VOOM"}
f(**o2)