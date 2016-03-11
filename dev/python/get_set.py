# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:17:09 2016

https://docs.python.org/2/library/functions.html#property
http://www.python-course.eu/python3_properties.php

@author: patricia
"""


class Test():
    def __init__(self, attr1, attr2):
        self.a = attr1
        self._b = attr2

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        if type(a) is not tuple:
            print('not a tuple')
        self._a = a

    @property
    def _b(self):
        return self.__b

    @_b.setter
    def _b(self, b):
        self.__b = b

    @property
    def c(self):
        return 2 * self.b[0]


class C(object):  # inherit from object is only important in Python 2
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x
