from distutils.core import setup
from Cython.Build import cythonize

# this lets you import all functions in the module into python, run:
# command line: python setup.py build_ext --inplace
# within python: from hello import say_hello_to
setup(
  name = 'Hello world app',
  ext_modules = cythonize("hello.pyx"),
)

