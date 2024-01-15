from setuptools import setup
from Cython.Build import cythonize
import numpy

# to compile, run
# python3 hde_setup.py build_ext --inplace

setup(
    name="hde_fast_embedding",
    ext_modules=cythonize(["hde_fast_embedding.pyx"], annotate=False),
    include_dirs=[numpy.get_include()],
)
