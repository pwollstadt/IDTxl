from setuptools import setup
from Cython.Build import cythonize
import numpy

# to compile, run
# python3 setup_hde_fast_embedding.py build_ext --inplace

setup(
    name="Speedy Module",
    ext_modules=cythonize(
        ["hde_fast_embedding.pyx"],
        compiler_directives={"language_level": "3"},
        annotate=False,
    ),
    include_dirs=[numpy.get_include()],
)
