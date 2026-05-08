from setuptools import setup, Extension
from pathlib import Path
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

extensions = [
    Extension(
        "idtxl.hde_fast_embedding",
        ["idtxl/hde_fast_embedding.pyx"], include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="idtxl",
    packages=["idtxl", "idtxl/knn"],
    include_package_data=True,
    version="1.8.2",
    description="Information Dynamics Toolkit xl",
    author="Patricia Wollstadt, Joseph T. Lizier, Raul Vicente, Conor Finn, Mario Martinez-Zarzuela, Pedro Mediano, Leonardo Novelli, Michael Wibral",
    author_email="p.wollstadt@gmail.com",
    url="https://github.com/pwollstadt/IDTxl",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
        "Environment :: Other Environment",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    ext_modules = cythonize(extensions, compiler_directives={"language_level": 3, "profile": False}),
)

