from distutils.core import setup

# http://www.diveintopython3.net/packaging.html
# https://pypi.python.org/pypi?:action=list_classifiers

with open('README.txt') as file:
    long_description = file.read()

setup(
    name='idtxl',
    packages=['idtxl'],
    version='1.0',
    description='Information Dynamics Toolkit xl',
    author='Patricia Wollstadt, Joseph T. Lizier, Raul Vicente, Conor Finn, Mario Martinez-Zarzuela, Pedro Mediano, Leonardo Novelli, Michael Wibral',
    author_email='p.wollstadt@gmail.com',
    url='https://github.com/pwollstadt/IDTxl',
    long_description=long_description,
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
    ]
)
