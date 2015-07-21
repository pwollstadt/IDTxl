from distutils.core import setup

# http://www.diveintopython3.net/packaging.html
# https://pypi.python.org/pypi?:action=list_classifiers

with open('README.txt') as file:
    long_description = file.read()

setup(
    name = 'trentoolxl',
    packages = ['trentoolxl'],
    version = '1.0',
    description = 'Multivariate transfer entropy estimator',
    author = 'Patricia Wollstadt',
    author_email = 'p.wollstadt@gmail.com',
    url = 'www.trentool.de',
    long_description = long_description,
    classifiers = [
	"Programming Language :: Python",
	"Programming Language :: Python :: 3",
	"Development Status :: 4 - Beta",
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
