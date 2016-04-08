IDTxl Packaging
===============

First make IDTxl importable. For that to work, we use the common 
folder structure for Python packages:

```python
IDTxl/
    setup.py   # your setuptools Python package metadata
    MANIFEST.in  # spefify non py-files that should be installed
    COPYING.txt  # license text
    README.txt  # description of the package and its installation
    README.md   # the same in markdown for GitHub
    idtxl/     # actual code
        __init__.py  # for now, this is empty
        multivariate_te.py
        ...
    test/      # unit and system tests
        test_multivariate_te.py
        ...
```

Make sure, every module uses explicit relative imports, otherwise
Python will choke. Relative imports look like this:

```python
from . import stats
from .network_analysis import Network_analysis
```

Include the JIDT `jar`-file in the `MANIFEST.in`. The `MANIFEST.in` lists
all non-`.py` files that should still go into the installed package 
(see [here](http://python-packaging.readthedocs.org/en/latest/non-code-files.html) for further info). Python calls such files *Resources*.

To access ressources from package modules one can use the `pkg_resources` 
API, which provides convenience functions to load and find ressources in
Python packages (see also 
[here](http://peak.telecommunity.com/DevCenter/PythonEggs#accessing-package-resources) 
and [here](https://pythonhosted.org/setuptools/pkg_resources.html#resourcemanager-api))

For example, the JIDT `jar`-file is loaded by the `estimators_cmi.py` like
this:

```python
from pkg_resources import resource_filename
jarLocation = resource_filename(__name__, 'infodynamics.jar')
if not jp.isJVMStarted():
    jp.startJVM(jp.getDefaultJVMPath(), '-ea', ('-Djava.class.path=' +
                jarLocation))
```

Check that everything is correct by going into the top-folder `IDTxl`
and run `import idtxl` in a Python session.

To be able to call `import idtxl` from everywhere on the system, install
the package using `pip3` (from the IDTxl folder):

    $ pip3 install .

As long as IDTxl is still under construction you can use

    $ pip3 install -e .   # install package using setup.py in editable mode

instead. This will install IDTxl in 'editable' or development mode, which 
means pip creates a soft link in Python's site-packages to the current IDTxl 
location (see [here](https://pythonhosted.org/setuptools/setuptools.html#development-mode)
for details). This way you can keep making changes in the Package without the
need to reinstall it after each change.

It is also possible to make an editable install of a reporistory:

    $ pip install -e git+http://repo/my_project.git#egg=SomeProject

See [here](https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support) for 
further info.
https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support
