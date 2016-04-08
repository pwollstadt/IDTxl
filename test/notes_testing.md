IDTxl Testing
=============

Unittests
---------

This is the folder structure of the IDTxl package:

```python
IDTxl/
    setup.py   # your setuptools Python package metadata
    idtxl/     # actual code
        __init__.py
        multivariate_te.py
        ...
    test/      # unit and system tests
        test_multivariate_te.py
        ...
```

Go to the `test` folder and call

    $ py.test   # execute py.test with Python 3

This starts the py.test module, which will automatically collect all 
`test_*.py` or `*_test.py` files and execute them (see 
[here](https://pytest.org/latest/goodpractices.html#test-discovery) for 
py.test's test detection rules).

Before this works, you have to make sure IDTxl is importable, i.e., it has to
be installed system-wide. This can be done by navigating into the IDTxl folder
(where `setup.py` is) and running

    $ pip3 install -e .   # install package using setup.py in editable mode

This will install IDTxl in 'editable' or development mode, which means pip
creates a soft link in Python's site-packages to the current IDTxl location 
(see [here](https://pythonhosted.org/setuptools/setuptools.html#development-mode)
for details). This way you can keep making changes in the Package without the
need to reinstall it after each change.

Some more ways to invoke unit tests with py.test (go 
[here](https://pytest.org/latest/usage.html) for a full documentation):

```bash
    $ py.test -x            # stop after first failure
    $ py.test --maxfail=2   # stop after two failures
    $ py.test test_mod.py   # run tests in given module
```


Systemtests
-----------

folder contains example data from an experiment described in Gruetzner C, Uhlhaas PJ, Genc E, Kohler A, Singer W, et al. (2010), J Neurosci 30:8342–8352. Files with extension .csv contain individual matrices, either raw data or embedded point sets (*_ps*.csv). Data were embedded for the first three trials using a dim of 5, tau of 3 and delay of 3 (see 'embed_data.m').
