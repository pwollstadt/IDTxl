# https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
# http://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
import warnings


class IdtxlError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class IdtxlParamError(Exception):

    def __init__(self, value, missing_param):
        self.value = value
        self.missing_param = missing_param


def opencl_missing(message):
    warnings.simplefilter('always', ImportWarning)
    warnings.warn(message, ImportWarning, stacklevel=2)

def jpype_missing(message):
    warnings.simplefilter('always', ImportWarning)
    warnings.warn(message, ImportWarning, stacklevel=2)    

def n_replications_low(message):
    warnings.simplefilter('always', RuntimeWarning)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
