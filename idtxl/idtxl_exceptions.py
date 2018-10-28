"""Provide error handling and warnings."""
import traceback


def package_missing(err, message):
    """Report a missing optional package upon import."""
    print(message)
    traceback.print_tb(err.__traceback__)
    print()
    # warnings.simplefilter('always', ImportWarning)
    # warnings.warn(message, ImportWarning, stacklevel=2)


class AlgorithmExhaustedError(Exception):
    """Exception raised to signal that the estimators can no longer be used
    for this particular target (e.g. because of memory errors in high
    dimensions) but that the estimation could continue for others.
    
    Attributes:
        message -- explanation of the error
    """
    def __init__(self, message):
        self.message = message


class JidtOutOfMemoryError(AlgorithmExhaustedError):
    """Exception raised to signal a Java OutOfMemoryException.
       It is a child class of AlgorithmExhaustedError.
    
    Attributes:
        message -- explanation of the error
    """
    
    def __init__(self, message):
        super().__init__(message)


class BROJA_2PID_Exception(Exception):
    pass
