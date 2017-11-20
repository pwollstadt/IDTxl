"""Provide error handling and warnings."""
import traceback


def package_missing(err, message):
    """Report a missing optional package upon import."""
    print(message)
    traceback.print_tb(err.__traceback__)
    print()
    # warnings.simplefilter('always', ImportWarning)
    # warnings.warn(message, ImportWarning, stacklevel=2)


class BROJA_2PID_Exception(Exception):
    pass
