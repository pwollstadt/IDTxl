# https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions

class IdtxlError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
    return repr(self.value)
