from idtxl.estimator import Estimator

class MockCMI(Estimator):
    """ Mock estimator for CMI

    This estimator is used for testing purposes only. It returns the first value of the first
    dimension of the first variable passed to it and evaluates all (possibly lazy) array arguments
    """
    def __init__(self, settings: dict=None):
        pass

    def estimate(self, var1, var2, conditional=None):

        # Evaluate all lazy arrays
        var1 = var1[:]
        var2 = var2[:]

        if conditional is not None:
            conditional = conditional[:]

        return var1[:][0]
    
    def is_analytic_null_estimator(self):
        return False
    
    def is_parallel(self):
        return False