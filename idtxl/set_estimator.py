import types
import random
import math as math
import numpy as np
from . import estimators_te
from . import estimators_cmi
from . import estimators_mi


class Estimator(object):
    """Set the estimator requested by the user."""

    @classmethod
    def removeMethod(cls, name):
        return delattr(cls, name)

    @classmethod
    def addMethodAs(cls, func, new_name=None):
        if new_name is None:
            new_name = func.__name__
        return setattr(cls, new_name, types.MethodType(func, cls))

    def get_estimator(self):
        return self.estimator_name


class Estimator_te(Estimator):
    """Set the requested transfer entropy estimator."""

    def __init__(self, estimator_name):
        try:
            estimator = getattr(estimators_te, estimator_name)
        except AttributeError:
            raise AttributeError('The requested TE estimator "{0}" was not '
                                 'found.'.format(estimator_name))
        else:
            self.estimator_name = estimator_name
            self.addMethodAs(estimator, "estimate")


class Estimator_cmi(Estimator):
    """Set the requested conditional mutual information estimator."""

    def __init__(self, estimator_name):
        try:
            estimator = getattr(estimators_cmi, estimator_name)
        except AttributeError:
            raise AttributeError('The requested CMI estimator "{0}" was not '
                                 'found.'.format(estimator_name))
        else:
            self.estimator_name = estimator_name
            self.addMethodAs(estimator, "estimate")


class Estimator_mi(Estimator):
    """Set the requested mutual information estimator."""

    def __init__(self, estimator_name):
        try:
            estimator = getattr(estimators_mi, estimator_name)
        except AttributeError:
            raise AttributeError('The requested MI estimator "{0}" was not '
                                 'found.'.format(estimator_name))
        else:
            self.estimator_name = estimator_name
            self.addMethodAs(estimator, "estimate")


if __name__ == "__main__":
    """ Do a quick check if eveything is called correctly."""

    te_estimator = Estimator_te("jidt_kraskov")
    mi_estimator = Estimator_mi("jidt_kraskov")
    cmi_estimator = Estimator_cmi("jidt_kraskov")

    n_obs = 10000
    covariance = 0.4
    dim = 5
    source = [random.normalvariate(0, 1) for r in range(n_obs)]
    target = [0] + [sum(pair) for pair in zip(
                    [covariance * y for y in source[0:n_obs-1]],
                    [(1-covariance) * y for y in [
                        random.normalvariate(0, 1) for r in range(n_obs-1)]])]
    var1 = [[random.normalvariate(0, 1) for x in range(dim)] for
            x in range(n_obs)]
    var2 = [[random.normalvariate(0, 1) for x in range(dim)] for
            x in range(n_obs)]
    conditional = [[random.normalvariate(0, 1) for x in range(dim)] for
                   x in range(n_obs)]

    knn = 4
    history_length = 1
    options = {
        'kraskov_k': 4,
        'history_target': 3
        }

    te = te_estimator.estimate(np.array(source), np.array(target), options)
    expected_te = math.log(1 / (1 - math.pow(covariance, 2)))
    print('TE estimator is {0}'.format(te_estimator.get_estimator()))
    print('TE result: {0:.4f} nats; expected to be close to {1:.4f} nats '
          'for correlated Gaussians.'.format(te, expected_te))

    mi = mi_estimator.estimate(np.array(var1), np.array(var2), options)
    print('MI estimator is ' + mi_estimator.get_estimator())
    print('MI result: %.4f nats.' % mi)

    cmi = cmi_estimator.estimate(np.array(var1), np.array(var2),
                                 np.array(conditional), options)
    print('Estimator is ' + cmi_estimator.get_estimator())
    print('CMI result: %.4f nats.' % cmi)

    # CMI testcases with correlated variables
    # Generate some random normalised data.

    # set options for maximal comparability between JIDT and opencl
    #options = {
    #    'kraskov_k': 4, 'noise_level': 0, 'debug': True, 'theiler_t': 0
    #    }

    numObservations = 10000
    covariance = 0.4

    # Source array of random normals:
    var1 = np.random.randn(numObservations, 1)
    # Destination array of random normals with partial correlation to previous
    # value of sourceArray
    var2 = [sum(pair) for pair in zip([covariance*y for y in var1],
            [(1-covariance)*y for y in
            [random.normalvariate(0, 1) for r in range(numObservations)]])]
    var2 = np.array(var2)
    # Uncorrelated conditionals:
    conditional = np.random.randn(numObservations, 1)

    # casting data to single and back for comparison with single precision
    # computations on the GPU
    cmi_jidt = cmi_estimator.estimate(np.array(var1).astype('float32').astype('float64'),
                                 np.array(var2).astype('float32').astype('float64'),
                                 np.array(conditional).astype('float32').astype('float64'),
                                 options)
    print('Estimator is ' + cmi_estimator.get_estimator())
#    print('CMI result: %.4f nats.' % cmi)
    print("CMI result %.4f nats; expected to be close to %.4f nats for these correlated Gaussians" % \
          (cmi_jidt, math.log(1/(1-math.pow(covariance, 2)))))

#    # casting data to single and back for comparison with single precision
#    # computations on the GPU
#    mi_jidt = mi_estimator.estimate(np.array(var1).astype('float32').astype('float64'),
#                                 np.array(var2).astype('float32').astype('float64'),
#                                 options)
#    print('Estimator is ' + mi_estimator.get_estimator())
#    print("MI result %.4f nats; expected to be close to %.4f nats for these correlated Gaussians" % \
#          mi_jidt, math.log(1/(1-math.pow(covariance, 2))))


    cmi_estimator = Estimator_cmi("opencl_kraskov")
    cmi_ocl = cmi_estimator.estimate(np.array(var1), np.array(var2),
                                 np.array(conditional), options)
    print('Estimator is ' + cmi_estimator.get_estimator())
    print('CMI result: %.4f nats.' % cmi_ocl)

    assert int(cmi_jidt*10000) == int(cmi_ocl*10000) , "JIDT and opencl estmator results mismatch"


    mi_estimator = Estimator_mi("opencl_kraskov")
    mi_ocl = mi_estimator.estimate(np.array(var1), np.array(var2), options)
    print('Estimator is ' + mi_estimator.get_estimator())
    print('MI result: %.4f nats.' % mi_ocl)

#    assert int(mi_jidt*10000) == int(mi_ocl*10000) , "JIDT and opencl estmator results mismatch"
