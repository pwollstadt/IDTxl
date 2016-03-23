import types
import random
import numpy as np
import estimators_te
import estimators_cmi
import estimators_mi


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

    n_obs = 1000
    covariance = 0.4
    source = [random.normalvariate(0, 1) for r in range(n_obs)]
    target = [0] + [sum(pair) for pair in zip(
                    [covariance * y for y in source[0:n_obs-1]],
                    [(1-covariance) * y for y in [
                        random.normalvariate(0, 1) for r in range(n_obs-1)]])]
    var1 = [[random.normalvariate(0, 1) for x in range(5)] for
            x in range(n_obs)]
    var2 = [[random.normalvariate(0, 1) for x in range(5)] for
            x in range(n_obs)]
    conditional = [[random.normalvariate(0, 1) for x in range(5)] for
                   x in range(n_obs)]
    knn = 4
    history_length = 1
    options = {
        'kraskov_k': 4
        }

    te = te_estimator.estimate(source, target, knn, history_length)
    print('Estimator is {0}'.format(te_estimator.get_estimator()))
    print('TE result: %.4f nats.' % te)

    mi = mi_estimator.estimate(var1, var2, knn)
    print('Estimator is ' + mi_estimator.get_estimator())
    print('MI result: %.4f nats.' % mi)
    cmi = cmi_estimator.estimate(np.array(var1), np.array(var2),
                                 np.array(conditional), options)
    print('Estimator is ' + cmi_estimator.get_estimator())
    print('CMI result: %.4f nats.' % cmi)
