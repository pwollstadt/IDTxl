"""Manage different estimators for information theoretic measures."""
import types
from . import estimators_te
from . import estimators_cmi
from . import estimators_mi


class Estimator(object):
    """Set the estimator requested by the user."""

    def __init__(self):
        self.estimator_name = None

    def estimate(self):
        """Stub for the estimator method."""
        print('No estimator set. Use "add_method".')

    def add_method(self, func, new_name=None):
        """Set the estimator for the 'estimate' method."""
        if new_name is None:
            new_name = func.__name__
        self.estimate = types.MethodType(func, self)
        self.estimator_name = new_name

    @property
    def estimator_name(self):
        """Name of the estimator set for the 'estimate' method."""
        return self._estimator_name

    @estimator_name.setter
    def estimator_name(self, estimator_name):
        self._estimator_name = estimator_name


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
            self.add_method(estimator, estimator_name)


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
            self.add_method(estimator, estimator_name)


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
            self.add_method(estimator, estimator_name)
