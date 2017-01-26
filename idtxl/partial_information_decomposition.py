"""Estimate partial information decomposition (PID).

Estimate PID for two source and one target process using different estimators.

Note:
    Written for Python 3.4+

@author: patricia
"""
import numpy as np
from . import stats
from .network_inference import Network_inference
from .set_estimator import Estimator_pid

VERBOSE = True


class Partial_information_decomposition(Network_inference):
    """Set up network analysis using partial information decomposition.

    Set parameters necessary to infer partial information decomposition (PID)
    for two source and one target process. Estimate unique, shared, and
    synergistic information in the two sources about the target.
    """

    def __init__(self, options):
        try:
            self.calculator_name = options['pid_calc_name']
        except KeyError:
            raise KeyError('Calculator name was not specified!')
        print('\n\nSetting calculator to: {0}'.format(self.calculator_name))
        self._pid_calculator = Estimator_pid(self.calculator_name)
        super().__init__(None, None, options)

    def analyse_single_target(self, data, target, sources):
        """Return PID for two sources and a target."""
        source1_realisations = data.get_realisations(self.current_value,
                                                     [sources[0]])
        source2_realisations = data.get_realisations(self.current_value,
                                                     [sources[1]])
        target_realisations = data.get_realisations(self.current_value,
                                                    [target])
        results = self._pid_calculator.estimate(source1_realisations,
                                                source2_realisations,
                                                target_realisations,
                                                self.options)
        results['options'] = self.options
        results['target'] = self.target
        results['source_1'] = sources[0]
        results['source_2'] = sources[1]
        return results

    # def _initialise(self, data, sources, target):
