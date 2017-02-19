"""Provide unit tests for high-level PID estimation.

@author: patricia
"""
import pytest
from idtxl.partial_information_decomposition import (
                                        Partial_information_decomposition)
from idtxl.data import Data


def test_pid_mute_data():
    """Test basic calls to PID class."""
    dat = Data()
    dat.generate_mute_data()
    analysis_opts = {'pid_calc_name': 'pid_sydney'}

    # TODO: the lags make more sense as an argument to analyse_single_target,
    # not as a global option -> figure this out; especially how this fits into
    # the network inference class, maybe PID should go into the single
    # process category after all
    pid = Partial_information_decomposition(options=analysis_opts,
                                            lags_sources=[2, 4])
    est = pid.analyse_single_target(data=dat, target=0, sources=[1, 2])


    # TODO: est = pid.analyse_network(data=dat, target=0)

if __name__ == '__main__':
    test_pid_mute_data()
