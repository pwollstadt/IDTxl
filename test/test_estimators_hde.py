"""Test HDE estimators.

This module provides unit tests for OpenCL estimators. Estimators are tested
against JIDT estimators.
"""
import math
import pytest
import numpy as np
from idtxl.estimators_hd import hdEstimatorShuffling
from test_estimators_jidt import _get_gauss_data

settings = {'debug': True, 'number_of_bootstraps_R_max': 100, 'verbose_output': "False"}
expected_mi, source, source_uncorr, target = _get_gauss_data(seed=0)

est = hdEstimatorShuffling()

est.estimate(source)

"""
embedding_past_range_set : [0.005, 0.00998, 0.01991, 0.03972, 0.07924, 0.15811, 0.31548, 0.62946, 1.25594, 2.50594, 5.0]
estimation_method : all
number_of_bootstraps_R_max : 10
number_of_bootstraps_R_tot : 10
auto_MI_bin_size_set : [0.01, 0.025, 0.05, 0.25]
persistent_analysis: False
"""