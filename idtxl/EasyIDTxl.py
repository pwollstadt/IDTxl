"""EasyIDTxl

EasyIDTxl can be used to easily generate analysis scripts for 
several idtxl estimators and analysis types:
    - mutual information (MI)
    - transfer entropy (TE)
    - conditional MI and TE
    - bi- and multivariate MI and TE
    - network analysis
    - active information storage (AIS)
    - partial information decomposition (PID)

Requirements:
    - working python environment where IDTxl is running properly.
    - additionally:
        pyqt6 (developed and tested with version 6.9.1 in python 3.12)

Start EasyIDTxl using: python EasyIDTxl.py

Michael Lindner, 2025
"""


import os
import sys
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QPushButton, 
    QMainWindow, 
    QGridLayout,
    QVBoxLayout, 
    QHBoxLayout, 
    QApplication,
    QComboBox,
    QFileDialog,
    QLineEdit,
    QTextEdit,
    QLabel,
    QTableWidget,
    QRadioButton,
    QCheckBox,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QWidget)


# ----------------------------------------------------------
# Definition of general variables - Start
# ----------------------------------------------------------

# name of ui
ui_name = "EasyIDTxl"

# define ui colors
background = "#c2b4a9"
yellow = "#f5a66e"
lightblue = "#8899ad"
darkblue = "#3498a8"
red = "#b8888d"
brown = "#ab998f"
darkbrown = "#6c5a50"

# set default windows size
window_width = 1200
window_higth = 800

# define available estimators for each analysis type
MI_estimators = ["JidtKraskovMI", "JidtDiscreteMI", "JidtGaussianMI", 
    "OpenCLKraskovMI", "RudeltNSBEstimatorSymbolsMI",
    "RudeltPluginEstimatorSymbolsMI", "RudeltBBCEstimator"]
TE_estimators = ["JidtKraskovTE", "JidtDiscreteTE", "JidtGaussianTE"]
CMI_estimators = ["JidtKraskovCMI", "JidtDiscreteCMI", "JidtGaussianCMI", 
    "OpenCLKraskovCMI", "PythonKraskovCMI"]
AIS_estimators = ["ActiveInformationStorage", "JidtKraskovAIS", 
    "JidtDiscreteAIS", "JidtGaussianAIS", "OptimizationRudelt", 
    "RudeltShufflingEstimator"]
AIS_CMI_estimators = CMI_estimators
PID_estimators = ["I_BROJA(via_permutation)", 
    "I_BROJA(via_convex_optimization)", "I_sx(multivariate)", 
    "BivariatePID", "MultivariatePID"]
PID_single_estimators = ["SydneyPID", "TartuPID", "SxPID"]
PID_bimulti_single_estimators = ["SydneyPID", "TartuPID"]
Estimator_dict = {
    "I_BROJA(via_permutation)": "SydneyPID",
    "I_BROJA(via_convex_optimization)": "TartuPID",
    "I_sx(multivariate)": "SxPID",
}
Multivariate_estimators = ["JidtKraskovCMI", "JidtDiscreteCMI", 
    "JidtGaussianCMI", "OpenCLKraskovCMI", "PythonKraskovCMI"]
Nonlinear_analysis = ["JidtGaussianCMI"]
Nonlinear_BiMulti_type = ["MultivariateTE", "BivariateTE"]
Network_analysis = ["JidtKraskovCMI", "JidtDiscreteCMI", "JidtGaussianCMI", 
    "OpenCLKraskovCMI", "PythonKraskovCMI"]
BiMulti_type = ["MultivariateMI", "MultivariateTE", 
    "BivariateMI", "BivariateTE"]
MPI_estimators = ["JidtKraskovMI", "JidtDiscreteMI", "JidtGaussianMI",
    "JidtKraskovTE", "JidtDiscreteTE", "JidtGaussianTE",
    "JidtKraskovCMI", "JidtDiscreteCMI", "JidtGaussianCMI"]

# define allowed input data orders
data_order_all = ["psr","prs","spr","srp","rps","rsp", "ps", "sp"]
data_order_2d = ["ps", "sp"]
data_order_2dx = ["ps", "s"]


# define estimator sources (files in which estimators are stored)
estimator_source = {
    "JidtKraskovMI": "estimators_jidt",
    "JidtDiscreteMI": "estimators_jidt",
    "JidtGaussianMI": "estimators_jidt",
    "OpenCLKraskovMI": "estimators_opencl",
    "JidtKraskovTE": "estimators_jidt", 
    "JidtDiscreteTE": "estimators_jidt",
    "JidtGaussianTE": "estimators_jidt",
    "JidtKraskovCMI": "estimators_jidt",
    "JidtDiscreteCMI": "estimators_jidt",
    "JidtGaussianCMI": "estimators_jidt",
    "OpenCLKraskovCMI": "estimators_opencl",
    "PythonKraskovCMI": "estimators_python",
    "JidtKraskovAIS": "estimators_jidt",
    "JidtDiscreteAIS": "estimators_jidt",
    "JidtGaussianAIS": "estimators_jidt",
    "ActiveInformationStorage": "active_information_storage",
    "SydneyPID": "estimators_pid",
    "SxPID": "estimators_multivariate_pid",
    "BivariatePID": "bivariate_pid",
    "MultivariatePID": "multivariate_pid",    
    "TartuPID": "estimators_pid",
    "BivariatePID": "bivariate_pid",
    "MultivariatePID": "multivariate_pid",
    "MultivariateMI": "multivariate_mi",
    "MultivariateTE": "multivariate_te",
    "BivariateMI": "bivariate_mi",
    "BivariateTE": "bivariate_te",
    "RudeltNSBEstimatorSymbolsMI": "estimators_Rudelt",
    "RudeltPluginEstimatorSymbolsMI": "estimators_Rudelt",
    "RudeltBBCEstimator": "estimators_Rudelt",
    "RudeltShufflingEstimator": "estimators_Rudelt",
    "OptimizationRudelt": "embedding_optimization_ais_Rudelt"
}

# define estimator tooltips
estimator_tooltips = {
    "JidtKraskovMI": "Estimate MI with JIDT's Kraskov implementation.\n Calculate the MI between two variables.",
    "JidtDiscreteMI": "Estimate MI with JIDT's discrete-variable implementation.\n Calculate the MI between two variables.",
    "JidtGaussianMI": "Estimate MI with JIDT's Gaussian implementation.\n Calculate the MI between two variables.",
    "OpenCLKraskovMI": "Estimate MI with OpenCL Kraskov implementation.\n Calculate the MI between two variables using OpenCL GPU-code.",
    "JidtKraskovTE": "Estimate TE with JIDT's Kraskov implementation.\n Calculate TE between a source and a target variable. TE is defined as the conditional mutual information between the source's past state and the target's current value, conditional on the target's past.", 
    "JidtDiscreteTE": "Estimate TE with JIDT's discrete-variable implementation.\n Calculate TE between a source and a target variable. TE is defined as the conditional mutual information between the source's past state and the target's current value, conditional on the target's past.", 
    "JidtGaussianTE": "Estimate TE with JIDT's Gaussian implementation.\n Calculate TE between a source and a target variable. TE is defined as the conditional mutual information between the source's past state and the target's current value, conditional on the target's past.", 
    "JidtKraskovCMI": "Estimate CMI with JIDT's Kraskov implementation.\n Calculate the CMI between three variables.\nCall JIDT via jpype and use the Kraskov 1 estimator. If no conditional is given (is None), the function returns the MI between source and target.",
    "JidtDiscreteCMI": "Estimate CMI with JIDT's discrete-variable implementation.\n Calculate the CMI between two variables given the third.",
    "JidtGaussianCMI": "Calculate CMI with JIDT's Gaussian implementation.\n Computes the differential CMI of two multivariate sets of observations, conditioned on another, assuming that the probability distribution function for these observations is a multivariate Gaussian distribution.",
    "OpenCLKraskovCMI": "Estimate CMI with OpenCL Kraskov implementation.\n Calculate the CMI between two variables given the third.",
    "PythonKraskovCMI": "Estimate CMI with Kraskov (1) implementation.\n Calculate the CMI between two variables given the third.",
    "JidtKraskovAIS": "Estimate AIS with JIDT's Kraskov implementation.\n Calculate AIS for some process using JIDT's implementation of the Kraskov type 1 estimator. AIS is defined as the MI between the processes' past state and current value.",
    "JidtDiscreteAIS": "Estimate AIS with JIDT's discrete-variable implementation.\n Calculate the AIS for some processes using JIDT's implementation of the discrtete estimator. AIS is defined as the mutual information between the processes' past state and current value.",
    "JidtGaussianAIS": "Estimate AIS with JIDT's Gaussian implementation.\n Calculate AIS for some processes using JIDT's implementation of the Gaussian estimator. AIS is defined as the mutual information between the processes' past state and current value.",
    "ActiveInformationStorage": "Analysis of active information storage (AIS) in individual processes of a network.\nanalyse_single_process: Estimate active information storage for one process in the network.\nnetwork_analysis: Estimate active information storage for all or a subset of processes in the network.",
    "SydneyPID": "Estimate PID of discrete variables.\nFast implementation of the BROJA partial information decomposition (PID) estimator for discrete data",
    "SxPID": "Estimate partial information decomposition for multiple inputs.\n\nImplementation of the multivariate partial information decomposition (PID)\nestimator for discrete data with (up to 4 inputs) and one output. The\nestimator finds shared information, unique information and synergistic\ninformation between the multiple inputs s1, s2, ..., sn with respect to the\noutput t for each realization (t, s1, ..., sn) and then average them\naccording to their distribution weights p(t, s1, ..., sn). Both the\npointwise (on the realization level) PID and the averaged PID are returned\n(see the 'return' of 'estimate()').\n\nThe algorithm uses recursion to compute the partial information decomposition.",
    "TartuPID": "Estimate PID for two inputs and one output",
    "BivariatePID": "Perform partial information decomposition for individual processes.\nPerform partial information decomposition (PID) for two source processes and one target process in the network. Estimate unique, shared, and synergistic information in the two sources about the target. Call analyse_network() on the whole network or a set of nodes or call analyse_single_target() to estimate PID for a single process. See docstrings of the two functions for more information.",
    "MultivariatePID": "Perform partial information decomposition for individual processes.\nPerform partial information decomposition (PID) for multiple source processes (up to 4 sources) and a target process in the network. Estimate unique, shared, and synergistic information in the multiple sources about the target. Call analyse_network() on the whole network or a set of nodes or call analyse_single_target() to estimate PID for a single process. See docstrings of the two functions for more information.",
    "MultivariateMI": "Perform network inference using multivariate MI.",
    "MultivariateTE": "Perform network inference using multivariate TE.",

    "RudeltNSBEstimatorSymbolsMI": "History dependence NSB estimator\n\nCalculate the mutual information (MI) of one variable depending on its past\nusing NSB estimator.",
    "RudeltPluginEstimatorSymbolsMI": "Plugin History dependence estimator\n\nCalculate the mutual information (MI) of one variable depending on its past\nusing plugin estimator.",
    "RudeltBBCEstimator": "Bayesian bias criterion (BBC) Estimator using NSB and Plugin estimator\n\nCalculate the mutual information (MI) of one variable depending on its past\nusing nsb and plugin estimator and check if bias criterion is passed.",
    "RudeltShufflingEstimator": "Estimate the history dependence in a spike train using the shuffling estimator.",
    "OptimizationRudelt": "Optimization of embedding parameters of spike times using the history dependence estimators."
}

# general attention text for tooltips for list and list of lists
gen_att = "ATTENTION: In case of 'list' or 'list of lists': \nDO NOT the enter outer square brackets!\ne.g.\nfor list:\n   1,2,3 instead of [1,2,3]\nfor list of list:\n [1,2][3,4] instead of [[1,2][3,4]]"

# define parameter tooltips
parameter_tooltips =  {
    "history_target": "[int] number of samples in the target's past used as embedding",
    "history_source": "[int] number of samples in the source's past used as embedding (default = same as the target history)",
    "tau_source": "[int] [optional] source's embedding delay (default=1)",
    "tau_target": "[int] [optional] target's embedding delay (default=1)",
    "source_target_delay": "[int] [optional] information transfer delay between source and target (default=1)",
    "algorithm_num": "[int] [optional] which Kraskov algorithm (1 or 2) to use (default=1)",
    "local_values": "[optional] [bool] return local TE instead of average TE (default=False)",
    "debug": "[bool] [optional] return debug information when calling JIDT (default=False)",
    
    "discretise_method": "[str] [optional] if and how to discretise incoming continuous data, can be 'max_ent' for maximum entropy binning, 'equal' for equal size bins, and 'none' if no binning is required. (default=none)",
    "n_discrete_bins": "[int] [optional] number of discrete bins/levels or the base of each dimension of the discrete variables. If set, this parameter overwrites/sets alph1 and alph2. (default=2)",
    "alph1": "[int] [optional] number of discrete bins/levels for source (default=2, or the value set for n_discrete_bins) (>= 2)",
    "alph2": "[int] [optional] number of discrete bins/levels for target (default=2, or the value set for n_discrete_bins) (>= 2)",
    "alphc": "[int] [optional] number of discrete bins/levels for conditional (default=2, or the value set for n_discrete_bins) (>= 2)",

    "kraskov_k": "[int] [optional] number of nearest neighbours for KNN search (default=4)",
    "theiler_t": "[int][optional] number of next temporal neighbours ignored in KNN and range searches (default=0)",
    "lag_mi": "[int] [optional] time difference in samples to calculate the lagged MI between processes (default=0)",
    "noise_level": "[float] [optional] random noise added to the data (default=1e-8)",
    "normalise": "[bool] [optional] z-standardise data ",
    "num_threads": "[int | str] [optional] number of threads used for estimation (default='USE_ALL', note that this uses *all* available threads on the current machine).",

    "gpuid": "[int] [optional] device ID used for estimation (if more than one device is available on the current platform) (default=0)",
    "return_counts": "[bool] [optional] return intermediate results, i.e. neighbour counts from range searches and KNN distances (default=False)",
    "padding": "[bool] [optional] pad data to a length that is a multiple of 1024 (default=True)",

    "knn_finder": "[str] [optional] knn algorithm to use, can be 'scipy_kdtree' (default), 'sklearn_kdtree', or 'sklearn_balltree'",
    "rng_seed": "[int | None] [optional] random seed if noise level > 0 (defaults=None)",
    "base": "[float] [optional] base of returned values (default=np.e)",

    "n_perm_max_stat": "[int] [optional] number of max permutations (default=500)",
    "n_perm_min_stat": "[int] [optional] number of min permutations (default=500)",
    "n_perm_omnibus": "[int] [optional] number of permutations for omnibus test (default=500)",
    "n_perm_max_seq": "[int] [optional] number of permutations (default=500)",

    "max_lag_sources": "[int] maximum temporal search depth for candidates in the sources' past in samples",
    "min_lag_sources": "[int] minimum temporal search depth for candidates in the sources' past in samples",
    "max_lag_target": "[int] maximum temporal search depth for candidates in the target's past in samples (default = same as max_lag_sources)",
    "tau_sources": "[int] [optional] spacing between candidates in the sources' past in samples (default=1)",
    "tau_target": "[int] [optional] spacing between candidates in the target's past in samples (default=1)",
    "alpha_min_stats": "[float] [optional] critical alpha level for statistical significance (default=0.05)",
    "alpha_max_stats": "[float] [optional] critical alpha level for statistical significance (default=0.05)",
    "alpha_omnibus_stats": "[float] [optional] critical alpha level for statistical significance (default=0.05)",
    "add_conditionals": "[list of tuples | str] [optional] force the estimator to add these conditionals when estimating TE; can either be a list of variables, where each variable is described as (idx process, lag wrt to current value) or can be a string: 'faes' for Faes-Method",
    "permute_in_time": "[bool] [optional] force surrogate creation by shuffling realisations in time instead of shuffling replications; see documentation of Data.permute_samples() for further settings (default=False)",
    "verbose": "[bool] [optional] toggle console output (default=True)",
    "write_ckp": "[bool] [optional] enable checkpointing, writes analysis state to disk every time a variable is selected; resume crashed analysis using network_analysis.resume_checkpoint() (default=False)",
    "filename_ckp": "[str] [optional] checkpoint file name (without extension)",

    "history": "[int] number of samples in the processes' past used as embedding",
    "tau": "[int] [optional] the processes' embedding delay (default=1)",
    "alph": "[int] [optional] number of discrete bins/levels for var1 (default=2 , or the value set for n_discrete_bins). (>= 2)",

    "alph_s1": "[int] alphabet size of s1",
    "alph_s2": "[int] alphabet size of s2",
    "alph_t": "[int] alphabet size of t",
    "max_unsuc_swaps_row_parm": "[int] soft limit for virtualised swaps based on the number of unsuccessful swaps attempted in a row. If there are too many unsuccessful swaps in a row, then it will break the inner swap loop; the outer loop decrements the size of the probability mass increment and then attemps virtualised swaps again with the smaller probability increment. The exact number of unsuccessful swaps allowed before breaking is the total number of possible swaps (given our alphabet sizes) times the control parameter max_unsuc_swaps_row_parm, e.g., if the parameter is set to 3, this gives a high degree of confidence that nearly (if not) all of the possible swaps have been attempted before this soft limit breaks the swap loop.",
    "num_reps": "[int] number of times the outer loop will halve the size of the probability increment used for the virtualised swaps. This is in direct correspondence with the number of times the empirical data was replicated in your original implementation.",
    "max_iters": "[int] provides a hard upper bound on the number of times it will attempt to perform virtualised swaps in the inner loop. However, this hard limit is (practically) never used as it should always hit the soft limit defined above (parameter may be removed in the future).",
    
    "cone_solver": "[str] [optional] which cone solver to use (default='ECOS')",
    "solver_args": "[dict] solver arguments (default={})",

    "n": "[int] number of pid sources",
    "pdf_orig": "[dict] the original joint distribution of the inputs and the output (realizations are the keys). It doesn't have to be a full support distribution, i.e., it can contain realizations with 'zero' mass probability",
    "chld": "[dict] list of children for each node in the redundancy lattice (nodes are the keys)",
    "achain": "[tuple] tuple of all the nodes (antichains) in the redundacy lattice",
    "printing": "[bool] If True (default) prints the results using PrettyTables",

    "max_lag": "[int] maximum temporal search depth for candidates in the processes' past in samples",
    "tau": "[int] [optional] spacing between candidates in the sources' past in samples (default=1)",
    "alpha_mi": "[float] critical alpha level for statistical significance (default=0.05)",

    "lags_pid": "[list of lists of ints] [optional] - lags in samples between sources and target (default=[[1, 1], [1, 1] ...])", 

    "n_chunks": "[int] - number of data chunks\nno. data points has to be the same for each chunk",

    "embedding_step_size": "[float] [optional] Step size delta t (in seconds) with which the window is slid through the data (default = 0.005).",
    "return_averaged_R": "[bool] [optional] If set to True, compute R̂tot as the average over R̂(T ) for T ∈ [T̂D, Tmax ] instead of R̂tot = R(T̂D ). If set to True, the setting for number_of_bootstraps_R_tot is ignored and set to 0 (default=True)",

    "estimation_method": "[string] The method to be used to estimate the history dependence 'bbc' or 'shuffling'.",
    "embedding_step_size": "[float] [optional] Step size delta t (in seconds) with which the window is slid through the data. (default: 0.005)",
    "embedding_number_of_bins_set": "[list of ints] [optional] Set of values for d, the number of bins in the embedding. (default: [1, 2, 3, 4, 5])",
    "embedding_past_range_set": "[list of floats] [optional] Set of values for T, the past range (in seconds) to be used for embeddings.\n(default: [0.005, 0.00561, 0.00629, 0.00706, 0.00792, 0.00889,\n0.00998, 0.01119, 0.01256, 0.01409, 0.01581, 0.01774, 0.01991,\n0.02233, 0.02506, 0.02812, 0.03155, 0.0354, 0.03972, 0.04456,\n0.05, 0.0561, 0.06295, 0.07063, 0.07924, 0.08891, 0.09976,\n0.11194, 0.12559, 0.14092, 0.15811, 0.17741, 0.19905, 0.22334,\n0.25059, 0.28117, 0.31548, 0.35397, 0.39716, 0.44563, 0.5,\n0.56101, 0.62946, 0.70627, 0.79245, 0.88914, 0.99763, 1.11936,\n1.25594, 1.40919, 1.58114, 1.77407, 1.99054, 2.23342, 2.50594,\n2.81171, 3.15479, 3.53973, 3.97164, 4.45625, 5.0])",
    "embedding_scaling_exponent_set": "[dict] [optional] Set of values for kappa, the scaling exponent for the bins in the embedding.\nShould be a python-dictionary with the three entries 'number_of_scalings', 'min_first_bin_size' and 'min_step_for_scaling'.\ndefaults: {'number_of_scalings': 10, 'min_first_bin_size': 0.005, 'min_step_for_scaling': 0.01})",
    "bbc_tolerance" : "[float] [optional] The tolerance for the Bayesian Bias Criterion. Influences which embeddings are discarded from the analysis. (default: 0.05)",
    "return_averaged_R": "[bool] [optional] Return R_tot as the average over R(T) for T in [T_D, T_max], instead of R_tot = R(T_D).\nIf set to True, the setting for number_of_bootstraps_R_tot (see below) is ignored and set to 0 and CI bounds are not calculated. (default: True)",
    "timescale_minimum_past_range": "[float] [optional] Minimum past range T_0 (in seconds) to take into consideration for the estimation of the information timescale tau_R. (default: 0.01)",
    "number_of_bootstraps_R_max": "[int] [optional] The number of bootstrap re-shuffles that should be used to determine the optimal embedding. (Bootstrap the estimates of R_max to determine R_tot.)\nThese are computed during the 'history-dependence' task because they are essential to obtain R_tot. (default: 250)",
    "number_of_bootstraps_R_tot": "[int] [optional] The number of bootstrap re-shuffles that should be used to estimate the confidence interval of the optimal embedding. (Bootstrap the estimates of R_tot = R(T_D) to obtain a confidence interval for R_tot.).\nThese are computed during the 'confidence-intervals' task. \nThe setting return_averaged_R (see above) needs to be set to False for this setting to take effect. (default: 250)",
    "number_of_bootstraps_nonessential": "[int] [optional] The number of bootstrap re-shuffles that should be used to estimate the confidence intervals for embeddings other than the optimal one. (Bootstrap the estimates of R(T) for all other T.)\n(These are not necessary for the main analysis and therefore default to 0.)",
    "symbol_block_length": "[int] [optional] The number of symbols that should be drawn in each block for bootstrap resampling If it is set to None (recommended), the length is automatically chosen, based on heuristics (default: None)",
    "bootstrap_CI_use_sd": "[bool] [optional] Most of the time we observed normally-distributed bootstrap replications, so it is sufficient (and more efficient) to compute confidence intervals based on the standard deviation (default: True)",
    "bootstrap_CI_percentile_lo": "[float] [optional] The lower percentile for the confidence interval.\nThis has no effect if bootstrap_CI_use_sd is set to True (default: 2.5)",
    "bootstrap_CI_percentile_hi": "[float] [optional] The upper percentiles for the confidence interval.\nThis has no effect if bootstrap_CI_use_sd is set to True (default: 97.5)",
    "analyse_auto_MI": "[bool] [optional] perform calculation of auto mutual information of the spike train (default: True)",
    "auto_MI_bin_size_set": "[list of float] [optional] Set of values for the sizes of the bins (in seconds). (default: [0.005, 0.01, 0.025, 0.05, 0.25, 0.5])",
    "auto_MI_max_delay": "[int] [optional] The maximum delay (in seconds) between the past bin and the response. (default: 5)",
    "visualization": "[bool] [optional] create .eps output image showing the optimization values and graphs for the history dependence and the auto mutual information (default: False)",
    "output_path": "[String] [optional] if visualization is True. Path where the .eps images should be saved.",
    "output_prefix": "[String]  [optional] if visualization is True. Prefix of the output images e.g. <output_prefix>_process0.eps",

}

# define parameters and defaults for each estimator
parameters = {}
parameters["JidtKraskovTE"] = {
    "history_target" : "",
    "history_source" : "",
    "tau_source" : 1,
    "tau_target" : 1,
    "source_target_delay" : 1,
    "algorithm_num": 1,
    "local_values": False,
    "debug": False
}
parameters["JidtDiscreteTE"] = {
    "history_target" : "",
    "history_source" : "",
    "tau_source" : 1,
    "tau_target" : 1,
    "source_target_delay" : 1,
    "discretise_method": "none",
    "n_discrete_bins": 2,
    "alph1": 2,
    "alph2": 2,
    "local_values": False,
    "debug": False
}
parameters["JidtGaussianTE"] = {
    "history_target" : "",
    "history_source" : "",
    "tau_source" : 1,
    "tau_target" : 1,
    "source_target_delay" : 1,
    "local_values": False,
    "debug": False
}
parameters["JidtKraskovMI"] = {
    "kraskov_k" : 4,
    "theiler_t": 1,
    "lag_mi" : 0,
    "noise_level": 1e-8,
    "algorithm_num": 1,
    "normalise": False,
    "local_values": False,
    "debug": False,
    "num_threads": 0,
}
parameters["JidtDiscreteMI"] = {
    "discretise_method": "none",
    "n_discrete_bins": 2,
    "alph1": 2,
    "alph2": 2,
    "lag_mi": 0,
    "local_values": False,
    "debug": False
}
parameters["JidtGaussianMI"] = {
    "lag_mi" : 1,
    "local_values": False,
    "debug": False
}
parameters["OpenCLKraskovMI"] = {
    "gpuid": 0,
    "kraskov_k" : 4,
    "theiler_t": 1,
    "lag_mi" : 0,
    "padding": True,
    "noise_level": 1e-8,
    "normalise": False,
    "return_counts": False,
    "debug": False,
    "n_chunks": 0,
}
parameters["JidtKraskovCMI"] = {
    "kraskov_k" : 4,
    "theiler_t": 1,
    "lag_mi" : 0,
    "noise_level": 1e-8,
    "algorithm_num": 1,
    "normalise": False,
    "local_values": False,
    "debug": False,
    "num_threads": "USE_ALL",
}
parameters["JidtDiscreteCMI"] = {
    "discretise_method": "none",
    "n_discrete_bins": 2,
    "alph1": 2,
    "alph2": 2,
    "alphc": 2,
    "local_values": False,
    "debug": False
}
parameters["JidtGaussianCMI"] = {
    "lag_mi" : 1,
    "local_values": False,
    "debug": False
}
parameters["OpenCLKraskovCMI"] = {
    "gpuid": 0,
    "kraskov_k" : 4,
    "theiler_t": 1,
    "lag_mi" : 0,
    "padding": True,
    "noise_level": 1e-8,
    "normalise": False,
    "return_counts": False,
    "debug": False,
    "n_chunks": 0,
}
parameters["PythonKraskovCMI"] = {
    "kraskov_k" : 4,
    "base": np.e,
    "knn_finder": "scipy_kdtree",
    "noise_level": 1e-8,
    "normalise": False,
    "rng_seed": None,
    "local_values": False,
    "debug": False,
    "num_threads": "USE_ALL",
}
parameters["BivariateTE"] = {
    "max_lag_sources" : "",
    "min_lag_sources" : "",
    "max_lag_target" : "",
    "tau_sources" : 1,
    "tau_target" : 1,
    "add_conditionals": "",
    "alpha_min_stats": 0.05,
    "alpha_max_stats": 0.05,
    "alpha_omnibus_stats": 0.05,
    "permute_in_time": False,
    "verbose": True,
    "write_ckp": False,
    "filename_ckp": "./idtxl_checkpoint",
}
parameters["BivariateMI"] = {
    "max_lag_sources" : "",
    "min_lag_sources" : "",
    "tau_sources" : 1,
    "tau_target" : 1,
    "add_conditionals": "",
    "alpha_min_stats": 0.05,
    "alpha_max_stats": 0.05,
    "alpha_omnibus_stats": 0.05,
    "permute_in_time": False,
    "verbose": True,
    "write_ckp": False,
    "filename_ckp": "./idtxl_checkpoint",
}
parameters["MultivariateTE"] = {
    "max_lag_sources" : "",
    "min_lag_sources" : "",
    "max_lag_target" : "",
    "tau_sources" : 1,
    "tau_target" : 1,
    "add_conditionals": "",
    "alpha_min_stats": 0.05,
    "alpha_max_stats": 0.05,
    "alpha_omnibus_stats": 0.05,
    "permute_in_time": False,
    "verbose": True,
    "write_ckp": False,
    "filename_ckp": "./idtxl_checkpoint",
}
parameters["MultivariateMI"] = {
    "max_lag_sources" : "",
    "min_lag_sources" : "",
    "tau_sources" : 1,
    "tau_target" : 1,
    "add_conditionals": "",
    "alpha_min_stats": 0.05,
    "alpha_max_stats": 0.05,
    "alpha_omnibus_stats": 0.05,
    "permute_in_time": False,
    "verbose": True,
    "write_ckp": False,
    "filename_ckp": "./idtxl_checkpoint",
}
parameters["permutations"] = {
    "n_perm_max_stat": 500,
    "n_perm_min_stat": 200,
    "n_perm_omnibus": 500,
    "n_perm_max_seq": 500,
}
parameters["JidtKraskovAIS"] = {
    "history": "",
    "kraskov_k": 4,
    "tau": 1,
    "theiler_t": 1,
    "noise_level": 1e-8,
    "normalise": False,
    "debug": False,
    "local_values": False,
    "num_threads": "USE_ALL",
    "algorithm_num": 1,
}
parameters["JidtDiscreteAIS"] = {
    "history": "",
    "discretise_method": "none",
    "n_discrete_bins": 2,
    "alph": 2,
    "debug": False,
    "local_values": False,
}
parameters["JidtGaussianAIS"] = {
    "history": "",
    "tau": 1,
    "debug": False,
    "local_values": False,
}
parameters["ActiveInformationStorage"] = {
    "max_lag": "",
    "tau": 1,
    "add_conditionals": "",
    "alpha_min_stats": 0.05,
    "alpha_max_stats": 0.05,
    "alpha_mi": 0.05,
    "permute_in_time": False,
    "verbose": True,
    "write_ckp": False,
    "filename_ckp": "./idtxl_checkpoint",
}
parameters["SydneyPID"] = {
    "alph_s1": "",
    "alph_s2": "",
    "alph_t": "",
    "max_unsuc_swaps_row_parm": "",
    "num_reps": "",
    "max_iters": "",
    "verbose": False,   
}
parameters["TartuPID"] = {
    "cone_solver": "ECOS",
    "solver_args": "{}",
    "verbose": False,
}
parameters["SxPID"] = {
    "verbose": False,
}
parameters["BivariatePID"] = {
    "lags_pid": "",
}
parameters["MultivariatePID"] = {
    "lags_pid": "",
}

parameters["RudeltNSBEstimatorSymbolsMI"] = {
    "embedding_step_size": 0.005,
    "normalise": True,
    "return_averaged_R": True,
}
parameters["RudeltPluginEstimatorSymbolsMI"] = {
    "embedding_step_size": 0.005,
    "normalise": True,
    "return_averaged_R": True,
}
parameters["RudeltBBCEstimator"] = {
    "embedding_step_size": 0.005,
    "normalise": True,
    "return_averaged_R": True,
}
parameters["RudeltShufflingEstimator"] = {
}
parameters["OptimizationRudelt"] = {
    "estimation_method": "",
    "embedding_step_size": 0.005,
    "embedding_number_of_bins_set": [1, 2, 3, 4, 5],
    "embedding_past_range_set": [0.005, 0.00561, 0.00629, 0.00706, 0.00792, 0.00889,
                0.00998, 0.01119, 0.01256, 0.01409, 0.01581, 0.01774, 0.01991,
                0.02233, 0.02506, 0.02812, 0.03155, 0.0354, 0.03972, 0.04456,
                0.05, 0.0561, 0.06295, 0.07063, 0.07924, 0.08891, 0.09976,
                0.11194, 0.12559, 0.14092, 0.15811, 0.17741, 0.19905, 0.22334,
                0.25059, 0.28117, 0.31548, 0.35397, 0.39716, 0.44563, 0.5,
                0.56101, 0.62946, 0.70627, 0.79245, 0.88914, 0.99763, 1.11936,
                1.25594, 1.40919, 1.58114, 1.77407, 1.99054, 2.23342, 2.50594,
                2.81171, 3.15479, 3.53973, 3.97164, 4.45625, 5.0],
    "embedding_scaling_exponent_set": "{'number_of_scalings': 10, 'min_first_bin_size': 0.005, 'min_step_for_scaling': 0.01}",
    "bbc_tolerance": 0.05,
    "return_averaged_R": True,
    "timescale_minimum_past_range": 0.01,
    "number_of_bootstraps_R_max": 250,
    "number_of_bootstraps_R_tot": 250,
    "number_of_bootstraps_nonessential": 0,
    "symbol_block_length": None,
    "bootstrap_CI_use_sd": True,
    "bootstrap_CI_percentile_lo": 2.5,
    "bootstrap_CI_percentile_hi": 97.5,
    "analyse_auto_MI": True,
    "auto_MI_bin_size_set": [0.005, 0.01, 0.025, 0.05, 0.25, 0.5],
    "auto_MI_max_delay": 5,
    "visualization": False,
    "output_path": "",
    "output_prefix": "",
    "debug": False,
}

# ----------------------------------------------------------
# Definition of general variables - End
# ----------------------------------------------------------


# define helper class
class ClickableLineEdit(QLineEdit):
    """Line edit class for example inputs which remove by clicking on it."""
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        self.clicked.emit()
        QLineEdit.mousePressEvent(self, event)


# define Windows classes from here
# ----------------------------------------------------------
class Window(QMainWindow):
    """Abstract class for input windows
    All user input elements (such as layouts, checkboxes, buttons, 
    Line edits, etc) and their functions are predefined here.
    Later the appropriate input sections will be made visible (or not)
    depending on if they are needed (or not).
    """
    def __init__(self):
        super().__init__()

        # set window size
        self.resize(window_width, window_higth)

        # posible data types (numpy and pickle)
        self.data_type_list = ["npy", "p"]

        self.dataorder=""
        self.svn=""

        # default setting for NOT putting default parameters in script
        self.parameter_defaults = False

        # define main layout
        self.pagelayout = QHBoxLayout()
        self.inputlayout = QVBoxLayout()
        self.inputlayout.setContentsMargins(10, 10, 10, 10)
        self.inputlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scriptlayout = QVBoxLayout()
        self.scriptlayout.setContentsMargins(10, 10, 10, 10)

        # input layout (left)
        self.window_label = QLabel()
        self.window_label.setText("")
        self.window_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #self.window_label.setContentsMargins(50,0,50,0)
        self.window_label.setStyleSheet(f"background-color: {background}; color: black; font-size: 14pt; font-weight: bold; border: 2px {yellow}")
        self.inputlayout.addWidget(self.window_label)

        self.header = QLabel()
        self.header.setText("Calculation settings")
        self.header.setStyleSheet("color: black; font-weight: bold;")
        #self.header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.inputlayout.addWidget(self.header)

        # estimator and label
        self.est_label = QLabel()
        self.est_label.setStyleSheet("color: black;")
        self.est_label.setAlignment(Qt.AlignmentFlag.AlignRight)    
        self.estimator = QComboBox()
        self.estimator.activated.connect(self.loadSettings)
        self.estimator.setStyleSheet(f"color: black; \
            background-color: {red};")

        # bi vs multivariate and label
        self.bimulti_label = QLabel()
        self.bimulti_label.setStyleSheet("color: black;")
        self.bimulti_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.bimulti_label.setVisible(False)
        self.bimulti = QComboBox()
        self.bimulti.activated.connect(self.bi_multi_type)
        self.bimulti.setStyleSheet(f"color: black; \
            background-color: {red};")
        self.bimulti.setVisible(False)
        
        # checkbox for nonlinear granger (yes/no) - for Bi-Multivariate or network analysis with JidtGaussianCMI
        self.nonlin_granger_box = QCheckBox(
            "use nonlinear granger.") 
        self.nonlin_granger_box.setStyleSheet(
            "QCheckBox::indicator { border : 1px solid black }"
            "QCheckBox::indicator { background-color : white }"
            "QCheckBox::indicator::checked { background-color : green }"
            "QCheckBox { color: black }")       
        self.nonlin_granger_box.clicked.connect(self.nonlin_granger)
        self.nonlin_granger_box.setChecked(False)
        self.nonlin_granger_box.setVisible(False)

        # single vs network for nonlin
        self.singlevsnetwork = QComboBox()
        self.singlevsnetwork.activated.connect(self.setSingleVsNetwork)
        self.singlevsnetwork.setStyleSheet(f"color: black; \
            background-color: {red};")
        self.singlevsnetwork.setVisible(False)

        self.aisestimator = QComboBox()
        self.aisestimator.activated.connect(self.getAISEstimator)
        self.aisestimator.setStyleSheet(f"color: black; \
            background-color: {red};")
        self.aisestimator.addItems(CMI_estimators)
        self.aisestimator.setVisible(False)

        self.pidestimator = QComboBox()
        self.pidestimator.activated.connect(self.getPIDEstimator)
        self.pidestimator.setStyleSheet(f"color: black; \
            background-color: {red};")
        self.pidestimator.addItems(PID_bimulti_single_estimators)
        self.pidestimator.setVisible(False)

        # create estimator layout
        self.estimatorlayout = QGridLayout()
        self.estimatorlayout.addWidget(self.est_label, 1, 0, 1, 1)
        self.estimatorlayout.addWidget(self.estimator, 1, 1, 1, 1)
        self.estimatorlayout.addWidget(self.bimulti_label, 0, 0, 1, 1)
        self.estimatorlayout.addWidget(self.bimulti, 0, 1, 1, 1)
        self.estimatorlayout.addWidget(self.singlevsnetwork, 2, 1, 1, 1)
        self.estimatorlayout.addWidget(self.aisestimator, 3, 1, 1, 1)
        self.estimatorlayout.addWidget(self.pidestimator, 4, 1, 1, 1)
        self.estimatorlayout.addWidget(self.nonlin_granger_box,5,1,1,1)
        # add estimator layout to inputlayout
        self.inputlayout.addLayout(self.estimatorlayout)

        # data layout grid 2xn
        self.data_layout = QGridLayout()

        self.load_button = QPushButton("Select data file", self)
        self.load_button.clicked.connect(self.openFileDialog)
        self.load_button.setStyleSheet(
            f"background-color: {red}; color: black;")
        self.load_button.setToolTip("Select a raw data file.\nIt can be a numpy (.npy) or a pickle (.p) file\ncontaining the array (2D or 3D) of your data")

        self.data_label = QLabel()
        self.data_label.setText("Data")
        self.data_label.setStyleSheet("color: black;")
        self.data_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # data file
        self.datafile_label = QLabel()
        self.datafile_label.setText("Data file:")
        self.datafile_label.setStyleSheet("color: black;")
        self.datafile_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.data_file = QLineEdit()
        self.data_file.setText("No data file selected yet")
        self.data_file.setStyleSheet(f"background-color: white; \
            color: red; border: 2px inset gray")
        
        # data order
        self.data_order_label = QLabel()
        self.data_order_label.setText("data order:")
        self.data_order_label.setStyleSheet("color: black;")
        self.data_order_label.setAlignment(Qt.AlignmentFlag.AlignRight) 
        self.data_order_box = QComboBox()
        self.data_order_box.setStyleSheet(f"background-color: {red};")
        self.data_order_box.setToolTip("Specify the order of your data:\
            \n'p' - Processes\n's' - samples\n'r' - replications")
        self.data_order_box.activated.connect(self.getDataOrder)
        self.data_order = QHBoxLayout()
        self.data_order.addWidget(self.data_order_label, stretch=1)
        self.data_order.addWidget(self.data_order_box, stretch=1)
        
        # set source, target, cond, reps layout
        self.stclayout = QHBoxLayout()

        # source
        self.source_label = QLabel()
        self.source_label.setText("Source:")
        self.source_label.setStyleSheet("color: black;")
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignRight) 
        self.source = QLineEdit()
        self.source.setText("")
        self.source.setStyleSheet("background-color: white; border: 2px inset gray")
        self.stclayout.addWidget(self.source_label)
        self.stclayout.addWidget(self.source)  

        # target
        self.target_label = QLabel()
        self.target_label.setText("Target:")
        self.target_label.setStyleSheet("color: black;")
        self.target_label.setAlignment(Qt.AlignmentFlag.AlignRight)   
        self.target = QLineEdit()
        self.target.setText("")
        self.target.setStyleSheet("background-color: white; border: 2px inset gray")
        self.stclayout.addWidget(self.target_label)
        self.stclayout.addWidget(self.target)

        # process for AIS and PID
        self.process_label = QLabel()
        self.process_label.setText("Processes:")
        self.process_label.setStyleSheet("color: black;")
        self.process_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.process_label.setVisible(False)
        self.process = ClickableLineEdit()
        self.process.setText("")
        self.process.clicked.connect(self.cleanProcess)
        self.process.setStyleSheet("background-color: white; border: 2px inset gray")
        self.process.setVisible(False)
        self.stclayout.addWidget(self.process_label)
        self.stclayout.addWidget(self.process)

        # conditional
        self.cond_label = QLabel()
        self.cond_label.setText("Conditional:")
        self.cond_label.setStyleSheet("color: black;")
        self.cond_label.setAlignment(Qt.AlignmentFlag.AlignRight) 
        self.conditional = QLineEdit()
        self.conditional.setText("")
        self.conditional.setStyleSheet("background-color: white; border: 2px inset gray")
        self.conditional.setToolTip("Specify the process(es) of your data that should be used as conditioning variable. If no conditional is provided the results are the MI between source and target")
        self.stclayout.addWidget(self.cond_label)
        self.stclayout.addWidget(self.conditional)

        # replication (for AIS and a datadim of 3 in TE and MI etc)
        self.rep_label = QLabel()
        self.rep_label.setText("Replications:")
        self.rep_label.setStyleSheet("color: black;")
        self.rep_label.setAlignment(Qt.AlignmentFlag.AlignRight) 
        self.rep_label.setVisible(False)
        self.replication = QLineEdit()
        self.replication.setText("")
        self.replication.setStyleSheet("background-color: white; border: 2px inset gray")
        self.replication.setVisible(False)
        self.replication.setToolTip("The dimendion of your input data is 3, but the chosen estimator accepts only 1D inputs.\nSpecify the replication of your data you want to estimate.\n\nInput can be an INT (of the replication)\n\nException: In case of MI estimation you can set : to estimate MI over all replications.")
        self.stclayout.addWidget(self.rep_label)
        self.stclayout.addWidget(self.replication)

        # lags (for PID)
        self.lag_label = QLabel()
        self.lag_label.setText("Lags:")
        self.lag_label.setStyleSheet("color: black;")
        self.lag_label.setAlignment(Qt.AlignmentFlag.AlignRight) 
        self.lag_label.setVisible(False)
        self.lags = QLineEdit()
        self.lags.setText("")
        self.lags.setStyleSheet("background-color: white; border: 2px inset gray")
        self.lags.setVisible(False)
        self.lags.setToolTip("")
        self.stclayout.addWidget(self.lag_label)
        self.stclayout.addWidget(self.lags)


        # nchunks
        self.nchunks_label = QLabel()
        self.nchunks_label.setText("Num. chuncks:")
        self.nchunks_label.setStyleSheet("color: black;")
        self.nchunks_label.setAlignment(Qt.AlignmentFlag.AlignRight) 
        self.nchunks = QLineEdit()
        self.nchunks.setText("")
        self.nchunks.setStyleSheet("background-color: white; border: 2px inset gray")
        self.nchunks.setToolTip("In case of OpenCL estimators you need to specify the number of data chunks (number of data points has to be the same for each chunk!)")
        
        # add data inputs to datalayout
        self.data_layout.addWidget(self.data_label, 0, 0, 1, 1)
        self.data_layout.addWidget(self.load_button, 0, 1, 1, 1)
        self.data_layout.addWidget(self.datafile_label, 1, 0, 1, 1)
        self.data_layout.addWidget(self.data_file, 1, 1, 1, 1)
        self.data_layout.addLayout(self.data_order, 2, 1, 1, 1)
        self.data_layout.addLayout(self.stclayout, 3, 0, 1, 2)
        self.data_layout.addWidget(self.nchunks_label, 7, 0, 1, 1)
        self.data_layout.addWidget(self.nchunks, 7, 1, 1, 1)
        self.data_layout.setContentsMargins(0, 10, 0, 0)
        # add datalayout to inputlayout
        self.inputlayout.addLayout(self.data_layout)

        # parameter layout
        self.paramlayout = QHBoxLayout()
        # label
        self.parameter_label = QLabel()
        self.parameter_label.setText("Parameters")
        self.parameter_label.setStyleSheet("color: black;")
        self.parameter_label.setAlignment(Qt.AlignmentFlag.AlignLeft) 
        self.paramlayout.addWidget(self.parameter_label)

        # checkbox to add default parameters to script
        self.parameter_default_box = QCheckBox(
            "Add default parameters to settings in script.") 
        self.parameter_default_box.setStyleSheet(
            "QCheckBox::indicator { border : 1px solid black }"
            "QCheckBox::indicator { background-color : white }"
            "QCheckBox::indicator::checked { background-color : green }"
            "QCheckBox { color: black }")       
        self.parameter_default_box.clicked.connect(
            self.parameter_defaults_in_settings)
        self.parameter_default_box.setChecked(False)
        self.parameter_default_box.setVisible(True)
        self.paramlayout.addWidget(self.parameter_default_box)

        self.paramlayout.setContentsMargins(0, 10, 0, 0)        
        self.inputlayout.addLayout(self.paramlayout)

        # parameter table
        self.parameters = QTableWidget()
        self.parameters.setStyleSheet(f"background-color: white; color: white; border: 1px inset grey;")
        self.parameters.setColumnCount(1)
        self.parameters.setHorizontalHeaderLabels(["value"])
        h = self.parameters.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.inputlayout.addWidget(self.parameters)

        # save results
        self.savelayout = QHBoxLayout()
        self.save_file_layout = QVBoxLayout()
        self.save_results = QCheckBox("Save results") 
        self.save_results.setStyleSheet(
            "QCheckBox::indicator { border : 1px solid black }"
            "QCheckBox::indicator { background-color : white }"
            "QCheckBox::indicator::checked { background-color : green }"
            "QCheckBox { color: black }")       
        self.save_results.clicked.connect(self.saveRes)
        self.save_results.setChecked(True)
        self.save_results.setVisible(True)
        self.savelayout.addWidget(self.save_results)

        self.save_res_button = QPushButton("Specify output file", self)
        self.save_res_button.clicked.connect(self.saveResDialog)
        self.save_res_button.setStyleSheet(
            f"background-color: {red}; color: black;")
        self.save_file_layout.addWidget(self.save_res_button)

        self.save_file = QLineEdit()
        self.save_file.setText("")
        self.save_file.setStyleSheet("background-color: white; border: 2px inset gray")
        self.save_file_layout.addWidget(self.save_file)
        self.savelayout.addLayout(self.save_file_layout)
        self.inputlayout.addLayout(self.savelayout)

        # create buttons
        self.createlayout = QHBoxLayout()
        self.create_button = QPushButton("Create script", self)
        self.create_button.clicked.connect(self.createScript)
        self.create_button.setStyleSheet(f"background-color: {yellow};")
        self.create_button.setEnabled(False)
        self.createlayout.addWidget(self.create_button)
        self.create_mpi_button = QPushButton("Create MPI script", self)
        self.create_mpi_button.clicked.connect(self.createMPIScript)
        self.create_mpi_button.setStyleSheet(f"background-color: {yellow};")
        self.create_mpi_button.setEnabled(False)
        self.createlayout.addWidget(self.create_mpi_button)
        self.inputlayout.addLayout(self.createlayout)

        # script layout (right)
        self.header2 = QLabel()
        self.header2.setText("Generated script")
        self.header2.setStyleSheet("color: black; font-weight: bold;")
        self.header2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.scriptlayout.addWidget(self.header2)

        # script
        self.script = QTextEdit()
        self.cleanScript()
        self.scriptlayout.addWidget(self.script)

        # buttons
        self.buttonlayout = QHBoxLayout()

        # edit
        self.edit_button = QPushButton("Edit", self)
        self.edit_button.clicked.connect(self.editScript)
        self.edit_button.setStyleSheet(f"background-color: {darkblue};")
        self.edit_button.setEnabled(False)
        self.buttonlayout.addWidget(self.edit_button)
        # save
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.saveScript)
        self.save_button.setStyleSheet(f"background-color: {darkblue};")
        self.save_button.setEnabled(False)
        self.buttonlayout.addWidget(self.save_button)
        # run
        self.run_button = QPushButton("Run", self)
        self.run_button.clicked.connect(self.runScript)
        self.run_button.setStyleSheet(f"background-color: {darkblue};")
        self.run_button.setEnabled(False)
        self.run_button.setToolTip("You need to:\n1) save the script before you can run it\nAND\n2) save results need to be set (otherwise the calculation is useless)!\n\nWhen scripts is finished, the message '- DONE' is shown in the terminal.\n\nATTENTION: Depending on the choosen analysis type (e.g. network analysis)\nthe calculation can take a long time.\nIn this cases it is recommended to start the script in the console,\nbecause during calculation the GUI is not usable and a button press can stop the script calculation!")
        self.buttonlayout.addWidget(self.run_button)
        # add buttonlayout to scriptlayout
        self.scriptlayout.addLayout(self.buttonlayout)

        # combine input- (left) and script- (right) layouts    
        self.pagelayout.addLayout(self.inputlayout, stretch=1)
        self.pagelayout.addLayout(self.scriptlayout, stretch=1)

        widget = QWidget()
        widget.setLayout(self.pagelayout)

        self.setCentralWidget(widget)
        self.setStyleSheet(f"background-color: {background};")

    def openFileDialog(self):
        """get input data file"""
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Select data file")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        # set name filter for allowed data types
        f = "Data ("
        for item in self.data_type_list:
            f += f"*.{item} "
        f += ")"
        file_dialog.setNameFilter(f)

        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()
            self.load_button.setStyleSheet(
            f"background-color: {lightblue}; color: black;")
            self.data_file.setText(selected_file[0])
            self.data_file.setStyleSheet(
                "background-color: white; color: green; border: 2px inset gray")
            filename, file_extension = os.path.splitext(selected_file[0])
            self.datatype = str(file_extension)
            self.cleanOutput()
            self.data_order_box.setStyleSheet(f"background-color: {red};")

    def loadSettings(self, AIS=None, PID=None):
        """load parameter, input fields, tooltips etc. depending on label and selected estimator"""
        # get estimator
        self.estimator.setStyleSheet(
            f"color: black; background-color: {lightblue};")
        self.selectedEstimator = self.estimator.currentText()
        # check estimator dict for renaming
        if self.selectedEstimator in Estimator_dict.keys():
            self.selectedEstimator = Estimator_dict[self.selectedEstimator]

        # clear script
        self.cleanScript()
            
        # activate generate script button
        self.create_button.setEnabled(True)

        # activate generate mpi script button
        self.create_mpi_button.setEnabled(True)

        # add use nonlinear granger box for JidtGaussianCMI (network and bi- multivariate)
        if self.label in ["network_analysis", "multivariate"]:
            if self.selectedBiMulti in Nonlinear_BiMulti_type:
                if self.selectedEstimator in Nonlinear_analysis:
                    self.nonlin_granger_box.setVisible(True)
                else:
                    self.nonlin_granger_box.setVisible(False)
                    self.nonlin_granger_box.setChecked(False)
            else:
                self.nonlin_granger_box.setVisible(False)
                self.nonlin_granger_box.setChecked(False)

        # activate nchuck button for opencl estimators
        if str(self.selectedEstimator[:6]) == "OpenCL":
            self.nchunks_label.setVisible(True)
            self.nchunks.setVisible(True)
            self.nchunks.setText("1")
        else:
            self.nchunks_label.setVisible(False)
            self.nchunks.setVisible(False)
            
        # set stylesheet for parameter table
        self.parameters.setStyleSheet(
            "QHeaderView::section {background-color : #8899ad; color: black }"
            "QTableView::item {background-color: white; color: black; border: 1px inset gray}"
            "QTableWidget {background-color: white; color: black; border: 1px inset gray}") 

        # load appropriate parameters for each estimator and set buttons if neccessary  
        if self.label in ["network_analysis", "multivariate"]:
            params = parameters[self.selectedBiMulti] | parameters["permutations"]

        elif self.label == "ais":

            # set process ais tooltips
            self.setAISProcessTooltips()
            
            if self.selectedEstimator == "ActiveInformationStorage":
                # AIS network or single process analysis
                # set data order all
                self.data_order_box.clear()
                self.data_order_box.addItems(data_order_all)

                # set button etc          
                if not AIS:
                    self.singlevsnetwork.setVisible(True)
                    self.singlevsnetwork.clear()
                    self.singlevsnetwork.addItems(["analyse_single_process", "network_analysis"])
                    self.singlevsnetwork.setStyleSheet(f"color: black; background-color: {red};")
                self.aisestimator.setVisible(True)
                if not AIS:
                    self.aisestimator.setStyleSheet(f"color: black; background-color: {red};")

                # get parameter
                params = parameters[self.selectedEstimator] | parameters["permutations"]

            elif self.selectedEstimator == "OptimizationRudelt":
                
                # AIS optimization Rudelt
                # set data order 2D
                self.data_order_box.clear()
                self.data_order_box.addItems(data_order_2dx)
                # set buttons etc
                self.singlevsnetwork.setVisible(False)
                self.aisestimator.setVisible(False)
                
                # set process tooltips
                self.setAISProcessTooltips()
                
                # get parameter
                params = parameters[self.selectedEstimator]
            
            else:
                # AIS estimators (Jidt and RudeltShuffling) 
                # set data order 2D
                self.data_order_box.clear()
                self.data_order_box.addItems(data_order_2d)
                # set buttons etc
                self.singlevsnetwork.setVisible(False)
                self.aisestimator.setVisible(False)

                # get parameter
                params = parameters[self.selectedEstimator]

        elif self.label == "pid":

            # set process pid tooltips
            self.setPIDProcessTooltips()

            if self.selectedEstimator in ["MultivariatePID", "BivariatePID"]:
                # PID network or single process analysis
                # set button etc
                if not PID:
                    self.singlevsnetwork.setVisible(True)
                    self.singlevsnetwork.clear()
                    self.singlevsnetwork.addItems(["analyse_single_target", "network_analysis"])
                    self.singlevsnetwork.setStyleSheet(f"color: black; background-color: {red};")
                self.pidestimator.setVisible(True)
                if not PID:
                    self.pidestimator.setStyleSheet(f"color: black; background-color: {red};")
                # switch to source and target
                self.process_label.setVisible(False)
                self.process.setVisible(False)
                self.lag_label.setVisible(False)
                self.lags.setVisible(False)
                self.source_label.setVisible(True)
                self.source.setVisible(True)
                self.target_label.setVisible(True)
                self.target.setVisible(True)

                # get parameter          
                if not PID:
                    self.selectedPIDEstimator = self.pidestimator.currentText()
                    if self.selectedPIDEstimator in Estimator_dict.keys():
                        self.selectedPIDEstimator = Estimator_dict[self.selectedPIDEstimator]
                params = parameters[self.selectedEstimator] | parameters[self.selectedPIDEstimator] | parameters["permutations"]

            else:
                # PID estimators (Goettingen, Sydney, Tartu)
                # set buttons etc
                self.singlevsnetwork.setVisible(False)
                self.pidestimator.setVisible(False)
                # switch to process and lags
                self.process_label.setVisible(True)
                self.process.setVisible(True)
                self.lag_label.setVisible(True)
                self.lags.setVisible(True)
                self.source_label.setVisible(False)
                self.source.setVisible(False)
                self.target_label.setVisible(False)
                self.target.setVisible(False)

                # set sdt example for process
                if self.selectedEstimator in PID_single_estimators:
                    if self.selectedEstimator == "SxPID":
                        self.process.setText("s1,s2,....up to s4,t")
                    else: 
                        self.process.setText("s1,s2,t")
                    self.process.setStyleSheet("background-color: white; color: red; border: 2px inset gray")
                
                # get parameter
                params = parameters[self.selectedEstimator]

        elif self.label=="MIestimator" and self.selectedEstimator in ["RudeltNSBEstimatorSymbolsMI", "RudeltPluginEstimatorSymbolsMI", "RudeltBBCEstimator"]: 
            # switch to processes
            self.process_label.setVisible(True)
            self.process.setVisible(True)
            self.process.setText("symbol, past_symbol, current_symbol")
            self.process.setStyleSheet("background-color: white; color: red; border: 2px inset gray")
            self.process.setToolTip("[list of ints] indices of three processes \n(in order: symbol_array, past_symbol_array, current_symbol_array)\n\n{gen_att}")
            self.lag_label.setVisible(False)
            self.lags.setVisible(False)
            self.source_label.setVisible(False)
            self.source.setVisible(False)
            self.target_label.setVisible(False)
            self.target.setVisible(False)

            # get parameter
            params = parameters[self.selectedEstimator]

        else:
            self.process_label.setVisible(False)
            self.process.setVisible(False)
            self.source_label.setVisible(True)
            self.source.setVisible(True)
            self.target_label.setVisible(True)
            self.target.setVisible(True)

            # get parameter for Mi, TE, CMI
            params = parameters[self.selectedEstimator]

        # prepare table
        self.parameters.setRowCount(len(params.keys()))
        
        # add parameters to table and get appropr. tooltips
        x = 0
        head = []
        tooltip = "Input of estimator parameters need to be given as the appr. estimator expects. E.g. Lists WITH outer squared brackets.\n\n"
        for keys in params.keys():
            head.append(keys) 
            self.parameters.setItem(x,0,QTableWidgetItem(str(params[keys])))
            tooltip = tooltip + f" - {keys}:\t{parameter_tooltips[keys]}\n"
            x += 1
        self.parameters.setVerticalHeaderLabels(head)

        # set tooltips
        self.parameters.setToolTip(tooltip)

        # clean output filename
        self.cleanOutput()

        # deactivate edit and save button
        self.save_button.setEnabled(False)
        self.edit_button.setEnabled(False)

        # deactivate nonlin granger
        self.nonlin_granger_box.setChecked(False)

    def bi_multi_type(self):
        self.bimulti.setStyleSheet(
            f"color: black; background-color: {lightblue};") 
        self.selectedBiMulti = self.bimulti.currentText()
        self.estimator.setStyleSheet(
            f"color: black; background-color: {red};")
        self.estimator.setEnabled(True)

    def setSingleVsNetwork(self):
        self.singlevsnetwork.setStyleSheet(
            f"color: black; background-color: {lightblue};")
        self.svn = self.singlevsnetwork.currentText()
        if self.svn == "network_analysis":
            if self.label in ["ais", "pid"]:
                self.process_label.setText("Processes:")
                self.process.setText("all")
                if self.label == "ais":
                    self.setAISProcessTooltips()
                    self.aisestimator.setStyleSheet(
                        f"color: black; background-color: {red};")
                    self.cleanScript()
                elif self.label == "pid":
                    self.setPIDProcessTooltips()
                    self.pidestimator.setStyleSheet(
                        f"color: black; background-color: {red};")
                    self.cleanScript()
            else:
                self.source_label.setVisible(True)
                self.source_label.setText("Source:")
                self.source.setVisible(True)
                self.source.setText("all")
                self.target_label.setVisible(True)
                self.target.setVisible(True)
                self.target.setText("all")
        elif self.svn == "analyse_single_target":
            if self.label not in ["ais", "pid"]:
                self.source_label.setVisible(True)
                self.source_label.setText("Source:")
                self.source.setVisible(True)
                self.target_label.setVisible(True)
                self.target.setVisible(True)
            elif self.label == "pid":
                self.setPIDProcessTooltips()
                self.pidestimator.setStyleSheet(
                        f"color: black; background-color: {red};")
                self.cleanScript()
        elif self.svn == "analyse_single_process":
            self.process_label.setText("Process:")
            self.process.setText("")
            self.setAISProcessTooltips()
            self.aisestimator.setStyleSheet(
                    f"color: black; background-color: {red};")
            self.cleanScript()

        # clean output filename
        self.cleanOutput()

    def nonlin_granger(self):
        if self.nonlin_granger_box.isChecked():
            self.source.setToolTip(f"[list of int | int] [optional]\nsingle index or list of indices of source processes\n(default=empty),\n if '', all network nodes excluding the\ntarget node are considered as potential sources.\n\n{gen_att}")
            self.source.setText("")
            # clean output filename
            self.cleanOutput()
        else:
            self.source.setToolTip(f"[list of int | int | 'all'] [optional]\nsingle index or list of indices of source processes\n(default='all'),\n if 'all', all network nodes excluding the\ntarget node are considered as potential sources.\n\n{gen_att}")
            self.source.setText("all")
            # clean output filename
            self.cleanOutput()

    def getEstimatorToolTips(self, estimators):
        # check estimator dict for renaming
        for index, est in enumerate(estimators):
            if est in Estimator_dict.keys():
                estimators[index] = Estimator_dict[est] 
        tt = ""
        for i in range(len(estimators)):
            tt += f"{estimators[i]}:\n {estimator_tooltips[estimators[i]]}\n\n"
        return tt

    def getAISEstimator(self):
        self.aisestimator.setStyleSheet(f"color: black; background-color: {lightblue};")
        self.selectedAISEstimator = self.aisestimator.currentText()
        if self.selectedAISEstimator in Estimator_dict.keys():
            self.selectedAISEstimator = Estimator_dict[self.selectedAISEstimator]
        self.loadSettings(AIS=True)

    def setAISProcessTooltips(self):
        # set process tooltips for AIS depending on selected estimator
        if self.selectedEstimator == "ActiveInformationStorage":
            if self.svn == "analyse_single_process":
                self.process.setToolTip("[int] index of process")
            elif self.svn == "network_analysis":
                self.process.setToolTip(f"[list of int | 'all'] index of processes (default='all')\n\nif 'all', AIS is estimated for all processes\nif list of int, AIS is estimated for processes specified in\nthe list.\n\n{gen_att}")
        elif self.selectedEstimator == "OptimizationRudelt":
            self.process.setToolTip("[list of int] indices of processes - spike times are optimized all processes specified in the list separately.")      
        else:
            self.process.setToolTip("[int] index of process")

    def getPIDEstimator(self):
        self.pidestimator.setStyleSheet(f"color: black; background-color: {lightblue};")
        self.selectedPIDEstimator = self.pidestimator.currentText()
        if self.selectedPIDEstimator in Estimator_dict.keys():
            self.selectedPIDEstimator = Estimator_dict[self.selectedPIDEstimator]
        self.loadSettings(PID=True)

    def setPIDProcessTooltips(self):
        """set source, target, process and lag tooltips for PID depending on selected estimator"""
        if self.selectedEstimator in ["BivariatePID", "MultivariatePID"]:
            if self.svn == "analyse_single_target":
                self.target.setToolTip("[int] index of target process")
                self.source.setToolTip(f"[list of ints] indices of the two source processes\n\n{gen_att}")
            elif self.svn == "network_analysis":
                self.target.setToolTip("[list of ints] indices of target processes")
                self.source.setToolTip(f"[list of lists] indices of the two source processes for each target,\n e.g. [0, 2], [1, 0], \nmust have the same length as targets\n\n{gen_att}")
        else:
            if self.selectedEstimator in ["SydneyPID", "TartuPID"]:
                self.process.setToolTip(f"[list of ints] indices of the processes (in order: s1,s2,t)\n\n{gen_att}")
                self.lags.setToolTip(f"[list of ints] [optional] lags in samples (in the orer s1,s2,t) the processes should be shifted\nmust have the same length as processes\n\n{gen_att}")
            elif self.selectedEstimator == "SxPID":
                self.process.setToolTip(f"[list of ints] indices of the processes (in order: s1,s2,..up to s4,t)\n\n{gen_att}")
                self.lags.setToolTip(f"[list of ints] [optional] lags in samples (in the orer s1,s2,..up to s4,t) the processes should be shifted\n\nmust have the same length as processes\n\n{gen_att}")   
    
    def getDataOrder(self):
        """get len of data order and show replication, id neccessary"""
        self.datadim = len(str(self.data_order_box.currentText()))
        self.dataorder = str(self.data_order_box.currentText())
        self.data_order_box.setStyleSheet(f"background-color: {lightblue};")
        if self.label in ["MIestimator", "TEestimator"]:
            if self.datadim == 3:
                self.rep_label.setVisible(True)
                self.replication.setVisible(True)
            elif self.datadim == 2:
                self.rep_label.setVisible(False)
                self.replication.setVisible(False)

    def saveRes(self):
        """switch save conditions"""
        if self.save_results.isChecked():
            self.save_res_button.setVisible(True)
            self.save_file.setVisible(True)
        else:
            self.save_res_button.setVisible(False)
            self.save_file.setVisible(False)

    def parameter_defaults_in_settings(self):
        if self.parameter_default_box.isChecked():
            self.parameter_defaults = True
        else:
            self.parameter_defaults = False

    def saveResDialog(self):
        """get result file name"""
        # propose output file name
        # get subtext on conditionas
        if self.label == "ais" and self.selectedEstimator == "ActiveInformationStorage":
            subtext = f"_{self.svn}_{self.selectedAISEstimator}"
        elif self.label == "pid" and self.selectedEstimator in ["BivariatePID", "MultivariatePID"]:
            subtext = f"_{self.svn}_{self.pidestimator.currentText()}"
        elif self.label in ["multivariate", "network_analysis"]:
            if self.nonlin_granger_box.isChecked():
                nl="_nonlinear"
            else:
                nl=""
            subtext = f"{nl}_{self.selectedBiMulti}"
        else:
            subtext = ""
        try:
            if self.data_file.text() != "No data file selected yet":
                resultfilename = f"results_{self.label}_{str(self.estimator.currentText())}{subtext}_{os.path.split(self.data_file.text())[1].split('.')[0]}.p"
            else:
                resultfilename = ""
        except:
            resultfilename = ""

        # start ui
        save_file = QFileDialog.getSaveFileName(self, 'Specify output file', resultfilename)
        if save_file[0] != "":
            self.save_res_button.setStyleSheet(
                f"background-color: {lightblue}; color: black;")
            self.save_file.setText(save_file[0])
            self.save_file.setStyleSheet("background-color: white; color: green;")

    def messageBox(self, text):
        """message box for missing inputs or input errors"""
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Input Error")
        dlg.setStyleSheet(f"background-color: {red};")
        dlg.setText(text)
        button = dlg.exec()

    def checkSTPLInput(self):
        """check all STPLR input strings for correct format"""
        testres = [False] * 5
        # test source
        testres[0] = self.checkStringInput(self.source.text())
        # test target
        testres[1] = self.checkStringInput(self.target.text())
        # test process
        testres[2] = self.checkStringInput(self.process.text())
        # test lags
        testres[3] = self.checkStringInput(self.lags.text())
        # test replications
        testres[4] = self.checkStringInput(self.replication.text())
        # check if all tests True
        if all(testres) == True:
            return True
        else:
            return False

    def checkStringInput(self, inputstring):
        """check input string concerning correct input of int, list of ints, list of lists of ints"""
        
        # skip test if input is empty
        if inputstring == "":
            return True
        else:
            teststring = inputstring

        # replace whitespaces
        teststring = teststring.replace(" ","")

        # test single integer input
        if teststring.isdigit():
            return True

        # test allowed strings
        if teststring in ["all"]:
            return True

        # test input for ints if not single int
        t = teststring
        t = t.replace(",","")
        t = t.replace("[","")
        t = t.replace("]","")
        if not t.isdigit():
            self.messageBox(f"{inputstring} is not a proper input!\n\nInputs need to ints, Lists of ints or lists of lists of ints.\n\nSee tooltips for more details.")
            return False

        # test bracket missmatch    
        if teststring.count("[") != teststring.count("]"):
            self.messageBox(f"{inputstring} is not a proper list or list of list input!\n\nbrackets missmatch.")
            return False
    
        # test double squared brackets
        if "[[" and "]]" in teststring:
            self.messageBox(f"{inputstring} is not a proper list or list of list input!\n\nLists and Lists of Lists must be defined WITHOUT outer squared brackets.\n\nSee tooltips for more details.")
            return False
    
        # test if input is list
        testlist = "[" + teststring + "]"
        try:
            x=eval(testlist)
        except:
            self.messageBox(f"{inputstring} is not a proper list or list of list input!\n\nSyntax error")
            return False

        # test if input is list
        if not isinstance(x, list):
            self.messageBox(f"{inputstring} is not a proper list or list of list input!")
            return False

        # check if list contains ints or lists
        if isinstance(x[0], int):
            return True
        elif isinstance(x[0], list):
            # check if list contains single list
            if len(x) == 1:
                self.messageBox(f"{inputstring} is not a proper list input!\n\nLists must be defined WITHOUT outer squared brackets.\n\nSee tooltips for more details.")
                return False
            # check if all lists have same length
            elif len(x) > 1:
                for i in range(len(x)):
                    if i > 0:
                        if len(x[i]) != len(x[i-1]):
                            self.messageBox(f"{inputstring} is not a proper list of list input!\n\nAll lists in list must have same length.")
                            return False
                return True

    def cleanProcess(self):
        self.process.setText("")
        self.process.setStyleSheet("background-color: white; color: black; border: 2px inset gray")

    def cleanOutput(self):
        """clean outpufilename and set specify output file button to red"""
        self.save_file.setText("")
        self.save_res_button.setStyleSheet(
            f"background-color: {red}; color: black;")

    def cleanScript(self):
        """reset script"""
        self.script.setText(
            "Specify calculation parameters and press \"Create Script\".")
        self.script.setStyleSheet(f"background-color: {brown}; border: 2px inset {darkbrown}")
        self.script.setReadOnly(True)
        
    def getScriptImports(self):
        """generate part of Scripts: input"""
        # standard imports
        self.script_standard_import = "# import\nimport os\nimport time\nimport pickle\nimport numpy as np\nfrom idtxl.data import Data\n"

        # get estimator
        if self.label in ["network_analysis", "multivariate"]:
            self.script_estimator_import = f"from idtxl.{estimator_source[self.selectedBiMulti]} import {self.selectedBiMulti}\n\n" 
            #self.script_estimator_import = f"from idtxl.{estimator_source[self.selectedEstimator]} import {self.selectedEstimator}\n\n"
        elif self.label == "ais":
            if self.selectedEstimator == "ActiveInformationStorage":
                estimator = f"from idtxl.{estimator_source[self.selectedEstimator]} import {self.selectedEstimator}"
                aisestimator = f"from idtxl.{estimator_source[self.selectedAISEstimator]} import {self.selectedAISEstimator}\n\n"
                self.script_estimator_import = estimator + "\n" + aisestimator
            elif self.selectedEstimator == "OptimizationRudelt":
                loadspiketrian = "from idtxl.data_spiketime import Data_spiketime\n"
                loadestimator = f"from idtxl.{estimator_source[self.selectedEstimator]} import {self.selectedEstimator}\n\n"
                self.script_estimator_import = loadspiketrian + loadestimator
            else:
                self.script_estimator_import = f"from idtxl.{estimator_source[self.selectedEstimator]} import {self.selectedEstimator}\n\n"
            
        else:
                self.script_estimator_import = f"from idtxl.{estimator_source[self.selectedEstimator]} import {self.selectedEstimator}\n\n"

    def getScriptLoadData(self, MPI=False):
        """generate part of Scripts: load .npy or .p data"""
        # set init string depending MPI or not
        if MPI:
            initstring = "    "
        else:
            initstring = ""

        # if no data selected
        if self.data_file.text() == "No data file selected yet":
            self.messageBox("Input data is not specified.")
            return "stop"
        
        # load data
        if self.datatype == ".npy":
            ld = f"{initstring}dat = np.load(\"{self.data_file.text()}\")"
        elif self.datatype == ".p":
            ld = f"{initstring}with open(r\"{self.data_file.text()}\", \"rb\") as file:\n{initstring}\tdat = pickle.load(file)"

        script_load_data = f"{initstring}# load {self.datatype} data\n" + ld + "\n\n"

        # create idtxl data object
        if self.nonlin_granger_box.isChecked():
            da = f"{initstring}data = Data(dat, \"{self.dataorder}\", normalise=False)"
        elif self.label == "pid":
            da = f"{initstring}data = Data(dat, \"{self.dataorder}\", normalise=False)"
        elif self.selectedEstimator in [ "RudeltNSBEstimatorSymbolsMI", "RudeltPluginEstimatorSymbolsMI", "RudeltBBCEstimator", "RudeltShufflingEstimator"]:
            da = f"{initstring}data = Data(dat, \"{self.dataorder}\", normalise=False)"
        elif self.selectedEstimator in [ "OptimizationRudelt"]:    
            da = f"{initstring}data = Data_spiketime(data=dat)"
        else:
            da = f"{initstring}data = Data(dat, \"{self.dataorder}\")"

        script_idtxl_data = f"{initstring}# create idtxl data object\n" + da + "\n\n"

        if self.label == "pid":
            lags = self.lags.text()
            lags = lags.replace(" ","")
            if lags != "":
                if self.selectedEstimator in PID_single_estimators:
                    procs = self.process.text()
                    procs = procs.replace(" ","")
                    if len(list(map(int, procs.split(",")))) != len(list(map(int, lags.split(",")))):
                        self.messageBox("Processes and lags need to have the same number of comma seperated input values!")
                        return "stop", "stop"
                    else:
                        script_idtxl_data = script_idtxl_data + self.getScriptPIDlag(initstring)
        
        return script_load_data, script_idtxl_data

    def getScriptPIDlag(self, initstring):
        """create script part for PID to cut data depending on lags"""
        l0 = f"{initstring}# create PID data depending on processes and lags\n"
        l1 = f"{initstring}processes = [{str(self.process.text())}]\n"
        l2 = f"{initstring}lags = [{str(self.lags.text())}]\n"
        l3 = f"{initstring}maxlags = max(lags)\n"
        l4 = f"{initstring}dat = np.empty((data.n_processes, int(data.n_samples)-maxlags,1))\n"
        l5 = f"{initstring}for i in range(len(processes))\n"
        l6 = f"{initstring}    dat[i,:,0] = data.data[procs[i],maxlags-lags[i]:data.n_samples-lags[i],0]\n"
        l7 = f"{initstring}data.set_data(dat, {self.dataorder}, normalise=False)\n\n"
        return l0+l1+l2+l3+l4+l5+l6+l7
        
    def getScriptSettings(self, MPI=False):
        """generate part of Scripts: settings"""       
        # set init string depending MPI or not
        if MPI:
            initstring = "    "
        else:
            initstring = ""

        settings = ""
        if self.label in ["network_analysis", "multivariate"]:
            settings += f"{initstring}    \"cmi_estimator\": \"{self.selectedEstimator}\",\n"
            # add source and target for nonlinear analysis
            if self.nonlin_granger_box.isChecked():
                target = str(self.target.text())
                if target == "":
                    self.messageBox("target is not specified.")
                    return "stop"
                settings += f"{initstring}    \"target\": {target},\n"
                
                source = str(self.source.text())
                if source == "all":
                    source = ""
                if source != "":
                    if source.isdigit():
                        settings += f"{initstring}    \"sources\": {source},\n"
                    else:
                        settings += f"{initstring}    \"sources\": [{source}],\n"

            # combine bi/multivariate settings
            params = parameters[self.selectedBiMulti] | parameters["permutations"]

        elif self.label == "ais":
            if self.selectedEstimator == "ActiveInformationStorage":
                settings += f"{initstring}    \"cmi_estimator\": \"{self.selectedAISEstimator}\",\n"
                params = parameters[self.selectedEstimator] | parameters["permutations"]
            else:
                params = parameters[self.selectedEstimator]

        elif self.label == "pid":            
            if self.selectedEstimator in ["MultivariatePID", "BivariatePID"]:
                settings += f"{initstring}    \"pid_estimator\": \"{self.selectedPIDEstimator}\",\n"
                params = parameters[self.selectedEstimator] | parameters[self.selectedPIDEstimator] | parameters["permutations"]
            else:
                params = parameters[self.selectedEstimator]

        else:
            params = parameters[self.selectedEstimator]

        # get parameters and and input values
        row = 0
        for key in params.keys():
            value = str(self.parameters.item(row, 0).text())
            # check if all mandatory paramters are specified
            # and add exceptions for not filled in parameters
            if value == "" and key not in ["add_conditionals", "lags_pid", "output_path", "output_prefix"]:
                self.messageBox(f"Parameter {key} is not specified.")
                return "stop"

            if self.parameter_defaults:
                # add all parameters to script
                if value == "" and key == "add_conditionals":
                    settings = settings
                else:
                    settings += f"{initstring}    \"{key}\": {value},\n"
            else:
                # add only modified parameters to script
                if value != str(params[key]):
                    settings += f"{initstring}    \"{key}\": {value},\n"
            row += 1

        # add MPI settings if MPI script is generated
        if MPI:
            settings += f"{initstring}    \"MPI\": max_workers > 0,\n"
            settings += f"{initstring}    \"max_workers\": max_workers,\n"
        
        return f"{initstring}# settings\n{initstring}" + "settings = {\n" + settings + f"{initstring}" + "}\n\n"

    def getScriptEstimator(self, MPI=False):
        """generate part of Scripts: get esstimator (if neccessary)"""
        # set init string depending MPI or not
        if MPI:
            initstring = "    "
        else:
            initstring = ""

        if self.label in ["network_analysis", "multivariate"]:
            return f"{initstring}# get estimator\n{initstring}{self.label} = {self.selectedBiMulti}()\n\n"

        elif self.label == "ais":
            if self.selectedEstimator == "ActiveInformationStorage":
                return f"{initstring}# get estimator\n{initstring}{self.label} = {self.selectedEstimator}()\n\n"
            else:
                return f"{initstring}# get estimator\n{initstring}{self.label} = {self.selectedEstimator}(settings)\n\n"

        elif self.label == "pid":
            if self.selectedEstimator in ["MultivariatePID", "BivariatePID"]:
                return f"{initstring}# get estimator\n{initstring}{self.label} = {self.selectedEstimator}()\n\n"
            else:
                return f"{initstring}# get estimator\n{initstring}{self.label} = {self.selectedEstimator}(settings)\n\n"
        else:
            return f"{initstring}# get estimator\n{initstring}{self.label} = {self.selectedEstimator}(settings)\n\n"

    def getScriptStartAnalysis(self, MPI=False):
        """generate part of Scripts: start analysis"""
        # set init string depending MPI or not
        if MPI:
            initstring = "    "
        else:
            initstring = ""

        if self.datadim == 2:
            dt = ":"
        elif self.datadim == 3:
            dt = ":,:"

        #empty sprep if NOT nonlinear analyis
        sprep = ""

        if self.label == "network_analysis":
            # get sources and targets
            source = str(self.source.text())
            if source == "" or source == "all":  
                source = "all"
            else:
                source = "[" + source + "]"
            
            target = str(self.target.text())
            if target == "" or target == "all":
                target = "all"
            else:
                target = "[" + target + "]"

            # add nonlinear data preparation 
            if self.nonlin_granger_box.isChecked():
                sprep = f"{initstring}# prepare data object for nonlinear analysis\n{initstring}settings, data = data.prepare_nonlinear(settings, data)\n\n"
 
            # start analysis
            if self.nonlin_granger_box.isChecked():
                if target == "all" and source == "all":
                    sa = f"results = {self.label}.analyse_network(settings, data)"
                elif target != "all" and source != "all":
                    sa = f"results = {self.label}.analyse_network(settings, data,\n{initstring}    target=settings[\"nonlinear_settings\"][\"nonlinear_target_predictors\"],\n{initstring}    sources=settings[\"nonlinear_settings\"][\"nonlinear_source_predictors\"])"
                else:
                    self.messageBox("Input error:\nsource and target can be both all or none of them.")
                    return "stop"
            else:
                if target == "all" and source == "all":
                    sa = f"results = {self.label}.analyse_network(settings, data)"
                elif target != "all" and source != "all":
                    sa = f"results = {self.label}.analyse_network(settings, data, targets={target}, sources={source})"
                else:
                    self.messageBox("Input error:\nsource and target can be both all or none of them.")
                    return "stop"

        elif self.label == "multivariate":
            # get sources and targets
            source = str(self.source.text())
            if source == "" or source == "all":  
                source = "all"
            else:
                source = "[" + source + "]"
            target = str(self.target.text())
            if target == "":   
                self.messageBox("target is not specified.")
                return "stop"
            if target.isdigit():
                target = target
            else:
                target = "[" + target + "]"

            # add nonliner data preparation 
            if self.nonlin_granger_box.isChecked():
                sprep = f"{initstring}# prepare data object for nonlinear analysis\n{initstring}settings, data = data.prepare_nonlinear(settings, data)\n\n"
                
            # start analysis
            if self.nonlin_granger_box.isChecked():
                sa = f"results = {self.label}.analyse_single_target(settings, data,\n{initstring}    target=settings[\"nonlinear_settings\"][\"nonlinear_target_predictors\"],\n{initstring}    sources=settings[\"nonlinear_settings\"][\"nonlinear_source_predictors\"])"

            else:
                if source == "all":
                    sa = f"results = {self.label}.analyse_single_target(settings, data, target={target})"
                else:
                    sa = f"results = {self.label}.analyse_single_target(settings, data, target={target}, sources={source})"

        elif self.label == "CMIestimator":
            # get sources, targets and conds
            source = str(self.source.text())
            if source == "":
                self.messageBox("source is not specified.")
                return "stop"
            target = str(self.target.text())
            if target == "":   
                self.messageBox("target is not specified.")
                return "stop"
            cond = str(self.conditional.text())

            # start analysis
            if cond == "":
                cond_text = ""
            else:
                cond_text = f", conditional=data.data[{cond},:,:]"

            if str(self.selectedEstimator[:6]) == "OpenCL":
                nchunks = str(self.nchunks.text())
                sa = f"results = {self.label}.estimate(data.data[{source},{dt}], data.data[{target},{dt}]{cond_text}, n_chunks={nchunks})"
            else:
                sa = f"results = {self.label}.estimate(data.data[{source},{dt}], data.data[{target},{dt}]{cond_text})"

        elif self.label == "ais":
            # get process
            process = str(self.process.text())
            if process == "":   
                self.messageBox("Process is not specified.")
                return "stop"

            # start analysis
            # ais network or single proces analysis
            if self.selectedEstimator == "ActiveInformationStorage":
                if self.svn == "analyse_single_process":
                    sa = f"results = {self.label}.analyse_single_process(settings, data, process={process})"

                elif self.svn == "network_analysis": 
                    if process == "all":
                        sa = f"results = {self.label}.analyse_network(settings, data)"
                    else:
                        sa = f"results = {self.label}.analyse_network(settings, data, process=[{process}])"

            elif self.selectedEstimator == "OptimizationRudelt":

                sa = f"results = {self.label}.optimize(data, [{process}])" 
            
            # ------------------------------------------------------------------------------------------ Rudelt opt

            elif self.selectedEstimator == "RudeltShufflingEstimator":

                sa = f"results = {self.label}.estimate(np.squeeze(data.data[{process},:]).astype(int))"

            # Jidt AIS estimator
            else:
                if self.datadim == 2:
                    sa = f"{initstring}# start analysis\n{initstring}results = {self.label}.estimate(data.data[{process},{dt}])"
                elif self.datadim == 3:
                    rep = str(self.replication.text())
                    if rep == "":   
                        self.messageBox("Replications are not specified.")
                        return "stop"
                    else:
                        sa = f"results = {self.label}.estimate(data.data[{process},:,{rep}])"

        elif self.label == "pid":

            if self.selectedEstimator in ["MultivariatePID", "BivariatePID"]:
                #get target and source
                source = str(self.source.text())
                if source == "":
                    self.messageBox("source is not specified.")
                    return "stop"
                target = str(self.target.text())
                if target == "":   
                    self.messageBox("target is not specified.")
                    return "stop"
                # start analysis
                if self.svn == "analyse_single_target":
                    sa = f"results = {self.label}.analyse_single_target(settings, data, target={target}, sources=[{source}])"
                elif self.svn == "network_analysis":
                    sa = f"results = {self.label}.analyse_network(settings, data, [{target}], [{source}])"

            elif self.selectedEstimator in ["SydneyPID", "TartuPID"]:
                # get processes
                process = str(self.process.text())
                if process == "":
                    self.messageBox("Processes are not specified.")
                    return "stop"

                # get list from input process string
                procs = process.replace(" ","")
                procs = list(map(int, procs.split(",")))

                # check process input
                if len(procs) != 3:
                    self.messageBox(f"Processes needs to a list of 3 process indices. Given: {len(procs)}")
                    return "stop"

                #start analysis 
                sa = f"results = {self.label}.estimate(data.data[{procs[0]},:], data.data[{procs[1]},:], data.data[{procs[2]},:])"

            elif self.selectedEstimator == "SxPID":
                # get processes
                process = str(self.process.text())
                if process == "":
                    self.messageBox("Processes are not specified.")
                    return "stop"

                # get list from input process string
                procs = process.replace(" ","")
                procs = list(map(int, procs.split(",")))

                # check len of inputs
                if len(procs) < 3:
                    self.messageBox("Processes need at least 3 inputs.\ne.g.: s1,s2,t.")
                    return "stop"
                elif len(procs) > 5:
                    self.messageBox("Processes can have max 5 inputs.\ne.g.: s1,s2,s3,s4,t")
                    return "stop"

                # replace proces with 1 to n in case of lags are set
                if self.lags.text() != "":
                    procs = range(len(procs))

                # loop over sn
                sn = "["
                for i in range(len(procs)-1):
                    sn += f"data.data[{str(i)},:],"
                # remove last comma from loop
                sn = sn[:-1]
                # add target
                sn += f"], data.data[{procs[-1]},:]"

                # start analysis
                sa = f"results = {self.label}.estimate(" + sn + ")"

        else: # TE and MI

            if self.selectedEstimator in [ "RudeltNSBEstimatorSymbolsMI", "RudeltPluginEstimatorSymbolsMI", "RudeltBBCEstimator"]:
                # get processes
                process = str(self.process.text())
                if process == "":
                    self.messageBox("Processes are not specified.")
                    return "stop"

                # get list from input process string
                procs = process.replace(" ","")
                procs = list(map(int, procs.split(",")))

                # check process input
                if len(procs) != 3:
                    self.messageBox(f"Processes needs to a list of 3 process indices. Given: {len(procs)}")
                    return "stop"

                #start analysis 
                sa = f"results = {self.label}.estimate(np.squeeze(data.data[{procs[0]},:]).astype(int), np.squeeze(data.data[{procs[1]},:]).astype(int), np.squeeze(data.data[{procs[2]},:]).astype(int))"
                
            else:
                # get sources and targets
                source = str(self.source.text())
                if source == "":   
                    self.messageBox("source is not specified.")
                    return "stop"
                target = str(self.target.text())
                if target == "":   
                    self.messageBox("target is not specified.")
                    return "stop"
                if self.datadim == 3:
                    rep = str(self.replication.text())
                    if rep == "":   
                        self.messageBox("Replications are not specified.")
                        return "stop"

                # start analysis
                if str(self.selectedEstimator[:6]) == "OpenCL":
                    nchunks = str(self.nchunks.text())
                    if nchunks == "":
                        self.messageBox("num chunks is not specified")
                        return "stop"
                    if self.datadim == 3:
                        sa = f"results = {self.label}.estimate(data.data[{source},:,{rep}], data.data[{target},:,{rep}], n_chunks={nchunks})"
                    else:
                        sa = f"results = {self.label}.estimate(data.data[{source},{dt}], data.data[{target},:], n_chunks={nchunks})"
                else:
                    if self.datadim == 3:
                        sa = f"results = {self.label}.estimate(data.data[{source},:,{rep}], data.data[{target},:,{rep}])"
                    else:
                        sa = f"results = {self.label}.estimate(data.data[{source},:], data.data[{target},:])"

        return sprep + f"{initstring}# start analysis\n{initstring}" + sa + "\n\n"

    def getScriptSaveResults(self, MPI=False):
        """generate part of Scripts: save results"""      
        # set init string depending MPI or not
        if MPI:
            initstring = "    "
        else:
            initstring = ""

        if self.save_results.isChecked():
            output_file_name = str(self.save_file.text())
            if output_file_name == "":   
                self.messageBox("No output file specified.")
                return "stop"

            sr = f"{initstring}pickle.dump(results, open(\"{output_file_name}\", \"wb\"))"
            
            script_save_results = f"{initstring}# save results\n" + sr + "\n\n"
            self.save_res = True
        else:
            script_save_results = "\n"
            self.save_res = False

        return script_save_results

    def createScript(self):
        """create analysis script."""
        
        # check dataorder
        if self.dataorder == "":
            self.messageBox("data order is not selected.")
            return
        
        # check STPL inputs for correct input formats
        if not self.checkSTPLInput():
            return

        # clean mpi script
        self.mpiScript = ""

        # header
        self.script_header = f"\"\"\"Script generated by {ui_name}\"\"\"\n\n" 

        # get imports
        self.getScriptImports()

        # load and set data
        self.script_load_data, self.script_idtxl_data = self.getScriptLoadData()
        
        # get settings
        self.script_settings = self.getScriptSettings()

        # estimator
        self.script_get_estimator = self.getScriptEstimator()

        # start analysis
        self.script_start_analysis = self.getScriptStartAnalysis()

        # save results
        self.script_save_results = self.getScriptSaveResults()

        # define order of script text blocks
        scriptorder = [self.script_header,
            self.script_standard_import,
            self.script_estimator_import,
            self.script_load_data,
            self.script_idtxl_data,
            self.script_settings,
            self.script_get_estimator,
            self.script_start_analysis,
            self.script_save_results,
            ]

        # concatenate text blocks
        self.analysisScript = ""
        for item in scriptorder:
            if str(item) == "stop":
                self.analysisScript = ""
                return
            else:
                self.analysisScript += str(item)
        
        # show script if not empty
        if self.analysisScript != "":        
            # show text
            self.script.setText(self.analysisScript)
            self.script.setReadOnly(True)
            self.script.setStyleSheet(f"background-color: {brown}; border: 2px inset {darkbrown}")

            # activate edit and save button
            self.edit_button.setEnabled(True)
            self.save_button.setEnabled(True)

    def createMPIScript(self):
        """create MPI script"""

        # check dataorder
        if self.dataorder == "":
            self.messageBox("data order is not selected.")
            return

        # check STPL inputs for correct input formats
        if not self.checkSTPLInput():
            return

        # clean analysis script
        self.analysisScript = ""

        # header
        self.script_header = f"\"\"\"MPI script generated by {ui_name}\n\n\
        Start script using (depending on your OS and installed MPI implementation and your number of threads (-n <your_num_threads> python <your_script_name.py> <your_num_threads>)):\n\n\
        e.g. with num_threads = 16:\n\
            mpirun -n 16 python <your_script_name.py> 16\n\
            srun -n 16 python <your_script_name.py> 16\n\
            mpiexec -n 16 python <your_script_name.py> 16\n\"\"\"\n\n"

        # get imports
        self.getScriptImports()

        # import MPI
        self.script_mpi_import = "from mpi4py import MPI\n"

        # def main
        self.script_mpi_main = "def main(args):\n    assert MPI.COMM_WORLD.Get_rank() == 0\n    max_workers = int(args[1])\n"

        # load and set data
        self.script_load_data, self.script_idtxl_data = self.getScriptLoadData(MPI=True)

        # settings
        self.script_settings = self.getScriptSettings(MPI=True)

        # estimator
        self.script_get_estimator = self.getScriptEstimator(MPI=True)

        # start analysis
        self.script_start_analysis = self.getScriptStartAnalysis(MPI=True)

        # save results
        self.script_save_results = self.getScriptSaveResults(MPI=True)

        # mpi def init
        self.script_mpi_init = "if __name__ == \"__main__\":\n    main(sys.argv)\n" 

        # define order of script text blocks
        scriptorder = [self.script_header,
            self.script_standard_import,
            self.script_mpi_import,
            self.script_estimator_import,
            self.script_mpi_main, 
            self.script_load_data,
            self.script_idtxl_data,
            self.script_settings,
            self.script_get_estimator,
            self.script_start_analysis,
            self.script_save_results,
            self.script_mpi_init
            ]
        
        # concatenate text blocks
        self.mpiScript = ""
        for item in scriptorder:
            if str(item) == "stop":
                self.mpiScript = ""
                return
            else:
                self.mpiScript += str(item)

        # show script if not empty
        if self.mpiScript != "":  
            # show scripts
            self.script.setText(self.mpiScript)
            self.script.setReadOnly(True)
            self.script.setStyleSheet(f"background-color: {brown}; border: 2px inset {darkbrown}")

            # activate edit and save button
            self.edit_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.run_button.setEnabled(False)

    def editScript(self):
        """make script editable"""
        self.script.setStyleSheet(f"background-color: white; border: 2px inset {darkbrown}")
        self.script.setReadOnly(False)

    def saveScript(self):
        """save script dialog and propose filename"""
        # propose scripts name
        # get subtext on conditionas
        if self.label == "ais" and self.selectedEstimator == "ActiveInformationStorage":
            subtext = f"_{self.svn}_{self.selectedAISEstimator}"
        elif self.label == "pid" and self.selectedEstimator in ["BivariatePID", "MultivariatePID"]:
            subtext = f"_{self.svn}_{self.pidestimator.currentText()}"
        elif self.label in ["multivariate", "network_analysis"]:
            if self.nonlin_granger_box.isChecked():
                nl = "_nonlinear"
            else:
                nl = ""
            subtext = f"{nl}_{self.selectedBiMulti}"
        else:
            subtext = ""
        # set proposed script name
        if self.mpiScript != "":
            scriptname = f"mpiscript_{self.label}_{str(self.estimator.currentText())}{subtext}_{os.path.split(self.data_file.text())[1].split('.')[0]}.py"
        elif self.analysisScript != "":
            scriptname = f"script_{self.label}_{str(self.estimator.currentText())}{subtext}_{os.path.split(self.data_file.text())[1].split('.')[0]}.py"

        # start ui
        name = QFileDialog.getSaveFileName(
            self, 'Save File', scriptname, "Python Files (*.py)")
        if name[0] != "":
            self.last_script_name = name[0]
            file = open(name[0],'w')
            text = self.script.toPlainText()
            file.write(text)
            file.close()
            if self.analysisScript != "":
                # activate run button if save results are set. 
                # Otherwise calculation useless.
                if self.save_res:
                    self.run_button.setEnabled(True)
                else:
                    self.run_button.setEnabled(False)
            else:
                self.run_button.setEnabled(False)

        self.script.setReadOnly(True)
        self.script.setStyleSheet(f"background-color: {brown}; border: 2px inset {darkbrown}")

    def runScript(self):
        """run last saved script"""
        print(f"Run: {self.last_script_name}")
        os.system(f"python {self.last_script_name}")
        print("- DONE")


class TEWindow(Window):
    """Window for transfer entropy (TE) analysis"""
    def __init__(self):
        super().__init__()

        # set window title and label
        self.setWindowTitle(f"{ui_name} Transfer Entropy")
        self.window_label.setText("Transfer Entropy")
        self.label = "TEestimator"

        # set estimators
        self.est_label.setText("TE estimator:")   
        self.estimator.addItems(TE_estimators)
        self.estimator.setToolTip(self.getEstimatorToolTips(TE_estimators));
        
        # set inputs and tooltips
        self.source.setToolTip("[int] index of source process.")
        self.target.setToolTip("[int] index of target process.")
        self.data_order_box.addItems(data_order_all)
        self.cond_label.setVisible(False)
        self.conditional.setVisible(False)
        self.nchunks_label.setVisible(False)
        self.nchunks.setVisible(False)
        
        
class MIWindow(Window):
    """Window for mutual information (MI) analysis"""
    def __init__(self):
        super().__init__()

        # set window title and label
        self.setWindowTitle(f"{ui_name} Mutual Information")
        self.window_label.setText("Mutual Information")
        self.label = "MIestimator"

        # set estimator and tooltips
        self.est_label.setText("MI estimator:")   
        self.estimator.addItems(MI_estimators)
        self.estimator.setToolTip(self.getEstimatorToolTips(MI_estimators))

        # set inputs and tooltips
        self.source.setToolTip("[int] index of source process.")
        self.target.setToolTip("[int] index of target process.")
        self.data_order_box.addItems(data_order_all)
        self.cond_label.setVisible(False)
        self.conditional.setVisible(False)
        self.nchunks_label.setVisible(False)
        self.nchunks.setVisible(False)


class CMIWindow(Window):
    """Window for conditional mutual information (CMI) analysis"""
    def __init__(self):
        super().__init__()

        # set window title and label
        self.setWindowTitle(f"{ui_name} Conditional MI and TE")
        self.window_label.setText("Conditional MI and TE")
        self.label = "CMIestimator"

        # set estimator and tooltips
        self.est_label.setText("CMI estimator:")   
        self.estimator.addItems(CMI_estimators)
        self.estimator.setToolTip(self.getEstimatorToolTips(CMI_estimators))
        
        # set inputs and tooltips
        self.source.setToolTip("[int] index of source process.")
        self.target.setToolTip("[int] index of target process.")
        self.conditional.setToolTip("[int] index of condtitional process.")
        self.data_order_box.addItems(data_order_all)
        self.nchunks_label.setVisible(False)
        self.nchunks.setVisible(False)


class NetworkWindow(Window):
    """Window for network analysis"""
    def __init__(self):
        super().__init__()

        # set window title and label
        self.setWindowTitle(f"{ui_name} Network Analysis")
        self.window_label.setText("Network Analysis")
        self.label = "network_analysis"

        # set estimators (and related fields) and their tooltips
        self.est_label.setText("Estimator:")   
        self.estimator.addItems(Network_analysis)
        self.estimator.setToolTip(self.getEstimatorToolTips(Network_analysis))
        self.estimator.setEnabled(False)
        self.bimulti_label.setVisible(True)
        self.bimulti_label.setText("Bi- or Multivariate TE/MI:")  
        self.bimulti.setVisible(True)
        self.bimulti.addItems(BiMulti_type)
        self.bimulti.setToolTip("MultivariateTE: Estimate multivariate transfer entropy between all nodes in the network or between selected sources and targets\n\nMultivariateMI: Estimate multivariate mutual information between all nodes in the network or between selected sources and targets\n\nBivariateTE: Estimate bivariate transfer entropy between all nodes in the network or between selected sources and targets\n\nBivariateMI: Estimate bivariate mutual information between all nodes in the network or between selected sources and targets\n\n")

        # set inputs and tooltips
        self.source.setToolTip(f"[list of int | list of list | 'all'] [optional]\nindices of source processes for each target (default='all').\n\nif 'all', all network nodes excluding the target node are\nconsidered as potential sources and tested\ņif list of int, the source specified by each int is tested as\na potential source for the target with the same index or a\nsingle target\nif list of list, sources specified in each inner list are\ntested for the target with the same index\n\n{gen_att}")
        self.target.setToolTip(f"[list of int | 'all'] [optional] index of target processes (default='all').\n\n{gen_att}")
        self.data_order_box.addItems(data_order_all)
        self.target.setText("all")
        self.source.setText("all")
        self.cond_label.setVisible(False)
        self.conditional.setVisible(False)
        self.nchunks_label.setVisible(False)
        self.nchunks.setVisible(False)


class MultivariateWindow(Window):  
    """Window for bi- and multivariate analysis"""
    def __init__(self):
        super().__init__()

        # set window title and label
        self.setWindowTitle(f"{ui_name} Multivariate Analysis")
        self.window_label.setText("Multivariate Analysis")
        self.label = "multivariate"

        # set estimators (and related fields) and their tooltips
        self.est_label.setText("Estimator:")   
        self.estimator.addItems(Multivariate_estimators)
        self.estimator.setToolTip(
            self.getEstimatorToolTips(Multivariate_estimators))
        self.estimator.setEnabled(False)
        self.bimulti_label.setVisible(True)
        self.bimulti_label.setText("Bi- or Multivariate TE/MI:")  
        self.bimulti.setVisible(True)
        self.bimulti.addItems(BiMulti_type)
        self.bimulti.setToolTip("MultivariateTE: Estimate multivariate transfer entropy between sources and a target.\n\nMultivariateMI: Estimate multivariate mutual information between sources and a target.\n\nBivariateTE: Estimate bivariate transfer entropy between sources and a target.\n\nBivariateMI: Estimate bivariate mutual information between sources and a target.\n\n")

        # set inputs and tooltips
        self.source.setToolTip(f"[list of int | int | 'all'] [optional]\nsingle index or list of indices of source processes\n(default='all'),\n if 'all', all network nodes excluding the\ntarget node are considered as potential sources.\n\n{gen_att}")
        self.source.setText("all")
        self.target.setToolTip(f"[int] index of target process")
        self.data_order_box.addItems(data_order_all)
        self.cond_label.setVisible(False)
        self.conditional.setVisible(False)
        self.nchunks_label.setVisible(False)
        self.nchunks.setVisible(False)


class AISWindow(Window):
    """Window for active information storage (AIS)"""
    def __init__(self):
        super().__init__()

        # set window title and label
        self.setWindowTitle(f"{ui_name} Active Information Storage")
        self.window_label.setText("Active Information Storage")
        self.label = "ais"

        # set estimators (and related fields) and their tooltips
        self.est_label.setText("Estimator:")   
        self.estimator.addItems(AIS_estimators)
        self.estimator.setToolTip(self.getEstimatorToolTips(AIS_estimators))
        self.singlevsnetwork.setToolTip("Choose between:\nanalyse_single_process: Estimate active information storage for one process in the network.\nand\nnetwork_analysis: Estimate active information storage for all or a subset of processes in the network.")
        self.aisestimator.setToolTip(
            self.getEstimatorToolTips(AIS_CMI_estimators))
        
        # set inputs
        self.data_order_box.addItems(data_order_2d)
        self.process_label.setVisible(True)
        self.process.setVisible(True)
        self.source_label.setVisible(False)
        self.source.setVisible(False)
        self.target_label.setVisible(False)
        self.target.setVisible(False)
        self.cond_label.setVisible(False)
        self.conditional.setVisible(False)
        self.nchunks_label.setVisible(False)
        self.nchunks.setVisible(False)


class PIDWindow(Window):
    """Window for partial information decomposition (PID)"""
    def __init__(self):
        super().__init__()

        # set window title and label
        self.setWindowTitle(f"{ui_name} Partial Information Decomposition")
        self.window_label.setText("Partial Information Decomposition")
        self.label = "pid"

        # set estimators (and related fields) and their tooltips
        self.est_label.setText("Estimator:")   
        self.estimator.addItems(PID_estimators)
        self.estimator.setToolTip(self.getEstimatorToolTips(PID_estimators)) 
        self.singlevsnetwork.setToolTip("Choose between:\nanalyse_single_process: Estimate partial information decomposition (PID) for a target node in the network. \nand\nnetwork_analysis: Estimate partial information decomposition (PID) for multiple nodes in the network")
        self.pidestimator.setToolTip(
            self.getEstimatorToolTips(PID_bimulti_single_estimators))
 
        # set inputs
        self.data_order_box.addItems(data_order_2d)
        self.target_label.setVisible(False)
        self.target.setVisible(False)
        self.source_label.setVisible(False)
        self.source.setVisible(False)
        self.cond_label.setVisible(False)
        self.conditional.setVisible(False)
        self.process_label.setVisible(True)
        self.process.setVisible(True)
        self.lag_label.setVisible(True)
        self.lags.setVisible(True)
        self.nchunks_label.setVisible(False)
        self.nchunks.setVisible(False)


class MainWindow(QMainWindow):
    """Main Window with buttons to select analysis type 
    e.g. AIS, PID, network analysis, etc
    """
    def __init__(self):
        super().__init__()

        # set window title and size
        self.setWindowTitle(f"{ui_name} Launcher")
        self.setStyleSheet(f"background-color: {yellow}")
        self.resize(400, 400)

        # Define buttons  
        # mi button
        mi_button = QPushButton("Mutual Information (MI)")
        mi_button.setFixedHeight(40)
        mi_button.setStyleSheet(f"background-color: {lightblue};")
        mi_button.pressed.connect(self.mi_button)

        # te button
        te_button = QPushButton("Transfer Entropy (TE)")
        te_button.setFixedHeight(40)
        te_button.setStyleSheet(f"background-color: {lightblue};")
        te_button.pressed.connect(self.te_button)

        # cmi button
        cmi_button = QPushButton("Conditional Mutual Information")
        cmi_button.setFixedHeight(40)
        cmi_button.setStyleSheet(f"background-color: {lightblue};")
        cmi_button.pressed.connect(self.cmi_button)

        # multivariate button
        mmi_button = QPushButton("Bi- or Multivariate MI and TE")
        mmi_button.setFixedHeight(40)
        mmi_button.setStyleSheet(f"background-color: {lightblue};")
        mmi_button.pressed.connect(self.mmi_button)

        # network button
        net_button = QPushButton("Network Analysis")
        net_button.setFixedHeight(40)
        net_button.setStyleSheet(f"background-color: {lightblue};")
        net_button.pressed.connect(self.net_button)

        # ais button
        ais_button = QPushButton("Active Information Storage")
        ais_button.setFixedHeight(40)
        ais_button.setStyleSheet(f"background-color: {lightblue};")
        ais_button.pressed.connect(self.ais_button)

        # pid button
        pid_button = QPushButton("Partial Information Decomposition")
        pid_button.setFixedHeight(40)
        pid_button.setStyleSheet(f"background-color: {lightblue};")
        pid_button.pressed.connect(self.pid_button)

        # define layout
        layout = QVBoxLayout()

        # add buttons to layout
        layout.addWidget(mi_button)
        layout.addWidget(te_button)
        layout.addWidget(cmi_button)
        layout.addWidget(mmi_button)
        layout.addWidget(net_button)
        layout.addWidget(ais_button)
        layout.addWidget(pid_button)
        
        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
        self.setStyleSheet(f"background-color: {background};")

    # button defs to start windows
    def te_button(self):
        # start TE window
        self.te = TEWindow()
        self.te.show()

    def mi_button(self):
        # start MI window
        self.mi = MIWindow()
        self.mi.show()

    def mmi_button(self):
        # start MMI window
        self.mmi = MultivariateWindow()
        self.mmi.show()

    def cmi_button(self):
        # start CMI window
        self.cmi = CMIWindow()
        self.cmi.show()

    def ais_button(self):
        # start AIS window
        self.ais = AISWindow()
        self.ais.show()

    def pid_button(self):
        # start PID window
        self.pid = PIDWindow()
        self.pid.show()

    def net_button(self):
        # start NET window
        self.net = NetworkWindow()
        self.net.show()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()

