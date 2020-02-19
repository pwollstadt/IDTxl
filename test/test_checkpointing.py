"""Provide unit tests for IDTxl checkpointing."""
import os
import subprocess
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.network_comparison import NetworkComparison
import copy
import time
from idtxl.idtxl_utils import calculate_mi
from idtxl.estimators_jidt import JidtDiscreteCMI
from idtxl.multivariate_mi import MultivariateMI
from idtxl.bivariate_mi import BivariateMI


def test_checkpoint_resume():
    """Test resuming from manually generated checkpoint."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=5)


    # Initialise analysis object and define settings
    filename_ckp = './my_checkpoint'
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'write_ckp': True,
                'filename_ckp': filename_ckp}

    # Manually create checkpoint files for two targets.
    # Note that variables are added as absolute time indices and not lags wrt.
    # the current value. Internally, IDTxl operates with abolute indices, but in
    # all results and console outputs, variables are described by their lags
    # wrt. to the current value.
    sources = [0, 1, 2]
    targets = [3, 4]
    add_vars = [[(0, 1), (0, 2), (3, 1)], [(0, 1), (0, 2), (1, 3), (4, 2)]]
    network_analysis._set_checkpointing_defaults(
        settings, data, sources, targets)
    for i in range(len(targets)):
        network_analysis._initialise(settings, data, sources, targets[i])
        print(i)
        network_analysis.selected_vars_full = add_vars[i]
        network_analysis._update_checkpoint('{}.ckp'.format(filename_ckp))

    # Resume analysis.
    network_analysis_res = MultivariateTE()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp)

    # Test if variables were added correctly
    #for i in range(len(targets)):
    #    network_analysis_res._initialise(settings, data, sources, targets[i])
    #    assert network_analysis_res.selected_vars_full == add_vars[i], (
    #        'Variables were not added correctly for target {0}.'.format(
    #            targets[i]))

    # Test if analysis runs from resumed checkpoint.
    results = network_analysis_res.analyse_network(
        settings=settings, data=data, targets=targets, sources=sources)

    # Remove test files.
    #os.remove('{0}.ckp'.format(filename_ckp))
    #os.remove('{0}.ckp.old'.format(filename_ckp))
    #os.remove('{0}.dat'.format(filename_ckp))
    #os.remove('{0}.json'.format(filename_ckp))

def test_checkpoint_copy():
    """Assert that a copy of the current checkpoint is written upon update."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=5)

    # Initialise analysis object and define settings
    filename_ckp = './my_checkpoint'
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'write_ckp': True,
                'filename_ckp': filename_ckp}

    # Manually create checkpoint files for two targets.
    # Note that variables are added as absolute time indices and not lags wrt.
    # the current value. Internally, IDTxl operates with abolute indices, but in
    # all results and console outputs, variables are described by their lags
    # wrt. to the current value.
    sources = [0, 1, 2]
    targets = [3, 4]
    #add_vars = [[(0, 1), (0, 2), (3, 1)], [(0, 1), (0, 2), (1, 3), (4, 2)]]
    network_analysis._initialise(settings, data, sources, targets)
    network_analysis._write_checkpoint()

    #assert os.path.isfile('{0}.ckp'.format(filename_ckp)), (
    #    'No checkpoint file {}.ckp was written.'.format(filename_ckp))
    #assert os.path.isfile('{0}.ckp.old'.format(filename_ckp)), (
    #    'No checkpoint file {}.ckp.old was written.'.format(filename_ckp))

    # Remove test files.
    #os.remove('{0}.ckp'.format(filename_ckp))
    #os.remove('{0}.ckp.old'.format(filename_ckp))
    #os.remove('{0}.dat'.format(filename_ckp))
    #os.remove('{0}.json'.format(filename_ckp))

def test_checkpoint_equal():
    """Assert that a result from a resumed checkpoint is equal to the original start."""
    filename_ckp = './my_checkpoint'
    filepath = "test/save_all_checkpoints.sh"
    # Running Script to save all Checkpoints
    subprocess.Popen(['./' + filepath, filename_ckp  + ".ckp.old"])

    """Test resuming from manually generated checkpoint."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=5)


    # Initialise analysis object and define settings
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'write_ckp': True,
                'filename_ckp': filename_ckp}

    # Setting sources and targets for the analyis
    sources = [0, 1, 2]
    targets = [3, 4]
    #add_vars = [[(0, 1), (0, 2), (3, 1)], [(0, 1), (0, 2), (1, 3), (4, 2)]]
    #network_analysis._initialise(settings, data, sources, targets[0])
    #network_analysis._initialise(settings, data, sources, targets[1])

    # Starting Analysis of the Network
    results = network_analysis.analyse_network(
        settings=settings, data=data, targets=targets, sources='all')

    #Closing running script which saves all checkpoints seperately
    os.system("ps -ef | grep save_all_checkpoints.sh | grep -v grep | awk '{print $2}' | xargs kill")
    proc = subprocess.Popen(["ls -dq *my_checkpoint.ckp.old* | wc -l "], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    checkpoints = int(out)

    #Resuming Analysis from every checkpoint created
    #os.system("mv my_checkpoint.ckp cmp_checkpoint.ckp")
    for i in range(2, checkpoints + 1):
        check = './my_checkpoint.ckp.old'

        # cannot resume first checkpoint properly
        print(i)
        #print(check+str(1))
        data, settings, targets, sources = network_analysis.resume_checkpoint(check+str(i))

    #data, settings, targets, sources = network_analysis.resume_checkpoint("my_checkpoint.ckp.old14")
        print()
        results = network_analysis.analyse_network(
            settings=settings, data=data, targets=targets, sources=sources)
        os.system("mv my_checkpoint.ckp cmp_checkpoint.ckp" + str(i))

     #first checkpoint does not include any sources


# TODO: testing with other estimators estimators_jidt.py
# analytically with JAVA changeable in settings for Gaussian s.t. not random

def JidtDiscreteCMI_MMI_checktest():

    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = './run_checkpoint'
    filename_ckp2 = './resume_checkpoint'

    """Test running analyis without any checkpoint setting."""
    # Generate Gaussian test data
    covariance = 0.4
    n = 100
    delay = 1
    sources = np.random.normal(0, 1, size=n)
    targets = (covariance * sources + (1 - covariance) *
              np.random.normal(0, 1, size=n))
    corr_expected = covariance / (
        1 * np.sqrt(covariance**2 + (1-covariance)**2))
    expected_mi = calculate_mi(corr_expected)
    sources = sources[delay:]
    targets = targets[:-delay]


    print("sources:")
    print(sources)
    #print(sources[5000])
    print("targets:")
    print(targets)
    #print(targets[5000])

    # Discretise data
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 5}
    est = JidtDiscreteCMI(settings)
    source_dis, target_dis = est._discretise_vars(var1=sources, var2=targets)
    data = Data(np.vstack((source_dis, target_dis)),
                dim_order='ps', normalise=False)


    # Initialise analysis object and define settings
    network_analysis1 = MultivariateMI()
    network_analysis2 = MultivariateMI()
    network_analysis3 = MultivariateMI()
    network_analysis4 = MultivariateMI()

    #Settings without checkpointing
    settings1 = {
                'cmi_estimator': 'JidtDiscreteCMI',
                'discretise_method': 'none',
                'n_discrete_bins': 5,
                'n_perm_max_stat': 21,
                'n_perm_omnibus': 30,
                'n_perm_max_seq': 30,
                'min_lag_sources': 1,
                'max_lag_sources': 2
                }

    #settings with checkpointing
    settings2 = {
                'cmi_estimator': 'JidtDiscreteCMI',
                'discretise_method': 'none',
                'n_discrete_bins': 5,
                'n_perm_max_stat': 21,
                'n_perm_omnibus': 30,
                'n_perm_max_seq': 30,
                'min_lag_sources': 1,
                'max_lag_sources': 2,
                'write_ckp': True,
                'filename_ckp': filename_ckp1
                }

    #settings resuming from checkpoint
    settings3 = {
                'cmi_estimator': 'JidtDiscreteCMI',
                'discretise_method': 'none',
                'n_discrete_bins': 5,
                'n_perm_max_stat': 21,
                'n_perm_omnibus': 30,
                'n_perm_max_seq': 30,
                'min_lag_sources': 1,
                'max_lag_sources': 2,
                'write_ckp': True,
                'filename_ckp': filename_ckp2
                }

    # Starting Analysis of the Network
    # results of a network analyis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data)

    # results of a network analysis with checkpointing
    results2 = network_analysis1.analyse_network(
        settings=settings2, data=data)

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    file = open(filename_ckp1 + ".ckp", "r+")

    tarsource = file.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    lines = open(filename_ckp2 + ".ckp").read().splitlines()
    lines[8] = tarsource
    open(filename_ckp2 + ".ckp",'w').write('\n'.join(lines))
    print(lines)

    #Resume analysis.
    network_analysis_res = MultivariateMI()
    data, settings, sources, networks = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analyis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data)

    # results of a network analyis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data)

    # results of a network analysis just like results 1
    results5 = network_analysis4.analyse_network(
        settings=settings1, data=data)

    adj_matrix1 = results1.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='max_te_lag', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("printing results")
    print("result 1: without checkpoint")
    print(result1)
    print("result 2: with checkpoint")
    print(result2)
    print("result 3: resuming from checkpoint")
    print(result3)
    print("result 4: without checkpoint with different targets and sources")
    print(result4)
    print("result 5: without checkpoint")
    print(result5)

    print("printing if results are equal")
    print("comparing result 1 and 2")
    print(result1 == result2)
    print("comparing result 1 and 3")
    print(result1 == result3)
    print("comparing result 1 and 4")
    print(result1 == result4)
    print("comparing result 1 and 5")
    print(result1 == result5)

    print("printing length")
    print("length of result1:")
    len1 = len(result1)
    print(len1)
    print("length of result2:")
    len2 = len(result2)
    print(len2)
    print("length of result3:")
    len3 = len(result3)
    print(len3)
    print("length of result4:")
    len4 = len(result4)
    print(len4)
    print("length of result5:")
    len5 = len(result5)
    print(len5)

    cmp1 = [i for i, j in zip(result1, result2) if i == j]
    cmp2 = [i for i, j in zip(result1, result3) if i == j]
    cmp3 = [i for i, j in zip(result1, result4) if i == j]
    cmp4 = [i for i, j in zip(result1, result5) if i == j]

    print("printing comparison of length")
    print("length of comparison 1-2 to 1")
    print(len(cmp1) == len1)
    print("length of comparison 1-3 to 1")
    print(len(cmp2) == len1)
    print("length of comparison 1-4 to 1")
    print(len(cmp3) == len1)
    print("length of comparison 1-5 to 1")
    print(len(cmp4) == len1)

    print("set intersection of results")
    print("elements for comparison between result 1 and 2")
    print(cmp1)
    print("elements for comparison between result 1 and 3")
    print(cmp2)
    print("elements for comparison between result 1 and 4")
    print(cmp3)
    print("elements for comparison between result 1 and 5")
    print(cmp4)

    os.system("./rm_check.sh")
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Running with checkpoints does not give the expected results")


def JidtGaussianCMI_MMI_checktest():

    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = './run_checkpoint'
    filename_ckp2 = './resume_checkpoint'

    """Test running analyis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=5)


    # Initialise analysis object and define settings
    network_analysis1 = MultivariateMI()
    network_analysis2 = MultivariateMI()
    network_analysis3 = MultivariateMI()
    network_analysis4 = MultivariateMI()

    #Settings without checkpointing
    settings1 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0
                }

    #settings with checkpointing
    settings2 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp1}

    #settings resuming from checkpoint
    settings3 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analyis
    sources = [0, 1]
    targets = [2, 3, 4]

    sources2 = [0, 1, 2]
    targets2 = [3, 4]

    # Starting Analysis of the Network
    # results of a network analyis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources='all')

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    file = open(filename_ckp1 + ".ckp", "r+")

    tarsource = file.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    lines = open(filename_ckp2 + ".ckp").read().splitlines()
    lines[8] = tarsource
    open(filename_ckp2 + ".ckp",'w').write('\n'.join(lines))
    print(lines)

     #Resume analysis.
    network_analysis_res = MultivariateMI()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analyis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources='all')

    # results of a network analyis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources='all')

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    adj_matrix1 = results1.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='max_te_lag', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("printing results")
    print("result 1: without checkpoint")
    print(result1)
    print("result 2: with checkpoint")
    print(result2)
    print("result 3: resuming from checkpoint")
    print(result3)
    print("result 4: without checkpoint with different targets and sources")
    print(result4)
    print("result 5: without checkpoint")
    print(result5)

    print("printing if results are equal")
    print("comparing result 1 and 2")
    print(result1 == result2)
    print("comparing result 1 and 3")
    print(result1 == result3)
    print("comparing result 1 and 4")
    print(result1 == result4)
    print("comparing result 1 and 5")
    print(result1 == result5)

    print("printing length")
    print("length of result1:")
    len1 = len(result1)
    print(len1)
    print("length of result2:")
    len2 = len(result2)
    print(len2)
    print("length of result3:")
    len3 = len(result3)
    print(len3)
    print("length of result4:")
    len4 = len(result4)
    print(len4)
    print("length of result5:")
    len5 = len(result5)
    print(len5)

    cmp1 = [i for i, j in zip(result1, result2) if i == j]
    cmp2 = [i for i, j in zip(result1, result3) if i == j]
    cmp3 = [i for i, j in zip(result1, result4) if i == j]
    cmp4 = [i for i, j in zip(result1, result5) if i == j]

    print("printing comparison of length")
    print("length of comparison 1-2 to 1")
    print(len(cmp1) == len1)
    print("length of comparison 1-3 to 1")
    print(len(cmp2) == len1)
    print("length of comparison 1-4 to 1")
    print(len(cmp3) == len1)
    print("length of comparison 1-5 to 1")
    print(len(cmp4) == len1)

    print("final")
    print("elements for comparison between result 1 and 2")
    print(cmp1)
    print("elements for comparison between result 1 and 3")
    print(cmp2)
    print("elements for comparison between result 1 and 4")
    print(cmp3)
    print("elements for comparison between result 1 and 5")
    print(cmp4)

    os.system("./rm_check.sh")
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Gaussian MMI Running with checkpoints does not give the expected results")


def JidtKraskovCMI_MMI_checktest():

    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = './run_checkpoint'
    filename_ckp2 = './resume_checkpoint'

    """Test running analyis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=100, n_replications=5)


    # Initialise analysis object and define settings
    network_analysis1 = MultivariateMI()
    network_analysis2 = MultivariateMI()
    network_analysis3 = MultivariateMI()
    network_analysis4 = MultivariateMI()

    #Settings without checkpointing
    settings1 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0
                }

    #settings with checkpointing
    settings2 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp1}

    #settings resuming from checkpoint
    settings3 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analyis
    sources = [0, 1]
    targets = [2, 3, 4]

    sources2 = [0, 1, 2]
    targets2 = [3, 4]

    # Starting Analysis of the Network
    # results of a network analyis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources='all')

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    file = open(filename_ckp1 + ".ckp", "r+")

    tarsource = file.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    lines = open(filename_ckp2 + ".ckp").read().splitlines()
    lines[8] = tarsource
    open(filename_ckp2 + ".ckp",'w').write('\n'.join(lines))
    print(lines)

     #Resume analysis.
    network_analysis_res = MultivariateMI()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analyis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources='all')

    # results of a network analyis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources='all')

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    adj_matrix1 = results1.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='max_te_lag', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("printing results")
    print("result 1: without checkpoint")
    print(result1)
    print("result 2: with checkpoint")
    print(result2)
    print("result 3: resuming from checkpoint")
    print(result3)
    print("result 4: without checkpoint with different targets and sources")
    print(result4)
    print("result 5: without checkpoint")
    print(result5)

    print("printing if results are equal")
    print("comparing result 1 and 2")
    print(result1 == result2)
    print("comparing result 1 and 3")
    print(result1 == result3)
    print("comparing result 1 and 4")
    print(result1 == result4)
    print("comparing result 1 and 5")
    print(result1 == result5)

    print("printing length")
    print("length of result1:")
    len1 = len(result1)
    print(len1)
    print("length of result2:")
    len2 = len(result2)
    print(len2)
    print("length of result3:")
    len3 = len(result3)
    print(len3)
    print("length of result4:")
    len4 = len(result4)
    print(len4)
    print("length of result5:")
    len5 = len(result5)
    print(len5)

    cmp1 = [i for i, j in zip(result1, result2) if i == j]
    cmp2 = [i for i, j in zip(result1, result3) if i == j]
    cmp3 = [i for i, j in zip(result1, result4) if i == j]
    cmp4 = [i for i, j in zip(result1, result5) if i == j]

    print("printing comparison of length")
    print("length of comparison 1-2 to 1")
    print(len(cmp1) == len1)
    print("length of comparison 1-3 to 1")
    print(len(cmp2) == len1)
    print("length of comparison 1-4 to 1")
    print(len(cmp3) == len1)
    print("length of comparison 1-5 to 1")
    print(len(cmp4) == len1)

    print("final")
    print("elements for comparison between result 1 and 2")
    print(cmp1)
    print("elements for comparison between result 1 and 3")
    print(cmp2)
    print("elements for comparison between result 1 and 4")
    print(cmp3)
    print("elements for comparison between result 1 and 5")
    print(cmp4)

    os.system("./rm_check.sh")
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Kraskov MMI running with checkpoints does not give the expected results")


def JidtGaussianCMI_MTE_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = './run_checkpoint'
    filename_ckp2 = './resume_checkpoint'

    """Test running analyis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=10000, n_replications=5)


    # Initialise analysis object and define settings
    network_analysis1 = MultivariateTE()
    network_analysis2 = MultivariateTE()
    network_analysis3 = MultivariateTE()
    network_analysis4 = MultivariateTE()

    #Settings without checkpointing
    settings1 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0
                }

    #settings with checkpointing
    settings2 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp1}

    #settings resuming from checkpoint
    settings3 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analyis
    sources = [0, 1]
    targets = [2, 3, 4]

    sources2 = [0, 1, 2]
    targets2 = [3, 4]

    # Starting Analysis of the Network
    # results of a network analyis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources='all')

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    file = open(filename_ckp1 + ".ckp", "r+")

    tarsource = file.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    lines = open(filename_ckp2 + ".ckp").read().splitlines()
    lines[8] = tarsource
    open(filename_ckp2 + ".ckp",'w').write('\n'.join(lines))
    print(lines)

     #Resume analysis.
    network_analysis_res = MultivariateTE()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analyis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources='all')

    # results of a network analyis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources='all')

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    adj_matrix1 = results1.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='max_te_lag', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("printing results")
    print("result 1: without checkpoint")
    print(result1)
    print("result 2: with checkpoint")
    print(result2)
    print("result 3: resuming from checkpoint")
    print(result3)
    print("result 4: without checkpoint with different targets and sources")
    print(result4)
    print("result 5: without checkpoint")
    print(result5)

    print("printing if results are equal")
    print("comparing result 1 and 2")
    print(result1 == result2)
    print("comparing result 1 and 3")
    print(result1 == result3)
    print("comparing result 1 and 4")
    print(result1 == result4)
    print("comparing result 1 and 5")
    print(result1 == result5)

    print("printing length")
    print("length of result1:")
    len1 = len(result1)
    print(len1)
    print("length of result2:")
    len2 = len(result2)
    print(len2)
    print("length of result3:")
    len3 = len(result3)
    print(len3)
    print("length of result4:")
    len4 = len(result4)
    print(len4)
    print("length of result5:")
    len5 = len(result5)
    print(len5)

    cmp1 = [i for i, j in zip(result1, result2) if i == j]
    cmp2 = [i for i, j in zip(result1, result3) if i == j]
    cmp3 = [i for i, j in zip(result1, result4) if i == j]
    cmp4 = [i for i, j in zip(result1, result5) if i == j]

    print("printing comparison of length")
    print("length of comparison 1-2 to 1")
    print(len(cmp1) == len1)
    print("length of comparison 1-3 to 1")
    print(len(cmp2) == len1)
    print("length of comparison 1-4 to 1")
    print(len(cmp3) == len1)
    print("length of comparison 1-5 to 1")
    print(len(cmp4) == len1)

    print("final")
    print("elements for comparison between result 1 and 2")
    print(cmp1)
    print("elements for comparison between result 1 and 3")
    print(cmp2)
    print("elements for comparison between result 1 and 4")
    print(cmp3)
    print("elements for comparison between result 1 and 5")
    print(cmp4)

    os.system("./rm_check.sh")
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Gaussian MTE running with checkpoints does not give the expected results")


def JidtKraskovCMI_MTE_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = './run_checkpoint'
    filename_ckp2 = './resume_checkpoint'

    """Test running analyis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=1000, n_replications=5)


    # Initialise analysis object and define settings
    network_analysis1 = MultivariateTE()
    network_analysis2 = MultivariateTE()
    network_analysis3 = MultivariateTE()
    network_analysis4 = MultivariateTE()

    #Settings without checkpointing
    settings1 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0
                }

    #settings with checkpointing
    settings2 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp1}

    #settings resuming from checkpoint
    settings3 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analyis
    sources = [0, 1]
    targets = [2, 3, 4]

    sources2 = [0, 1, 2]
    targets2 = [3, 4]

    # Starting Analysis of the Network
    # results of a network analyis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources='all')

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    file = open(filename_ckp1 + ".ckp", "r+")

    tarsource = file.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    lines = open(filename_ckp2 + ".ckp").read().splitlines()
    lines[8] = tarsource
    open(filename_ckp2 + ".ckp",'w').write('\n'.join(lines))
    print(lines)

     #Resume analysis.
    network_analysis_res = MultivariateTE()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analyis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources='all')

    # results of a network analyis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources='all')

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    adj_matrix1 = results1.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='max_te_lag', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("printing results")
    print("result 1: without checkpoint")
    print(result1)
    print("result 2: with checkpoint")
    print(result2)
    print("result 3: resuming from checkpoint")
    print(result3)
    print("result 4: without checkpoint with different targets and sources")
    print(result4)
    print("result 5: without checkpoint")
    print(result5)

    print("printing if results are equal")
    print("comparing result 1 and 2")
    print(result1 == result2)
    print("comparing result 1 and 3")
    print(result1 == result3)
    print("comparing result 1 and 4")
    print(result1 == result4)
    print("comparing result 1 and 5")
    print(result1 == result5)

    print("printing length")
    print("length of result1:")
    len1 = len(result1)
    print(len1)
    print("length of result2:")
    len2 = len(result2)
    print(len2)
    print("length of result3:")
    len3 = len(result3)
    print(len3)
    print("length of result4:")
    len4 = len(result4)
    print(len4)
    print("length of result5:")
    len5 = len(result5)
    print(len5)

    cmp1 = [i for i, j in zip(result1, result2) if i == j]
    cmp2 = [i for i, j in zip(result1, result3) if i == j]
    cmp3 = [i for i, j in zip(result1, result4) if i == j]
    cmp4 = [i for i, j in zip(result1, result5) if i == j]

    print("printing comparison of length")
    print("length of comparison 1-2 to 1")
    print(len(cmp1) == len1)
    print("length of comparison 1-3 to 1")
    print(len(cmp2) == len1)
    print("length of comparison 1-4 to 1")
    print(len(cmp3) == len1)
    print("length of comparison 1-5 to 1")
    print(len(cmp4) == len1)

    print("final")
    print("elements for comparison between result 1 and 2")
    print(cmp1)
    print("elements for comparison between result 1 and 3")
    print(cmp2)
    print("elements for comparison between result 1 and 4")
    print(cmp3)
    print("elements for comparison between result 1 and 5")
    print(cmp4)

    os.system("./rm_check.sh")
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Kraskov MTE Running with checkpoints does not give the expected results")


def JidtGaussianCMI_BTE_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = './run_checkpoint'
    filename_ckp2 = './resume_checkpoint'

    """Test running analyis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=10000, n_replications=5)


    # Initialise analysis object and define settings
    network_analysis1 = BivariateMI()
    network_analysis2 = BivariateMI()
    network_analysis3 = BivariateMI()
    network_analysis4 = BivariateMI()

    #Settings without checkpointing
    settings1 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0
                }

    #settings with checkpointing
    settings2 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp1}

    #settings resuming from checkpoint
    settings3 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analyis
    sources = [0, 1]
    targets = [2, 3, 4]

    sources2 = [0, 1, 2]
    targets2 = [3, 4]

    # Starting Analysis of the Network
    # results of a network analyis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources='all')

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    file = open(filename_ckp1 + ".ckp", "r+")

    tarsource = file.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    lines = open(filename_ckp2 + ".ckp").read().splitlines()
    lines[8] = tarsource
    open(filename_ckp2 + ".ckp",'w').write('\n'.join(lines))
    print(lines)

     #Resume analysis.
    network_analysis_res = BivariateMI()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analyis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources='all')

    # results of a network analyis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources='all')

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    adj_matrix1 = results1.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='max_te_lag', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("printing results")
    print("result 1: without checkpoint")
    print(result1)
    print("result 2: with checkpoint")
    print(result2)
    print("result 3: resuming from checkpoint")
    print(result3)
    print("result 4: without checkpoint with different targets and sources")
    print(result4)
    print("result 5: without checkpoint")
    print(result5)

    print("printing if results are equal")
    print("comparing result 1 and 2")
    print(result1 == result2)
    print("comparing result 1 and 3")
    print(result1 == result3)
    print("comparing result 1 and 4")
    print(result1 == result4)
    print("comparing result 1 and 5")
    print(result1 == result5)

    print("printing length")
    print("length of result1:")
    len1 = len(result1)
    print(len1)
    print("length of result2:")
    len2 = len(result2)
    print(len2)
    print("length of result3:")
    len3 = len(result3)
    print(len3)
    print("length of result4:")
    len4 = len(result4)
    print(len4)
    print("length of result5:")
    len5 = len(result5)
    print(len5)

    cmp1 = [i for i, j in zip(result1, result2) if i == j]
    cmp2 = [i for i, j in zip(result1, result3) if i == j]
    cmp3 = [i for i, j in zip(result1, result4) if i == j]
    cmp4 = [i for i, j in zip(result1, result5) if i == j]

    print("printing comparison of length")
    print("length of comparison 1-2 to 1")
    print(len(cmp1) == len1)
    print("length of comparison 1-3 to 1")
    print(len(cmp2) == len1)
    print("length of comparison 1-4 to 1")
    print(len(cmp3) == len1)
    print("length of comparison 1-5 to 1")
    print(len(cmp4) == len1)

    print("final")
    print("elements for comparison between result 1 and 2")
    print(cmp1)
    print("elements for comparison between result 1 and 3")
    print(cmp2)
    print("elements for comparison between result 1 and 4")
    print(cmp3)
    print("elements for comparison between result 1 and 5")
    print(cmp4)

    os.system("./rm_check.sh")
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Gaussian BTE running with checkpoints does not give the expected results")


def JidtKraskovCMI_BTE_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = './run_checkpoint'
    filename_ckp2 = './resume_checkpoint'

    """Test running analyis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=100, n_replications=5)


    # Initialise analysis object and define settings
    network_analysis1 = BivariateMI()
    network_analysis2 = BivariateMI()
    network_analysis3 = BivariateMI()
    network_analysis4 = BivariateMI()

    #Settings without checkpointing
    settings1 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0
                }

    #settings with checkpointing
    settings2 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp1}

    #settings resuming from checkpoint
    settings3 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analyis
    sources = [0, 1]
    targets = [2, 3, 4]

    sources2 = [0, 1, 2]
    targets2 = [3, 4]

    # Starting Analysis of the Network
    # results of a network analyis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources='all')

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    file = open(filename_ckp1 + ".ckp", "r+")

    tarsource = file.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    lines = open(filename_ckp2 + ".ckp").read().splitlines()
    lines[8] = tarsource
    open(filename_ckp2 + ".ckp",'w').write('\n'.join(lines))
    print(lines)

     #Resume analysis.
    network_analysis_res = BivariateMI()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analyis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources='all')

    # results of a network analyis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources='all')

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    adj_matrix1 = results1.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='max_te_lag', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("printing results")
    print("result 1: without checkpoint")
    print(result1)
    print("result 2: with checkpoint")
    print(result2)
    print("result 3: resuming from checkpoint")
    print(result3)
    print("result 4: without checkpoint with different targets and sources")
    print(result4)
    print("result 5: without checkpoint")
    print(result5)

    print("printing if results are equal")
    print("comparing result 1 and 2")
    print(result1 == result2)
    print("comparing result 1 and 3")
    print(result1 == result3)
    print("comparing result 1 and 4")
    print(result1 == result4)
    print("comparing result 1 and 5")
    print(result1 == result5)

    print("printing length")
    print("length of result1:")
    len1 = len(result1)
    print(len1)
    print("length of result2:")
    len2 = len(result2)
    print(len2)
    print("length of result3:")
    len3 = len(result3)
    print(len3)
    print("length of result4:")
    len4 = len(result4)
    print(len4)
    print("length of result5:")
    len5 = len(result5)
    print(len5)

    cmp1 = [i for i, j in zip(result1, result2) if i == j]
    cmp2 = [i for i, j in zip(result1, result3) if i == j]
    cmp3 = [i for i, j in zip(result1, result4) if i == j]
    cmp4 = [i for i, j in zip(result1, result5) if i == j]

    print("printing comparison of length")
    print("length of comparison 1-2 to 1")
    print(len(cmp1) == len1)
    print("length of comparison 1-3 to 1")
    print(len(cmp2) == len1)
    print("length of comparison 1-4 to 1")
    print(len(cmp3) == len1)
    print("length of comparison 1-5 to 1")
    print(len(cmp4) == len1)

    print("final")
    print("elements for comparison between result 1 and 2")
    print(cmp1)
    print("elements for comparison between result 1 and 3")
    print(cmp2)
    print("elements for comparison between result 1 and 4")
    print(cmp3)
    print("elements for comparison between result 1 and 5")
    print(cmp4)

    os.system("./rm_check.sh")
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Kraskov BTE running with checkpoints does not give the expected results")


def JidtGaussianCMI_BMI_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = './run_checkpoint'
    filename_ckp2 = './resume_checkpoint'

    """Test running analyis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=10000, n_replications=5)


    # Initialise analysis object and define settings
    network_analysis1 = BivariateMI()
    network_analysis2 = BivariateMI()
    network_analysis3 = BivariateMI()
    network_analysis4 = BivariateMI()

    #Settings without checkpointing
    settings1 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0
                }

    #settings with checkpointing
    settings2 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp1}

    #settings resuming from checkpoint
    settings3 = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analyis
    sources = [0, 1]
    targets = [2, 3, 4]

    sources2 = [0, 1, 2]
    targets2 = [3, 4]

    # Starting Analysis of the Network
    # results of a network analyis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources='all')

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    file = open(filename_ckp1 + ".ckp", "r+")

    tarsource = file.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    lines = open(filename_ckp2 + ".ckp").read().splitlines()
    lines[8] = tarsource
    open(filename_ckp2 + ".ckp",'w').write('\n'.join(lines))
    print(lines)

     #Resume analysis.
    network_analysis_res = BivariateMI()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analyis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources='all')

    # results of a network analyis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources='all')

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    adj_matrix1 = results1.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='max_te_lag', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("printing results")
    print("result 1: without checkpoint")
    print(result1)
    print("result 2: with checkpoint")
    print(result2)
    print("result 3: resuming from checkpoint")
    print(result3)
    print("result 4: without checkpoint with different targets and sources")
    print(result4)
    print("result 5: without checkpoint")
    print(result5)

    print("printing if results are equal")
    print("comparing result 1 and 2")
    print(result1 == result2)
    print("comparing result 1 and 3")
    print(result1 == result3)
    print("comparing result 1 and 4")
    print(result1 == result4)
    print("comparing result 1 and 5")
    print(result1 == result5)

    print("printing length")
    print("length of result1:")
    len1 = len(result1)
    print(len1)
    print("length of result2:")
    len2 = len(result2)
    print(len2)
    print("length of result3:")
    len3 = len(result3)
    print(len3)
    print("length of result4:")
    len4 = len(result4)
    print(len4)
    print("length of result5:")
    len5 = len(result5)
    print(len5)

    cmp1 = [i for i, j in zip(result1, result2) if i == j]
    cmp2 = [i for i, j in zip(result1, result3) if i == j]
    cmp3 = [i for i, j in zip(result1, result4) if i == j]
    cmp4 = [i for i, j in zip(result1, result5) if i == j]

    print("printing comparison of length")
    print("length of comparison 1-2 to 1")
    print(len(cmp1) == len1)
    print("length of comparison 1-3 to 1")
    print(len(cmp2) == len1)
    print("length of comparison 1-4 to 1")
    print(len(cmp3) == len1)
    print("length of comparison 1-5 to 1")
    print(len(cmp4) == len1)

    print("final")
    print("elements for comparison between result 1 and 2")
    print(cmp1)
    print("elements for comparison between result 1 and 3")
    print(cmp2)
    print("elements for comparison between result 1 and 4")
    print(cmp3)
    print("elements for comparison between result 1 and 5")
    print(cmp4)

    os.system("./rm_check.sh")
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Gaussian BMI running with checkpoints does not give the expected results")


def JidtKraskovCMI_BMI_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = './run_checkpoint'
    filename_ckp2 = './resume_checkpoint'

    """Test running analyis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=100, n_replications=5)


    # Initialise analysis object and define settings
    network_analysis1 = BivariateMI()
    network_analysis2 = BivariateMI()
    network_analysis3 = BivariateMI()
    network_analysis4 = BivariateMI()

    #Settings without checkpointing
    settings1 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0
                }

    #settings with checkpointing
    settings2 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp1}

    #settings resuming from checkpoint
    settings3 = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag_sources': 5,
                'min_lag_sources': 1,
                'noise_level': 0,
                'write_ckp': True,
                'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analyis
    sources = [0, 1]
    targets = [2, 3, 4]

    sources2 = [0, 1, 2]
    targets2 = [3, 4]

    # Starting Analysis of the Network
    # results of a network analyis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources='all')

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    file = open(filename_ckp1 + ".ckp", "r+")

    tarsource = file.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    lines = open(filename_ckp2 + ".ckp").read().splitlines()
    lines[8] = tarsource
    open(filename_ckp2 + ".ckp",'w').write('\n'.join(lines))
    print(lines)

     #Resume analysis.
    network_analysis_res = BivariateMI()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analyis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources='all')

    # results of a network analyis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources='all')

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources='all')

    adj_matrix1 = results1.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='max_te_lag', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='max_te_lag', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("printing results")
    print("result 1: without checkpoint")
    print(result1)
    print("result 2: with checkpoint")
    print(result2)
    print("result 3: resuming from checkpoint")
    print(result3)
    print("result 4: without checkpoint with different targets and sources")
    print(result4)
    print("result 5: without checkpoint")
    print(result5)

    print("printing if results are equal")
    print("comparing result 1 and 2")
    print(result1 == result2)
    print("comparing result 1 and 3")
    print(result1 == result3)
    print("comparing result 1 and 4")
    print(result1 == result4)
    print("comparing result 1 and 5")
    print(result1 == result5)

    print("printing length")
    print("length of result1:")
    len1 = len(result1)
    print(len1)
    print("length of result2:")
    len2 = len(result2)
    print(len2)
    print("length of result3:")
    len3 = len(result3)
    print(len3)
    print("length of result4:")
    len4 = len(result4)
    print(len4)
    print("length of result5:")
    len5 = len(result5)
    print(len5)

    cmp1 = [i for i, j in zip(result1, result2) if i == j]
    cmp2 = [i for i, j in zip(result1, result3) if i == j]
    cmp3 = [i for i, j in zip(result1, result4) if i == j]
    cmp4 = [i for i, j in zip(result1, result5) if i == j]

    print("printing comparison of length")
    print("length of comparison 1-2 to 1")
    print(len(cmp1) == len1)
    print("length of comparison 1-3 to 1")
    print(len(cmp2) == len1)
    print("length of comparison 1-4 to 1")
    print(len(cmp3) == len1)
    print("length of comparison 1-5 to 1")
    print(len(cmp4) == len1)

    print("final")
    print("elements for comparison between result 1 and 2")
    print(cmp1)
    print("elements for comparison between result 1 and 3")
    print(cmp2)
    print("elements for comparison between result 1 and 4")
    print(cmp3)
    print("elements for comparison between result 1 and 5")
    print(cmp4)

    os.system("./rm_check.sh")
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Kraskov BMI running with checkpoints does not give the expected results")


if __name__ == '__main__':
    #test_checkpoint_copy()
    #test_checkpoint_resume()
    #test_checkpoint_equal()
    JidtDiscreteCMI_MMI_checktest()
    #JidtGaussianCMI_MMI_checktest()
    #JidtKraskovCMI_MMI_checktest()
    #JidtGaussianCMI_MTE_checktest()
    #JidtKraskovCMI_MTE_checktest()
    #JidtGaussianCMI_BTE_checktest()
    #JidtKraskovCMI_BTE_checktest()
    #JidtGaussianCMI_BMI_checktest()
    #JidtKraskovCMI_BMI_checktest()
