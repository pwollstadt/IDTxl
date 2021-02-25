"""Provide unit tests for IDTxl checkpointing."""
import os
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.multivariate_mi import MultivariateMI
from idtxl.bivariate_mi import BivariateMI
from idtxl.bivariate_te import BivariateTE
from test_checkpointing import _clear_ckp

N_PERM = 21  # this no. permutations is usually sufficient to find links in the MUTE data
N_SAMPLES = 10000
SEED = 0
GAUSS_COV = 0.4
DELAY = 1


def test_JidtKraskovCMI_MMI_checkpoint():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data(seed=SEED)
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    network_analysis1 = MultivariateMI()
    network_analysis2 = MultivariateMI()
    network_analysis3 = MultivariateMI()
    network_analysis4 = MultivariateMI()

    # Settings without checkpointing.
    settings1 = {'cmi_estimator': 'JidtKraskovCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0
                 }

    # Settings with checkpointing.
    settings2 = {'cmi_estimator': 'JidtKraskovCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp1}

    # Settings resuming from checkpoint.
    settings3 = {'cmi_estimator': 'JidtKraskovCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analysis
    sources = [0, 1]
    targets = [2, 3]
    targets2 = [3]

    # Starting Analysis of the Network
    # results of a network analysis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources=sources)

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    with open(filename_ckp1 + ".ckp", "r+") as f:
        tarsource = f.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    with open(filename_ckp2 + ".ckp") as f:
        lines = f.read().splitlines()
    lines[8] = tarsource
    with open(filename_ckp2 + ".ckp", 'w') as f:
        f.write('\n'.join(lines))
    print(lines)

    # Resume analysis.
    network_analysis_res = MultivariateMI()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analysis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources=sources)

    # results of a network analysis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def test_JidtKraskovCMI_MTE_checkpoint():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data(seed=SEED)
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    network_analysis1 = MultivariateTE()
    network_analysis2 = MultivariateTE()
    network_analysis3 = MultivariateTE()
    network_analysis4 = MultivariateTE()

    # Settings without checkpointing.
    settings1 = {'cmi_estimator': 'JidtKraskovCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0
                 }

    # Settings with checkpointing.
    settings2 = {'cmi_estimator': 'JidtKraskovCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp1}

    # Settings resuming from checkpoint.
    settings3 = {'cmi_estimator': 'JidtKraskovCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analysis
    sources = [0, 1]
    targets = [2, 3]
    targets2 = [3]

    # Starting Analysis of the Network
    # results of a network analysis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources=sources)

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    with open(filename_ckp1 + ".ckp", "r+") as f:
        tarsource = f.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    with open(filename_ckp2 + ".ckp") as f:
        lines = f.read().splitlines()
    lines[8] = tarsource
    with open(filename_ckp2 + ".ckp", 'w') as f:
        f.write('\n'.join(lines))
    print(lines)

    # Resume analysis.
    network_analysis_res = MultivariateTE()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analysis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources=sources)

    # results of a network analysis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def test_JidtKraskovCMI_BTE_checkpoint():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data(seed=SEED)
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    network_analysis1 = BivariateTE()
    network_analysis2 = BivariateTE()
    network_analysis3 = BivariateTE()
    network_analysis4 = BivariateTE()

    # Settings without checkpointing.
    settings1 = {'cmi_estimator': 'JidtKraskovCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0
                 }

    # Settings with checkpointing.
    settings2 = settings1
    settings2['write_ckp'] = True
    settings2['filename_ckp'] = filename_ckp1

    # Settings resuming from checkpoint.
    settings3 = settings1
    settings3['write_ckp'] = True
    settings3['filename_ckp'] = filename_ckp2

    # Setting sources and targets for the analysis
    sources = [0, 1]
    targets = [2, 3]
    targets2 = [3]

    # Starting Analysis of the Network
    # results of a network analysis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources=sources)

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    with open(filename_ckp1 + ".ckp", "r+") as f:
        tarsource = f.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])
    print(tarsource)

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    with open(filename_ckp2 + ".ckp") as f:
        lines = f.read().splitlines()
    lines[8] = tarsource
    with open(filename_ckp2 + ".ckp", 'w') as f:
        f.write('\n'.join(lines))
    print(lines)

    # Resume analysis.
    network_analysis_res = BivariateMI()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analysis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources=sources)

    # results of a network analysis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def test_JidtKraskovCMI_BMI_checkpoint():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data(seed=SEED)
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    network_analysis1 = BivariateMI()
    network_analysis2 = BivariateMI()
    network_analysis3 = BivariateMI()
    network_analysis4 = BivariateMI()

    # Settings without checkpointing.
    settings1 = {'cmi_estimator': 'JidtKraskovCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0
                 }

    # Settings with checkpointing.
    settings2 = {'cmi_estimator': 'JidtKraskovCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp1}

    # Settings resuming from checkpoint.
    settings3 = {'cmi_estimator': 'JidtKraskovCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analysis
    sources = [0]
    targets = [2, 3]
    targets2 = [3]

    # Starting Analysis of the Network
    # results of a network analysis without checkpointing
    results1 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_network(
        settings=settings2, data=data, targets=targets, sources=sources)

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    with open(filename_ckp1 + ".ckp", "r+") as f:
        tarsource = f.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources, targets)

    with open(filename_ckp2 + ".ckp") as f:
        lines = f.read().splitlines()
    lines[8] = tarsource
    with open(filename_ckp2 + ".ckp", 'w') as f:
        f.write('\n'.join(lines))
    print(lines)

    # Resume analysis.
    network_analysis_res = BivariateMI()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp2)

    # results of a network analysis resuming from checkpoint
    results3 = network_analysis_res.analyse_network(
        settings=settings3, data=data, targets=targets, sources=sources)

    # results of a network analysis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_network(
        settings=settings1, data=data, targets=targets2, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


if __name__ == '__main__':
    test_JidtKraskovCMI_MMI_checkpoint()
    test_JidtKraskovCMI_MTE_checkpoint()
    test_JidtKraskovCMI_BMI_checkpoint()
    test_JidtKraskovCMI_BTE_checkpoint()
