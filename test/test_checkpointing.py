"""Provide unit tests for IDTxl checkpointing."""
import os
import numpy as np
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.idtxl_utils import calculate_mi
from idtxl.estimators_jidt import JidtDiscreteCMI
from idtxl.multivariate_mi import MultivariateMI
from idtxl.bivariate_mi import BivariateMI
from idtxl.bivariate_te import BivariateTE

N_PERM = 21  # this no. permutations is usually sufficient to find links in the MUTE data
N_SAMPLES = 1000


def _clear_ckp(filename):
    # Delete created checkpoint from disk.
    os.remove(f'{filename}.ckp')
    try:
        os.remove(f'{filename}.ckp.old')
    except FileNotFoundError:
        # If algorithm ran for only one iteration, no old checkpoint exists.
        print(f'No file {filename}.ckp.old')
    os.remove(f'{filename}.dat')
    os.remove(f'{filename}.json')


def test_checkpoint_defaults():
    """Test if checkpointing defaults are set and used correctly."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 3,
                'min_lag_sources': 2,
                'verbose': True,
                'write_ckp': True}
    sources = [0, 1]
    targets = [3]
    network_analysis = MultivariateTE()
    network_analysis._set_checkpointing_defaults(settings, data, sources, targets)
    ckp_file = os.path.join(os.path.dirname(__file__), 'idtxl_checkpoint')
    assert os.path.isfile(f'{ckp_file}.ckp'), 'Did not write default checkpoint file'
    assert os.path.isfile(f'{ckp_file}.dat'), 'Did not write default checkpoint data'
    assert os.path.isfile(f'{ckp_file}.json'), 'Did not write default checkpoint settings'

    # Check if
    data_res, settings_res, targets_res, sources_res = network_analysis.resume_checkpoint(
        ckp_file)
    for k in settings:
        assert settings[k] == settings_res[k], f'Unequal entries in settings: key {k}'
    assert sources == sources_res, 'Sources not the same'
    assert targets == targets_res, 'Targets not the same'

    _clear_ckp(ckp_file)


def test_checkpoint_resume():
    """Test resuming from manually generated checkpoint."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    filename_ckp = os.path.join(
        os.path.dirname(__file__), 'data', 'my_checkpoint')
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 3,
                'min_lag_sources': 2,
                'write_ckp': True,
                'filename_ckp': filename_ckp}

    # Manually create checkpoint files for two targets.
    # Note that variables are added as absolute time indices and not lags wrt.
    # the current value. Internally, IDTxl operates with abolute indices, but
    # in all results and console outputs, variables are described by their lags
    # wrt. to the current value.
    sources = [0, 1, 2]
    targets = [3, 4]
    add_vars = [[(0, 1), (0, 2), (3, 1)], [(0, 1), (0, 2), (1, 3), (4, 2)]]
    network_analysis._set_checkpointing_defaults(
        settings, data, sources, targets)
    for i in range(len(targets)):
        network_analysis._initialise(settings, data, sources, targets[i])
        network_analysis.selected_vars_full = add_vars[i]
        network_analysis._update_checkpoint('{}.ckp'.format(filename_ckp))

    # Resume analysis.
    network_analysis_res = MultivariateTE()
    data, settings, targets, sources = network_analysis_res.resume_checkpoint(
        filename_ckp)

    # Test if variables were added correctly
    for i in range(len(targets)):
        network_analysis_res._initialise(settings, data, sources, targets[i])
        assert network_analysis_res.selected_vars_full == add_vars[i], (
            'Variables were not added correctly for target {0}.'.format(
                targets[i]))

    # Test if analysis runs from resumed checkpoint.
    network_analysis_res.analyse_network(
        settings=settings, data=data, targets=targets, sources=sources)

    _clear_ckp(filename_ckp)


def test_checkpoint_copy():
    """Assert that a copy of the current checkpoint is written upon update."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    filename_ckp = os.path.join(
        os.path.dirname(__file__), 'data', 'my_checkpoint')
    network_analysis = MultivariateTE()
    settings = {'cmi_estimator': 'JidtGaussianCMI',
                'max_lag_sources': 3,
                'min_lag_sources': 2,
                'write_ckp': True,
                'filename_ckp': filename_ckp}

    # Manually create checkpoint files for two targets.
    # Note that variables are added as absolute time indices and not lags wrt.
    # the current value. Internally, IDTxl operates with abolute indices, but
    # in all results and console outputs, variables are described by their lags
    # wrt. to the current value.
    sources = [0, 1, 2]
    targets = [3, 4]
    network_analysis._initialise(settings, data, sources, targets[0])
    network_analysis._write_checkpoint()

    assert os.path.isfile('{0}.ckp'.format(filename_ckp)), (
        'No checkpoint file {}.ckp was written.'.format(filename_ckp))
    assert os.path.isfile('{0}.ckp.old'.format(filename_ckp)), (
        'No checkpoint file {}.ckp.old was written.'.format(filename_ckp))

    _clear_ckp(filename_ckp)


def JidtGaussianCMI_MMI_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    network_analysis1 = MultivariateMI()
    network_analysis2 = MultivariateMI()
    network_analysis3 = MultivariateMI()
    network_analysis4 = MultivariateMI()

    # Settings without checkpointing.
    settings1 = {'cmi_estimator': 'JidtGaussianCMI',
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0
                 }

    # Settings with checkpointing.
    settings2 = {'cmi_estimator': 'JidtGaussianCMI',
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp1}

    # Settings resuming from checkpoint.
    settings3 = {'cmi_estimator': 'JidtGaussianCMI',
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analysis
    sources = [0, 1]
    targets = [2, 3, 4]
    targets2 = [3, 4]

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

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtKraskovCMI_MMI_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data()
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
    targets = [2, 3, 4]
    targets2 = [3, 4]

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

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtDiscreteCMI_MMI_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test multivariate TE estimation from discrete data."""
    # Generate Gaussian test data
    covariance = 0.4
    n = N_SAMPLES
    delay = 1
    source = np.random.normal(0, 1, size=n)
    target = (covariance * source + (1 - covariance) *
              np.random.normal(0, 1, size=n))
    corr_expected = covariance / \
        (1 * np.sqrt(covariance**2 + (1-covariance)**2))
    expected_mi = calculate_mi(corr_expected)
    source = source[delay:]
    target = target[:-delay]

    # Discretise data
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 5}
    est = JidtDiscreteCMI(settings)
    source_dis, target_dis = est._discretise_vars(var1=source, var2=target)
    data = Data(np.vstack((source_dis, target_dis)),
                dim_order='ps', normalise=False)

    network_analysis1 = MultivariateMI()
    network_analysis2 = MultivariateMI()
    network_analysis3 = MultivariateMI()
    network_analysis4 = MultivariateMI()

    settings1 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1}

    settings2 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1,
        'write_ckp': True,
        'filename_ckp': filename_ckp1}

    settings3 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1,
        'write_ckp': True,
        'filename_ckp': filename_ckp2}

    # Starting Analysis of the Network
    # results of a network analysis without checkpointing
    results1 = network_analysis1.analyse_single_target(settings1, data,
                                                       target=1)

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_single_target(settings2, data,
                                                       target=1)

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    with open(filename_ckp1 + ".ckp", "r+") as f:
        tarsource = f.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources=tarsource, target=1)

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
    results3 = network_analysis_res.analyse_single_target(
        settings=settings, data=data, target=1)

    # results of a network analysis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_single_target(
        settings=settings1, data=data, target=0)

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_single_target(settings1, data,
                                                       target=1)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtGaussianCMI_MTE_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    network_analysis1 = MultivariateTE()
    network_analysis2 = MultivariateTE()
    network_analysis3 = MultivariateTE()
    network_analysis4 = MultivariateTE()

    # Settings without checkpointing.
    settings1 = {'cmi_estimator': 'JidtGaussianCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0
                 }

    # Settings with checkpointing.
    settings2 = {'cmi_estimator': 'JidtGaussianCMI',
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp1}

    # Settings resuming from checkpoint.
    settings3 = {'cmi_estimator': 'JidtGaussianCMI',
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analysis
    sources = [0, 1]
    targets = [2, 3, 4]
    targets2 = [3, 4]

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

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtKraskovCMI_MTE_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data()
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
    targets = [2, 3, 4]
    targets2 = [3, 4]

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

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtDiscreteCMI_MTE_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test multivariate TE estimation from discrete data."""
    # Generate Gaussian test data
    covariance = 0.4
    n = N_SAMPLES
    delay = 1
    source = np.random.normal(0, 1, size=n)
    target = (covariance * source + (1 - covariance) *
              np.random.normal(0, 1, size=n))
    corr_expected = covariance / \
        (1 * np.sqrt(covariance**2 + (1-covariance)**2))
    expected_mi = calculate_mi(corr_expected)
    source = source[delay:]
    target = target[:-delay]

    # Discretise data
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 5}
    est = JidtDiscreteCMI(settings)
    source_dis, target_dis = est._discretise_vars(var1=source, var2=target)
    data = Data(np.vstack((source_dis, target_dis)),
                dim_order='ps', normalise=False)
    settings1 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1}

    settings2 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1,
        'write_ckp': True,
        'filename_ckp': filename_ckp1}

    settings3 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1,
        'write_ckp': True,
        'filename_ckp': filename_ckp2}

    network_analysis1 = MultivariateTE()
    network_analysis2 = MultivariateTE()
    network_analysis3 = MultivariateTE()
    network_analysis4 = MultivariateTE()

    # Starting Analysis of the Network
    # results of a network analysis without checkpointing
    results1 = network_analysis1.analyse_single_target(settings1, data,
                                                       target=1)

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_single_target(settings2, data,
                                                       target=1)

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    with open(filename_ckp1 + ".ckp", "r+") as f:
        tarsource = f.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources=tarsource, target=1)

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
    results3 = network_analysis_res.analyse_single_target(
        settings=settings, data=data, target=1)

    # results of a network analysis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_single_target(
        settings=settings1, data=data, target=0)

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_single_target(settings1, data,
                                                       target=1)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtGaussianCMI_BTE_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=N_SAMPLES*10, n_replications=1)

    # Initialise analysis object and define settings
    network_analysis1 = BivariateTE()
    network_analysis2 = BivariateTE()
    network_analysis3 = BivariateTE()
    network_analysis4 = BivariateTE()

    # Settings without checkpointing.
    settings1 = {'cmi_estimator': 'JidtGaussianCMI',
                 'n_perm_max_stat': N_PERM,
                 'n_perm_min_stat': N_PERM,
                 'n_perm_max_seq': N_PERM,
                 'n_perm_omnibus': N_PERM,
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0
                 }

    # Settings with checkpointing.
    settings2 = {'cmi_estimator': 'JidtGaussianCMI',
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
    settings3 = {'cmi_estimator': 'JidtGaussianCMI',
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
    targets = [2, 3, 4]
    targets2 = [3, 4]

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

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtKraskovCMI_BTE_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data()
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
    targets = [2, 3, 4]
    targets2 = [3, 4]

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

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtDiscreteCMI_BTE_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test multivariate TE estimation from discrete data."""
    # Generate Gaussian test data
    covariance = 0.4
    n = N_SAMPLES*10
    delay = 1
    source = np.random.normal(0, 1, size=n)
    target = (covariance * source + (1 - covariance) *
              np.random.normal(0, 1, size=n))
    corr_expected = covariance / \
        (1 * np.sqrt(covariance**2 + (1-covariance)**2))
    expected_mi = calculate_mi(corr_expected)
    source = source[delay:]
    target = target[:-delay]

    # Discretise data
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 5}
    est = JidtDiscreteCMI(settings)
    source_dis, target_dis = est._discretise_vars(var1=source, var2=target)
    data = Data(np.vstack((source_dis, target_dis)),
                dim_order='ps', normalise=False)
    settings1 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1}

    settings2 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1,
        'write_ckp': True,
        'filename_ckp': filename_ckp1}

    settings3 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1,
        'write_ckp': True,
        'filename_ckp': filename_ckp2}

    network_analysis1 = BivariateTE()
    network_analysis2 = BivariateTE()
    network_analysis3 = BivariateTE()
    network_analysis4 = BivariateTE()

    # Starting Analysis of the Network
    # results of a network analysis without checkpointing
    results1 = network_analysis1.analyse_single_target(settings1, data,
                                                       target=1)

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_single_target(settings2, data,
                                                       target=1)

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    with open(filename_ckp1 + ".ckp", "r+") as f:
        tarsource = f.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources=tarsource, target=1)

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
    results3 = network_analysis_res.analyse_single_target(
        settings=settings, data=data, target=1)

    # results of a network analysis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_single_target(
        settings=settings1, data=data, target=0)

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_single_target(settings1, data,
                                                       target=1)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtGaussianCMI_BMI_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data()
    data.generate_mute_data(n_samples=N_SAMPLES, n_replications=1)

    # Initialise analysis object and define settings
    network_analysis1 = BivariateMI()
    network_analysis2 = BivariateMI()
    network_analysis3 = BivariateMI()
    network_analysis4 = BivariateMI()

    # Settings without checkpointing.
    settings1 = {'cmi_estimator': 'JidtGaussianCMI',
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0
                 }

    # Settings with checkpointing.
    settings2 = {'cmi_estimator': 'JidtGaussianCMI',
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp1}

    # Settings resuming from checkpoint.
    settings3 = {'cmi_estimator': 'JidtGaussianCMI',
                 'max_lag_sources': 3,
                 'min_lag_sources': 2,
                 'noise_level': 0,
                 'write_ckp': True,
                 'filename_ckp': filename_ckp2}

    # Setting sources and targets for the analysis
    sources = [0, 1]
    targets = [2, 3, 4]
    targets2 = [3, 4]

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
        tarsource = tarsource = f.readlines()
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

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtKraskovCMI_BMI_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    """Test running analysis without any checkpoint setting."""
    # Generate test data
    data = Data()
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

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_network(
        settings=settings1, data=data, targets=targets, sources=sources)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


def JidtDiscreteCMI_BMI_checktest():
    """ run test without checkpointing, with checkpointing without resume and checkpointing with resume to compare the results"""
    filename_ckp1 = os.path.join(
        os.path.dirname(__file__), 'data', 'run_checkpoint')
    filename_ckp2 = os.path.join(
        os.path.dirname(__file__), 'data', 'resume_checkpoint')

    # Generate Gaussian test data
    covariance = 0.4
    n = N_SAMPLES*10
    delay = 1
    source = np.random.normal(0, 1, size=n)
    target = (covariance * source + (1 - covariance) *
              np.random.normal(0, 1, size=n))
    corr_expected = covariance / \
        (1 * np.sqrt(covariance**2 + (1-covariance)**2))
    expected_mi = calculate_mi(corr_expected)
    source = source[delay:]
    target = target[:-delay]

    # Discretise data
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 5}
    est = JidtDiscreteCMI(settings)
    source_dis, target_dis = est._discretise_vars(var1=source, var2=target)
    data = Data(np.vstack((source_dis, target_dis)),
                dim_order='ps', normalise=False)
    settings1 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1}

    settings2 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1,
        'write_ckp': True,
        'filename_ckp': filename_ckp1}

    settings3 = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,
        'n_perm_max_stat': 21,
        'n_perm_omnibus': 30,
        'n_perm_max_seq': 30,
        'min_lag_sources': 2,
        'max_lag_sources': 2,
        'max_lag_target': 1,
        'write_ckp': True,
        'filename_ckp': filename_ckp2}

    # Initialise analysis object and define settings
    network_analysis1 = BivariateMI()
    network_analysis2 = BivariateMI()
    network_analysis3 = BivariateMI()
    network_analysis4 = BivariateMI()

    # Starting Analysis of the Network
    # results of a network analysis without checkpointing
    results1 = network_analysis1.analyse_single_target(settings1, data,
                                                       target=1)

    # results of a network analysis with checkpointing
    results2 = network_analysis2.analyse_single_target(settings2, data,
                                                       target=1)

    # Creating a checkpoint similar to the above settings where the targets of
    # of the first source have been already analyzed
    with open(filename_ckp1 + ".ckp", "r+") as f:
        tarsource = f.readlines()
    tarsource = tarsource[8]
    tarsource = (tarsource[:-1])

    network_analysis3._set_checkpointing_defaults(
        settings3, data, sources=tarsource, target=1)

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
    results3 = network_analysis_res.analyse_single_target(
        settings=settings, data=data, target=1)

    # results of a network analysis without checkpointing but other targets/sources
    results4 = network_analysis4.analyse_single_target(
        settings=settings1, data=data, target=0)

    # results of a network analysis just like results 1
    results5 = network_analysis1.analyse_single_target(settings1, data,
                                                       target=1)

    adj_matrix1 = results1.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix2 = results2.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix3 = results3.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix4 = results4.get_adjacency_matrix(weights='binary', fdr=False)
    adj_matrix5 = results5.get_adjacency_matrix(weights='binary', fdr=False)

    result1 = adj_matrix1.get_edge_list()
    result2 = adj_matrix2.get_edge_list()
    result3 = adj_matrix3.get_edge_list()
    result4 = adj_matrix4.get_edge_list()
    result5 = adj_matrix5.get_edge_list()

    print("Printing results:")
    print("Result 1: without checkpoint")
    print(result1)
    print("Result 2: with checkpoint")
    print(result2)
    print("Result 3: resuming from checkpoint")
    print(result3)
    print("Result 4: without checkpoint and different targets and sources")
    print(result4)
    print("Result 5: without checkpoint and same targets and sources")
    print(result5)

    print("Comparing the results:")
    assert np.array_equal(result1, result2), 'Result 1 and 2 not equal!'
    assert np.array_equal(result1, result3), 'Result 1 and 3 not equal!'
    assert not np.array_equal(result1, result4), 'Result 1 and 4 equal, expected to be different!'
    assert np.array_equal(result1, result5), 'Result 1 and 5 not equal!'

    cmp1 = list(set(result1).intersection(result2))
    cmp2 = list(set(result1).intersection(result3))
    cmp3 = list(set(result1).intersection(result4))
    cmp4 = list(set(result1).intersection(result5))
    len1 = len(result1)
    assert len(cmp1) == len1 and len(cmp2) == len1 and len(cmp4) == len1 and len(cmp3) != len1, (
        "Discrete MultivariateMI Running with checkpoints does not give the expected results")

    print("Final")
    print("Elements of comparison between result 1 and 2")
    print(cmp1)
    print("Elements of comparison between result 1 and 3")
    print(cmp2)
    print("Elements of comparison between result 1 and 4")
    print(cmp3)
    print("Elements of comparison between result 1 and 5")
    print(cmp4)

    _clear_ckp(filename_ckp1)
    _clear_ckp(filename_ckp2)


if __name__ == '__main__':
    test_checkpoint_defaults()
    test_checkpoint_copy()
    test_checkpoint_resume()

    JidtKraskovCMI_MMI_checktest()
    JidtKraskovCMI_MTE_checktest()
    JidtKraskovCMI_BMI_checktest()  # passed
    JidtKraskovCMI_BTE_checktest()  # passed

    JidtDiscreteCMI_MMI_checktest()
    JidtDiscreteCMI_MTE_checktest()
    JidtDiscreteCMI_BMI_checktest()  # passed
    JidtDiscreteCMI_BTE_checktest()  # passed

    JidtGaussianCMI_MMI_checktest()
    JidtGaussianCMI_MTE_checktest()
    JidtGaussianCMI_BMI_checktest()  # passed
    JidtGaussianCMI_BTE_checktest()  # passed
